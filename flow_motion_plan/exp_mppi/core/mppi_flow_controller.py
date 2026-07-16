import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Callable
import time

from .energy_cost import EnergyCostFunction

class MPPIFlowController:
    def __init__(self, flow_model, config: dict):
        self.flow_model = flow_model
        self.config = config
        self.device = next(flow_model.parameters()).device

        self.horizon = config.get('horizon', 40)
        self.num_samples = config.get('num_samples', 1000)
        self.lambda_ = float(config.get('lambda', 0.1))
        self.dt = config.get('dt', 0.05)
        self.num_iterations = config.get('num_iterations', 30)

        self.noise_sigma = float(config.get('noise_sigma', 0.3))

        self.cost_weights = config.get('cost_weights', {})
        self.sigma = config.get('sigma', 2.0)
        self.use_perturbation_cost = config.get('use_perturbation_cost', True)

        self.maze_min = float(config.get('maze_min', 0.0))
        self.maze_max = float(config.get('maze_max', 5.0))
        self.perturbation_scale = float(config.get('perturbation_scale', 1.0))

        self.deviation_weight = float(config.get('deviation_weight', 5.0))

        self.gcar_follow_weight = float(config.get('gcar_follow_weight', 30.0))

        self.smoothing_kernel_size = config.get('smoothing_kernel_size', 5)
        self.smoothing_sigma = config.get('smoothing_sigma', 1.0)
        self.smoothing_kernel = self._create_smoothing_kernel(
            self.smoothing_kernel_size, self.smoothing_sigma
        )

        print(f"✅ MPPI控制器 (Fixed: [B,H,4] Output) 初始化完成")

    def _create_smoothing_kernel(self, kernel_size: int, sigma: float):
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1).repeat(2, 1, 1)

    def _get_base_traj(self, start_pos, goal_pos, walls):
        from experiments.utils.inference import FlowModelInference

        inferencer = FlowModelInference(None, None)
        inferencer.model = self.flow_model
        inferencer.horizon = self.horizon
        inferencer.device = self.device

        trajs, _ = inferencer.generate_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            wall_positions=walls,
            num_steps=20,
            dt=self.dt,
            num_samples=1,
            record_steps=False
        )

        return torch.tensor(trajs[0], dtype=torch.float32, device=self.device)

    def _rollout_trajectories(self,
                             start_pos: torch.Tensor,
                             base_traj: torch.Tensor,
                             perturbations: torch.Tensor) -> torch.Tensor:
        is_batch = perturbations.ndim == 3
        if not is_batch:
            perturbations = perturbations.unsqueeze(0)
        K = perturbations.shape[0]

        base = base_traj.unsqueeze(0).expand(K, -1, -1)

        deviation = torch.cumsum(perturbations * self.perturbation_scale * self.dt, dim=1)

        H = deviation.shape[1]
        ramp = torch.linspace(0.0, 1.0, H, device=deviation.device).view(1, H, 1)
        deviation = deviation - ramp * deviation[:, -1:, :]
        trajs = base + deviation

        trajs = trajs.clone()
        trajs[:, 0, :] = start_pos.view(1, 2)

        trajs = torch.clamp(trajs, self.maze_min, self.maze_max)

        return trajs if is_batch else trajs.squeeze(0)

    def _compute_mppi_weights(self, costs: torch.Tensor) -> torch.Tensor:
        beta = torch.min(costs)
        if self.lambda_ < 1e-6:
            norm_costs = torch.zeros_like(costs)
            norm_costs[costs > beta + 1e-6] = 1e8
        else:
            norm_costs = (costs - beta) / self.lambda_
        weights = torch.softmax(-norm_costs, dim=0)
        return weights

    def generate_trajectory(self, start_pos, goal_pos, walls, base_traj=None, gcar_traj=None, **kwargs):

        start_tensor = torch.tensor(start_pos, dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor(goal_pos, dtype=torch.float32, device=self.device)
        walls_tensor = torch.tensor(walls, dtype=torch.float32, device=self.device)

        if base_traj is not None:

            print("  [MPPI] 使用 g^car 修正轨迹作为基准流场 (MPPI + g^car)...")
            base_traj_t = torch.as_tensor(base_traj, dtype=torch.float32, device=self.device).view(self.horizon, 2)
        else:
            print("  [MPPI] 计算基准流场 (Base Flow)...")
            base_traj_t = self._get_base_traj(start_pos, goal_pos, walls)

        gcar_traj_t = None
        if gcar_traj is not None:
            gcar_traj_t = torch.as_tensor(gcar_traj, dtype=torch.float32, device=self.device).view(self.horizon, 2)

        mean_perturbation = torch.zeros(self.horizon, 2, device=self.device)

        cost_fn = EnergyCostFunction(
            goal_pos=goal_tensor, walls=walls_tensor,
            energy_centers=kwargs.get('energy_centers'),
            energy_scales=kwargs.get('energy_scales'),
            horizon=self.horizon, weights=self.cost_weights,
            maze_min=0.0, maze_max=5.0
        )

        print(f"  [MPPI] 优化流场扰动 (Optimizing Perturbations)...")

        for i in range(self.num_iterations):

            noise = torch.randn(self.num_samples, self.horizon, 2, device=self.device) * self.noise_sigma

            noise_perm = noise.permute(0, 2, 1)
            pad = self.smoothing_kernel_size // 2
            noise_padded = F.pad(noise_perm, (pad, pad), mode='replicate')
            smooth_noise = F.conv1d(noise_padded, self.smoothing_kernel, groups=2).permute(0, 2, 1)

            sampled_perts = mean_perturbation.unsqueeze(0) + smooth_noise

            sampled_trajs = self._rollout_trajectories(start_tensor, base_traj_t, sampled_perts)

            costs = cost_fn.compute_total_cost(sampled_trajs)

            if self.deviation_weight > 0:
                dev = ((sampled_trajs - base_traj_t.unsqueeze(0)) ** 2).sum(dim=(1, 2))
                costs = costs + self.deviation_weight * dev

            if gcar_traj_t is not None and self.gcar_follow_weight > 0:
                gfollow = ((sampled_trajs - gcar_traj_t.unsqueeze(0)) ** 2).sum(dim=(1, 2))
                costs = costs + self.gcar_follow_weight * gfollow

            if self.use_perturbation_cost:
                pert_cost = (self.lambda_ / (2.0 * self.noise_sigma ** 2)) * torch.sum(smooth_noise ** 2, dim=(1, 2))
                costs += pert_cost

            weights = self._compute_mppi_weights(costs)
            weighted_noise = torch.sum(weights.view(-1, 1, 1) * smooth_noise, dim=0)
            mean_perturbation = mean_perturbation + weighted_noise

            if (i+1) % 10 == 0:
                print(f"      Iter {i+1}: Min Cost = {torch.min(costs).item():.4f}")

        final_traj = self._rollout_trajectories(start_tensor, base_traj_t, mean_perturbation)
        return final_traj.cpu().numpy()
