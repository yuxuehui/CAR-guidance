import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List

from .base_guidance import BaseGuidance

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from diffuser.models.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except ImportError as e:
    print(f"⚠️  无法导入ConditionalUnet1D: {e}")
    ConditionalUnet1D = None

def _smootherstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

def _smootherstep_gate(conflict: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    conflict_norm = (conflict - (threshold - temperature)) / (2.0 * temperature + 1e-8)
    return _smootherstep(conflict_norm)

def make_guidance_fn(base_guidance: BaseGuidance, flow_model):
    def guidance_fn(x, t, v_uncond, conditions, wall_locations):

        conditions_clean = conditions
        if conditions is not None:
            if isinstance(conditions, dict):
                conditions_clean = {}
                for k, v in conditions.items():
                    if isinstance(v, torch.Tensor):
                        if hasattr(v, 'is_sparse') and v.is_sparse:
                            conditions_clean[k] = v.to_dense()
                        else:
                            conditions_clean[k] = v.contiguous() if v.is_contiguous() is False else v
                    else:
                        conditions_clean[k] = v
            elif isinstance(conditions, torch.Tensor):
                if hasattr(conditions, 'is_sparse') and conditions.is_sparse:
                    conditions_clean = conditions.to_dense()
                else:
                    conditions_clean = conditions.contiguous() if conditions.is_contiguous() is False else conditions

        wall_locations_clean = wall_locations
        if wall_locations is not None and isinstance(wall_locations, torch.Tensor):
            if hasattr(wall_locations, 'is_sparse') and wall_locations.is_sparse:
                wall_locations_clean = wall_locations.to_dense()
            else:
                wall_locations_clean = wall_locations.contiguous() if wall_locations.is_contiguous() is False else wall_locations

        guided_velocity = base_guidance(x, t, v_uncond, conditions_clean, wall_locations_clean)

        guidance = guided_velocity - v_uncond
        return guidance

    guidance_fn.base_guidance = base_guidance

    return guidance_fn

class GCovWrapper(nn.Module):

    def __init__(
        self,
        flow_model,
        base_guidance_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Optional[torch.Tensor]], torch.Tensor],
        horizon: int = 40,
        device: str = 'cuda',
        config: Optional[Dict[str, Any]] = None,
        traj_normalizer=None,
        conflict_compute_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        energy_centers: Optional[List[List[float]]] = None,
        energy_scales: Optional[List[float]] = None,
    ):
        super().__init__()

        if ConditionalUnet1D is None:
            raise ImportError("无法导入ConditionalUnet1D，请检查依赖")

        self.flow_model = flow_model
        self.base_guidance_fn = base_guidance_fn
        self.horizon = horizon
        self.device = device
        self.config = config or {}
        self.traj_normalizer = traj_normalizer
        self.conflict_compute_fn = conflict_compute_fn
        self.energy_centers = energy_centers
        self.energy_scales = energy_scales

        self.is_repulsion = True
        if energy_scales is not None and len(energy_scales) > 0:

            positive_count = sum(1 for s in energy_scales if s > 0)
            self.is_repulsion = positive_count < len(energy_scales) / 2

        reward_type = self.config.get('reward_type', None)
        if reward_type == 'repulsion':
            self.is_repulsion = True
        elif reward_type == 'attraction':
            self.is_repulsion = False

        self.goal_pos = None

        self.global_cond_dim = int(self.config.get("global_cond_dim", 16))

        self.learned_guidance_model = self._build_learned_guidance_model()

        self.learned_guidance_model = self.learned_guidance_model.to(self.device)

        self.step_data = []

    def _build_learned_guidance_model(self):
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')
        model_channels = int(self.config.get("residual_unet_channels", 64))
        down_dims = self.config.get("residual_unet_down_dims", [256, 512, 1024])

        if online_loss_type == 'gradient':

            return self._build_vector_residual_model(model_channels, down_dims)
        else:

            return self._build_scalar_residual_model(model_channels, down_dims)

    def _build_vector_residual_model(self, model_channels, down_dims):
        class UNetVectorResidual(nn.Module):
            def __init__(self, horizon, in_dim, model_channels, down_dims, global_cond_dim: int):
                super().__init__()
                self.horizon = horizon
                self.in_proj = nn.Linear(in_dim, model_channels)
                self.unet = ConditionalUnet1D(
                    input_dim=model_channels,
                    global_cond_dim=global_cond_dim,
                    down_dims=down_dims,
                    diffusion_step_embed_dim=256,
                )
                self.out_proj = nn.Linear(model_channels, in_dim)

            def forward(self, x_pos, t, global_cond=None):
                x_feat = self.in_proj(x_pos)
                y = self.unet(sample=x_feat, timestep=t, global_cond=global_cond)
                return self.out_proj(y)

        return UNetVectorResidual(
            horizon=self.horizon,
            in_dim=2,
            model_channels=model_channels,
            down_dims=down_dims,
            global_cond_dim=self.global_cond_dim,
        )

    def _build_scalar_residual_model(self, model_channels, down_dims):
        class UNetScalarResidual(nn.Module):
            def __init__(self, horizon, in_dim, model_channels, down_dims, global_cond_dim: int):
                super().__init__()
                self.horizon = horizon
                self.in_proj = nn.Linear(in_dim, model_channels)
                self.unet = ConditionalUnet1D(
                    input_dim=model_channels,
                    global_cond_dim=global_cond_dim,
                    down_dims=down_dims,
                    diffusion_step_embed_dim=256,
                )
                self.out_proj = nn.Linear(model_channels, 1)

            def forward(self, x_pos, t, global_cond=None):
                x_feat = self.in_proj(x_pos)
                y = self.unet(sample=x_feat, timestep=t, global_cond=global_cond)

                g_seq = self.out_proj(y).squeeze(-1)
                return g_seq.sum(dim=1)

        return UNetScalarResidual(
            horizon=self.horizon,
            in_dim=2,
            model_channels=model_channels,
            down_dims=down_dims,
            global_cond_dim=self.global_cond_dim,
        )

    def _build_global_cond(self, conditions, wall_locations, batch_size: int, device):
        dtype = torch.float32

        start = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        goal = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        if isinstance(conditions, dict) and conditions is not None:
            s = conditions.get(0, None)
            g = conditions.get(self.horizon - 1, None)
            if isinstance(s, torch.Tensor) and s.numel() > 0:
                s_t = s.to(device=device, dtype=dtype)
                if s_t.shape[0] != batch_size and batch_size % s_t.shape[0] == 0:
                    s_t = s_t.repeat_interleave(batch_size // s_t.shape[0], dim=0)
                start = s_t.view(batch_size, -1)[:, :2]
            if isinstance(g, torch.Tensor) and g.numel() > 0:
                g_t = g.to(device=device, dtype=dtype)
                if g_t.shape[0] != batch_size and batch_size % g_t.shape[0] == 0:
                    g_t = g_t.repeat_interleave(batch_size // g_t.shape[0], dim=0)
                goal = g_t.view(batch_size, -1)[:, :2]

        wall_flat = None
        if isinstance(wall_locations, torch.Tensor) and wall_locations.numel() > 0:
            wl = wall_locations.to(device=device, dtype=dtype)
            if wl.dim() == 2:
                wl = wl.unsqueeze(0)
            if wl.shape[0] != batch_size:
                if batch_size % wl.shape[0] == 0:
                    wl = wl.repeat_interleave(batch_size // wl.shape[0], dim=0)
                else:
                    raise ValueError(
                        f"wall_locations batch ({wl.shape[0]}) 与 batch_size ({batch_size}) 不匹配"
                    )
            wall_flat = wl.reshape(batch_size, -1)
        else:
            wall_flat = torch.zeros(batch_size, max(self.global_cond_dim - 4, 0), device=device, dtype=dtype)

        global_cond = torch.cat([start, goal, wall_flat], dim=-1)
        if global_cond.shape[-1] != self.global_cond_dim:
            raise ValueError(
                f"global_cond_dim 不匹配：得到 {global_cond.shape[-1]}，期望 {self.global_cond_dim}。"
                f"（请检查 wall_locations 展平长度以及 config.global_cond_dim）"
            )

        return global_cond

    def _extract_position(self, x):
        return x[:, :, :2]

    def _expand_to_full_state(self, pos_guidance):
        batch_size, horizon, _ = pos_guidance.shape
        full_guidance = torch.zeros(batch_size, horizon, 4, device=pos_guidance.device, dtype=pos_guidance.dtype)
        full_guidance[:, :, :2] = pos_guidance
        return full_guidance

    def _compute_conflict_score(self, x_pos, base_guidance_pos):
        if self.conflict_compute_fn is not None:

            return self.conflict_compute_fn(x_pos)

        if self.energy_centers is None or len(self.energy_centers) < 2:
            return torch.zeros(base_guidance_pos.shape[0], base_guidance_pos.shape[1], device=base_guidance_pos.device, dtype=base_guidance_pos.dtype)

        device = base_guidance_pos.device

        energy_centers = torch.tensor(self.energy_centers, dtype=torch.float32, device=device)
        if self.energy_scales is None or len(self.energy_scales) != len(self.energy_centers):
            energy_scales = torch.ones(len(self.energy_centers), dtype=torch.float32, device=device)
        else:
            energy_scales = torch.tensor(self.energy_scales, dtype=torch.float32, device=device)

        num_energy = energy_centers.shape[0]
        sigma = self.config.get('sigma', 2.0)

        if num_energy < 2:
            return torch.zeros(base_guidance_pos.shape[0], base_guidance_pos.shape[1], device=device, dtype=base_guidance_pos.dtype)

        B, H, D = x_pos.shape
        x_flat = x_pos.reshape(-1, D)

        diff_all = x_flat.unsqueeze(1) - energy_centers.unsqueeze(0)
        sq_dist_all = (diff_all ** 2).sum(dim=-1, keepdim=True)

        energy_all = torch.exp(-sq_dist_all / (sigma ** 2 + 1e-8))

        dist_all = torch.sqrt(sq_dist_all + 1e-8)
        dir_to_center = -diff_all / dist_all

        scales_expanded = energy_scales.view(1, -1, 1)
        grads_all = scales_expanded * energy_all * dir_to_center

        grads_all = grads_all.transpose(0, 1)

        grad_norms = torch.norm(grads_all, dim=-1, keepdim=True)
        grads_normalized = grads_all / (grad_norms + 1e-8)

        grads_expanded_i = grads_normalized.unsqueeze(1)
        grads_expanded_j = grads_normalized.unsqueeze(0)
        cos_sim = (grads_expanded_i * grads_expanded_j).sum(dim=-1)

        mask = torch.triu(torch.ones(num_energy, num_energy, device=device), diagonal=1)
        mask = mask.unsqueeze(-1)

        zero_thr = float(self.config.get('zero_gradient_threshold', 1e-2))
        norms = grad_norms.squeeze(-1)
        valid = (norms >= zero_thr).float()
        pair_valid = valid.unsqueeze(1) * valid.unsqueeze(0)
        mask = mask * pair_valid

        conflict_all = (1.0 - cos_sim) / 2.0 * mask

        num_pairs = mask.sum(dim=(0, 1))
        conflict_flat = conflict_all.sum(dim=(0, 1)) / (num_pairs + 1e-8)

        conflict = conflict_flat.reshape(B, H)

        return conflict

    def _compute_base_guidance(self, x, t, v_uncond, conditions, wall_locations):

        return self.base_guidance_fn(x, t, v_uncond, conditions, wall_locations)

    def _compute_learned_correction(self, x_pos, t_tensor, conditions=None, wall_locations=None):
        if self.learned_guidance_model is None:
            return torch.zeros_like(x_pos)

        x_pos = x_pos.to(self.device)
        t_tensor = t_tensor.to(self.device)

        online_loss_type = self.config.get('online_loss_type', 'mse_simple')

        global_cond = self._build_global_cond(
            conditions=conditions,
            wall_locations=wall_locations,
            batch_size=x_pos.shape[0],
            device=self.device,
        )

        if online_loss_type == 'gradient':

            with torch.no_grad():
                learned_correction_pos = self.learned_guidance_model(x_pos, t_tensor, global_cond)
        else:

            param_requires_grad = [p.requires_grad for p in self.learned_guidance_model.parameters()]
            try:
                for p in self.learned_guidance_model.parameters():
                    p.requires_grad_(False)
                with torch.enable_grad():
                    x_pos_req = x_pos.detach().requires_grad_(True)
                    g = self.learned_guidance_model(x_pos_req, t_tensor, global_cond)
                    grad_flat = torch.autograd.grad(g.sum(), x_pos_req, create_graph=False, retain_graph=False)[0]
                    learned_correction_pos = grad_flat.detach()
            finally:
                for p, rg in zip(self.learned_guidance_model.parameters(), param_requires_grad):
                    p.requires_grad_(rg)

        return learned_correction_pos

    def forward(self, x, t, conditions=None, wall_locations=None, record_step=False):

        conditions_processed = conditions
        if conditions is not None:

            if isinstance(conditions, dict):
                conditions_processed = {}
                for k, v in conditions.items():

                    if isinstance(v, torch.Tensor):
                        if hasattr(v, 'is_sparse') and v.is_sparse:
                            conditions_processed[k] = v.to_dense()
                        else:

                            conditions_processed[k] = v.contiguous() if v.is_contiguous() is False else v
                    else:
                        conditions_processed[k] = v
            elif isinstance(conditions, torch.Tensor):

                if hasattr(conditions, 'is_sparse') and conditions.is_sparse:
                    conditions_processed = conditions.to_dense()
                else:
                    conditions_processed = conditions.contiguous() if conditions.is_contiguous() is False else conditions

        wall_locations_processed = wall_locations
        if wall_locations is not None:
            if isinstance(wall_locations, torch.Tensor):
                if hasattr(wall_locations, 'is_sparse') and wall_locations.is_sparse:
                    wall_locations_processed = wall_locations.to_dense()
                else:
                    wall_locations_processed = wall_locations.contiguous() if wall_locations.is_contiguous() is False else wall_locations

        if conditions_processed is not None and isinstance(conditions_processed, dict):
            conditions_final = {}
            for k, v in conditions_processed.items():
                if isinstance(v, torch.Tensor):

                    v_clean = torch.from_numpy(v.cpu().detach().numpy()).to(v.device, dtype=v.dtype)
                    conditions_final[k] = v_clean
                else:
                    conditions_final[k] = v
            conditions_processed = conditions_final

        if wall_locations_processed is not None and isinstance(wall_locations_processed, torch.Tensor):

            wall_locations_processed = torch.from_numpy(
                wall_locations_processed.cpu().detach().numpy()
            ).to(wall_locations_processed.device, dtype=wall_locations_processed.dtype)

        v_uncond = self.flow_model.velocity_field(x, t, conditions_processed, wall_locations_processed)

        if record_step:
            if not hasattr(self, 'step_data'):
                self.step_data = []

            if hasattr(self.base_guidance_fn, 'base_guidance'):
                base_guidance_obj = self.base_guidance_fn.base_guidance
                if hasattr(base_guidance_obj, 'config'):
                    base_guidance_obj.config['record_step'] = True

        g_total = self._compute_guidance_for_rollout(x, t, v_uncond, conditions_processed, wall_locations_processed)
        base_guidance = self._compute_base_guidance(x, t, v_uncond, conditions_processed, wall_locations_processed)
        learned_correction = g_total - base_guidance
        g_total = base_guidance + learned_correction

        if record_step:

            base_guidance = self._compute_base_guidance(x, t, v_uncond, conditions_processed, wall_locations_processed)

            individual_grads = []
            if hasattr(self.base_guidance_fn, 'base_guidance'):
                base_guidance_obj = self.base_guidance_fn.base_guidance
                if hasattr(base_guidance_obj, '_last_individual_grads'):
                    individual_grads = base_guidance_obj._last_individual_grads

            step_info = {
                'trajectory': x.detach().cpu().numpy(),
                'v_uncond': v_uncond.detach().cpu().numpy(),
                'guidance_grad': g_total.detach().cpu().numpy(),
                'base_guidance': base_guidance.detach().cpu().numpy(),
                't': t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t,
                'individual_grads': individual_grads,
            }
            self.step_data.append(step_info)

        return v_uncond + g_total

    def _compute_guidance_for_rollout(self, x, t, v_uncond, conditions=None, wall_locations=None):

        conditions_clean = conditions
        if conditions is not None:
            if isinstance(conditions, dict):
                conditions_clean = {}
                for k, v in conditions.items():
                    if isinstance(v, torch.Tensor):
                        if hasattr(v, 'is_sparse') and v.is_sparse:
                            conditions_clean[k] = v.to_dense()
                        else:
                            conditions_clean[k] = v.contiguous() if v.is_contiguous() is False else v
                    else:
                        conditions_clean[k] = v
            elif isinstance(conditions, torch.Tensor):
                if hasattr(conditions, 'is_sparse') and conditions.is_sparse:
                    conditions_clean = conditions.to_dense()
                else:
                    conditions_clean = conditions.contiguous() if conditions.is_contiguous() is False else conditions

        wall_locations_clean = wall_locations
        if wall_locations is not None and isinstance(wall_locations, torch.Tensor):
            if hasattr(wall_locations, 'is_sparse') and wall_locations.is_sparse:
                wall_locations_clean = wall_locations.to_dense()
            else:
                wall_locations_clean = wall_locations.contiguous() if wall_locations.is_contiguous() is False else wall_locations

        base_guidance = self._compute_base_guidance(x, t, v_uncond, conditions_clean, wall_locations_clean)
        base_guidance_pos = self._extract_position(base_guidance)

        x_pos = self._extract_position(x)

        x_pos = x_pos.to(self.device)

        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_tensor = t.unsqueeze(0).expand(x.shape[0]).to(device=self.device)
            elif t.dim() == 1:
                t_tensor = t.view(-1).to(device=self.device)
            else:
                t_tensor = t.view(-1)[0].expand(x.shape[0]).to(device=self.device)
        else:
            t_tensor = torch.full((x.shape[0],), float(t), device=self.device, dtype=torch.float32)

        learned_correction_pos = self._compute_learned_correction(
            x_pos, t_tensor, conditions_clean, wall_locations_clean
        )

        conflict_score = self._compute_conflict_score(x_pos, base_guidance_pos)
        conflict_threshold = float(self.config.get('conflict_threshold', 0.5))
        conflict_temperature = float(self.config.get('conflict_temperature', 0.1))
        conflict_weight = _smootherstep_gate(conflict_score, conflict_threshold, conflict_temperature)

        learned_correction_pos = learned_correction_pos * conflict_weight.unsqueeze(-1)
        learned_correction = self._expand_to_full_state(learned_correction_pos)

        g_total = base_guidance + learned_correction

        return g_total

    def _compute_terminal_reward(self, x1_pos):
        with torch.no_grad():
            if self.is_repulsion:

                return self._compute_terminal_reward_repulsion(x1_pos)
            else:

                return self._compute_terminal_reward_attraction(x1_pos)

    def _compute_terminal_reward_repulsion(self, x1_pos):
        if self.energy_centers is None or len(self.energy_centers) == 0:
            return torch.zeros(x1_pos.shape[0], device=x1_pos.device)

        if self.traj_normalizer is not None:
            x1_pos_np = x1_pos.cpu().numpy()
            x1_pos_unnorm_np = self.traj_normalizer.unnormalize(x1_pos_np)
            x1_pos_unnorm = torch.from_numpy(x1_pos_unnorm_np).to(x1_pos.device).float()
        else:
            x1_pos_unnorm = x1_pos

        sigma = float(self.config.get('sigma', 2.0))

        centers = torch.tensor(self.energy_centers, dtype=torch.float32, device=x1_pos.device)
        scales_abs = torch.abs(torch.tensor(self.energy_scales, dtype=torch.float32, device=x1_pos.device))

        diff = x1_pos_unnorm.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)

        energy_all = torch.exp(-sq_dist / (sigma ** 2 + 1e-8))

        energy_per_center = energy_all.mean(dim=1)

        total_energy = (energy_per_center * scales_abs.unsqueeze(0)).sum(dim=-1)

        r1 = -total_energy

        return r1

    def _compute_terminal_reward_attraction(self, x1_pos):
        if self.energy_centers is None or len(self.energy_centers) == 0:

            return -x1_pos[:, -1, :].norm(dim=-1)

        if self.traj_normalizer is not None:
            x1_pos_np = x1_pos.cpu().numpy()
            x1_pos_unnorm_np = self.traj_normalizer.unnormalize(x1_pos_np)
            x1_pos_unnorm = torch.from_numpy(x1_pos_unnorm_np).to(x1_pos.device).float()
        else:
            x1_pos_unnorm = x1_pos

        sigma = float(self.config.get('sigma', 2.0))

        centers = torch.tensor(self.energy_centers, dtype=torch.float32, device=x1_pos.device)
        scales_abs = torch.abs(torch.tensor(self.energy_scales, dtype=torch.float32, device=x1_pos.device))

        diff = x1_pos_unnorm.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=-1)

        energy_all = torch.exp(-sq_dist / (sigma ** 2 + 1e-8))

        energy_per_center = energy_all.mean(dim=1)

        total_energy = (energy_per_center * scales_abs.unsqueeze(0)).sum(dim=-1)

        r1 = total_energy

        return r1

    def _scalar_predict(self, x_pos, t_tensor, conditions=None, wall_locations=None):
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')
        if online_loss_type == 'gradient':

            raise ValueError("gradient模式应该使用向量场模型")
        else:

            global_cond = self._build_global_cond(
                conditions=conditions,
                wall_locations=wall_locations,
                batch_size=x_pos.shape[0],
                device=x_pos.device,
            )
            return self.learned_guidance_model(x_pos, t_tensor, global_cond)

    def train_model(self, conditions, wall_locations):
        if not self.config.get('train_online', True):
            return

        steps = int(self.config.get('online_train_steps', 1000))
        batch_size = int(self.config.get('online_batch_size', 4))
        lr = float(self.config.get('online_lr', 1e-4))
        num_ode_steps = int(self.config.get('num_ode_steps', 20))
        dt = 1.0 / num_ode_steps
        conflict_threshold = float(self.config.get('conflict_threshold', 0.5))
        conflict_temperature = float(self.config.get('conflict_temperature', 0.1))

        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()

        print(f"[GCovWrapper] 开始在线训练 ({steps} 步)")
        print(f"  奖励类型: {'排斥场（远离能量中心）' if self.is_repulsion else '吸引场（到达终点）'}")

        train_conditions = None
        train_wall_locations = None

        if conditions is not None:
            base_start = conditions.get(0, None)
            base_goal = conditions.get(self.horizon - 1, None)

            if base_goal is not None and not self.is_repulsion:

                if self.traj_normalizer is not None:
                    goal_norm = base_goal[0:1].cpu().numpy()
                    goal_unnorm = self.traj_normalizer.unnormalize(goal_norm.reshape(1, 1, 2))[0, 0]
                    self.goal_pos = goal_unnorm.tolist()
                else:
                    self.goal_pos = base_goal[0].cpu().numpy().tolist()

            if base_start is not None and base_goal is not None:

                if base_start.is_sparse:
                    base_start = base_start.to_dense()
                if base_goal.is_sparse:
                    base_goal = base_goal.to_dense()

                base_start = base_start.to(self.device)[0:1]
                base_goal = base_goal.to(self.device)[0:1]
                start_rep = base_start.repeat(batch_size, 1)
                goal_rep = base_goal.repeat(batch_size, 1)

                train_conditions = {
                    0: start_rep,
                    self.horizon - 1: goal_rep
                }

        if wall_locations is not None:
            wall_locations = wall_locations.to(self.device)
            if wall_locations.dim() == 3 and wall_locations.size(0) == 1:
                train_wall_locations = wall_locations.repeat(batch_size, 1, 1)
            else:
                train_wall_locations = wall_locations

        try:
            from tqdm import tqdm
            use_tqdm = True
            progress_bar = tqdm(range(steps), desc="Training Online Guidance", unit="step", ncols=100)
        except ImportError:
            use_tqdm = False
            progress_bar = range(steps)

        for step in progress_bar:

            x0 = torch.randn(batch_size, self.horizon, 4, device=self.device)

            if train_conditions is not None:
                for timestep, condition in train_conditions.items():
                    if 0 <= timestep < self.horizon:
                        x0[:, timestep, :2] = condition[:batch_size, :2]

            traj_xs = []
            traj_ts = []
            curr_x = x0

            for t_step in range(num_ode_steps):
                t_val = t_step * dt
                t_tensor = torch.full((batch_size,), t_val, device=self.device)

                traj_xs.append(curr_x.clone())
                traj_ts.append(t_tensor)

                cond_to_use = train_conditions if train_conditions is not None else conditions
                walls_to_use = train_wall_locations if train_wall_locations is not None else wall_locations

                if cond_to_use is not None:

                    if isinstance(cond_to_use, dict):
                        cond_processed = {}
                        for k, v in cond_to_use.items():
                            if hasattr(v, 'is_sparse') and v.is_sparse:
                                cond_processed[k] = v.to_dense()
                            else:
                                cond_processed[k] = v
                        cond_to_use = cond_processed
                    elif isinstance(cond_to_use, torch.Tensor):

                        if hasattr(cond_to_use, 'is_sparse') and cond_to_use.is_sparse:
                            cond_to_use = cond_to_use.to_dense()

                if walls_to_use is not None and hasattr(walls_to_use, 'is_sparse') and walls_to_use.is_sparse:
                    walls_to_use = walls_to_use.to_dense()

                with torch.no_grad():
                    v_uncond = self.flow_model.velocity_field(
                        curr_x,
                        t_tensor,
                        cond_to_use,
                        walls_to_use,
                    )

                    g_total = self._compute_guidance_for_rollout(curr_x, t_tensor, v_uncond,
                                                                 cond_to_use, walls_to_use)
                    curr_x = curr_x + (v_uncond + g_total) * dt

                if train_conditions is not None:
                    for timestep, condition in train_conditions.items():
                        if 0 <= timestep < self.horizon:
                            curr_x[:, timestep, :2] = condition[:batch_size, :2]

            traj_xs.append(curr_x.clone())
            traj_ts.append(torch.full((batch_size,), num_ode_steps * dt, device=self.device))

            xs_stacked = torch.stack(traj_xs, dim=0)
            xs_pos_stacked = xs_stacked[:, :, :, :2]

            conflict_score = self._compute_conflict_score_from_base_guidance(xs_stacked[:-1], torch.stack(traj_ts, dim=0)[:-1],
                                                                           train_conditions, train_wall_locations)
            conflict_mask = _smootherstep_gate(conflict_score, conflict_threshold, conflict_temperature)

            active_ratio = (conflict_mask > 0.5).float().mean().item()
            if active_ratio < 1e-6:
                if use_tqdm:
                    progress_bar.set_postfix({'loss': 'skipped', 'conflict': f'{active_ratio:.1%}'})
                continue

            loss = self._compute_online_loss(
                xs_stacked,
                torch.stack(traj_ts, dim=0),
                conflict_mask,
                batch_size,
                num_ode_steps,
                train_conditions if train_conditions is not None else conditions,
                train_wall_locations if train_wall_locations is not None else wall_locations
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.learned_guidance_model.parameters(), max_norm=1.0)
            optimizer.step()

            if use_tqdm:
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}', 'conflict': f'{active_ratio:.1%}'})

            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{steps} | Loss: {loss.item():.6f} | Active Conflict: {active_ratio:.1%}")

        if use_tqdm:
            progress_bar.close()

        self.learned_guidance_model.eval()
        print("[GCovWrapper] 在线训练完成")

    def _compute_conflict_score_from_base_guidance(self, xs_stacked, ts_stacked, conditions, wall_locations):

        xs_pos_stacked = xs_stacked[:, :, :, :2]

        if self.conflict_compute_fn is not None:
            conflict_score = self.conflict_compute_fn(xs_pos_stacked)
            return conflict_score

        if self.energy_centers is None or len(self.energy_centers) < 2:

            T, B, H, _ = xs_stacked.shape
            return torch.zeros(T, B, H, device=xs_stacked.device, dtype=xs_stacked.dtype)

        device = xs_stacked.device
        energy_centers = torch.tensor(self.energy_centers, dtype=torch.float32, device=device)
        if self.energy_scales is None or len(self.energy_scales) != len(self.energy_centers):

            energy_scales = torch.ones(len(self.energy_centers), dtype=torch.float32, device=device)
        else:
            energy_scales = torch.tensor(self.energy_scales, dtype=torch.float32, device=device)

        num_energy = energy_centers.shape[0]
        sigma = self.config.get('sigma', 2.0)

        T, B, H, D = xs_pos_stacked.shape
        x_flat = xs_pos_stacked.reshape(-1, D)

        if num_energy < 2:
            return torch.zeros(T, B, H, device=device, dtype=xs_stacked.dtype)

        diff_all = x_flat.unsqueeze(1) - energy_centers.unsqueeze(0)
        sq_dist_all = (diff_all ** 2).sum(dim=-1, keepdim=True)

        energy_all = torch.exp(-sq_dist_all / (sigma ** 2 + 1e-8))

        dist_all = torch.sqrt(sq_dist_all + 1e-8)
        dir_to_center = -diff_all / dist_all

        scales_expanded = energy_scales.view(1, -1, 1)
        grads_all = scales_expanded * energy_all * dir_to_center

        grads_all = grads_all.transpose(0, 1)

        grad_norms = torch.norm(grads_all, dim=-1, keepdim=True)
        grads_normalized = grads_all / (grad_norms + 1e-8)

        grads_expanded_i = grads_normalized.unsqueeze(1)
        grads_expanded_j = grads_normalized.unsqueeze(0)
        cos_sim = (grads_expanded_i * grads_expanded_j).sum(dim=-1)

        mask = torch.triu(torch.ones(num_energy, num_energy, device=device), diagonal=1)
        mask = mask.unsqueeze(-1)

        zero_thr = float(self.config.get('zero_gradient_threshold', 1e-2))
        norms = grad_norms.squeeze(-1)
        valid = (norms >= zero_thr).float()
        pair_valid = valid.unsqueeze(1) * valid.unsqueeze(0)
        mask = mask * pair_valid

        conflict_all = (1.0 - cos_sim) / 2.0 * mask

        num_pairs = mask.sum(dim=(0, 1))
        conflict_flat = conflict_all.sum(dim=(0, 1)) / (num_pairs + 1e-8)

        conflict_score = conflict_flat.reshape(T, B, H)

        return conflict_score

    def _compute_online_loss(self, trajectories, ts, conflict_mask, batch_size, num_steps, conditions, wall_locations):
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')

        xs_pos_stacked = trajectories[:, :, :, :2]

        if online_loss_type == 'mse_simple':
            return self._compute_online_loss_mse_simple(
                xs_pos_stacked, ts, conflict_mask, batch_size, num_steps, conditions, wall_locations
            )
        elif online_loss_type == 'gradient':
            return self._compute_online_loss_gradient(trajectories, ts, conflict_mask, batch_size, num_steps, conditions, wall_locations)
        else:
            raise ValueError(f"不支持的 online_loss_type: {online_loss_type}")

    def _compute_online_loss_mse_simple(self, xs_pos_stacked, ts_stacked, conflict_mask, batch_size, num_steps, conditions, wall_locations):

        x1_pos = xs_pos_stacked[-1]
        r1 = self._compute_terminal_reward(x1_pos)

        x_flat = xs_pos_stacked[:-1].reshape(-1, self.horizon, 2)
        t_flat = ts_stacked[:-1].reshape(-1)

        pred_flat = self._scalar_predict(x_flat, t_flat, conditions, wall_locations)
        pred = pred_flat.view(num_steps, batch_size)

        target = r1.unsqueeze(0).expand(num_steps, batch_size).detach()

        if conflict_mask is not None:
            weight = conflict_mask.mean(dim=-1)
        else:
            weight = torch.ones_like(pred)

        import torch.nn.functional as F
        loss_unreduced = F.mse_loss(pred, target, reduction='none')
        loss = (loss_unreduced * weight).sum() / (weight.sum() + 1e-8)

        return loss

    def _compute_online_loss_gradient(self, xs_stacked, ts_stacked, conflict_mask, batch_size, num_steps, conditions, wall_locations):

        x1_pos = xs_stacked[-1, :, :, :2]
        xs_pos_stacked = xs_stacked[:-1, :, :, :2]

        actual_num_steps = xs_pos_stacked.shape[0]

        x_flat = xs_pos_stacked.reshape(-1, self.horizon, 2)
        t_flat = ts_stacked[:-1].reshape(-1)

        x1_pos_expanded = x1_pos.unsqueeze(0).expand(actual_num_steps, -1, -1, -1)
        t_expanded = ts_stacked[:-1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, 2)
        one_minus_t = (1.0 - t_expanded).clamp_min(1e-6)
        u_t = (x1_pos_expanded - xs_pos_stacked) / one_minus_t

        xs_stacked_flat = xs_stacked[:-1].reshape(-1, self.horizon, 4)
        t_flat_expanded = ts_stacked[:-1].reshape(-1)

        with torch.no_grad():
            if conditions is not None and wall_locations is not None:
                expanded_conditions = {}
                for k, v in conditions.items():
                    v_expanded = v.unsqueeze(0).expand(actual_num_steps, -1, -1).reshape(-1, v.shape[-1])
                    expanded_conditions[k] = v_expanded

                expanded_wall_locations = wall_locations.unsqueeze(0).expand(actual_num_steps, -1, -1, -1).reshape(-1, wall_locations.shape[1], wall_locations.shape[2])

                v_theta_full = self.flow_model.velocity_field(
                    xs_stacked_flat, t_flat_expanded, expanded_conditions, expanded_wall_locations
                )
                v_theta = v_theta_full[:, :, :2]
                v_theta = v_theta.view(actual_num_steps, batch_size, self.horizon, 2)
            else:
                v_theta = torch.zeros_like(u_t)

        target = (u_t - v_theta).detach()

        if self.config.get('online_loss_type') == 'gradient':
            global_cond = self._build_global_cond(
                conditions=conditions,
                wall_locations=wall_locations,
                batch_size=x_flat.shape[0],
                device=x_flat.device,
            )
            pred = self.learned_guidance_model(x_flat, t_flat, global_cond)
        else:
            x_flat_req = x_flat.detach().requires_grad_(True)
            g = self._scalar_predict(x_flat_req, t_flat, conditions, wall_locations)
            pred = torch.autograd.grad(g.sum(), x_flat_req, create_graph=True, retain_graph=True)[0]

        pred = pred.view(actual_num_steps, batch_size, self.horizon, 2)

        if conflict_mask is not None:
            weight = conflict_mask.unsqueeze(-1).expand(-1, -1, -1, 2).to(pred.device).float()
        else:
            weight = torch.ones_like(pred)

        loss_unreduced = (pred - target).pow(2)
        loss = (loss_unreduced * weight).sum() / (weight.sum() + 1e-8)

        return loss
