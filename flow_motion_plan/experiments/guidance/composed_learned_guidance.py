import torch
import sys
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any

from .base_guidance import BaseGuidance
from .gcov_wrapper import GCovWrapper, make_guidance_fn

from .energy_function import EnergyFunction

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from old_experiments.trajectory_g_cov_a_gm_online import TrajectoryGCovAGMOnlineGuidance
    from old_experiments.trajectory_dynamic_g_cov_a_gm_online import TrajectoryDynamicGCovAGMOnlineGuidance
except ImportError as e:

    TrajectoryGCovAGMOnlineGuidance = None
    TrajectoryDynamicGCovAGMOnlineGuidance = None

class ComposedLearnedGuidance(BaseGuidance):

    def __init__(self,
                 flow_model,
                 base_guidance: BaseGuidance,
                 traj_normalizer=None,
                 config: Optional[Dict] = None,
                 use_legacy_mode: bool = False,
                 energy_centers: Optional[List[List[float]]] = None,
                 energy_scales: Optional[List[float]] = None,
                 path_functions: Optional[List[Callable[[float], List[float]]]] = None):
        super().__init__(config)

        self.flow_model = flow_model
        self.base_guidance = base_guidance
        self.traj_normalizer = traj_normalizer
        self.use_legacy_mode = use_legacy_mode

        if use_legacy_mode:

            if TrajectoryGCovAGMOnlineGuidance is None:
                raise ImportError("无法导入TrajectoryGCovAGMOnlineGuidance，请检查old_experiments目录")

            self.energy_centers = energy_centers or []
            self.energy_scales = energy_scales or []
            self.path_functions = path_functions
            self.is_dynamic = path_functions is not None and len(path_functions) > 0

            self.online_config = {
                'online_train_steps': self.config.get('online_train_steps', 1000),
                'online_batch_size': self.config.get('online_batch_size', 4),
                'online_lr': self.config.get('online_lr', 1e-4),
                'num_ode_steps': self.config.get('num_ode_steps', 20),
                'conflict_threshold': self.config.get('conflict_threshold', 0.5),
                'conflict_temperature': self.config.get('conflict_temperature', 0.1),
                'online_loss_type': self.config.get('online_loss_type', 'mse_simple'),
                'residual_unet_channels': self.config.get('residual_unet_channels', 64),
                'residual_unet_down_dims': self.config.get('residual_unet_down_dims', [256, 512, 1024]),
                'use_learned_correction': self.config.get('train_online', True),
            }

            self._create_legacy_online_guidance()
        else:

            base_guidance_fn = make_guidance_fn(self.base_guidance, self.flow_model)

            horizon = getattr(self.flow_model, 'horizon', 40)
            device = str(next(self.flow_model.parameters()).device)

            conflict_compute_fn = None
            if hasattr(self.base_guidance, 'energy_centers') and hasattr(self.base_guidance, 'energy_scales'):
                energy_centers = self.base_guidance.energy_centers
                energy_scales = self.base_guidance.energy_scales
                if len(energy_centers) >= 2 and len(energy_scales) >= 2:

                    conflict_compute_fn = self._create_energy_based_conflict_fn(
                        energy_centers, energy_scales, device
                    )

            self.gcov_wrapper = GCovWrapper(
                flow_model=self.flow_model,
                base_guidance_fn=base_guidance_fn,
                horizon=horizon,
                device=device,
                config=self.config,
                traj_normalizer=self.traj_normalizer,
                conflict_compute_fn=conflict_compute_fn,
                energy_centers=energy_centers if energy_centers else None,
                energy_scales=energy_scales if energy_scales else None,
            )
            self.online_guidance = self.gcov_wrapper

        self.trained = False

    def _create_energy_based_conflict_fn(self, energy_centers, energy_scales, device):
        import torch

        centers_tensor = torch.tensor(energy_centers, dtype=torch.float32, device=device)
        scales_tensor = torch.tensor(energy_scales, dtype=torch.float32, device=device)
        num_energy = len(energy_centers)

        sigma = self.config.get('sigma', 2.0)

        def conflict_compute_fn(x_pos):

            if x_pos.dim() == 4:

                T, B, H, D = x_pos.shape
                x_flat = x_pos.reshape(-1, D)
            else:

                B, H, D = x_pos.shape
                x_flat = x_pos.reshape(-1, D)

            if num_energy < 2:

                if x_pos.dim() == 4:
                    return torch.zeros(T, B, H, device=x_pos.device)
                else:
                    return torch.zeros(B, H, device=x_pos.device)

            diff_all = x_flat.unsqueeze(1) - centers_tensor.unsqueeze(0)
            sq_dist_all = (diff_all ** 2).sum(dim=-1, keepdim=True)

            energy_all = torch.exp(-sq_dist_all / (sigma ** 2 + 1e-8))

            dist_all = torch.sqrt(sq_dist_all + 1e-8)
            dir_to_center = -diff_all / dist_all

            scales_expanded = scales_tensor.view(1, -1, 1)
            grads_all = scales_expanded * energy_all * dir_to_center

            grads_all = grads_all.transpose(0, 1)

            grad_norms = torch.norm(grads_all, dim=-1, keepdim=True)
            grads_normalized = grads_all / (grad_norms + 1e-8)

            grads_expanded_i = grads_normalized.unsqueeze(1)
            grads_expanded_j = grads_normalized.unsqueeze(0)
            cos_sim = (grads_expanded_i * grads_expanded_j).sum(dim=-1)

            mask = torch.triu(torch.ones(num_energy, num_energy, device=x_pos.device), diagonal=1)
            mask = mask.unsqueeze(-1)

            conflict_all = (1.0 - cos_sim) * mask

            num_pairs = mask.sum(dim=(0, 1))
            conflict_flat = conflict_all.sum(dim=(0, 1)) / (num_pairs + 1e-8)

            if x_pos.dim() == 4:
                conflict = conflict_flat.reshape(T, B, H)
            else:
                conflict = conflict_flat.reshape(B, H)

            return conflict

        return conflict_compute_fn

    def _create_legacy_online_guidance(self):
        print("使用旧模式: _create_legacy_online_guidance")

        """创建旧模式的在线guidance对象（向后兼容）"""
        horizon = getattr(self.flow_model, 'horizon', 40)
        device = str(next(self.flow_model.parameters()).device)

        if self.is_dynamic:
            if TrajectoryDynamicGCovAGMOnlineGuidance is None:
                raise ImportError("无法导入TrajectoryDynamicGCovAGMOnlineGuidance")

            self.online_guidance = TrajectoryDynamicGCovAGMOnlineGuidance(
                flow_model=self.flow_model,
                path_functions=self.path_functions,
                energy_scales=self.energy_scales,
                horizon=horizon,
                device=device,
                config=self.online_config,
                traj_normalizer=self.traj_normalizer,
            )
        else:
            if len(self.energy_centers) == 0:
                raise ValueError("静态能量场需要提供energy_centers")

            energy_functions = []
            for center in self.energy_centers:
                energy_functions.append(EnergyFunction(center))

            self.online_guidance = TrajectoryGCovAGMOnlineGuidance(
                flow_model=self.flow_model,
                energy_functions=energy_functions,
                energy_scales=self.energy_scales,
                horizon=horizon,
                device=device,
                config=self.online_config,
                traj_normalizer=self.traj_normalizer,
            )

    def train_online(self, conditions: Dict, wall_locations: torch.Tensor):
        if self.use_legacy_mode:
            if self.online_config.get('use_learned_correction', True):
                self.online_guidance.train_model(
                    conditions=conditions,
                    wall_locations=wall_locations
                )
                self.trained = True
        else:

            if self.config.get('train_online', True):
                self.gcov_wrapper.train_model(
                    conditions=conditions,
                    wall_locations=wall_locations
                )
                self.trained = True

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict,
                wall_locations: Optional[torch.Tensor] = None) -> torch.Tensor:

        return base_velocity

    def get_online_guidance(self):
        return self.online_guidance

    def get_base_guidance(self):
        return self.base_guidance
