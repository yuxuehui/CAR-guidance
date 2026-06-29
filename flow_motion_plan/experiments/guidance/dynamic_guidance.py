import torch
import numpy as np
from typing import Dict, Optional, List, Callable

from .base_guidance import BaseGuidance
from .pcgrad import pcgrad_combine

class DynamicGuidance(BaseGuidance):

    def __init__(self,
                 path_functions: List[Callable[[float], List[float]]],
                 config: Optional[Dict] = None):
        super().__init__(config)
        self.path_functions = path_functions
        self.energy_scales = self.config.get('energy_scales', [-1.0] * len(path_functions))
        self.sigma = self.config.get('sigma', 2.0)
        self.traj_normalizer = self.config.get('traj_normalizer', None)

        self.combine_method = self.config.get('combine_method', 'sum')

        assert len(self.energy_scales) == len(path_functions), \
            f"能量场缩放系数数量 ({len(self.energy_scales)}) 与路径函数数量 ({len(path_functions)}) 不匹配"

    def compute_reward_grads(self,
                             x: torch.Tensor,
                             t: torch.Tensor,
                             base_velocity: torch.Tensor,
                             conditions: Dict,
                             wall_locations: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_scalar = float(t.item())
            else:
                t_scalar = float(t[0].item())
        else:
            t_scalar = float(t)

        t_normalized = max(0.0, min(1.0, t_scalar))

        dynamic_centers = []
        for path_fn in self.path_functions:
            center = path_fn(t_normalized)
            dynamic_centers.append(center)

        if len(dynamic_centers) == 0:
            return []

        if isinstance(t, torch.Tensor):
            t_expanded = t.view(-1, 1, 1)
        else:
            t_expanded = torch.full((x.shape[0], 1, 1), t, device=x.device, dtype=x.dtype)

        x_1_pred = x + (1.0 - t_expanded) * base_velocity
        x_pos = x_1_pred[:, :, :2]

        if self.traj_normalizer is not None:
            x_pos_np = x_pos.detach().cpu().numpy()
            x_pos_unnorm_np = self.traj_normalizer.unnormalize(x_pos_np)
            x_pos_unnorm = torch.from_numpy(x_pos_unnorm_np).to(x_pos.device).float()
        else:
            x_pos_unnorm = x_pos

        per_reward_grads = []
        individual_grads = []

        for center, scale in zip(dynamic_centers, self.energy_scales):
            center_tensor = torch.tensor(center, dtype=torch.float32, device=x_pos_unnorm.device)

            diff = x_pos_unnorm - center_tensor
            sq_dist = torch.sum(diff ** 2, dim=-1, keepdim=True)

            energy = torch.exp(-sq_dist / (self.sigma ** 2 + 1e-8))

            dist = torch.sqrt(sq_dist + 1e-8)
            dir_to_center = -diff / dist

            guidance_pos_unnorm = scale * energy * dir_to_center

            if self.traj_normalizer is not None:
                scale_factor = 2.0 / (self.traj_normalizer.maxs - self.traj_normalizer.mins)
                scale_factor_tensor = torch.from_numpy(scale_factor).to(guidance_pos_unnorm.device).float()
                guidance_pos = guidance_pos_unnorm * scale_factor_tensor
            else:
                guidance_pos = guidance_pos_unnorm

            full_guidance = torch.zeros_like(x_1_pred)
            full_guidance[:, :, :2] = guidance_pos
            per_reward_grads.append(full_guidance)

            if self.config.get('record_step', False):
                individual_grads.append(guidance_pos_unnorm.detach().cpu().numpy())

        if self.config.get('record_step', False):
            self._last_individual_grads = individual_grads
        else:
            self._last_individual_grads = []

        return per_reward_grads

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict,
                wall_locations: Optional[torch.Tensor] = None) -> torch.Tensor:

        per_reward_grads = self.compute_reward_grads(
            x, t, base_velocity, conditions, wall_locations
        )
        if len(per_reward_grads) == 0:
            return base_velocity

        if self.combine_method == 'pcgrad' and len(per_reward_grads) > 1:
            total_guidance = pcgrad_combine(per_reward_grads)
        else:
            total_guidance = torch.stack(per_reward_grads, dim=0).sum(dim=0)

        return base_velocity + total_guidance
