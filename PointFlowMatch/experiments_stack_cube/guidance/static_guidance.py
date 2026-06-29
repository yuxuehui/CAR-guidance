import torch
import numpy as np
from typing import Dict, Optional, List
import os

from .base_guidance import BaseGuidance
from .pcgrad import pcgrad_combine

class StaticGuidance(BaseGuidance):

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        self.energy_centers = self.config.get('energy_centers', [])
        self.energy_scales = self.config.get('energy_scales', [])
        self.sigma = self.config.get('sigma', 0.1)
        self.max_guidance = self.config.get('max_guidance', None)
        self.norm_pcd_center = self.config.get('norm_pcd_center', None)

        self.combine_method = self.config.get('combine_method', 'sum')

    def compute_reward_grads(self,
                             x: torch.Tensor,
                             t: torch.Tensor,
                             base_velocity: torch.Tensor,
                             conditions: Dict) -> List[torch.Tensor]:
        if not self.energy_centers or len(self.energy_centers) == 0:
            return []

        assert len(self.energy_centers) == len(self.energy_scales), \
            f"能量中心数量 ({len(self.energy_centers)}) 与缩放系数数量 ({len(self.energy_scales)}) 不匹配"

        device = x.device
        dtype = x.dtype

        if base_velocity.device != device:
            base_velocity = base_velocity.to(device)

        if isinstance(t, torch.Tensor):
            if t.device != device:
                t = t.to(device)
            if t.dim() == 0:
                t_expanded = t.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, 1)
            else:
                t_expanded = t.view(-1, 1, 1)
            t_expanded = t_expanded.to(dtype=dtype)
        else:
            t_expanded = torch.full((x.shape[0], 1, 1), t, device=device, dtype=dtype)

        x_1_pred = x + (1.0 - t_expanded) * base_velocity

        x_pos = x_1_pred[:, :, :3]

        per_reward_grads = []
        individual_grads = []

        for idx, (center, scale) in enumerate(zip(self.energy_centers, self.energy_scales)):
            center_tensor = torch.tensor(center, dtype=torch.float32, device=x_pos.device)

            diff = x_pos - center_tensor
            sq_dist = torch.sum(diff ** 2, dim=-1, keepdim=True)

            energy = torch.exp(-sq_dist / (self.sigma ** 2 + 1e-8))

            dist = torch.sqrt(sq_dist + 1e-8)
            dir_to_center = -diff / dist

            guidance_pos = scale * energy * dir_to_center
            if self.max_guidance is not None:
                grad_mag = torch.norm(guidance_pos, dim=-1, keepdim=True)
                scale_factor = torch.clamp(self.max_guidance / (grad_mag + 1e-8), max=1.0)
                guidance_pos = guidance_pos * scale_factor

            full_guidance = torch.zeros_like(x_1_pred)
            full_guidance[:, :, :3] = guidance_pos
            per_reward_grads.append(full_guidance)

            if self.config.get('record_step', False):
                individual_grads.append(guidance_pos.detach().cpu().numpy())

        if self.config.get('record_step', False):
            self._last_individual_grads = individual_grads
        else:
            self._last_individual_grads = []

        return per_reward_grads

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict) -> torch.Tensor:

        if not self.energy_centers or len(self.energy_centers) == 0:
            return base_velocity

        if base_velocity.device != x.device:
            base_velocity = base_velocity.to(x.device)

        per_reward_grads = self.compute_reward_grads(x, t, base_velocity, conditions)
        if len(per_reward_grads) == 0:
            return base_velocity

        if self.combine_method == 'pcgrad' and len(per_reward_grads) > 1:

            total_guidance = pcgrad_combine(per_reward_grads)
        else:

            total_guidance = torch.stack(per_reward_grads, dim=0).sum(dim=0)

        return base_velocity + total_guidance
