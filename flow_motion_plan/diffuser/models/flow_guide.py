import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, Any

from inspect import isfunction
from einops import rearrange, repeat

from flow_matching.utils import ModelWrapper

from .diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from experiments.energy_guide import EnergyFunction, EnergyGuideVectorField, DynamicEnergyGuideVectorField

__all__ = ['TrajFlowModel', 'EnergyFunction', 'EnergyGuideVectorField', 'DynamicEnergyGuideVectorField', 'create_traj_flow_model']

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def compute_velocity_from_positions(positions: torch.Tensor, dt: float = 1.0,
                                   smooth: bool = True, method: str = 'central') -> torch.Tensor:
    batch_size, horizon, position_dim = positions.shape

    velocities = torch.zeros_like(positions)

    if method == 'forward':

        velocities[:, :-1] = (positions[:, 1:] - positions[:, :-1]) / dt
        velocities[:, -1] = velocities[:, -2]

    elif method == 'backward':

        velocities[:, 1:] = (positions[:, 1:] - positions[:, :-1]) / dt
        velocities[:, 0] = velocities[:, 1]

    elif method == 'central':

        if horizon >= 3:
            velocities[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2 * dt)

            velocities[:, 0] = (positions[:, 1] - positions[:, 0]) / dt
            velocities[:, -1] = (positions[:, -1] - positions[:, -2]) / dt
        else:

            velocities[:, :-1] = (positions[:, 1:] - positions[:, :-1]) / dt
            velocities[:, -1] = velocities[:, -2] if horizon > 1 else torch.zeros_like(velocities[:, -1])

    if smooth:

        kernel_size = min(5, horizon // 4) if horizon > 10 else 3
        kernel_size = max(3, kernel_size)

        if horizon >= kernel_size:

            sigma = kernel_size / 6.0
            x = torch.arange(kernel_size, dtype=torch.float32, device=positions.device)
            x = x - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, -1)

            velocities_smooth = torch.zeros_like(velocities)
            for dim in range(position_dim):
                vel_dim = velocities[..., dim].unsqueeze(1)
                vel_smooth = F.conv1d(vel_dim, kernel, padding=kernel_size//2)
                velocities_smooth[..., dim] = vel_smooth.squeeze(1)
            velocities = velocities_smooth

    return velocities

def augment_trajectory_with_velocity(positions: torch.Tensor, dt: float = 1.0,
                                   start_velocity: Optional[torch.Tensor] = None,
                                   end_velocity: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size, horizon, position_dim = positions.shape

    velocities = compute_velocity_from_positions(positions, dt, smooth=True, method='central')

    if start_velocity is not None:
        velocities[:, 0] = start_velocity
    if end_velocity is not None:
        velocities[:, -1] = end_velocity
    else:

        velocities[:, -1] = torch.zeros_like(velocities[:, -1])

    states = torch.cat([positions, velocities], dim=-1)

    return states

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t, **extras)

class TrajFlowModel(nn.Module):

    def __init__(self,
                 position_dim: int = 2,
                 horizon: int = 48,
                 hidden_dim: int = 512,
                 num_layers: int = 8,
                 time_embedding_dim: int = 256,
                 condition_dim: int = 4,
                 max_walls: int = 6,
                 wall_feature_dim: int = 512,
                 num_attention_heads: int = 8,
                 dropout: float = 0.15,

                 sigma: float = 0.01):
        super().__init__()

        self.position_dim = position_dim
        self.state_dim = position_dim * 2
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.max_walls = max_walls
        self.wall_feature_dim = wall_feature_dim
        self.condition_dim = condition_dim

        self.condition_encoder = nn.Sequential(
            nn.Linear(2 + 2 + 2 * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.unet = ConditionalUnet1D(
            input_dim=4,
            global_cond_dim=512,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        self.guide_model = None

    def velocity_field(self, x_t, t, conditions=None, wall_locations=None, mask=None):
        return self.forward(x_t, t, conditions, wall_locations, mask)

    def prepare_context(self, conditions: Dict[int, torch.Tensor],
                       wall_locations: Optional[torch.Tensor] = None,
                       mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = list(conditions.values())[0].shape[0]
        device = list(conditions.values())[0].device

        start_pos = conditions.get(0, torch.zeros(batch_size, self.position_dim, device=device))
        goal_pos = conditions.get(self.horizon - 1, torch.zeros(batch_size, self.position_dim, device=device))

        wall_locations = wall_locations.view(wall_locations.size(0), -1)
        condition = torch.cat([start_pos, goal_pos, wall_locations], dim=1)
        condition_features = self.condition_encoder(condition)

        return condition_features

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                conditions: Dict[int, torch.Tensor] = None,
                wall_locations: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, horizon, state_dim = x.shape
        device = x.device

        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.expand(batch_size).float()

        context = self.prepare_context(conditions, wall_locations, mask)

        velocity_field = self.unet(
            sample=x,
            timestep=t,
            global_cond=context,
        )

        return velocity_field

    def compute_loss(self,
                    x_target: torch.Tensor,
                    conditions: Dict[int, torch.Tensor],
                    wall_locations: Optional[torch.Tensor] = None,
                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x_target.shape[0]
        device = x_target.device

        x_target_state = augment_trajectory_with_velocity(x_target, dt=1.0)

        """
        # 1. sample data: x_1 是目标轨迹，x_0 是噪声
        x_1 = x_target_state  # [batch_size, horizon, state_dim]
        noise = torch.randn_like(x_1, device=device)  # 噪声源
        t = torch.rand(batch_size, device=device)  # [batch_size]
        t_expanded = t.view(-1, 1, 1)  # [batch_size, 1, 1]

        # 使用t对噪声和目标进行加权插值
        x_0 = (1 - t_expanded) * noise + t_expanded * x_1

        # GT
        target_velocity = x_1 - noise

        predicted_velocity = self.velocity_field(x_0, t, conditions, wall_locations, mask)
        """

        x_1 = x_target_state
        x_0 = torch.randn_like(x_1, device=device)

        t = torch.rand(batch_size, device=device)
        t_expanded = t.view(-1, 1, 1)

        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        target_velocity = x_1 - x_0

        predicted_velocity = self.velocity_field(x_t, t, conditions, wall_locations, mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_diff = torch.pow(
                predicted_velocity * mask_expanded - target_velocity * mask_expanded, 2
            )
            mask_sum = mask.sum()
            if mask_sum > 0:
                flow_loss = masked_diff.sum() / (mask_sum * self.state_dim)
            else:

                flow_loss = torch.pow(predicted_velocity - path_sample.dx_t, 2).mean()
        else:
            flow_loss = torch.pow(predicted_velocity - target_velocity, 2).mean()

        dynamics_loss = flow_loss
        position_loss = flow_loss
        velocity_loss = 0.0

        total_loss = dynamics_loss
        loss_dict = {
            'total_loss': total_loss.item(),
            'dynamics_loss': dynamics_loss.item()
        }

        return total_loss, loss_dict

    def apply_conditions(self, x: torch.Tensor, conditions: Dict[int, torch.Tensor]) -> torch.Tensor:
        x_conditioned = x.clone()

        for timestep, condition in conditions.items():
            if condition is not None and 0 <= timestep < self.horizon:

                x_conditioned[:, timestep, :self.position_dim] = condition

        return x_conditioned

    def sample_trajectory(self,
                         conditions: Dict[int, torch.Tensor],
                         wall_locations: Optional[torch.Tensor] = None,
                         num_steps: int = 50,
                         return_positions_only: bool = False,
                         energy_guide: bool =  False,
                         energy_function=None,
                         energy_scale=None,
                         record_steps: bool = False
                         ) -> torch.Tensor:
        if energy_guide:

            if not hasattr(self, 'guide_model') or self.guide_model is None:
                assert energy_function is not None, "能量引导函数不能为空"
                assert energy_scale is not None, "energy_scale不能为空"
                self.guide_model = EnergyGuideVectorField(self, energy_function, energy_scale)

        if record_steps and hasattr(self, 'guide_model') and self.guide_model is not None:
            if hasattr(self.guide_model, 'step_data'):
                self.guide_model.step_data = []

        batch_size = list(conditions.values())[0].shape[0]
        device = list(conditions.values())[0].device

        start_pos = conditions[0][..., :self.position_dim]
        goal_pos = conditions[self.horizon - 1][..., :self.position_dim]

        start_vel = torch.zeros_like(start_pos)
        goal_vel = torch.zeros_like(goal_pos)
        start_state = torch.cat([start_pos, start_vel], dim=-1)
        goal_state = torch.cat([goal_pos, goal_vel], dim=-1)

        x_0 = torch.randn(batch_size, self.horizon, self.state_dim, device=device)

        with torch.no_grad():
            x = x_0.clone()
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((batch_size,), t_val, device=device)

                if hasattr(self, 'guide_model') and self.guide_model is not None:

                    try:
                        from experiments.trajectory_g_cov_a_gm_online import TrajectoryGCovAGMOnlineGuidance
                        from experiments.guidance.gcov_wrapper import GCovWrapper

                        if isinstance(self.guide_model, TrajectoryGCovAGMOnlineGuidance):

                            v_uncond = self.velocity_field(x, t_tensor, conditions, wall_locations)
                            velocity_field = v_uncond + self.guide_model.compute_guidance(x, t_tensor, v_uncond)
                        elif isinstance(self.guide_model, GCovWrapper):

                            velocity_field = self.guide_model(x, t_tensor, conditions, wall_locations, record_step=record_steps)
                        else:

                            velocity_field = self.guide_model(x, t_tensor, conditions, wall_locations, record_step=record_steps)
                    except ImportError:

                        velocity_field = self.guide_model(x, t_tensor, conditions, wall_locations, record_step=record_steps)
                elif energy_guide:
                    velocity_field = self.guide_model(x, t_tensor, conditions, wall_locations, record_step=record_steps)
                else:
                    velocity_field = self.velocity_field(x, t_tensor, conditions, wall_locations)

                x = x + velocity_field * dt

        if return_positions_only:
            return x[..., :self.position_dim].unsqueeze(0)
        else:
            return x.unsqueeze(0)

def create_traj_flow_model(config: Dict[str, Any]) -> TrajFlowModel:
    model = TrajFlowModel(
        position_dim=config.get('position_dim', 2),
        horizon=config.get('horizon', 40),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 8),
        time_embedding_dim=config.get('time_embedding_dim', 256),
        condition_dim=config.get('condition_dim', 4),
        max_walls=config.get('max_walls', 10),
        wall_feature_dim=config.get('wall_feature_dim', 512),
        num_attention_heads=config.get('num_attention_heads', 8),
        dropout=config.get('dropout', 0.15),
        sigma=config.get('sigma', 0.01)
    )

    print(f"🚀 创建Flow Matching模型完成")
    print(f"   - σ = {config.get('sigma', 0.01)}")
    print(f"   - 条件维度: {config.get('condition_dim', 4)} (位置边界条件)")

    return model
