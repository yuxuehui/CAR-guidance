import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple, Any

from inspect import isfunction
from einops import rearrange, repeat

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver

from .diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def get_timesteps(schedule: str, k_steps: int, exp_scale: float = 1.0):
    t = torch.linspace(0, 1, k_steps + 1)[:-1]
    if schedule == "linear":
        dt = torch.ones(k_steps) / k_steps
    elif schedule == "cosine":
        dt = torch.cos(t * torch.pi) + 1
        dt /= torch.sum(dt)
    elif schedule == "exp":
        dt = torch.exp(-t * exp_scale)
        dt /= torch.sum(dt)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    t0 = torch.cat((torch.zeros(1), torch.cumsum(dt, dim=0)[:-1]))
    return t0, dt

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)

        t_expanded = t * freqs.unsqueeze(0)
        t_embed = torch.cat([torch.sin(t_expanded), torch.cos(t_expanded)], dim=-1)

        return t_embed

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if pos.dim() == 2:
            pos = pos.unsqueeze(1)

        batch_size, num_pos, _ = pos.shape

        x, y = pos[..., 0], pos[..., 1]

        half_dim = self.hidden_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim // 2, device=pos.device) / (half_dim // 2))

        x_expanded = x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        x_pos_embed = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)

        y_expanded = y.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        y_pos_embed = torch.cat([torch.sin(y_expanded), torch.cos(y_expanded)], dim=-1)

        pos_embed = torch.cat([x_pos_embed, y_pos_embed], dim=-1)

        if pos_embed.shape[1] == 1:
            pos_embed = pos_embed.squeeze(1)

        return pos_embed

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class ObstacleEncoder(nn.Module):

    def __init__(self, num_walls: int = 6, input_dim: int = 2, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.obstacle_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.obstacle_proj = nn.Linear(output_dim, output_dim)

    def forward(self, obstacle_locations: torch.Tensor,
                obstacle_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_walls, _ = obstacle_locations.shape

        if obstacle_mask is None:
            obstacle_mask = torch.any(obstacle_locations != 0, dim=-1)

        obstacle_features = self.obstacle_encoder(obstacle_locations)

        obstacle_features = obstacle_features * obstacle_mask.unsqueeze(-1).float()

        obstacle_features = torch.mean(obstacle_features, dim=1)

        obstacle_features = self.obstacle_proj(obstacle_features)

        return obstacle_features, obstacle_mask

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
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.max_walls = max_walls
        self.wall_feature_dim = wall_feature_dim
        self.condition_dim = condition_dim
        self.sigma = sigma

        self.flow_path = AffineProbPath(scheduler=CondOTScheduler())

        self.time_embedder = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        self.traj_proj = nn.Conv1d(
            in_channels=self.state_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.wall_encoder = ObstacleEncoder(
            num_walls=max_walls,
            input_dim=position_dim,
            hidden_dim=wall_feature_dim // 2,
            output_dim=wall_feature_dim
        )

        self.wall_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                query_dim=hidden_dim,
                context_dim=wall_feature_dim,
                heads=num_attention_heads,
                dim_head=hidden_dim // num_attention_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.wall_proj = nn.Linear(wall_feature_dim, hidden_dim)

        fusion_input_dim = hidden_dim * 2
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(2 + 2 + 2 * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.time_encoding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_proj= nn.Linear(time_embedding_dim, hidden_dim)

        combined_dim = self.hidden_dim + wall_feature_dim
        self.vf_predict_head = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.state_dim * self.horizon)
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

        x_1 = x_target_state
        noise = torch.randn_like(x_1, device=device)
        t = torch.rand(batch_size, device=device)
        t_expanded = t.view(-1, 1, 1)

        x_0 = (1 - t_expanded) * noise + t_expanded * x_1

        target_veloctiy = x_1 - noise

        predicted_velocity = self.velocity_field(x_0, t, conditions, wall_locations, mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_diff = torch.pow(
                predicted_velocity * mask_expanded - target_veloctiy * mask_expanded, 2
            )
            mask_sum = mask.sum()
            if mask_sum > 0:
                flow_loss = masked_diff.sum() / (mask_sum * self.state_dim)
            else:

                flow_loss = torch.pow(predicted_velocity - path_sample.dx_t, 2).mean()
        else:
            flow_loss = torch.pow(predicted_velocity - target_veloctiy, 2).mean()

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
                         return_positions_only: bool = False) -> torch.Tensor:
        batch_size = list(conditions.values())[0].shape[0]
        device = list(conditions.values())[0].device

        print("num_steps是", num_steps)

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

                velocity_field = self.velocity_field(x, t_tensor, conditions, wall_locations)

                x = x.detach().clone() + dt * velocity_field

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
