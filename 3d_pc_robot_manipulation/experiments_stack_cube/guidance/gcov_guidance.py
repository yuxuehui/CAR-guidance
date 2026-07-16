import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:

    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
except ImportError as e1:
    try:

        from diffuser.models.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    except ImportError as e2:
        ConditionalUnet1D = None
        print(f"Warning: Could not import ConditionalUnet1D")
        print(f"  Tried: diffusion_policy.model.diffusion.conditional_unet1d -> {e1}")
        print(f"  Tried: diffuser.models.diffusion_policy.model.diffusion.conditional_unet1d -> {e2}")
        print("  G_Cov online training will be disabled. Please install diffusion_policy package.")

from .base_guidance import BaseGuidance
from .static_guidance import StaticGuidance

def _smootherstep(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

def _smootherstep_gate(conflict: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    conflict_norm = (conflict - (threshold - temperature)) / (2.0 * temperature + 1e-8)
    return _smootherstep(conflict_norm)

class UNetScalarResidual(nn.Module):
    def __init__(
        self,
        horizon: int,
        in_dim: int = 3,
        model_channels: int = 64,
        diffusion_step_embed_dim: int = 256,
        down_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.model_channels = int(model_channels)
        self.in_proj = nn.Linear(in_dim, self.model_channels)

        if ConditionalUnet1D is None:
            raise ImportError(
                "ConditionalUnet1D is required for G_Cov online training but not available. "
                "Please ensure diffusion_policy package is installed and accessible."
            )

        self.unet = ConditionalUnet1D(
            input_dim=self.model_channels,
            global_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims or [256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        self.out_proj = nn.Linear(self.model_channels, 1)

    def forward(self, x_pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x_feat = self.in_proj(x_pos)

        y = self.unet(sample=x_feat, timestep=t, global_cond=None)
        g_seq = self.out_proj(y).squeeze(-1)
        return g_seq.sum(dim=1)

class UNetVectorResidual(nn.Module):
    def __init__(
        self,
        horizon: int,
        in_dim: int = 3,
        model_channels: int = 64,
        diffusion_step_embed_dim: int = 256,
        down_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.model_channels = int(model_channels)
        self.in_proj = nn.Linear(in_dim, self.model_channels)

        if ConditionalUnet1D is None:
            raise ImportError(
                "ConditionalUnet1D is required for G_Cov online training but not available. "
                "Please ensure diffusion_policy package is installed and accessible."
            )

        self.unet = ConditionalUnet1D(
            input_dim=self.model_channels,
            global_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims or [256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        self.out_proj = nn.Linear(self.model_channels, in_dim)

    def forward(self, x_pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x_feat = self.in_proj(x_pos)

        y = self.unet(sample=x_feat, timestep=t, global_cond=None)
        return self.out_proj(y)

class GCovGuidance(BaseGuidance):

    def __init__(
        self,
        base_guidance: StaticGuidance,
        horizon: int = 32,
        device: str = 'cuda',
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.base_guidance = base_guidance
        self.horizon = horizon
        self.device = device

        self.energy_centers = base_guidance.energy_centers
        self.energy_scales = base_guidance.energy_scales
        self.sigma = base_guidance.sigma
        self.num_energy = len(self.energy_centers)

        self.norm_pcd_center = base_guidance.norm_pcd_center if hasattr(base_guidance, 'norm_pcd_center') else None
        if self.norm_pcd_center is not None:
            self.norm_pcd_center = torch.tensor(self.norm_pcd_center, dtype=torch.float32, device=self.device)

        if self.num_energy > 0:
            self.energy_centers_tensor = torch.stack([
                torch.tensor(center, dtype=torch.float32, device=self.device)
                for center in self.energy_centers
            ])
            if self.norm_pcd_center is not None:
                self.energy_centers_tensor = self.energy_centers_tensor - self.norm_pcd_center.to(self.device)
            self.energy_scales_tensor = torch.tensor(
                self.energy_scales, dtype=torch.float32, device=self.device
            )
        else:
            self.energy_centers_tensor = None
            self.energy_scales_tensor = None

        self.learned_guidance_model = self._build_learned_guidance_model()

        self.trained = False

    def _build_learned_guidance_model(self):
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')
        model_channels = int(self.config.get("residual_unet_channels", 64))
        down_dims = self.config.get("residual_unet_down_dims", [256, 512, 1024])

        if online_loss_type == 'gradient':

            print(f"[GCovGuidance] 使用 gradient 模式: UNet 输出向量场 [B,H,3]")
            return UNetVectorResidual(
                horizon=self.horizon,
                in_dim=3,
                model_channels=model_channels,
                diffusion_step_embed_dim=256,
                down_dims=down_dims,
            ).to(self.device)
        else:

            print(f"[GCovGuidance] 使用 {online_loss_type} 模式: UNet 输出标量 [B]")
            return UNetScalarResidual(
                horizon=self.horizon,
                in_dim=3,
                model_channels=model_channels,
                diffusion_step_embed_dim=256,
                down_dims=down_dims,
            ).to(self.device)

    def _is_scalar_model(self) -> bool:
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')
        return online_loss_type != 'gradient'

    def _extract_position(self, x):

        return x[:, :, :3]

    def _expand_to_full_state(self, pos_guidance):

        batch_size, horizon, _ = pos_guidance.shape
        full_guidance = torch.zeros(batch_size, horizon, 10, device=pos_guidance.device, dtype=pos_guidance.dtype)
        full_guidance[:, :, :3] = pos_guidance
        return full_guidance

    def _compute_conflict_score(self, x_pos):
        if self.num_energy < 2:

            if x_pos.dim() == 4:
                T, B, H, _ = x_pos.shape
                return torch.zeros(T, B, H, device=x_pos.device)
            else:
                B, H, _ = x_pos.shape
                return torch.zeros(B, H, device=x_pos.device)

        if x_pos.dim() == 4:
            T, B, H, D = x_pos.shape
            x_flat = x_pos.reshape(-1, D)
        else:
            B, H, D = x_pos.shape
            x_flat = x_pos.reshape(-1, D)

        centers = self.energy_centers_tensor.to(x_flat.device)
        scales = self.energy_scales_tensor.to(x_flat.device)

        diff_all = x_flat.unsqueeze(1) - centers.unsqueeze(0)
        sq_dist_all = (diff_all ** 2).sum(dim=-1, keepdim=True)

        energy_all = torch.exp(-sq_dist_all / (self.sigma ** 2 + 1e-8))

        dist_all = torch.sqrt(sq_dist_all + 1e-8)
        dir_to_center = -diff_all / dist_all

        scales_expanded = scales.view(1, -1, 1)
        grads_all = scales_expanded * energy_all * dir_to_center

        grads_all = grads_all.transpose(0, 1)

        grad_norms = torch.norm(grads_all, dim=-1, keepdim=True)
        grads_normalized = grads_all / (grad_norms + 1e-8)

        grads_expanded_i = grads_normalized.unsqueeze(1)
        grads_expanded_j = grads_normalized.unsqueeze(0)
        cos_sim = (grads_expanded_i * grads_expanded_j).sum(dim=-1)

        mask = torch.triu(torch.ones(self.num_energy, self.num_energy, device=x_flat.device), diagonal=1)
        mask = mask.unsqueeze(-1)

        zero_thr = float(self.config.get('zero_gradient_threshold', 1e-2))
        norms = grad_norms.squeeze(-1)
        valid = (norms >= zero_thr).float()
        pair_valid = valid.unsqueeze(1) * valid.unsqueeze(0)
        mask = mask * pair_valid

        conflict_all = (1.0 - cos_sim) / 2.0 * mask

        num_pairs = mask.sum(dim=(0, 1))
        conflict_flat = conflict_all.sum(dim=(0, 1)) / (num_pairs + 1e-8)

        if x_pos.dim() == 4:
            conflict = conflict_flat.reshape(T, B, H)
        else:
            conflict = conflict_flat.reshape(B, H)

        return conflict

    def _compute_terminal_reward(self, x1_pos):
        with torch.no_grad():

            obstacle_reward = torch.zeros(x1_pos.shape[0], device=x1_pos.device)

            if self.num_energy > 0:
                centers = self.energy_centers_tensor.to(x1_pos.device)
                scales_abs = torch.abs(self.energy_scales_tensor.to(x1_pos.device))

                diff = x1_pos.unsqueeze(-2) - centers.unsqueeze(0).unsqueeze(0)
                sq_dist = (diff ** 2).sum(dim=-1)

                energy = torch.exp(-sq_dist / (self.sigma ** 2 + 1e-8))
                energy_per_center_max = energy.max(dim=1)[0]
                total_energy = (energy_per_center_max * scales_abs.unsqueeze(0)).sum(dim=-1)

                obstacle_reward = -total_energy

            goal_reward = torch.zeros(x1_pos.shape[0], device=x1_pos.device)

            if hasattr(self, 'current_goal_pos') and self.current_goal_pos is not None:
                end_effector_pos = x1_pos[:, -1, :]
                goal_pos = self.current_goal_pos.to(x1_pos.device)

                sq_dist_to_goal = torch.sum((end_effector_pos - goal_pos)**2, dim=-1)

                goal_sigma = float(self.config.get('goal_sigma', 0.5))
                goal_reward = torch.exp(-sq_dist_to_goal / (goal_sigma**2 + 1e-8))

            obstacle_weight = float(self.config.get('obstacle_reward_weight', 10.0))
            goal_weight = float(self.config.get('goal_reward_weight', 5.0))

            r1 = obstacle_weight * obstacle_reward + goal_weight * goal_reward

        return r1

    def _compute_base_guidance(self, x, t, base_velocity):

        record_needed = self.config.get('record_guidance_details', False)
        if record_needed:
            self.base_guidance.config['record_step'] = True

        conditions = {}
        guided_velocity = self.base_guidance.forward(x, t, base_velocity, conditions)
        base_guidance = guided_velocity - base_velocity

        if record_needed:
            self.base_guidance.config['record_step'] = False

        return base_guidance

    def _scalar_predict(self, x_pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.learned_guidance_model(x_pos, t)

    def _compute_learned_correction(self, x, t, force=False):
        if self.learned_guidance_model is None:
            return torch.zeros_like(x)
        if not force and not self.trained:
            return torch.zeros_like(x)

        x_pos = self._extract_position(x)

        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_tensor = t.unsqueeze(0).expand(x.shape[0]).to(device=x.device)
            elif t.dim() == 1:
                t_tensor = t.view(-1).to(device=x.device)
            else:
                t_tensor = t.view(-1)[0].expand(x.shape[0]).to(device=x.device)
        else:
            t_tensor = torch.full((x.shape[0],), float(t), device=x.device, dtype=torch.float32)

        if self._is_scalar_model():

            with torch.enable_grad():
                x_pos_req = x_pos.detach().requires_grad_(True)
                g = self._scalar_predict(x_pos_req, t_tensor)
                grad_flat = torch.autograd.grad(g.sum(), x_pos_req, create_graph=False, retain_graph=False)[0]
                learned_correction_pos = grad_flat.detach()

        else:

            with torch.no_grad():
                learned_correction_pos = self.learned_guidance_model(x_pos, t_tensor)

        conflict_score = self._compute_conflict_score(x_pos)
        conflict_threshold = float(self.config.get('conflict_threshold', 0.5))
        conflict_temperature = float(self.config.get('conflict_temperature', 0.1))
        conflict_weight = _smootherstep_gate(conflict_score, conflict_threshold, conflict_temperature)

        if self.config.get('debug_learned_correction', False):
            pass

        learned_correction_pos = learned_correction_pos * conflict_weight.unsqueeze(-1)

        if self.config.get('debug_learned_correction', False):
            pass

        correction_scale = float(self.config.get('learned_correction_scale', 1.0))
        learned_correction_pos = learned_correction_pos * correction_scale

        learned_correction = self._expand_to_full_state(learned_correction_pos)

        return learned_correction

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict) -> torch.Tensor:

        base_guidance = self._compute_base_guidance(x, t, base_velocity)

        learned_correction = self._compute_learned_correction(x, t)

        total_guidance = base_guidance + learned_correction

        if self.config.get('record_guidance_details', False):
            self._record_guidance_details(x, t, base_velocity, base_guidance, learned_correction, total_guidance)

        if self.config.get('save_visualization_data', False):
            if not hasattr(self, 'visualization_data'):
                self.visualization_data = []

            if isinstance(t, torch.Tensor):
                if t.dim() == 0:
                    t_val = float(t.item())
                else:
                    t_val = float(t.view(-1)[0].item())
            else:
                t_val = float(t)

            self.visualization_data.append({
                'x': x.detach().cpu().numpy(),
                't': t_val,
                'base_velocity': base_velocity.detach().cpu().numpy(),
                'base_guidance': base_guidance.detach().cpu().numpy(),
                'learned_correction': learned_correction.detach().cpu().numpy(),
                'total_guidance': total_guidance.detach().cpu().numpy(),
            })

        return base_velocity + total_guidance

    def save_visualization_data(self, save_path: str):
        if hasattr(self, 'visualization_data') and len(self.visualization_data) > 0:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'data': self.visualization_data,
                    'energy_centers': self.energy_centers,
                    'energy_scales': self.energy_scales,
                    'sigma': self.sigma,
                    'horizon': self.horizon,
                    'current_goal_pos': self.current_goal_pos.cpu().numpy() if hasattr(self, 'current_goal_pos') and self.current_goal_pos is not None else None,
                    'norm_pcd_center': self.norm_pcd_center.cpu().numpy() if hasattr(self, 'norm_pcd_center') and self.norm_pcd_center is not None else None,
                }, f)
            print(f"[GCovGuidance] 保存可视化数据到: {save_path}")

    def clear_visualization_data(self):
        if hasattr(self, 'visualization_data'):
            self.visualization_data = []

    def _record_guidance_details(self, x, t, base_velocity, base_guidance, learned_correction, total_guidance):
        if not hasattr(self, 'guidance_records'):
            self.guidance_records = []

        x_pos = x[:, :, :3]
        base_velocity_pos = base_velocity[:, :, :3]
        base_guidance_pos = base_guidance[:, :, :3]
        learned_correction_pos = learned_correction[:, :, :3]
        total_guidance_pos = total_guidance[:, :, :3]

        batch_size = x_pos.shape[0]
        horizon = x_pos.shape[1]

        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_val = float(t.item())
            else:
                t_val = float(t.view(-1)[0].item())
        else:
            t_val = float(t)

        for b in range(batch_size):
            positions = x_pos[b].detach().cpu().numpy()

            if self.norm_pcd_center is not None:
                positions_unnorm = positions + self.norm_pcd_center.cpu().numpy()
            else:
                positions_unnorm = positions

            record = {
                't': t_val,
                'batch': b,
                'individual_guidances': [],
                'total_guidance_norm': [],
                'learned_correction_norm': [],
            }

            individual_grads = getattr(self.base_guidance, '_last_individual_grads', [])

            if self.num_energy > 0:
                centers = self.energy_centers_tensor.cpu().numpy()
                scales = self.energy_scales_tensor.cpu().numpy()

                for h in range(horizon):
                    pos_norm = positions[h]
                    pos_unnorm = positions_unnorm[h]
                    base_vel = base_velocity_pos[b, h].detach().cpu().numpy()

                    individual_info = []
                    for e in range(self.num_energy):
                        center = centers[e]
                        scale = scales[e]

                        diff = pos_norm - center
                        dist = np.linalg.norm(diff)

                        energy = np.exp(-dist**2 / (self.sigma**2 + 1e-8))

                        if len(individual_grads) > e and individual_grads[e] is not None:
                            guidance_single = individual_grads[e][b, h, :]
                            guidance_norm = np.linalg.norm(guidance_single)
                        else:

                            dir_to_center = -diff / (dist + 1e-8)
                            guidance_single = scale * energy * dir_to_center
                            guidance_norm = np.linalg.norm(guidance_single)

                        if guidance_norm > 1e-6:

                            dir_to_center = -diff / (dist + 1e-8)
                            reverse_guidance = -guidance_single / guidance_norm
                            dot_product = np.dot(reverse_guidance, dir_to_center)

                            reverse_points_to_center = dot_product > 0
                            direction_correct = (scale < 0 and reverse_points_to_center) or (scale > 0 and not reverse_points_to_center)
                        else:

                            reverse_points_to_center = None
                            direction_correct = None

                        center_unnorm = center
                        if self.norm_pcd_center is not None:
                            center_unnorm = center + self.norm_pcd_center.cpu().numpy()

                        individual_info.append({
                            'energy_id': e,
                            'center': center_unnorm.tolist(),
                            'scale': float(scale),
                            'distance': float(dist),
                            'energy': float(energy),
                            'guidance_norm': float(guidance_norm),
                            'reverse_points_to_center': reverse_points_to_center,
                            'expected_behavior': 'repel (reverse->center)' if scale < 0 else 'attract (forward->center)',
                            'direction_correct': direction_correct,
                        })

                    base_vel_norm = float(np.linalg.norm(base_vel))

                    base_guid = base_guidance_pos[b, h].detach().cpu().numpy()
                    base_guid_norm = float(np.linalg.norm(base_guid))

                    total_guid_norm = float(np.linalg.norm(total_guidance_pos[b, h].detach().cpu().numpy()))
                    learned_corr_norm = float(np.linalg.norm(learned_correction_pos[b, h].detach().cpu().numpy()))

                    record['individual_guidances'].append({
                        'horizon_idx': h,
                        'position': pos_unnorm.tolist(),
                        'base_velocity_norm': base_vel_norm,
                        'base_guidance_norm': base_guid_norm,
                        'energies': individual_info,
                        'total_guidance_norm': total_guid_norm,
                        'learned_correction_norm': learned_corr_norm,
                    })

            self.guidance_records.append(record)

    def save_guidance_records(self, save_path: str):
        if not hasattr(self, 'guidance_records') or len(self.guidance_records) == 0:
            print("[GCovGuidance] 没有 guidance 记录可保存")
            return

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Guidance 详细记录\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"能量中心数量: {self.num_energy}\n")
            f.write(f"能量中心位置: {self.energy_centers}\n")
            f.write(f"能量 scales: {self.energy_scales}\n")
            f.write(f"高斯 sigma: {self.sigma}\n")
            f.write(f"Horizon: {self.horizon}\n\n")

            f.write("=" * 80 + "\n")
            f.write("详细记录\n")
            f.write("=" * 80 + "\n\n")

            for i, record in enumerate(self.guidance_records):
                f.write(f"\n{'='*60}\n")
                f.write(f"记录 {i+1}/{len(self.guidance_records)} | t={record['t']:.4f} | batch={record['batch']}\n")
                f.write(f"{'='*60}\n\n")

                for point_info in record['individual_guidances']:
                    h = point_info['horizon_idx']
                    pos = point_info['position']
                    base_vel_norm = point_info['base_velocity_norm']
                    base_guid_norm = point_info['base_guidance_norm']

                    f.write(f"  Horizon点 {h}: 位置=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n")
                    f.write(f"  Base Velocity 大小: {base_vel_norm:.6f}\n")
                    f.write(f"  Base Guidance 大小: {base_guid_norm:.6f} (来自所有能量场)\n")
                    f.write(f"  {'─'*56}\n")

                    for energy_info in point_info['energies']:
                        e_id = energy_info['energy_id']
                        center = energy_info['center']
                        scale = energy_info['scale']
                        dist = energy_info['distance']
                        energy = energy_info['energy']
                        guid_norm = energy_info['guidance_norm']
                        reverse_to_center = energy_info['reverse_points_to_center']
                        expected = energy_info['expected_behavior']
                        correct = energy_info['direction_correct']

                        f.write(f"    能量场 {e_id+1}:\n")
                        f.write(f"      中心位置: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]\n")
                        f.write(f"      Scale: {scale:.2f} ({'排斥场' if scale < 0 else '吸引场'})\n")
                        f.write(f"      距离: {dist:.6f}\n")
                        f.write(f"      高斯能量: {energy:.6f}\n")
                        f.write(f"      Guidance 大小: {guid_norm:.6f}\n")
                        if reverse_to_center is not None:
                            f.write(f"      Guidance 反向延长线指向中心: {reverse_to_center}\n")
                            f.write(f"      预期行为: {expected}\n")
                            f.write(f"      方向正确: {'✓' if correct else '✗'}\n")
                        else:
                            f.write(f"      Guidance 反向延长线指向中心: N/A (guidance太小)\n")
                            f.write(f"      预期行为: {expected}\n")
                            f.write(f"      方向正确: N/A (guidance太小)\n")
                        f.write(f"\n")

                    f.write(f"    总 Guidance 大小: {point_info['total_guidance_norm']:.6f} (= Base Guidance + Learned Correction)\n")
                    f.write(f"    Learned Correction 大小: {point_info['learned_correction_norm']:.6f}\n")
                    f.write(f"\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("统计信息\n")
            f.write("=" * 80 + "\n\n")

            all_individual_norms = {e: [] for e in range(self.num_energy)}
            all_total_norms = []
            all_learned_norms = []
            direction_correct_count = {e: 0 for e in range(self.num_energy)}
            direction_total_count = {e: 0 for e in range(self.num_energy)}

            for record in self.guidance_records:
                for point_info in record['individual_guidances']:
                    all_total_norms.append(point_info['total_guidance_norm'])
                    all_learned_norms.append(point_info['learned_correction_norm'])

                    for energy_info in point_info['energies']:
                        e_id = energy_info['energy_id']
                        all_individual_norms[e_id].append(energy_info['guidance_norm'])

                        if energy_info['direction_correct'] is not None:
                            direction_total_count[e_id] += 1
                            if energy_info['direction_correct']:
                                direction_correct_count[e_id] += 1

            for e in range(self.num_energy):
                norms = all_individual_norms[e]
                if len(norms) > 0:
                    f.write(f"能量场 {e+1}:\n")
                    f.write(f"  Guidance 大小: min={min(norms):.6f}, max={max(norms):.6f}, mean={np.mean(norms):.6f}\n")
                    if direction_total_count[e] > 0:
                        f.write(f"  方向正确率: {direction_correct_count[e]}/{direction_total_count[e]} ({direction_correct_count[e]/direction_total_count[e]*100:.1f}%)\n")
                    else:
                        f.write(f"  方向正确率: N/A (所有 guidance 都太小，无法检查方向)\n")
                    f.write("\n")

            if len(all_total_norms) > 0:
                f.write(f"总 Guidance:\n")
                f.write(f"  大小: min={min(all_total_norms):.6f}, max={max(all_total_norms):.6f}, mean={np.mean(all_total_norms):.6f}\n\n")

            if len(all_learned_norms) > 0:
                f.write(f"Learned Correction:\n")
                f.write(f"  大小: min={min(all_learned_norms):.6f}, max={max(all_learned_norms):.6f}, mean={np.mean(all_learned_norms):.6f}\n\n")

        print(f"[GCovGuidance] Guidance 详细记录已保存到: {save_path}")

    def clear_guidance_records(self):
        if hasattr(self, 'guidance_records'):
            self.guidance_records = []

    def train_model(
        self,
        flow_model,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        goal_pos: torch.Tensor,
        num_samples: Optional[int] = None,
    ):
        if not self.config.get('train_online', True):
            print("[GCovGuidance] 在线训练已禁用")
            return

        steps = int(self.config.get('online_train_steps', 1000))
        batch_size = int(self.config.get('online_batch_size', 4))
        if num_samples is not None:
            batch_size = int(num_samples)
        lr = float(self.config.get('online_lr', 1e-4))
        num_ode_steps = int(self.config.get('num_ode_steps', 20))
        conflict_threshold = float(self.config.get('conflict_threshold', 0.5))
        conflict_temperature = float(self.config.get('conflict_temperature', 0.1))

        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()

        print(f"[GCovGuidance] 开始在线训练 ({steps} 步)")
        print(f"  Horizon: {self.horizon}")
        print(f"  Batch size: {batch_size}")
        print(f"  ODE steps: {num_ode_steps}")

        if pcd.dim() != 4:
            raise ValueError(f"pcd must be [1,n_obs_steps,n_points,C], got {pcd.shape}")
        if robot_state_obs.dim() != 3:
            raise ValueError(f"robot_state_obs must be [1,n_obs_steps,10], got {robot_state_obs.shape}")
        if goal_pos.dim() != 3:
            raise ValueError(f"goal_pos must be [1,n_obs_steps,3], got {goal_pos.shape}")

        pcd_b = pcd.repeat(batch_size, 1, 1, 1).to(self.device)
        rs_b = robot_state_obs.repeat(batch_size, 1, 1).to(self.device)
        goal_b = goal_pos.repeat(batch_size, 1, 1).to(self.device)

        self.current_goal_pos = goal_pos[0, 0, :].to(self.device)
        if self.norm_pcd_center is not None:
            self.current_goal_pos = self.current_goal_pos - self.norm_pcd_center.to(self.device)

        if self.norm_pcd_center is not None:
            off = self.norm_pcd_center.to(self.device)
            pcd_b[..., :3] = pcd_b[..., :3] - off
            rs_b[..., :3] = rs_b[..., :3] - off
            goal_b[..., :3] = goal_b[..., :3] - off

        with torch.no_grad():
            nx = flow_model.obs_encoder(pcd_b, rs_b, goal_b)

        try:
            from tqdm import tqdm
            use_tqdm = True
            progress_bar = tqdm(range(steps), desc="Training G_Cov", unit="step", ncols=100)
        except ImportError:
            use_tqdm = False
            progress_bar = range(steps)

        from pfp.common.fm_utils import get_timesteps
        t0, dt = get_timesteps(
            getattr(flow_model, "flow_schedule", "linear"),
            num_ode_steps,
            exp_scale=getattr(flow_model, "exp_scale", None),
        )
        t0 = t0.to(self.device)
        dt = dt.to(self.device)

        pos_emb_scale = float(getattr(flow_model, "pos_emb_scale", 20))

        self._train_flow_model = flow_model
        self._train_nx = nx
        self._train_pos_emb_scale = pos_emb_scale

        for step in progress_bar:

            try:
                z = flow_model._init_noise(batch_size).to(self.device)
            except Exception:
                z = torch.randn(batch_size, self.horizon, 10, device=self.device)

            traj_xs = []
            traj_ts = []
            curr_x = z

            traj_xs.append(curr_x.clone())
            traj_ts.append(torch.ones((batch_size,), device=self.device) * t0[0])

            for t_step in range(num_ode_steps):
                t_tensor = torch.ones((batch_size,), device=self.device) * t0[t_step]

                with torch.no_grad():
                    timesteps = (t_tensor * pos_emb_scale).to(self.device)
                    v_uncond = flow_model.diffusion_net(curr_x, timesteps, global_cond=nx)

                    base_guidance = self._compute_base_guidance(curr_x, t_tensor, v_uncond)
                    learned_corr = self._compute_learned_correction(curr_x, t_tensor, force=True)

                    g_total = base_guidance + learned_corr

                    curr_x = curr_x + (v_uncond + g_total) * dt[t_step]

                traj_xs.append(curr_x.clone())
                traj_ts.append(t_tensor)

            xs_stacked = torch.stack(traj_xs, dim=0)
            xs_pos_stacked = xs_stacked[:, :, :, :3]

            conflict_score = self._compute_conflict_score(xs_pos_stacked[:-1])

            conflict_mask = _smootherstep_gate(conflict_score, conflict_threshold, conflict_temperature)

            active_ratio = (conflict_mask > 0.5).float().mean().item()

            if step == 0 and self.config.get('debug_training', False):
                print(f"\n[Debug] Training Step {step}:")
                print(f"  Conflict score: min={conflict_score.min().item():.4f}, max={conflict_score.max().item():.4f}, mean={conflict_score.mean().item():.4f}")
                print(f"  Active ratio (>{conflict_threshold}): {active_ratio:.2%}")
                print(f"  Conflict mask: min={conflict_mask.min().item():.4f}, max={conflict_mask.max().item():.4f}, mean={conflict_mask.mean().item():.4f}")

            if active_ratio < 1e-6:
                if use_tqdm:
                    progress_bar.set_postfix({'loss': 'skipped', 'conflict': f'{active_ratio:.1%}'})
                continue

            ts_stacked = torch.stack(traj_ts, dim=0)
            loss = self._compute_online_loss(
                xs_stacked, ts_stacked, conflict_mask, batch_size, num_ode_steps
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.learned_guidance_model.parameters(), max_norm=1.0)
            optimizer.step()

            if use_tqdm:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'conflict': f'{active_ratio:.1%}'
                })

            if (step + 1) % 100 == 0 or (step < 10 and self.config.get('debug_training', False)):
                print(f"Step {step+1}/{steps} | Loss: {loss.item():.6f} | Active Conflict: {active_ratio:.1%}")

        if use_tqdm:
            progress_bar.close()

        self.learned_guidance_model.eval()
        self.trained = True

        self._train_flow_model = None
        self._train_nx = None
        self._train_pos_emb_scale = None

        print("[GCovGuidance] 在线训练完成")

    def _compute_online_loss(self, xs_stacked, ts_stacked, conflict_mask, batch_size, num_steps):
        online_loss_type = self.config.get('online_loss_type', 'mse_simple')
        xs_pos_stacked = xs_stacked[:, :, :, :3]

        if online_loss_type == 'mse_simple':
            return self._compute_online_loss_mse_simple(
                xs_pos_stacked, ts_stacked, conflict_mask, batch_size, num_steps
            )
        elif online_loss_type == 'ground_truth':
            return self._compute_online_loss_ground_truth(
                xs_pos_stacked, ts_stacked, conflict_mask, batch_size, num_steps
            )
        elif online_loss_type == 'gradient':
            return self._compute_online_loss_gradient(
                xs_stacked, ts_stacked, conflict_mask, batch_size, num_steps
            )
        else:
            raise ValueError(f"Unsupported online_loss_type: {online_loss_type}")

    def _compute_online_loss_mse_simple(self, xs_pos_stacked, ts_stacked, conflict_mask, batch_size, num_steps):
        x1_pos = xs_pos_stacked[-1]
        r1 = self._compute_terminal_reward(x1_pos)

        tau = float(self.config.get('reward_temp', 1.0))
        reward_weight = F.softmax(r1 / tau, dim=0)

        x_flat = xs_pos_stacked[:-1].reshape(-1, self.horizon, 3)
        t_flat = ts_stacked[:-1].reshape(-1)
        pred_flat = self._scalar_predict(x_flat, t_flat)
        pred = pred_flat.view(num_steps, batch_size)

        target = r1.unsqueeze(0).expand(num_steps, batch_size).detach()

        if conflict_mask is not None:
            conflict_w = conflict_mask.mean(dim=-1)
        else:
            conflict_w = torch.ones_like(pred)

        reward_w = reward_weight.unsqueeze(0).expand(num_steps, batch_size)
        combined_weight = conflict_w * reward_w

        loss_unreduced = F.mse_loss(pred, target, reduction='none')
        loss = (loss_unreduced * combined_weight).sum() / (combined_weight.sum() + 1e-8)

        return loss

    def _compute_online_loss_ground_truth(self, xs_pos_stacked, ts_stacked, conflict_mask, batch_size, num_steps):
        x1_pos = xs_pos_stacked[-1]
        r1 = self._compute_terminal_reward(x1_pos)
        beta = float(self.config.get("energy_temperature", 1.0))
        logits = beta * (r1 - r1.max())
        w_eff = torch.softmax(logits, dim=0)

        x_flat = xs_pos_stacked[:-1].reshape(-1, self.horizon, 3)
        t_flat = ts_stacked[:-1].reshape(-1)

        pred_flat = self._scalar_predict(x_flat, t_flat)
        pred_mat = pred_flat.view(num_steps, batch_size)

        tau = float(self.config.get("policy_temperature", 1.0))
        log_prob = torch.log_softmax(pred_mat / max(tau, 1e-8), dim=1)

        if conflict_mask is not None:
            weight = conflict_mask.mean(dim=-1)
        else:
            weight = torch.ones_like(pred_mat)

        w_mat = w_eff.view(1, batch_size).expand(num_steps, batch_size)
        loss_weights = (w_mat * weight).detach()
        target_dist = loss_weights / (loss_weights.sum(dim=1, keepdim=True) + 1e-8)

        loss_t = -torch.sum(target_dist * log_prob, dim=1)
        return loss_t.mean()

    def _compute_online_loss_gradient(self, xs_stacked, ts_stacked, conflict_mask, batch_size, num_steps):
        x1_full = xs_stacked[-1]
        x1_pos = x1_full[:, :, :3]
        xs_pos_stacked = xs_stacked[:-1, :, :, :3]
        actual_T = xs_pos_stacked.shape[0]

        with torch.no_grad():
            r1 = self._compute_terminal_reward(x1_pos)

        tau = float(self.config.get('reward_temp', 1.0))
        reward_weight = F.softmax(r1 / tau, dim=0)

        x1_exp = x1_pos.unsqueeze(0).expand(actual_T, -1, -1, -1)
        t_exp = (
            ts_stacked[:-1]
            .unsqueeze(-1).unsqueeze(-1)
            .expand(-1, -1, self.horizon, 3)
        )
        one_minus_t = (1.0 - t_exp).clamp_min(1e-6)
        v_cond = (x1_exp - xs_pos_stacked) / one_minus_t

        xs_flat = xs_stacked[:-1].reshape(-1, self.horizon, xs_stacked.shape[-1])
        t_flat = ts_stacked[:-1].reshape(-1)

        with torch.no_grad():
            if hasattr(self, '_train_nx') and self._train_nx is not None:
                nx_expanded = (
                    self._train_nx
                    .unsqueeze(0)
                    .expand(actual_T, *self._train_nx.shape)
                    .reshape(-1, *self._train_nx.shape[1:])
                )
                timesteps = (t_flat * self._train_pos_emb_scale).to(self.device)
                v_theta_full = self._train_flow_model.diffusion_net(
                    xs_flat, timesteps, global_cond=nx_expanded
                )
                v_theta = (
                    v_theta_full[:, :, :3]
                    .view(actual_T, batch_size, self.horizon, 3)
                )
            else:
                v_theta = torch.zeros_like(v_cond)

        target = v_cond.detach()

        x_flat = xs_pos_stacked.reshape(-1, self.horizon, 3)

        if self._is_scalar_model():
            x_flat_req = x_flat.detach().requires_grad_(True)
            g = self._scalar_predict(x_flat_req, t_flat)
            g_phi = torch.autograd.grad(g.sum(), x_flat_req, create_graph=True, retain_graph=True)[0]
        else:
            g_phi = self.learned_guidance_model(x_flat, t_flat)

        g_phi = g_phi.view(actual_T, batch_size, self.horizon, 3)
        pred = g_phi + v_theta.detach()

        reward_w = reward_weight.view(1, batch_size, 1, 1).expand_as(pred)

        if conflict_mask is not None:
            conflict_w = (
                conflict_mask
                .unsqueeze(-1)
                .expand(-1, -1, -1, 3)
                .to(pred.device)
                .float()
            )
        else:
            conflict_w = torch.ones_like(pred)

        combined_weight = conflict_w * reward_w
        loss_unreduced = (pred - target).pow(2)
        loss = (loss_unreduced * combined_weight).sum() / (combined_weight.sum() + 1e-8)

        if self.config.get('debug_loss', False):
            with torch.no_grad():
                print(
                    f"  [MRGM] r1: mean={r1.mean():.3f} max={r1.max():.3f} min={r1.min():.3f} | "
                    f"reward_w: max={reward_weight.max():.3f} min={reward_weight.min():.3f} | "
                    f"loss={loss.item():.6f}"
                )

        return loss
