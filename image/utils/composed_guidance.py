import torch
import torch.nn as nn
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Map t from [0, 1] into a high-frequency feature space.

    This avoids time-blindness in the residual net. The scale controls the
    network's sensitivity to time changes and keeps the magnitudes stable
    without multiplying t by 999.
    """
    def __init__(self, embedding_size=256, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [B, 1]
        x_proj = x[:, 0:1] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# Vector-field network for gradient regression (output dim = in_channels).
class ImageResidualEnergyNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        time_dim = base_channels * 4
        self.t_proj = GaussianFourierProjection(embedding_size=time_dim, scale=30.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.inc = DoubleConv(base_channels, base_channels)      
        self.down1 = Down(base_channels, base_channels * 2)      
        self.down2 = Down(base_channels * 2, base_channels * 4)  
        self.bot1 = DoubleConv(base_channels * 4, base_channels * 4)
        self.bot2 = DoubleConv(base_channels * 4, base_channels * 4)
        self.up1 = Up(base_channels * 6, base_channels * 2) 
        self.up2 = Up(base_channels * 3, base_channels)     
        
        # Output a vector field with in_channels (e.g. 3), not a scalar.
        self.outc = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x, t):
        if t.dim() == 1: t = t.view(-1, 1)
        
        t_emb = self.t_proj(t)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        x_start = self.input_proj(x)
        x1 = self.inc(x_start)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = x3 + t_emb
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        
        # Directly output the vector field [B, C, H, W].
        out = self.outc(x) 
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(8, in_channels), 
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ImageGCovGGMOnlineGuidance:
    def __init__(self, base_model, loss_fns, scales, config, learnable=True, conflict_weight=0.1):
        self.flow_model = base_model
        self.classifiers = loss_fns    # plays the role of classifiers (here L_N_list)
        self.scales = scales
        self.cfg = config
        self.device = config.device
        
        # Dummy targets to match the 2D call signature.
        self.targets = [None] * len(loss_fns) 
        self.distribution = None       # no toy GT distribution in the image setting
        self.conflict_weight = conflict_weight
        
        # Residual net (3-channel images by default).
        self.learned_guidance_model = ImageResidualEnergyNet(in_channels=3).to(self.device)
        self._prepare_models_for_subclass()

    def __call__(self, x, t):
        """Callable wrapper so the field can be passed directly to generate_traj."""
        v_uncond = self.flow_model(x, t)
        g = self.compute_guidance(x, t, v_uncond)
        return v_uncond + g

    def _prepare_models_for_subclass(self):
        # Initialize last layer to zero for warm start (residual = 0 initially)
        if hasattr(self, 'learned_guidance_model'):
            nn.init.zeros_(self.learned_guidance_model.outc.weight)
            nn.init.zeros_(self.learned_guidance_model.outc.bias)

    def _compute_trajectory_conflict_mask(self, xs_stacked, ts_stacked, num_steps, batch_size):
        # ts_stacked shape: (T, B, 1), aligned with xs_stacked.

        conflict_list = []
        for step in range(num_steps):
            x_t = xs_stacked[step]              # [B, C, H, W]
            t_t = ts_stacked[step] * 999.0      # [B, 1], rescaled to 0~999

            with torch.no_grad():
                c_t = self._compute_conflict_score(x_t, self.targets, t=t_t)
            conflict_list.append(c_t)
            
        # Stack back to shape [T, B].
        conflict = torch.stack(conflict_list, dim=0)
        
        threshold, _ = self._get_conflict_threshold_and_temperature()
        
        conflict_mask = (conflict > threshold).float()
        active_ratio = conflict_mask.mean().item()
        return conflict_mask, active_ratio

    def _compute_trajectory_weights_ground_truth(self, x1, batch_size):
        """Compute target label distribution p*(i) from terminal reward."""
        t_final = torch.ones(batch_size, 1, device=self.device) * 999.0
        with torch.no_grad():
            # No GT distribution for images; use base_log_prob as the reward.
            v_final = self.flow_model(x1, t_final.view(-1))
            base_log_prob = self._compute_g_cov_g_energy(
                x1, t_final, v_final,
                self.classifiers, self.targets, self.scales, self.cfg
            ).flatten()
            # Penalize the terminal x1 by its conflict score.
            terminal_conflict = self._compute_conflict_score(x1, self.targets)
            
            cw = getattr(self.cfg, "conflict_weight", 10.0)
            
            # Final reward = semantic score - conflict penalty.
            r1 = base_log_prob - cw * terminal_conflict

            beta = getattr(self.cfg, "energy_temperature", 1.0)
            logits = beta * r1
            logits = logits - logits.max()
            w_eff = torch.softmax(logits, dim=0)
            w_eff = w_eff.unsqueeze(-1)

        return w_eff, r1

    def _compute_trajectory_weights(self, x1, batch_size):
        """Compute effective weights based on base quality."""
        t_final = torch.ones(batch_size, 1, device=self.device) * 999.0
        with torch.no_grad():
            v_final = self.flow_model(x1, t_final.view(-1))
            base_log_prob = self._compute_g_cov_g_energy(
                x1, t_final, v_final, self.classifiers, self.targets, self.scales, self.cfg
            ).flatten()
            
            if getattr(self.cfg, "weight_zscore", True):
                med = base_log_prob.median()
                mad = (base_log_prob - med).abs().median() + 1e-8
                r1_norm = (base_log_prob - med) / mad
            else:
                r1_norm = base_log_prob
            
            beta = getattr(self.cfg, "energy_temperature", 1.0)
            scores = -beta * r1_norm
            scores = scores - scores.max() 
            w_eff = torch.softmax(scores, dim=0).unsqueeze(-1)
            
            wu = getattr(self.cfg, "weight_uniform_mix", 0.01)
            if wu > 0:
                w_eff = (1 - wu) * w_eff + wu * (1.0 / batch_size)
            
            w_eff = torch.clamp(
                w_eff,
                min=getattr(self.cfg, "weight_min", 1e-4),
                max=getattr(self.cfg, "weight_clamp_max", 1e6),
            )
        
        return w_eff, base_log_prob

    def _compute_online_loss_gradient(self, xs_stacked, ts_stacked, vs_stacked,
                                    conflict_mask, num_steps, batch_size,
                                    x1_final=None, original_images=None):
        """
        MSE loss with internal backward and micro-batch gradient accumulation.

        Returns (loss_float, stats_dict) rather than a tensor. The caller must
        call optimizer.zero_grad() before this and optimizer.step() afterwards.
        """
        T, B, C, H, W = xs_stacked.shape
        space_dim = C * H * W
        total_samples = T * B

        xs_flat_img = xs_stacked.view(-1, C, H, W)
        ts_flat = ts_stacked.view(-1, 1)

        # 1. Compute the target (teacher) gradient: energy gradient at x1_final.
        if x1_final is None:
            x1_final = xs_stacked[-1]

        x1_req = x1_final.detach().requires_grad_(True)

        with torch.enable_grad():
            # Get the total energy scalar, then take its gradient w.r.t. x1.
            t_final = torch.ones(B, device=self.device) * 999.0
            v_final_dummy = torch.zeros_like(x1_req)  # no v_uncond needed at the endpoint

            # Average the teacher gradient over K passes to reduce noise: CLIP's
            # DiffAugment makes each grad(E(x1)) noisy, so we average the
            # gradient (not the energy). retain_graph=True for repeated grads.
            K = getattr(self.cfg, "reward_avg_K", 8)
            grad_energy = 0.0
            for _k in range(K):
                energy = self._compute_g_cov_g_energy(
                    x1_req, t_final, v_final_dummy,
                    self.classifiers, self.targets, self.scales, self.cfg
                )
                # Higher energy is better -> ascent direction is the target.
                g = torch.autograd.grad(energy.sum(), x1_req, retain_graph=True)[0]
                grad_energy = grad_energy + g
            grad_energy = grad_energy / K
            # target = grad(E(x1)): teach the net to move along increasing energy.
            clip_norm = grad_energy.reshape(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)

        total_grad = grad_energy.detach()

        # Target the net should regress onto this gradient direction.
        target_final    = total_grad                                        # [B, C, H, W]
        target_expanded = target_final.unsqueeze(0).expand(T, -1, -1, -1, -1)
        target_flat     = target_expanded.reshape(-1, space_dim)

        # 2. Train the student (micro-batch + internal backward).
        micro_batch  = batch_size
        num_chunks   = (total_samples + micro_batch - 1) // micro_batch
        total_loss_val = 0.0
        stats = {}

        for idx in range(0, total_samples, micro_batch):
            x_chunk    = xs_flat_img[idx : idx + micro_batch].detach()
            t_chunk    = ts_flat[idx : idx + micro_batch]
            targ_chunk = target_flat[idx : idx + micro_batch]

            pred_grad  = self.learned_guidance_model(x_chunk, t_chunk)
            pred_flat  = pred_grad.view(micro_batch, -1)
            targ_flat_view = targ_chunk.view(micro_batch, -1)

            # MSE Loss
            loss_chunk = ((pred_flat - targ_flat_view) ** 2).mean(dim=-1)

            # Weight by the conflict mask.
            if conflict_mask is not None:
                mask_full  = conflict_mask.view(-1)
                mask_chunk = mask_full[idx : idx + micro_batch].to(loss_chunk.device)
                weight     = mask_chunk
            else:
                weight = torch.ones_like(loss_chunk)

            loss_scalar  = (loss_chunk * weight).sum() / (weight.sum() + 1e-8)
            loss_backward = loss_scalar / num_chunks
            loss_backward.backward()          # internal backward, accumulate grads
            total_loss_val += loss_scalar.item()

            # Record stats for the first chunk.
            if idx == 0:
                with torch.no_grad():
                    p_vec = pred_flat.view(-1)
                    t_vec = targ_flat_view.view(-1)
                    cosine  = F.cosine_similarity(p_vec, t_vec, dim=0, eps=1e-8)
                    p_norm  = p_vec.norm().item()
                    tn_norm = t_vec.norm().item()
                    stats['cosine']      = cosine.item()
                    stats['pred_norm']   = p_norm
                    stats['targ_norm']   = tn_norm
                    stats['ratio']       = p_norm / (tn_norm + 1e-8)
                    stats['mask_ratio']  = weight.mean().item()

            del x_chunk, pred_grad, loss_backward

        avg_loss = total_loss_val / num_chunks
        return avg_loss, stats

    def _log_online_training_progress_ground_truth(self, step, total_steps, loss, active_ratio, w_eff, r1, angle=None):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt(
                    (w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8
                )
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0

            # Format r1 as a 2-decimal string list for readability.
            r1_str = "[" + ", ".join([f"{val:.2f}" for val in r1.tolist()]) + "]"
        print(
            f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
            f"Active Conflict: {active_ratio:.1%} | "
            f"corr(w_eff, reward): {corr:.3f}",
            f"Angle(Base, Res): {angle:.1f}°",
            f"r1: {r1_str}",
            flush=True
        )
    
    def _log_online_training_progress(self, step, total_steps, loss, active_ratio, w_eff, base_log_prob):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            neg_r1 = -base_log_prob
            
            if len(w_eff_flat) > 1:
                w_mean, r_mean = w_eff_flat.mean(), neg_r1.mean()
                w_centered = w_eff_flat - w_mean
                r_centered = neg_r1 - r_mean
                numerator = (w_centered * r_centered).sum()
                denominator = torch.sqrt((w_centered.pow(2).sum() * r_centered.pow(2).sum()) + 1e-8)
                corr = (numerator / denominator).item() if denominator > 1e-8 else 0.0
            else:
                corr = 0.0
        
        print(f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
              f"Active Conflict: {active_ratio:.1%} | "
              f"corr(w_eff, -r1): {corr:.3f}", flush=True)

    def train_model(self, z0, num_steps=100, steps=15):
        # NaN/Inf guard.
        if torch.isnan(z0).any() or torch.isinf(z0).any():
            print("!!! WARNING: z0 contains NaN/Inf! Clamping.")
            z0 = torch.nan_to_num(z0, nan=0.0)
            z0 = torch.clamp(z0, -5.0, 5.0)
        # batch_size must equal the real batch size of z0 (latent_batch).
        batch_size = z0.shape[0]  
        steps = getattr(self.cfg, "guidance_train_steps", 25) 
        lr = getattr(self.cfg, "guidance_lr", 1e-3)
        log_interval = 1  # image training is slow; log frequently
        
        # ODE solver params.
        num_steps = 10
        dt = 1.0 / num_steps
        eps = 1e-3
        
        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()
        
        print(f"[ImageGCovGGMOnlineGuidance] Training Online Residual model ({steps} steps)")
        
        for i in range(steps):
            self._epoch_angles = []  # reset per-epoch angle history
            curr_x = z0.detach().clone()
            traj_xs = []
            traj_ts = []
            traj_vs = []  # store v_uncond at each step
            
            for step in range(num_steps):
                # Match generate_traj's time scaling.
                t_norm = step / num_steps * (1. - eps) + eps
                t_tensor = torch.full((batch_size, 1), t_norm, device=self.device)
                t_model = t_tensor * 999.0
                
                traj_xs.append(curr_x.clone())
                # Store the 0~1 t_tensor to keep the later MSE loss well-scaled.
                traj_ts.append(t_tensor)
                
                # Keep the U-Net under no_grad: the algorithm needs no gradient
                # w.r.t. v_uncond, and building the graph wastes memory.
                x_in = curr_x.detach()
                with torch.no_grad():
                    v_uncond = self.flow_model(x_in, t_model.view(-1))
                if torch.isnan(v_uncond).any():
                    v_uncond = torch.nan_to_num(v_uncond, nan=0.0)
                traj_vs.append(v_uncond.detach().clone())
                should_log_norms = (i+1) % log_interval == 0 and step == 0  
                current_epoch = i + 1
                if (current_epoch == 1 or current_epoch == steps // 2 or current_epoch == steps) and step == 0:
                    vis_step_val = current_epoch
                else:
                    vis_step_val = None
                g_total = self.compute_guidance(x_in, t_model, v_uncond)
                d_x = v_uncond + g_total
                
                curr_x = curr_x + d_x.detach() * dt
            
            x1 = curr_x
            
            xs_stacked = torch.stack(traj_xs, dim=0)
            ts_stacked = torch.stack(traj_ts, dim=0)
            vs_stacked = torch.stack(traj_vs, dim=0)
            
            conflict_mask, active_ratio = self._compute_trajectory_conflict_mask(
                xs_stacked, ts_stacked, num_steps, batch_size
            )
            if conflict_mask is None or active_ratio < 1e-6:
                print(f"Online-Step {i+1}/{steps} Skipped: Low Conflict ({active_ratio:.1%})", flush=True)
                continue
            
            w_eff, r1 = self._compute_trajectory_weights_ground_truth(x1, batch_size)
            optimizer.zero_grad()   # must clear before compute_online_loss_gradient
            loss_val, stats = self._compute_online_loss_gradient(
                xs_stacked, ts_stacked, vs_stacked, conflict_mask,
                num_steps, batch_size, x1_final=x1            # endpoint for teacher gradient
            )

            # Average angle for the current epoch.
            avg_angle = sum(self._epoch_angles) / len(self._epoch_angles) if len(self._epoch_angles) > 0 else 0.0
            if log_interval and (i+1) % log_interval == 0:
                print(f"\n=== Online-Step {i+1}/{steps} ===")
                print(f"Loss (MSE): {loss_val:.6e}")
                print(f"Direction Cosine: {stats['cosine']:.4f} {'+' if stats['cosine'] > 0 else '-'}")
                print(f"Norms: Pred={stats['pred_norm']:.6f} | Targ={stats['targ_norm']:.6f} | Ratio={stats['ratio']:.4f}")
                print(f"Active Conflict: {active_ratio:.1%}")

                with torch.no_grad():
                    last_layer = self.learned_guidance_model.outc
                    b_val  = last_layer.bias.mean().item() if last_layer.bias is not None else 0.0
                    w_norm = last_layer.weight.norm().item()
                    if last_layer.weight.grad is not None:
                        grad_norm = last_layer.weight.grad.norm().item()
                        print(f"Backprop Grad Norm: {grad_norm:.6f}")
                    else:
                        print("!!! CRITICAL: No Gradient !!!")
                    print(f"[Trap Check] Bias Mean: {b_val:.4f} | Weight Norm: {w_norm:.6f}")
                print("="*40)



            # Gradient clipping to avoid NaNs.
            torch.nn.utils.clip_grad_norm_(self.learned_guidance_model.parameters(), 1.0)
            optimizer.step()

        self.learned_guidance_model.eval()

    def _prepare_input_for_grad(self, x, need_higher_order=False):
        """Prepare input tensor for gradient computation."""
        if need_higher_order:
            return x if x.requires_grad else x.clone().detach().requires_grad_(True)
        return x.detach().requires_grad_(True) if not x.requires_grad else x

    @torch.enable_grad()
    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        if not self.learned_guidance_model:
            return torch.zeros_like(x)

        x_req = self._prepare_input_for_grad(x, need_higher_order)
        t_vec = t.view(-1)
        t_model = t.view(-1, 1)

        base_energy = self._compute_g_cov_g_energy(
            x_req, t_vec, v_uncond.detach(),
            self.classifiers, self.targets, self.scales, self.cfg
        )
        g_base = torch.autograd.grad(base_energy.sum(), x_req, retain_graph=True)[0]

        eps = 1e-3
        t_norm = (t_model / 999.0 - eps) / (1.0 - eps)
        g_res_raw = self.learned_guidance_model(x_req, t_norm)

        # Normalize to a pure direction and apply a uniform lr_res magnitude
        # (paired with normalized-target training so the residual stays effective).
        g_res_flat = g_res_raw.view(x_req.shape[0], -1)
        g_res_norm = g_res_flat.norm(dim=1, keepdim=True).view(x_req.shape[0], 1, 1, 1) + 1e-8
        lr_res = getattr(self.cfg, "lr_res", 30.0)
        g_res = lr_res * (g_res_raw / g_res_norm)

        # Apply the conflict gate.
        if self.classifiers:
            with torch.no_grad():
                conflict = self._compute_conflict_score(x_req.detach(), self.targets, t=t_model)
            if conflict is not None:
                threshold, temperature = self._get_conflict_threshold_and_temperature()
                weight = torch.sigmoid((conflict - threshold) / temperature).view(-1, 1, 1, 1)
                grad = g_base + weight * g_res
            else:
                grad = g_base + g_res
        else:
            grad = g_base + g_res

        return grad

    def compute_direct_conflict_score(self, x, t=None, epsilon=1e-8):
        grads = []
        need_double_grad = x.requires_grad

        with torch.enable_grad():
            x_req = x if need_double_grad else x.detach().requires_grad_(True)

            # Re-run the flow model so the Jacobian is connected to x_req.
            if t is not None:
                t_vec = t.view(-1)
                t_4d  = t.view(-1, 1, 1, 1)
                eps   = 1e-3
                t_norm = (t_4d / 999.0 - eps) / (1.0 - eps)

                # v_t is computed from x_req to keep the graph connected.
                v_t    = self.flow_model(x_req, t_vec)
                x1_est = x_req + (1.0 - t_norm) * v_t
            else:
                # Fallback when t is not provided.
                x1_est = x_req

            for L_N in self.classifiers:
                loss        = L_N(x1_est)
                loss_scalar = loss.sum() if loss.dim() > 0 else loss
                g = torch.autograd.grad(
                    loss_scalar, x_req,
                    retain_graph=True,
                    create_graph=need_double_grad
                )[0]
                grads.append(g)

        return self.compute_conflict_score(grads, epsilon=epsilon)

    def compute_conflict_score(self, grads, epsilon=1e-8):
        if not grads:
            raise ValueError("No gradients provided.")

        first = grads[0]
        B = first.shape[0] if first.dim() > 1 else 1
        device = first.device

        if len(grads) < 2:
            return torch.zeros(B, device=device)

        pairwise_scores = []
        zero_thr = getattr(self.cfg, "zero_gradient_threshold_direct", 1e-6)

        for i in range(len(grads)):
            gi = grads[i].view(B, -1)
            norm_i = gi.norm(dim=-1, keepdim=True)
            gi_unit = gi / (norm_i + epsilon)

            for j in range(i + 1, len(grads)):
                gj = grads[j].view(B, -1)
                norm_j = gj.norm(dim=-1, keepdim=True)
                gj_unit = gj / (norm_j + epsilon)

                cos = (gi_unit * gj_unit).sum(dim=-1)

                norm_i_flat = norm_i.squeeze(-1)
                norm_j_flat = norm_j.squeeze(-1)
                near_zero = (norm_i_flat < zero_thr) | (norm_j_flat < zero_thr)

                conflict = -cos + 1.0
                conflict = torch.where(near_zero, torch.zeros_like(conflict), conflict)
                pairwise_scores.append(conflict)

        conflict_avg = torch.stack(pairwise_scores, dim=0).mean(dim=0)
        return conflict_avg

    def _compute_conflict_score(self, x: torch.Tensor, labels, t=None) -> torch.Tensor:
        return self.compute_direct_conflict_score(x, t=t)

    def _compute_g_cov_g_energy(self, x, t, v_uncond, classifiers, targets, scales, cfg):
        t_ = t.view(-1, 1, 1, 1)
        eps = 1e-3
        
        # Recover t_norm from t_model, where t_model = (t_norm*(1-eps)+eps)*999,
        # i.e. t_norm = (t_ / 999.0 - eps) / (1.0 - eps).
        t_norm = (t_ / 999.0 - eps) / (1.0 - eps)
        
        # Estimate the endpoint: x1_pred = x_t + v_t * (1.0 - t_norm).
        x1_est = x if getattr(cfg, "estimate_x1", False) else x + (1.0 - t_norm) * v_uncond
        
        scales = scales if scales and len(scales) == len(classifiers) else [1.0] * len(classifiers)

        total_obj = torch.zeros(x.shape[0], device=x.device)
        for clf, target, lam in zip(classifiers, targets, scales):
            # clf is a function in L_N_list returning the loss directly; it
            # already applies its own alpha scaling, so do not multiply again.
            loss = clf(x1_est)
            # Define the energy as E = -lam * loss so that grad(E) = -lam *
            # grad(loss), i.e. moving x along grad(E) decreases the loss.
            total_obj = total_obj - float(lam) * loss
        return total_obj

    def _get_conflict_threshold_and_temperature(self):
        threshold = getattr(self.cfg, "conflict_threshold", 0.1)
        temperature = getattr(self.cfg, "conflict_temperature", 0.1)
        return threshold, temperature