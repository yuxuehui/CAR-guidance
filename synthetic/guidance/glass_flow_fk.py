"""
GLASS Flow with Feynman-Kac (FK) Corrector.

Implements sequential Monte Carlo (SMC) sampling on top of the GLASS
transition kernel from the tutorial notebook.  The FK corrector adds
three things to the plain GLASS forward pass:

    1. Reward function  r(x̂₁)  evaluated at the denoised prediction.
    2. Importance weights  accumulated in log-space at every backbone step.
    3. Multinomial resampling that collapses low-weight particles.

This file is self-contained: it re-exports the notebook primitives
(`format_batch_variable`, `mult_first_dim`) and the `GlassFlow` base
class so that nothing depends on the notebook itself.

Typical usage
-------------
    glass = GlassFlow(fm_model=pretrained_flow)
    fk    = GlassFlowFK(glass)

    reward = MultiPromptCLIPReward(clip_model, prompts=["a dog", "outdoors"])
    t_backbone = torch.linspace(0, 1, 6, device=device)

    samples = fk.sample(
        K=16,
        reward_fn=reward,
        t_backbone=t_backbone,
        n_inner_steps=20,
        corr_rho=0.5,
        device=device,
        dtype=torch.float32,
    )   # shape: (K, ...)
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Notebook primitives (copied verbatim from glass_flows_tutorial.ipynb)
# ---------------------------------------------------------------------------

def format_batch_variable(t, x_t: Tensor) -> Tensor:
    t = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    if len(t) < x_t.shape[0]:
        assert len(t) == 1
        t = torch.ones(x_t.shape[0], device=x_t.device, dtype=x_t.dtype) * t
    return t


def mult_first_dim(x: Tensor, t: Tensor) -> Tensor:
    if t.ndim == 0:
        return t * x
    t = t.view(-1)
    if x.size(0) != t.size(0):
        raise ValueError("Size of t must match first dimension of x.")
    return x * t.view(-1, *([1] * (x.dim() - 1)))


# ---------------------------------------------------------------------------
# GlassFlow  (re-export of notebook Cell 6)
# ---------------------------------------------------------------------------

class GlassFlow(nn.Module):
    """
    Wraps a pre-trained flow matching model and exposes GLASS transition
    sampling.  Schedulers assume the standard linear path α_t = t,
    σ_t = 1 − t; override the four α/σ methods for other schedulers.
    """

    def __init__(
        self,
        fm_model: nn.Module,
        clip_val: float = 1e-8,
        t_min: float = 0.001,
        t_max: float = 0.999,
        eta_t_clip: float = 200.0,
    ):
        super().__init__()
        self.fm_model   = fm_model
        self.clip_val   = clip_val
        self.t_min      = t_min
        self.t_max      = t_max
        self.eta_t_clip = eta_t_clip

    # -- schedulers (linear path) ------------------------------------------
    def alpha_t(self, t: Tensor) -> Tensor:     return t
    def dot_alpha_t(self, t: Tensor) -> Tensor: return torch.ones_like(t)
    def sigma_t(self, t: Tensor) -> Tensor:     return 1 - t
    def dot_sigma_t(self, t: Tensor) -> Tensor: return -torch.ones_like(t)

    def g_t(self, t: Tensor) -> Tensor:
        return (self.sigma_t(t) / torch.clamp(self.alpha_t(t), min=self.clip_val)) ** 2

    def g_t_inv(self, inp: Tensor) -> Tensor:
        return 1.0 / (1.0 + torch.sqrt(inp))

    # -- denoiser  x̂₁ = D(x_t, t) -----------------------------------------
    def denoiser(self, x_t: Tensor, t, **kwargs) -> Tensor:
        t = format_batch_variable(t, x_t)
        velocity = self.fm_model(x_t=x_t, t=t, **kwargs)
        diff = (mult_first_dim(velocity, self.sigma_t(t))
                - mult_first_dim(x_t, self.dot_sigma_t(t)))
        denom = (self.dot_alpha_t(t) * self.sigma_t(t)
                 - self.dot_sigma_t(t) * self.alpha_t(t))
        return mult_first_dim(diff, 1.0 / torch.clamp(denom, min=self.clip_val))

    # -- internal helpers --------------------------------------------------
    def bar_alpha_s(self, s: Tensor, bar_alpha_final: float) -> Tensor:
        return bar_alpha_final * s

    def dot_bar_alpha_s(self, s: Tensor, bar_alpha_final: float) -> Tensor:
        return bar_alpha_final * torch.ones_like(s)

    def bar_sigma_s(self, s: Tensor, sigma_cond_final: float) -> Tensor:
        return s * sigma_cond_final + (1 - s)

    def dot_bar_sigma_s(self, s: Tensor, sigma_cond_final: float) -> Tensor:
        return torch.ones_like(s) * (sigma_cond_final - 1.0)

    def _stable_inv(self, M: Tensor) -> Tensor:
        return torch.linalg.inv(
            M + 0.0001 * self.clip_val * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        )

    def _glass_denoiser(
        self,
        mu_s: Tensor, Cov_s: Tensor,
        X_t: Tensor, bar_X_s: Tensor,
        dtype: torch.dtype, precdtype: torch.dtype,
        **kwargs,
    ) -> Tensor:
        inv_Cov_s = self._stable_inv(Cov_s)
        bproduct  = mu_s @ inv_Cov_s @ mu_s
        t_star    = self.g_t_inv(1.0 / torch.clamp(bproduct, min=self.clip_val))
        weights   = self.alpha_t(t_star) * (mu_s @ inv_Cov_s) / torch.clamp(bproduct, min=self.clip_val)
        suff_stat = weights[0] * X_t + weights[1] * bar_X_s
        return self.denoiser(x_t=suff_stat.to(dtype), t=t_star.to(dtype), **kwargs).to(dtype=precdtype)

    # -- core transition sampler -------------------------------------------
    def sample_glass_transition(
        self,
        X_t: Tensor,
        t_start: Tensor,
        t_end: Tensor,
        corr_rho: float,
        n_steps: int,
        dtype: torch.dtype,
        device: torch.device,
        return_traj: bool = False,
        precdtype: torch.dtype = torch.float64,
        **kwargs,
    ):
        s_vec = torch.linspace(self.t_min, self.t_max, n_steps + 1, dtype=precdtype, device=device)

        alpha_t_start = self.alpha_t(t_start).to(precdtype)
        alpha_t_end   = self.alpha_t(t_end).to(precdtype)
        sigma_t_start = self.sigma_t(t_start).to(precdtype)
        sigma_t_end   = self.sigma_t(t_end).to(precdtype)

        bar_gamma   = corr_rho * sigma_t_end / torch.clamp(sigma_t_start, min=self.clip_val)
        bar_X_s     = bar_gamma * X_t + torch.randn_like(X_t)

        bar_alpha_final = alpha_t_end - bar_gamma * alpha_t_start
        bar_sigma_final = torch.sqrt(
            torch.clamp(sigma_t_end ** 2 * (1 - corr_rho ** 2), min=0.0)
        )

        bar_alpha_s     = self.bar_alpha_s(s_vec, bar_alpha_final)
        dot_bar_alpha_s = self.dot_bar_alpha_s(s_vec, bar_alpha_final)
        bar_sigma_s     = self.bar_sigma_s(s_vec, bar_sigma_final)
        dot_bar_sigma_s = self.dot_bar_sigma_s(s_vec, bar_sigma_final)

        w_1 = dot_bar_sigma_s / torch.clamp(bar_sigma_s, min=self.clip_val)
        w_2 = dot_bar_alpha_s - w_1 * bar_alpha_s
        w_3 = -w_1 * bar_gamma

        X_t    = X_t.to(dtype=precdtype)
        traj   = [bar_X_s.cpu().detach().float()] if return_traj else None

        for i in range(len(s_vec) - 1):
            mu_s  = torch.tensor(
                [alpha_t_start, bar_alpha_s[i] + bar_gamma * alpha_t_start],
                dtype=precdtype, device=device,
            )
            Cov_s = torch.tensor(
                [[sigma_t_start ** 2, bar_gamma * sigma_t_start ** 2],
                 [bar_gamma * sigma_t_start ** 2,
                  bar_sigma_s[i] ** 2 + bar_gamma ** 2 * sigma_t_start ** 2]],
                dtype=precdtype, device=device,
            )
            glass_den = self._glass_denoiser(mu_s, Cov_s, X_t, bar_X_s, dtype, precdtype, **kwargs)
            velocity  = w_1[i] * bar_X_s + w_2[i] * glass_den + w_3[i] * X_t
            bar_X_s   = bar_X_s + (s_vec[i + 1] - s_vec[i]) * velocity
            if traj is not None:
                traj.append(bar_X_s.cpu().detach())

        return traj if return_traj else bar_X_s

    # -- DDPM-matched correlation ------------------------------------------
    def get_ddpm_corr(self, t_start: Tensor, t_end: Tensor) -> Tensor:
        return (
            self.alpha_t(t_start) * self.sigma_t(t_end)
            / torch.clamp(self.alpha_t(t_end) * self.sigma_t(t_start), min=self.clip_val)
        )

    def sample_glass_transition_ddpm(self, t_start, t_end, **kwargs):
        return self.sample_glass_transition(
            t_start=t_start, t_end=t_end,
            corr_rho=self.get_ddpm_corr(t_start, t_end),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# FK Corrector
# ---------------------------------------------------------------------------

class GlassFlowFK(nn.Module):
    """
    Feynman-Kac (FK) corrector wrapped around a `GlassFlow` instance.

    At each backbone step the FK corrector:
        (1) propagates all K particles through a GLASS transition,
        (2) evaluates the reward at the denoised prediction  x̂₁ = D(X_t, t),
        (3) resamples multinomially from the current importance weights.

    The result approximates sampling from  p(x₁) ∝ exp(Σ_t log r_t(x̂₁)).

    Args:
        glass_flow:  A `GlassFlow` (or compatible) instance.
        resample_every: Resample at every N backbone intervals (default 1,
                        i.e. after every transition).  Set > 1 to delay
                        resampling and reduce sample impoverishment.
    """

    def __init__(self, glass_flow: GlassFlow, resample_every: int = 1):
        super().__init__()
        self.glass_flow    = glass_flow
        self.resample_every = resample_every

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        K: int,
        reward_fn: Callable[[Tensor], Tensor],
        t_backbone: Tensor,
        n_inner_steps: int,
        corr_rho: float,
        device: torch.device,
        dtype: torch.dtype,
        x_init: Optional[Tensor] = None,
        return_log_weights: bool = False,
        return_intermediates: bool = False,
        verbose: bool = False,
        **glass_kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Run the FK corrector and return K (approximate) posterior samples.

        Args:
            K:               Number of particles.
            reward_fn:       Callable  x̂₁ (K, ...) -> log_r (K,).
                             Should return log-reward values (log-space scores).
            t_backbone:      1-D tensor of T+1 backbone times, e.g.
                             ``torch.linspace(0, 1, 6)``.
            n_inner_steps:   Euler steps inside each GLASS transition.
            corr_rho:        Transition correlation ρ  (0 = max stochastic,
                             1 = deterministic ODE).
            device / dtype:  Standard torch device/dtype.
            x_init:          Optional initial particles  (K, ...).  If None,
                             samples K particles from N(0,I).
            return_log_weights:   Also return the final log-weight tensor.
            return_intermediates: If True, also return a list of (K, ...) CPU
                             tensors — one snapshot per backbone step
                             (after resampling), including the initial noise.
                             List length = len(t_backbone).
            verbose:         Print ESS at every backbone step.
            **glass_kwargs:  Extra kwargs forwarded to `sample_glass_transition`.

        Returns:
            X  of shape (K, ...).
            If return_intermediates: additionally a list[Tensor] of snapshots.
            If return_log_weights:   additionally the log-weight tensor (K,).
        """
        t_backbone = t_backbone.to(device=device, dtype=dtype)
        n_steps    = len(t_backbone) - 1

        # ── 1. Initialise particles ──────────────────────────────────────
        if x_init is not None:
            X = x_init.to(device=device, dtype=dtype)
            assert X.shape[0] == K, f"x_init batch dim {X.shape[0]} != K={K}"
        else:
            shape = getattr(self.glass_flow, "data_shape", None)
            if shape is None:
                raise ValueError(
                    "Provide x_init or set glass_flow.data_shape = (C, H, W)."
                )
            X = torch.randn(K, *shape, device=device, dtype=dtype)

        log_weights   = torch.zeros(K, device=device, dtype=dtype)
        intermediates = [X.cpu().float()] if return_intermediates else None

        # ── 2. FK loop ───────────────────────────────────────────────────
        for idx in range(n_steps):
            t_start = t_backbone[idx]
            t_end   = t_backbone[idx + 1]

            # (a) Propagate
            X = self.glass_flow.sample_glass_transition(
                X_t=X,
                t_start=t_start,
                t_end=t_end,
                corr_rho=corr_rho,
                n_steps=n_inner_steps,
                dtype=dtype,
                device=device,
                **glass_kwargs,
            ).to(dtype=dtype)             # shape: (K, ...)

            # (b) Weight: reward at denoised prediction
            x1_hat      = self.glass_flow.denoiser(X, t=t_end).to(dtype=dtype)
            log_r       = reward_fn(x1_hat)                  # (K,)
            log_weights = log_weights + log_r

            # (c) Resample (every `resample_every` steps)
            if (idx + 1) % self.resample_every == 0:
                weights = torch.softmax(log_weights, dim=0)
                if verbose:
                    ess = 1.0 / (weights ** 2).sum().item()
                    print(f"[GlassFlowFK] step {idx+1}/{n_steps}  "
                          f"ESS={ess:.1f}/{K}  "
                          f"log_w: min={log_weights.min():.2f} "
                          f"max={log_weights.max():.2f}")
                indices     = torch.multinomial(weights, K, replacement=True)
                X           = X[indices]
                log_weights = torch.zeros(K, device=device, dtype=dtype)

            if intermediates is not None:
                intermediates.append(X.cpu().float())

        out = [X]
        if return_intermediates:
            out.append(intermediates)   # list of len(t_backbone) tensors
        if return_log_weights:
            out.append(log_weights)
        return out[0] if len(out) == 1 else tuple(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_batched(
        self,
        n_groups: int,
        K: int,
        reward_fn: Callable[[Tensor], Tensor],
        t_backbone: Tensor,
        n_inner_steps: int,
        corr_rho: float,
        device: torch.device,
        dtype: torch.dtype,
        return_intermediates: bool = False,
        verbose: bool = False,
        **glass_kwargs,
    ) -> "Tensor | tuple[Tensor, list]":
        """
        Batched FK corrector: ``n_groups`` independent runs in parallel, each
        carrying ``K`` particles.  Total forward-pass batch = ``n_groups * K``.
        Resampling is strictly within each group (no cross-contamination).

        Returns ``(n_groups, D)`` — one output sample per group, drawn uniformly
        from the K post-resampling particles.

        If ``return_intermediates=True``, also returns a list of ``(n_groups, D)``
        CPU tensors (one per backbone step, length = ``len(t_backbone)``),
        tracking particle-0 of each group as a marginal snapshot.

        Tip — OOM: for large ``n_groups * K`` reduce ``n_groups`` or call this
        method in chunks by splitting ``n_groups`` manually.
        """
        t_backbone = t_backbone.to(device=device, dtype=dtype)
        n_steps    = len(t_backbone) - 1
        G          = n_groups
        shape      = getattr(self.glass_flow, "data_shape", None)
        if shape is None:
            raise ValueError(
                "Set glass_flow.data_shape = (D,) or use sample() with x_init."
            )

        # ── 1. Init: (G*K, D) ───────────────────────────────────────────
        X           = torch.randn(G * K, *shape, device=device, dtype=dtype)
        log_weights = torch.zeros(G * K, device=device, dtype=dtype)

        # group_offsets[g] = g*K, shape (G, 1) — for index arithmetic
        group_offsets = torch.arange(G, device=device).unsqueeze(1) * K  # (G,1)

        # Snapshot particle-0 of each group as initial marginal
        intermediates = [X[::K].cpu().float()] if return_intermediates else None

        # ── 2. FK loop ───────────────────────────────────────────────────
        for idx in range(n_steps):
            t_start = t_backbone[idx]
            t_end   = t_backbone[idx + 1]

            # (a) Propagate — one call for the full (G*K, D) batch
            X = self.glass_flow.sample_glass_transition(
                X_t=X,
                t_start=t_start,
                t_end=t_end,
                corr_rho=corr_rho,
                n_steps=n_inner_steps,
                dtype=dtype,
                device=device,
                **glass_kwargs,
            ).to(dtype=dtype)

            # (b) Weight: reward at denoised prediction
            x1_hat      = self.glass_flow.denoiser(X, t=t_end).to(dtype=dtype)
            log_r       = reward_fn(x1_hat)          # (G*K,)
            log_weights = log_weights + log_r

            # (c) Per-group resample (every resample_every steps)
            if (idx + 1) % self.resample_every == 0:
                log_w_g = log_weights.view(G, K)              # (G, K)
                w_g     = torch.softmax(log_w_g, dim=1)       # normalise per group
                if verbose:
                    ess_g = 1.0 / (w_g ** 2).sum(dim=1)       # (G,)
                    print(f"[sample_batched] step {idx+1}/{n_steps}  "
                          f"ESS mean={ess_g.mean():.1f}  min={ess_g.min():.1f}/{K}")
                idx_g    = torch.multinomial(w_g, K, replacement=True)  # (G, K)
                flat_idx = (idx_g + group_offsets).view(G * K)           # (G*K,)
                X           = X[flat_idx]
                log_weights = torch.zeros(G * K, device=device, dtype=dtype)

            if intermediates is not None:
                # particle-0 of each group is a valid marginal sample post-resample
                intermediates.append(X[::K].cpu().float())   # (G, D)

        # ── 3. Pick 1 particle per group uniformly ───────────────────────
        pick      = torch.randint(0, K, (G,), device=device)       # (G,)
        flat_pick = pick + torch.arange(G, device=device) * K      # (G,)
        out       = X[flat_pick]                                    # (G, D)

        if return_intermediates:
            return out, intermediates
        return out

    # ------------------------------------------------------------------
    def ess(self, log_weights: Tensor) -> float:
        """Effective Sample Size from log-weights."""
        w = torch.softmax(log_weights, dim=0)
        return float(1.0 / (w ** 2).sum())


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

class MultiPromptCLIPReward(nn.Module):
    """
    Log-reward  Σ_i score(x̂₁, prompt_i)  for multi-prompt FK sampling.

    Summing CLIP scores in log-space is equivalent to the product of
    per-prompt probabilities, which means the FK corrector will find
    images that satisfy all prompts simultaneously without any gradient
    conflict between them.

    Args:
        clip_model: Any object with a `.score(images, text) -> (B,)` method
                    returning log-probabilities (or any monotone reward).
        prompts:    List of text prompts.
        weights:    Optional per-prompt weights.  Defaults to uniform.
    """

    def __init__(
        self,
        clip_model,
        prompts: List[str],
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.prompts    = prompts
        self.weights    = (
            [1.0] * len(prompts) if weights is None else list(weights)
        )
        assert len(self.weights) == len(prompts)

    def forward(self, x1_hat: Tensor) -> Tensor:
        """
        Args:
            x1_hat: Denoised images  (K, C, H, W) or (K, D).
        Returns:
            log_r: (K,) summed log-reward.
        """
        total = torch.zeros(x1_hat.shape[0], device=x1_hat.device, dtype=x1_hat.dtype)
        for prompt, w in zip(self.prompts, self.weights):
            total = total + w * self.clip_model.score(x1_hat, prompt)
        return total


class GaussianReward(nn.Module):
    """
    Isotropic Gaussian log-reward  −‖x̂₁ − μ‖² / (2σ²).

    Useful for synthetic experiments where the target is a known point.
    """

    def __init__(self, mu: Tensor, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("mu", mu)
        self.sigma = sigma

    def forward(self, x1_hat: Tensor) -> Tensor:
        diff = x1_hat - self.mu.to(x1_hat)
        return -0.5 * (diff ** 2).sum(dim=tuple(range(1, diff.dim()))) / self.sigma ** 2
