"""Energy visualization helpers."""

import os
from typing import Callable, Optional, Sequence, Dict

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


def _smootherstep(x, edge0=0.0, edge1=1.0):
    """
    Smooth interpolation function (Ken Perlin's smootherstep).
    Returns a smooth S-curve interpolation between 0 and 1.
    More smooth than sigmoid, with zero first and second derivatives at edges.
    
    Args:
        x: Input tensor (can be any value)
        edge0: Lower edge (output is 0 when x <= edge0)
        edge1: Upper edge (output is 1 when x >= edge1)
    
    Returns:
        Smoothly interpolated value between 0 and 1
    """
    edge_diff = edge1 - edge0
    if isinstance(edge_diff, torch.Tensor):
        edge_diff = torch.clamp(edge_diff, min=1e-8)
    elif abs(edge_diff) < 1e-8:
        edge_diff = 1e-8
    
    x_normalized = (x - edge0) / edge_diff
    x_clamped = torch.clamp(x_normalized, 0.0, 1.0)
    # Smootherstep: 6t^5 - 15t^4 + 10t^3
    return x_clamped * x_clamped * x_clamped * (x_clamped * (x_clamped * 6.0 - 15.0) + 10.0)


class EnergyVisualizer:
    """Encapsulates energy computation and plotting utilities."""

    def __init__(self, device: torch.device, cfg, num_points_fn: Callable[[], int]):
        self.device = device
        self.cfg = cfg
        self._num_points_fn = num_points_fn
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.default_guidance_ckpt_dir = os.path.join(
            base_dir, "guidance", "pretrained_guidance", "guidance_ckpt"
        )
        self._guidance_matching_cache: Dict[str, nn.Module] = {}

    @property
    def num_points(self) -> int:
        return self._num_points_fn()

    def _resolve_guidance_mode(self, guided_field) -> str:
        if guided_field is None:
            return "none"
        identifier = getattr(guided_field, "guidance_identifier", None)
        if isinstance(identifier, str):
            return identifier.lower()
        if callable(identifier):
            return getattr(identifier, "__name__", "g_cov_g").lower()
        return "g_cov_g"

    def _compute_classifier_guidance_energy(self, x1: torch.Tensor, guided_field) -> torch.Tensor:
        """
        Compute classifier guidance ENERGY: E_clf = Σ λ * (-log p(y|x))
        Convention: lower is better (energy)
        """
        if (
            guided_field is None
            or not hasattr(guided_field, "classifiers")
            or not guided_field.classifiers
        ):
            return torch.zeros(x1.shape[0], device=x1.device)

        scales = getattr(guided_field, "scales", None)
        if not scales or len(scales) != len(guided_field.classifiers):
            scales = [1.0] * len(guided_field.classifiers)

        # no_grad to avoid building a computation graph and memory issues
        with torch.no_grad():
            energy = torch.zeros(x1.shape[0], device=x1.device)
            for clf, y_tar, lam in zip(guided_field.classifiers, guided_field.targets, scales):
                logits = clf(x1)
                logp = torch.log_softmax(logits, dim=1)[:, int(y_tar)]
                energy = energy + float(lam) * (-logp)  # -log p is energy
        return energy

    def _compute_classifier_guidance_reward(self, x1: torch.Tensor, guided_field) -> torch.Tensor:
        """
        Compute classifier guidance REWARD: R_clf = Σ λ * log p(y|x)
        Convention: higher is better (reward)
        This is simply -E_clf
        """
        if (
            guided_field is None
            or not hasattr(guided_field, "classifiers")
            or not guided_field.classifiers
        ):
            return torch.zeros(x1.shape[0], device=x1.device)

        scales = getattr(guided_field, "scales", None)
        if not scales or len(scales) != len(guided_field.classifiers):
            scales = [1.0] * len(guided_field.classifiers)

        with torch.no_grad():
            reward = torch.zeros(x1.shape[0], device=x1.device)
            for clf, y_tar, lam in zip(guided_field.classifiers, guided_field.targets, scales):
                logits = clf(x1)
                logp = torch.log_softmax(logits, dim=1)[:, int(y_tar)]
                reward = reward + float(lam) * logp  # log p is reward (higher is better)
        return reward

    def _effective_guidance_scale(self, guided_field) -> float:
        default_scale = float(getattr(self.cfg, "guidance_scale", 1.0))
        if guided_field is None:
            return default_scale

        scales = getattr(guided_field, "scales", None)
        if scales is None:
            return default_scale

        def _to_float(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu().item())
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        if isinstance(scales, torch.Tensor):
            values = [_to_float(v) for v in scales.view(-1)]
        elif isinstance(scales, (list, tuple)):
            values = [_to_float(v) for v in scales]
        else:
            single = _to_float(scales)
            return single if single is not None else default_scale

        values = [v for v in values if v is not None]
        if not values:
            return default_scale
        uniques = sorted(set(values))
        if len(uniques) == 1:
            return uniques[0]
        return sum(values) / len(values)

    def _compute_guidance_matching_energy(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        guided_field,
    ) -> Optional[torch.Tensor]:
        """
        Compute guidance matching energy for visualization.
        
        Note: z_model is trained via MSE to predict importance weights, not probabilities.
        The -log(z) transformation is used as a proxy for visualization purposes only
        (making higher weights correspond to lower "energy" regions).
        This is NOT a proper energy/log-probability in a probabilistic sense.
        """
        z_model = getattr(guided_field, "learned_z_model", None)
        if z_model is None:
            z_model = self._load_guidance_matching_model(guided_field)
            if z_model is None:
                return None

        was_training = z_model.training
        z_model.eval()
        if t.dim() == 1:
            t = t.view(-1, 1)
        if t.shape[0] == 1 and positions.shape[0] > 1:
            t = t.expand(positions.shape[0], -1)
        inputs = torch.cat([positions, t], dim=-1)
        with torch.no_grad():
            weights = z_model(inputs)
        if was_training:
            z_model.train()

        if weights.dim() > 1 and weights.shape[1] == 1:
            weights = weights.squeeze(-1)
        weights = weights.clamp_min(1e-8)
        # -log(z) as visualization proxy: higher weight → lower "energy"
        energy = -torch.log(weights)

        scale = self._effective_guidance_scale(guided_field)
        if scale > 0:
            energy = energy * scale
        return energy

    def _integrate_gradient_to_scalar(self, gradient_field: torch.Tensor, positions: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Attempt to integrate a gradient field to recover scalar potential/energy.
        
        Args:
            gradient_field: (B, D) tensor of gradients at each position
            positions: (B, D) tensor of positions
            
        Returns:
            (B,) tensor of scalar energy values, or None if integration fails
            
        Methods:
        1. Line integral from origin (assumes conservative field)
        2. Least-squares fitting (for non-conservative fields)
        3. Helmholtz decomposition (advanced, not implemented yet)
        
        Note: This is an approximation. For accurate results, the gradient field
        should be conservative (curl-free). Non-conservative components will be
        projected out in the least-squares method.
        """
        B, D = gradient_field.shape
        device = gradient_field.device
        
        if D != 2:
            # Only implemented for 2D for now
            print(f"[Warning] Gradient integration only implemented for 2D, got D={D}")
            return None
        
        # Method 1: Simple line integral from origin along straight paths
        # E(x) = - integral_0^x grad · dx
        # Using straight line: x(s) = s * x_target, s in [0,1]
        # dx/ds = x_target
        # integral = integral_0^1 grad(s*x_target) · x_target ds
        
        # For simplicity, use trapezoidal rule with the gradient at the final position
        # This is approximate but fast: E(x) ≈ -grad(x) · x
        energy_approx = -(gradient_field * positions).sum(dim=-1)  # (B,)
        
        return energy_approx
    
    def _compute_ceg_energy(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        guided_field,
    ) -> Optional[torch.Tensor]:
        """
        Compute learned guidance contribution for visualization.
        
        Model output interpretation depends on training mode:
        - Scalar output: treated as reward/log-prob (higher = better)
        - Vector field output: returns None. Vector-valued guidance should only
          affect the flow field, not the scalar energy visualization.

        Returns: energy-like scalar for visualization (lower = better, negated from reward)
        """
        ceg_model = getattr(guided_field, "learned_guidance_model", None)
        
        if ceg_model is None:
            # Don't override output dimension; let it be determined from the checkpoint.
            # If the model outputs a vector field (gradient type), we detect it and return None.
            ceg_model = self._load_guidance_matching_model(guided_field, suffix="__guidance.pth", out_dim_override=None)
            if ceg_model is None:
                return None

        was_training = ceg_model.training
        ceg_model.eval()
        if t.dim() == 1:
            t = t.view(-1, 1)
        if t.shape[0] == 1 and positions.shape[0] > 1:
            t = t.expand(positions.shape[0], -1)

        inputs = torch.cat([positions, t], dim=-1)
        with torch.no_grad():
            model_output = ceg_model(inputs)

        if was_training:
            ceg_model.train()

        if model_output.dim() > 1:  # Output has shape (B, D)
            if model_output.shape[-1] == 1:
                # Scalar output (B, 1): model learns reward/log-prob (higher = better);
                # negate to get energy (lower = better) for visualization.
                reward_val = model_output.squeeze(-1)  # (B,)
                return -reward_val
            else:
                # Vector field output (B, D), D > 1: learned correction g_phi = nabla E
                integrate_flag = getattr(self.cfg, "integrate_gradient_to_energy", False)

                if integrate_flag:
                    try:
                        energy = self._integrate_gradient_to_scalar(model_output, positions)
                        if energy is not None:
                            print(f"[SUCCESS] Integrated gradient to energy! energy.shape: {energy.shape}")
                            return energy
                    except Exception as e:
                        print(f"[WARNING] Failed to integrate gradient to energy: {e}")
                        import traceback
                        traceback.print_exc()

                return None
        else:
            # 1D scalar output (B,)
            return -model_output

    def _compute_ceg_reward(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        guided_field,
    ) -> Optional[torch.Tensor]:
        """
        Compute learned guidance REWARD for visualization.
        
        Convention: higher is better (reward)
        Model output is score/reward (higher = better), return directly without negation.
        
        Returns: reward scalar for visualization (higher = better)
        """
        ceg_model = getattr(guided_field, "learned_guidance_model", None)
        if ceg_model is None:
            # Don't override output dimension; let it be determined from the checkpoint.
            # If the model outputs a vector field (gradient type), we detect it and return None.
            ceg_model = self._load_guidance_matching_model(guided_field, suffix="__guidance.pth", out_dim_override=None)
            if ceg_model is None:
                return None

        was_training = ceg_model.training
        ceg_model.eval()
        if t.dim() == 1:
            t = t.view(-1, 1)
        if t.shape[0] == 1 and positions.shape[0] > 1:
            t = t.expand(positions.shape[0], -1)

        inputs = torch.cat([positions, t], dim=-1)
        with torch.no_grad():
            model_output = ceg_model(inputs)

        if was_training:
            ceg_model.train()

        if model_output.dim() > 1:
            if model_output.shape[-1] == 1:
                # Scalar output: model learns reward/log-prob (higher = better); return as-is.
                return model_output.squeeze(-1)  # (B,)
            else:
                # Vector field output: not a scalar reward
                return None
        else:
            # 1D scalar output
            return model_output

    def _map_classifiers_to_indices(self, classifiers, distribution):
        """Map classifiers to indices in distribution._guidance_models."""
        if not hasattr(distribution, "_guidance_models"):
            return None
        
        import re
        clf_items = list(distribution._guidance_models.items())
        classifier_indices = []
        
        for clf in classifiers:
            found = False
            # Strategy 1: Match by guidance_name (exact match)
            if hasattr(clf, 'guidance_name') and clf.guidance_name:
                for idx, (name, model) in enumerate(clf_items):
                    if name == clf.guidance_name:
                        classifier_indices.append(idx)
                        found = True
                        break
                
                # Strategy 1b: Extract index from guidance_name (e.g., "Classifier_0" -> 0)
                if not found:
                    match = re.search(r'(\d+)$', clf.guidance_name)
                    if match:
                        guidance_idx = int(match.group(1))
                        if 0 <= guidance_idx < len(clf_items):
                            classifier_indices.append(guidance_idx)
                            found = True
            
            # Strategy 2: Match by object identity
            if not found:
                for idx, (name, model) in enumerate(clf_items):
                    if clf is model:
                        classifier_indices.append(idx)
                        found = True
                        break
            
            # Strategy 3: Match by name or checkpoint_name attribute
            if not found:
                clf_name = getattr(clf, 'name', None) or getattr(clf, 'checkpoint_name', None)
                if clf_name:
                    for idx, (name, model) in enumerate(clf_items):
                        if name == clf_name:
                            classifier_indices.append(idx)
                            found = True
                            break
            
            if not found:
                # If can't map, return None to fall back to default behavior
                return None
        
        return classifier_indices if len(classifier_indices) == len(classifiers) else None

    def _load_guidance_matching_model(self, guided_field, suffix=None, out_dim_override=None) -> Optional[nn.Module]:
        signature = getattr(guided_field, "guidance_signature", "guidance_matching")
        ckpt_dir = getattr(
            guided_field,
            "guidance_ckpt_dir",
            self.default_guidance_ckpt_dir,
        )
        
        if suffix:
            candidate_suffixes = [suffix]
        else:
            candidate_suffixes = ["__z.pth", "__guidance.pth"]
            
        ckpt_path = None
        for s in candidate_suffixes:
            candidate = os.path.join(ckpt_dir, f"{signature}{s}")
            if os.path.exists(candidate):
                ckpt_path = candidate
                break
        if ckpt_path is None:
            return None
        if ckpt_path in self._guidance_matching_cache:
            return self._guidance_matching_cache[ckpt_path]

        state = torch.load(ckpt_path, map_location=self.device)
        layer0_key = next((k for k in state.keys() if k.endswith("0.weight")), None)
        layer4_key = next((k for k in state.keys() if k.endswith("4.weight")), None)
        if layer0_key is None or layer4_key is None:
            return None
        in_dim = state[layer0_key].shape[1]
        hidden_dim = state[layer0_key].shape[0]
        if out_dim_override is not None:
            out_dim = out_dim_override
        else:
            out_dim = state[layer4_key].shape[0]

        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        ).to(self.device)
        model.load_state_dict(state)
        model.eval()
        self._guidance_matching_cache[ckpt_path] = model
        return model

    def _approx_x1(self, positions: torch.Tensor, t: torch.Tensor, vf) -> torch.Tensor:
        """Approximate terminal state x1 from (x_t, t) via linear extrapolation."""
        with torch.no_grad():
            v_uncond = vf(positions, t)
            if v_uncond.dim() > 2:
                v_uncond = v_uncond.mean(dim=1)
            elif v_uncond.dim() == 1:
                v_uncond = v_uncond.reshape(positions.shape[0], -1)
            return positions + (1.0 - t.view(-1, 1)) * v_uncond

    def compute_residual_components(self, positions: torch.Tensor, t: torch.Tensor, vf, guided_field) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute base and residual (learned) energy components separately for learnable guidance modes.
        
        Returns energy in visualization convention (lower = better):
        - base_guidance: Σ λ * (-log p(y|x)) from classifiers (cost/energy)
        - learned_residual: -model_output from learned model (negated reward → energy)
                            For vector field outputs (gradient methods), returns zeros
        
        Supported modes:
        - car_guidance: CAR guidance with conflict-gated learned residual
        """
        mode = self._resolve_guidance_mode(guided_field)
        if mode != "car_guidance":
            return None

        with torch.no_grad():
            x1 = self._approx_x1(positions, t, vf)
            classifier_energy = self._compute_classifier_guidance_energy(x1, guided_field)
            learned_energy = self._compute_ceg_energy(positions, t, guided_field)

            if learned_energy is None:
                learned_energy = torch.zeros_like(classifier_energy)

            return {
                "base_guidance": classifier_energy,
                "learned_residual": learned_energy.to(classifier_energy.device)
            }

    def compute_target_energy(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        vf,
        guided_field=None,
    ) -> torch.Tensor:
        """
        Compute total energy for visualization (lower = better density).
        
        E(x) = -log p_base(x) + Σ λ_j * (-log p(y_j|x)) + learned_energy
        
        All components use energy convention (lower = better):
        - neg_log_p_base: -log p_base(x) from flow model
        - classifier_energy: Σ λ * (-log p(y|x)) from classifiers  
        - learned_energy: negated model output (see _compute_ceg_energy)
        """
        x1 = self._approx_x1(positions, t, vf)
        gaussian_log_density = dist.Independent(
            dist.Normal(torch.zeros(x1.shape[-1], device=self.device),
                        torch.ones(x1.shape[-1],  device=self.device)), 1
        ).log_prob

        base_solver = ODESolver(velocity_model=ModelWrapper(vf))
        with torch.no_grad():
            _, log_p_base = base_solver.compute_likelihood(
                x_1=x1,
                method='midpoint',
                step_size=float(self.cfg.step_size),
                exact_divergence=True,
                log_p0=gaussian_log_density,
            )

        neg_log_p_base = -log_p_base
        guidance_mode = self._resolve_guidance_mode(guided_field)
        guidance_energy = torch.zeros_like(neg_log_p_base)

        if guided_field:
            classifier_energy = self._compute_classifier_guidance_energy(x1, guided_field)

            if guidance_mode == "g_cov_g":
                # Pure classifier-based guidance (no learned component)
                guidance_energy = classifier_energy

            elif guidance_mode == "guidance_matching":
                gm_energy = self._compute_guidance_matching_energy(positions, t, guided_field)
                if gm_energy is not None:
                    guidance_energy = classifier_energy + gm_energy.to(neg_log_p_base.device)
                else:
                    guidance_energy = classifier_energy

            elif guidance_mode == "car_guidance":
                learned_energy = self._compute_ceg_energy(positions, t, guided_field)
                
                mask = 1.0
                distribution = getattr(guided_field, "distribution", None)
                targets = list(getattr(guided_field, "targets", []))
                classifiers = getattr(guided_field, "classifiers", [])
                if distribution is not None and targets and len(classifiers) >= 2:
                    score = None

                    if hasattr(distribution, "compute_direct_conflict_score"):
                        classifier_indices = self._map_classifiers_to_indices(classifiers, distribution)

                        if classifier_indices is not None and len(classifier_indices) >= 2:
                            # _map_classifiers_to_indices returns indices in classifiers-list order,
                            # so targets (also in classifiers order) aligns with classifier_indices.
                            if len(targets) == 1:
                                label_arg = int(targets[0])  # broadcasts to all classifiers
                            else:
                                label_arg = [int(t) for t in targets]

                            # Use x1 (or positions if estimate_x1), matching the classifier
                            # guidance input used in composed_guidance.py.
                            estimate_x1 = bool(getattr(self.cfg, "estimate_x1", False))
                            x_for_conflict = positions if estimate_x1 else x1

                            score = distribution.compute_direct_conflict_score(
                                x_for_conflict, label=label_arg, classifier_indices=classifier_indices,
                            )

                    if score is not None:
                        # conflict score in [0, 2]: 0 = classifiers aligned (same direction),
                        # 1 = perpendicular, 2 = opposite (maximal conflict).
                        conflict = score

                        # Expect (B,); if not, take the mean over extra dims.
                        if conflict.dim() != 1:
                            import warnings
                            warnings.warn(
                                f"[energy.py] Unexpected conflict shape {conflict.shape}, expected 1D. "
                                f"This may indicate an upstream bug. Taking mean to get (B,)."
                            )
                            conflict = conflict.view(conflict.shape[0], -1).mean(dim=-1)  # (B,)

                        # Smooth conflict-zone gating.
                        threshold = getattr(self.cfg, "conflict_threshold", 0.9)
                        temperature = getattr(self.cfg, "conflict_temperature", 0.15)
                        blend_type = getattr(self.cfg, "blend_function", "smootherstep")

                        if blend_type == "smootherstep":
                            # Normalize conflict to [threshold - temperature, threshold + temperature].
                            conflict_normalized = (conflict - (threshold - temperature)) / (2 * temperature + 1e-8)
                            weight = _smootherstep(conflict_normalized, edge0=0.0, edge1=1.0)
                        else:
                            weight = torch.sigmoid((conflict - threshold) / (temperature + 1e-8))
                        mask = weight.to(neg_log_p_base.device)

                        # Align mask to (B,) so it broadcasts against the energy tensors.
                        if mask.dim() > 1:
                            mask = mask.squeeze()
                        if mask.shape != classifier_energy.shape:
                            if mask.numel() == classifier_energy.numel():
                                mask = mask.view(classifier_energy.shape)

                if learned_energy is not None:
                    # Ensure all tensors have compatible shapes for broadcasting
                    learned_energy_aligned = learned_energy.to(neg_log_p_base.device)
                    if learned_energy_aligned.shape != classifier_energy.shape:
                        if learned_energy_aligned.numel() == classifier_energy.numel():
                            learned_energy_aligned = learned_energy_aligned.view(classifier_energy.shape)
                        elif learned_energy_aligned.dim() == 0:
                            learned_energy_aligned = learned_energy_aligned.expand_as(classifier_energy)
                        else:
                            learned_energy_aligned = learned_energy_aligned.squeeze()
                            if learned_energy_aligned.shape != classifier_energy.shape:
                                learned_energy_aligned = learned_energy_aligned.flatten()[:classifier_energy.numel()].view(classifier_energy.shape)

                    guidance_energy = classifier_energy + mask * learned_energy_aligned
                else:
                    guidance_energy = classifier_energy

            else:
                # Fallback: unknown mode, use classifier energy only
                guidance_energy = classifier_energy

        return neg_log_p_base + guidance_energy

    def compute_target_reward(
        self,
        positions: torch.Tensor,
        t: torch.Tensor,
        vf,
        guided_field=None,
    ) -> torch.Tensor:
        """
        Compute total REWARD for visualization (Option B: higher is better).
        
        R_total(x,t) = log p_base(x1) + R_clf(x1) + R_learned(x,t)
        
        All components use reward convention (higher = better):
        - log_p_base: base density log-likelihood (higher = more likely)
        - R_clf: Σ λ * log p(y|x) (higher = more confident in target class)
        - R_learned: model score output (higher = better)
        
        Returns:
        - Total reward (higher = better regions for sampling)
        """
        x1 = self._approx_x1(positions, t, vf)
        gaussian_log_density = dist.Independent(
            dist.Normal(torch.zeros(x1.shape[-1], device=self.device),
                        torch.ones(x1.shape[-1], device=self.device)), 1
        ).log_prob

        base_solver = ODESolver(velocity_model=ModelWrapper(vf))
        with torch.no_grad():
            _, log_p_base = base_solver.compute_likelihood(
                x_1=x1,
                method='midpoint',
                step_size=float(self.cfg.step_size),
                exact_divergence=True,
                log_p0=gaussian_log_density,
            )

        # Base reward: log p_base (higher = more likely, NO negation)
        R = log_p_base.clone()
        
        guidance_mode = self._resolve_guidance_mode(guided_field)
        
        if guided_field:
            # Classifier reward: Σ λ * log p(y|x) (higher = more confident)
            classifier_reward = self._compute_classifier_guidance_reward(x1, guided_field)

            if guidance_mode == "g_cov_g":
                # Pure classifier-based guidance (no learned component)
                R = R + classifier_reward

            elif guidance_mode == "guidance_matching":
                gm_energy = self._compute_guidance_matching_energy(positions, t, guided_field)
                if gm_energy is not None:
                    # gm_energy is -log(z), negate to get reward
                    gm_reward = -gm_energy.to(log_p_base.device)
                    R = R + classifier_reward + gm_reward
                else:
                    R = R + classifier_reward

            elif guidance_mode == "car_guidance":
                learned_reward = self._compute_ceg_reward(positions, t, guided_field)
                
                # Compute conflict mask (same logic as energy version)
                mask = 1.0
                distribution = getattr(guided_field, "distribution", None)
                targets = list(getattr(guided_field, "targets", []))
                classifiers = getattr(guided_field, "classifiers", [])
                
                if distribution is not None and targets and len(classifiers) >= 2:
                    score = None
                    if hasattr(distribution, "compute_direct_conflict_score"):
                        classifier_indices = self._map_classifiers_to_indices(classifiers, distribution)
                        
                        if classifier_indices is not None and len(classifier_indices) >= 2:
                            if len(targets) == 1:
                                label_arg = int(targets[0])
                            else:
                                label_arg = [int(t_) for t_ in targets]
                            
                            estimate_x1 = bool(getattr(self.cfg, "estimate_x1", False))
                            x_for_conflict = positions if estimate_x1 else x1
                            
                            score = distribution.compute_direct_conflict_score(
                                x_for_conflict, label=label_arg, classifier_indices=classifier_indices,
                            )
                    
                    if score is not None:
                        conflict = score
                        if conflict.dim() != 1:
                            conflict = conflict.view(conflict.shape[0], -1).mean(dim=-1)
                        
                        threshold = getattr(self.cfg, "conflict_threshold", 0.9)
                        temperature = getattr(self.cfg, "conflict_temperature", 0.15)
                        blend_type = getattr(self.cfg, "blend_function", "smootherstep")
                        
                        if blend_type == "smootherstep":
                            conflict_normalized = (conflict - (threshold - temperature)) / (2 * temperature + 1e-8)
                            weight = _smootherstep(conflict_normalized, edge0=0.0, edge1=1.0)
                        else:
                            weight = torch.sigmoid((conflict - threshold) / (temperature + 1e-8))
                        mask = weight.to(log_p_base.device)
                        
                        if mask.dim() > 1:
                            mask = mask.squeeze()
                        if mask.shape != classifier_reward.shape:
                            if mask.numel() == classifier_reward.numel():
                                mask = mask.view(classifier_reward.shape)

                if learned_reward is not None:
                    learned_reward_aligned = learned_reward.to(log_p_base.device)
                    if learned_reward_aligned.shape != classifier_reward.shape:
                        if learned_reward_aligned.numel() == classifier_reward.numel():
                            learned_reward_aligned = learned_reward_aligned.view(classifier_reward.shape)
                        elif learned_reward_aligned.dim() == 0:
                            learned_reward_aligned = learned_reward_aligned.expand_as(classifier_reward)
                        else:
                            learned_reward_aligned = learned_reward_aligned.squeeze()
                            if learned_reward_aligned.shape != classifier_reward.shape:
                                learned_reward_aligned = learned_reward_aligned.flatten()[:classifier_reward.numel()].view(classifier_reward.shape)

                    R = R + classifier_reward + mask * learned_reward_aligned
                else:
                    R = R + classifier_reward

            else:
                R = R + classifier_reward

        return R

    def plot_energy_landscape_3d(
        self,
        ax,
        positions,
        t,
        vf,
        guided_field=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        clip_percentiles: Optional[Sequence[float]] = (1.0, 99.5),
        elev: float = 30,
        azim: float = 45,
    ):
        """3D energy surface without log compression."""
        if not hasattr(ax, "plot_surface"):
            fig = ax.figure
            ss = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(ss, projection='3d')

        with torch.no_grad():
            E = self.compute_target_energy(positions, t, vf, guided_field)
            E = E.reshape(self.num_points, self.num_points).detach().cpu().numpy()
            if np.all(np.isnan(E)):
                ax.set_title('Energy Landscape (unavailable: all-NaN)')
                return
            E = E - np.nanmin(E)

            x = np.linspace(x_min, x_max, self.num_points)
            y = np.linspace(y_min, y_max, self.num_points)
            X, Y = np.meshgrid(x, y, indexing="xy")

            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                vmin = float(np.percentile(E, p_low))
                vmax = float(np.percentile(E, p_high))
            else:
                vmin = float(np.nanmin(E))
                vmax = float(np.nanmax(E))

            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = 1.0
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6

            norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)

            ax.plot_surface(X, Y, E, cmap='viridis', linewidth=0, antialiased=True, norm=norm)

            title_suffix = ' (with guidance)' if guided_field else ''
            ax.set_title('3D Target Energy Landscape' + title_suffix)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Energy')
            ax.view_init(elev=elev, azim=azim)
            ax.set_zlim(vmin, vmax)

            mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            mappable.set_array([])
            plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, pad=0.05, label='Energy')

    def plot_energy_landscape(
        self,
        ax,
        positions,
        t,
        vf,
        guided_field=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        clip_percentiles: Optional[Sequence[float]] = (1.0, 99.5),
    ):
        """2D energy heatmap aligned with E(x) definition."""
        with torch.no_grad():
            E = self.compute_target_energy(positions, t, vf, guided_field)
            E_vis = E.reshape(self.num_points, self.num_points).detach().cpu().numpy()
            E_vis = E_vis - float(np.nanmin(E_vis))

            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                vmin = float(np.percentile(E_vis, p_low))
                vmax = float(np.percentile(E_vis, p_high))
            else:
                vmin = float(np.nanmin(E_vis))
                vmax = float(np.nanmax(E_vis))
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = 1.0
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6

            norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
            im = ax.imshow(
                E_vis,
                extent=(x_min, x_max, y_min, y_max),
                origin='lower',
                cmap='viridis',
                norm=norm,
            )
            title_suffix = ' (with guidance)' if (
                guided_field and hasattr(guided_field, 'classifiers')
            ) else ''
            ax.set_title('Target Energy Landscape' + title_suffix)
            ax.grid(False)
            plt.colorbar(im, ax=ax, orientation='horizontal', label='Energy')

            x = np.linspace(x_min, x_max, self.num_points)
            y = np.linspace(y_min, y_max, self.num_points)
            X, Y = np.meshgrid(x, y, indexing="xy")
            # Use guided_field for streamplot when available; otherwise the heatmap shows
            # guided energy while the streamlines would show only the base vf.
            field_fn = guided_field if guided_field is not None else vf
            field = field_fn(positions, t)
            if field.dim() > 2:
                field = field.mean(dim=1)
            elif field.dim() == 1:
                field = field.reshape(positions.shape[0], -1)
            U = field[:, 0].reshape(self.num_points, self.num_points).cpu().numpy()
            V = field[:, 1].reshape(self.num_points, self.num_points).cpu().numpy()
            ax.streamplot(X, Y, U, V, color='white', linewidth=0.5, density=1.5, arrowsize=0.5)

    def plot_energy_contour(
        self,
        ax,
        positions,
        t,
        vf,
        guided_field=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        clip_percentiles: Optional[Sequence[float]] = (1.0, 99.5),
    ):
        """Contour view of the energy landscape."""
        with torch.no_grad():
            E = self.compute_target_energy(positions, t, vf, guided_field)
            E_vis = E.reshape(self.num_points, self.num_points).detach().cpu().numpy()
            if np.all(np.isnan(E_vis)):
                ax.set_title('Energy Landscape Contour (unavailable: all-NaN)')
                return
            E_vis = E_vis - float(np.nanmin(E_vis))

            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                vmin = float(np.percentile(E_vis, p_low))
                vmax = float(np.percentile(E_vis, p_high))
            else:
                vmin = float(np.nanmin(E_vis))
                vmax = float(np.nanmax(E_vis))
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6

            x = np.linspace(x_min, x_max, self.num_points)
            y = np.linspace(y_min, y_max, self.num_points)
            X, Y = np.meshgrid(x, y, indexing="xy")

            levels = np.linspace(vmin, vmax, 13)
            contours = ax.contour(X, Y, E_vis, levels=levels, colors='black', alpha=0.7)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            contourf = ax.contourf(X, Y, E_vis, levels=levels, cmap='viridis', alpha=0.6)
            plt.colorbar(contourf, ax=ax, orientation='horizontal', label='Energy')

            title_suffix = ' Contour' + (
                ' (with guidance)' if (guided_field and hasattr(guided_field, 'classifiers')) else ''
            )
            ax.set_title('Energy Landscape' + title_suffix)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_aspect('equal')

    # ==================== REWARD-BASED VISUALIZATION (Option B) ====================
    # Convention: higher reward = better (should see high values in target regions)
    
    def plot_reward_landscape(
        self,
        ax,
        positions,
        t,
        vf,
        guided_field=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        clip_percentiles: Optional[Sequence[float]] = (0.5, 99.5),
    ):
        """
        2D reward heatmap (Option B: higher is better).
        
        High reward regions = high base likelihood + high classifier confidence + high learned score
        """
        with torch.no_grad():
            R = self.compute_target_reward(positions, t, vf, guided_field)
            R_vis = R.reshape(self.num_points, self.num_points).detach().cpu().numpy()

            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                vmin = float(np.percentile(R_vis, p_low))
                vmax = float(np.percentile(R_vis, p_high))
            else:
                vmin = float(np.nanmin(R_vis))
                vmax = float(np.nanmax(R_vis))
            if not np.isfinite(vmin):
                vmin = -10.0
            if not np.isfinite(vmax):
                vmax = 10.0
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6

            norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
            # Use 'plasma' colormap - high values (yellow) = good, low values (purple) = bad
            im = ax.imshow(
                R_vis,
                extent=(x_min, x_max, y_min, y_max),
                origin='lower',
                cmap='plasma',
                norm=norm,
            )
            title_suffix = ' (with guidance)' if (
                guided_field and hasattr(guided_field, 'classifiers')
            ) else ''
            ax.set_title('Target Reward Landscape' + title_suffix)
            ax.grid(False)
            plt.colorbar(im, ax=ax, orientation='horizontal', label='Reward (higher=better)')

            x = np.linspace(x_min, x_max, self.num_points)
            y = np.linspace(y_min, y_max, self.num_points)
            X, Y = np.meshgrid(x, y, indexing="xy")
            # Use guided_field for streamplot when available
            field_fn = guided_field if guided_field is not None else vf
            field = field_fn(positions, t)
            if field.dim() > 2:
                field = field.mean(dim=1)
            elif field.dim() == 1:
                field = field.reshape(positions.shape[0], -1)
            U = field[:, 0].reshape(self.num_points, self.num_points).cpu().numpy()
            V = field[:, 1].reshape(self.num_points, self.num_points).cpu().numpy()
            ax.streamplot(X, Y, U, V, color='white', linewidth=0.5, density=1.5, arrowsize=0.5)

    def plot_weighted_density(
        self,
        ax,
        vf,
        guided_field,
        x_min,
        x_max,
        y_min,
        y_max,
        bins: int = 200,
    ):
        """Visualize posterior proportional to exp(-E) at t=1."""
        x_grid = torch.linspace(x_min, x_max, bins, device=self.device)
        y_grid = torch.linspace(y_min, y_max, bins, device=self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        positions = torch.stack([X_mesh.ravel(), Y_mesh.ravel()], dim=1)
        t1 = torch.ones((positions.shape[0], 1), device=self.device)

        with torch.no_grad():
            E = self.compute_target_energy(positions, t1, vf, guided_field)
            posterior = torch.exp(-E).reshape(bins, bins).detach().cpu().numpy()

        posterior = np.clip(posterior, 1e-12, None)
        vmin = np.percentile(posterior, 1)
        vmax = np.percentile(posterior, 99.5)
        if not np.isfinite(vmin) or vmin <= 0:
            positive_vals = posterior[posterior > 0]
            if positive_vals.size == 0:
                positive_vals = np.array([1e-12])
            vmin = max(float(np.min(positive_vals)), 1e-12)
        if not np.isfinite(vmax):
            vmax = vmin * 10.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        im = ax.imshow(
            posterior,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='viridis',
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_title(r'Weighted Density $e^{-E(x_1)}$ (log color)')
        ax.grid(False)
        plt.colorbar(im, ax=ax, orientation='horizontal', label='density')

        E_vis = (E - E.min()).reshape(bins, bins).cpu().numpy()
        # Note: No transpose needed - flatten order is consistent with meshgrid(indexing='xy')
        ax.contour(
            x_grid.cpu().numpy(),
            y_grid.cpu().numpy(),
            E_vis,
            levels=15,
            colors='white',
            linewidths=0.6,
            alpha=0.8,
        )

    def plot_weighted_density_ori(
        self,
        ax,
        vf,
        guided_field,
        x_min,
        x_max,
        y_min,
        y_max,
        bins: int = 200,
    ):
        """Plot weighted density p_base(x1) * exp(-sum J(x1)) at t=1."""
        x_grid = torch.linspace(x_min, x_max, bins).to(self.device)
        y_grid = torch.linspace(y_min, y_max, bins).to(self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x1 = torch.stack([X_mesh.ravel(), Y_mesh.ravel()], dim=1)

        gaussian_log_density = dist.Independent(
            dist.Normal(torch.zeros(2, device=self.device),
                        torch.ones(2, device=self.device)), 1
        ).log_prob
        base_solver = ODESolver(velocity_model=ModelWrapper(vf))
        with torch.no_grad():
            _, log_p_base = base_solver.compute_likelihood(
                x_1=x1,
                method='midpoint',
                step_size=float(self.cfg.step_size),
                exact_divergence=True,
                log_p0=gaussian_log_density,
            )
        p_base_vals = torch.exp(log_p_base)

        total_J = torch.zeros_like(p_base_vals)
        if guided_field and hasattr(guided_field, 'classifiers'):
            scales = getattr(guided_field, 'scales', [1.0]*len(guided_field.classifiers))
            for clf, target, lam in zip(guided_field.classifiers, guided_field.targets, scales):
                with torch.no_grad():
                    logits = clf(x1)
                    log_probs = torch.log_softmax(logits, dim=1)
                    J_x = -log_probs[:, target]
                    total_J += lam * J_x

        unnorm_posterior = p_base_vals * torch.exp(-total_J)
        posterior_grid = unnorm_posterior.reshape(bins, bins).cpu().numpy()

        im = ax.imshow(
            posterior_grid,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='viridis',
        )
        ax.set_title(r'Weighted Density $p(x_1)e^{-\sum J(x_1)}$')
        plt.colorbar(im, ax=ax, orientation='horizontal', label='Density')
        ax.grid(False)
