"""
Composed guidance module.

Supported guidance methods (``--guidance_fn`` / ``cfg.guidance_fn``):
  - ``g_cov_g``           — non-learnable approximate classifier guidance g^approx
  - ``guidance_matching`` — learnable guidance-matching baseline
  - ``car_guidance``      — CAR guidance (ours): g^car = g^approx + w_t · g_psi

Class hierarchy
---------------
ComposedGuidance  (factory base)
├── GCovGGuidance             – g_cov_g
├── GuidanceMatching          – guidance_matching
└── GCovGGMOnlineGuidance     – car_guidance (online-trained residual g_psi)
"""

from typing import Callable, Optional, Sequence, Union, Dict, Tuple
import copy
import numbers
import os
import re
import torch
from torch import nn
import torch.nn.functional as F

GUIDANCE_FN_REGISTRY = {}
_GUIDANCE_MODEL_CACHE: Dict[str, nn.Module] = {}
_GUIDANCE_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GUIDANCE_CKPT_DIR = os.path.join(_GUIDANCE_BASE_DIR, "guidance", "pretrained_guidance", "guidance_ckpt_new")
# Directory holding shipped, pretrained guidance networks. It is searched as a
# fallback when restoring models so a matching network is loaded instead of
# being retrained (see ComposedGuidance._pretrained_ckpt_path).
PRETRAINED_GUIDANCE_DIR = os.path.join(_GUIDANCE_BASE_DIR, "guidance", "pretrained_guidance")

# Canonical public method names.
GUIDANCE_G_COVG = "g_cov_g"
GUIDANCE_MATCHING = "guidance_matching"
GUIDANCE_CAR = "car_guidance"


def _normalize_tensor_shapes(x, t):
    """
    Normalize x and t tensors to consistent shapes.

    - x: (B, ...) -> (B, D) by flattening all non-batch dims.
    - t: scalar / (B,) / (B,1) / (1,1) -> (B,1)
    """
    if x.dim() == 0:
        x_in = x.view(1, 1)
    elif x.dim() == 1:
        x_in = x.view(1, -1)
    else:
        x_in = x.view(x.shape[0], -1)

    if t.dim() == 0:
        t_ = t.view(1, 1).expand(x_in.shape[0], 1)
    elif t.dim() == 1:
        t_ = t.view(-1, 1)
        if t_.shape[0] == 1 and x_in.shape[0] > 1:
            t_ = t_.expand(x_in.shape[0], 1)
    else:
        t_ = t
        if t_.shape[0] == 1 and x_in.shape[0] > 1:
            t_ = t_.expand(x_in.shape[0], 1)

    return x_in, t_


def _prepare_input_for_grad(x, need_higher_order=False):
    """Prepare input tensor for gradient computation."""
    if need_higher_order:
        return x if x.requires_grad else x.clone().detach().requires_grad_(True)
    return x.detach().requires_grad_(True) if not x.requires_grad else x

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
    # Handle edge case where edge0 == edge1
    edge_diff = edge1 - edge0
    if isinstance(edge_diff, torch.Tensor):
        edge_diff = torch.clamp(edge_diff, min=1e-8)
    elif abs(edge_diff) < 1e-8:
        edge_diff = 1e-8

    x_normalized = (x - edge0) / edge_diff
    x_clamped = torch.clamp(x_normalized, 0.0, 1.0)
    # Smootherstep polynomial: 6t^5 - 15t^4 + 10t^3
    return x_clamped * x_clamped * x_clamped * (x_clamped * (x_clamped * 6.0 - 15.0) + 10.0)

def _compute_g_cov_g_energy(x, t, v_uncond, classifiers, targets, scales, cfg):
    """Computes the 'base' energy used by g_cov_g: sum(lambda * log_prob(y|x_est))"""
    t_ = t.view(-1, 1) if t.dim() == 1 else t
    x1_est = x if getattr(cfg, "estimate_x1", False) else x + (1.0 - t_) * v_uncond
    scales = scales if scales and len(scales) == len(classifiers) else [1.0] * len(classifiers)

    total_obj = torch.zeros(x.shape[0], device=x.device)
    for clf, y_tar, lam in zip(classifiers, targets, scales):
        logits = clf(x1_est)
        log_probs = torch.log_softmax(logits, dim=1)
        if isinstance(y_tar, torch.Tensor):
            y_tar_int = int(y_tar.item())
        else:
            y_tar_int = int(y_tar)
        total_obj = total_obj + float(lam) * log_probs[:, y_tar_int]
    return total_obj

def _resolve_classifier_name(clf, idx: int) -> str:
    name = getattr(clf, "guidance_name", None) or getattr(clf, "name", None) or getattr(clf, "checkpoint_name", None)
    return name or f"{clf.__class__.__name__}_{idx}"

def _signature_from_components(classifiers, targets, identifier: str) -> str:
    parts = [identifier or "guidance_matching"]
    if not classifiers: parts.append("unguided")
    for idx, (clf, target) in enumerate(zip(classifiers, targets)):
        parts.append(f"{_resolve_classifier_name(clf, idx)}_y{int(target)}")
    return "__".join(parts)

def _load_guidance_matching_model(ckpt_dir, signature, device):
    ckpt_path = os.path.join(ckpt_dir, f"{signature}__guidance.pth")
    if not os.path.exists(ckpt_path): return None
    state = torch.load(ckpt_path, map_location=device)
    try:
        first_k = next(k for k in state if k.endswith("0.weight"))
        last_k = next(k for k in state if k.endswith("4.weight"))
        in_dim = state[first_k].shape[1]
        hidden_dim = state[first_k].shape[0]
        out_dim = state[last_k].shape[0]
    except StopIteration: return None
    
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, out_dim)
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

class ComposedGuidance:
    """Base class for composed guidance."""
    def __new__(cls, flow_model, classifiers, targets, scales, cfg, guidance_fn=None, learnable=False, learnable_model=None, distribution=None, **kwargs):
        if cls is ComposedGuidance:
            g_fn = guidance_fn or getattr(cfg, "guidance_fn", GUIDANCE_G_COVG)
            if g_fn == GUIDANCE_MATCHING:
                return super().__new__(GuidanceMatching)
            if g_fn == GUIDANCE_CAR:
                return super().__new__(GCovGGMOnlineGuidance)
            if g_fn == GUIDANCE_G_COVG:
                return super().__new__(GCovGGuidance)
        return super().__new__(cls)

    def __init__(self, flow_model, classifiers, targets, scales, cfg, guidance_fn=None, learnable=False, learnable_model=None, distribution=None):
        self.flow_model = flow_model
        self.cfg = cfg
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.distribution = distribution

        to_list = lambda x: list(x) if isinstance(x, (list, tuple)) else [x]
        self.classifiers = to_list(classifiers)
        if targets is None: raise ValueError("Targets required")
        self.targets = [int(t.item()) if isinstance(t, torch.Tensor) else int(t) for t in to_list(targets)]
        scales_list = to_list(scales) if scales is not None else []
        self.scales = [float(s.item()) if isinstance(s, torch.Tensor) else float(s) for s in scales_list]

        n_t = len(self.targets)
        if len(self.classifiers) == 1 and n_t > 1: self.classifiers *= n_t
        if not self.scales: self.scales = [1.0] * n_t
        elif len(self.scales) == 1 and n_t > 1: self.scales *= n_t
        if len(self.classifiers) != n_t or len(self.scales) != n_t: raise ValueError("Component length mismatch")

        self.guidance_identifier = guidance_fn or getattr(cfg, "guidance_fn", GUIDANCE_G_COVG)
        self.learnable = learnable or (learnable_model is not None)
        self.learned_guidance_model = learnable_model.to(self.device) if learnable_model else None
        self.learned_z_model = None
        
        self.guidance_ckpt_dir = getattr(cfg, "guidance_ckpt_dir", DEFAULT_GUIDANCE_CKPT_DIR)
        os.makedirs(self.guidance_ckpt_dir, exist_ok=True)
        self.guidance_signature = _signature_from_components(
            self.classifiers, self.targets, self.guidance_identifier
        )
        print(f"[ComposedGuidance] Init signature: {self.guidance_signature}")
        
        self._setup_training()

    def _setup_training(self):
        bs_fn = getattr(self.cfg, "guidance_sample_fn", None)
        if bs_fn and not callable(bs_fn): raise TypeError("guidance_sample_fn must be callable")
        
        self.sample_batch_fn = None if not bs_fn else (lambda b, d: bs_fn(b, d, targets=self.targets, classifiers=self.classifiers, scales=self.scales))

        if self.learnable:
            if not self.sample_batch_fn: raise ValueError("Learnable guidance requires sample_batch_fn")
            
            self._prepare_models_for_subclass()
            
            restored = self._restore_models()
            if not restored:
                self.train_model()

    def _prepare_models_for_subclass(self):
        # Default: output scalar energy. Subclasses override for other output dims.
        self._prepare_models(self.sample_batch_fn, output_dim_override=1)
        print(f"[ComposedGuidance] Default: output_dim=1 (scalar)")
    
    def _map_classifiers_to_indices(self):
        """Map self.classifiers to indices in distribution._guidance_models,
        preserving the order of self.classifiers.

        Returns:
            List of indices in distribution._guidance_models, in the same order as self.classifiers.
        """
        clf_items = list(self.distribution._guidance_models.items())
        classifier_indices = []

        clf_names_in_order = [getattr(c, 'guidance_name', None) for c in self.classifiers]
        model_keys_in_order = [name for name, _ in clf_items]

        for clf in self.classifiers:
            idx = self._find_classifier_index(clf, clf_items)
            if idx is None:
                available = list(self.distribution._guidance_models.keys())
                guidance_name = getattr(clf, 'guidance_name', None)
                raise ValueError(
                    f"Could not map classifier {clf} to any classifier in distribution._guidance_models. "
                    f"Available classifiers: {available}. "
                    f"Classifier guidance_name: {guidance_name}"
                )
            classifier_indices.append(idx)

        if len(classifier_indices) != len(self.classifiers):
            raise ValueError(
                f"Mapped {len(classifier_indices)} indices but expected {len(self.classifiers)}. "
                f"Classifier names: {clf_names_in_order}, Model keys: {model_keys_in_order}, "
                f"Indices: {classifier_indices}"
            )
        
        return classifier_indices
    
    def _find_classifier_index(self, clf, clf_items):
        """Find index of classifier in clf_items using multiple matching strategies."""
        # Strategy 1: match by guidance_name, else extract trailing index (e.g. "Classifier_0" -> 0)
        if hasattr(clf, 'guidance_name') and clf.guidance_name:
            for idx, (name, model) in enumerate(clf_items):
                if name == clf.guidance_name:
                    return idx

            match = re.search(r'(\d+)$', clf.guidance_name)
            if match:
                guidance_idx = int(match.group(1))
                if 0 <= guidance_idx < len(clf_items):
                    return guidance_idx

        # Strategy 2: match by object identity
        for idx, (name, model) in enumerate(clf_items):
            if clf is model:
                return idx

        # Strategy 3: match by name or checkpoint_name attribute
        clf_name = getattr(clf, 'name', None) or getattr(clf, 'checkpoint_name', None)
        if clf_name:
            for idx, (name, model) in enumerate(clf_items):
                if name == clf_name:
                    return idx
        
        return None

    def _get_conflict_threshold_and_temperature(self):
        """Get conflict threshold and temperature from config."""
        threshold = getattr(self.cfg, "conflict_threshold", 0.9)
        temperature = getattr(self.cfg, "conflict_temperature", 0.1)
        return threshold, temperature
    
    def _compute_gcar_weight(self, conflict) -> torch.Tensor:
        """Unified weight w ∈ [0, 1] for g^car = g_approx + w(x_t) · g_ψ.

        Used by BOTH training (conflict mask) and inference (guidance weight),
        so the two are always consistent.

        conflict_mask_type (cfg):
            'smootherstep' (default) — smooth gate, aligned at threshold ± temperature
            'hard'                   — binary gate: 1 if conflict > threshold else 0
            'soft'                   — continuous weight: conflict  (range [0,1])
        fixed_guidance_weight (cfg, float in [0,1]):
            when set, always returns this constant weight (overrides all other modes)
        disable_conflict_guidance_weight (cfg):
            True → equivalent to fixed_guidance_weight=1.0
        """
        fixed_w = getattr(self.cfg, "fixed_guidance_weight", None)
        if fixed_w is not None:
            return torch.full_like(conflict, float(fixed_w))

        if getattr(self.cfg, "disable_conflict_guidance_weight", False):
            return torch.ones_like(conflict)

        threshold, temperature = self._get_conflict_threshold_and_temperature()
        mask_type = getattr(self.cfg, "conflict_mask_type", "smootherstep")

        if mask_type == "hard":
            return (conflict > threshold).float()
        elif mask_type == "soft":
            # conflict ∈ [0, 1]: 0=aligned, 1=anti-parallel
            return conflict
        else:  # "smootherstep": smooth transition around threshold
            conflict_normalized = (conflict - (threshold - temperature)) / (2 * temperature + 1e-8)
            return _smootherstep(conflict_normalized, edge0=0.0, edge1=1.0)

    def _apply_gcar(self, base_guidance, learned_guidance, x) -> torch.Tensor:
        """Combine base and learned guidance via the conflict-aware gate.

        g^car = g_approx + w_t(x) · g_ψ

        Args:
            base_guidance:    g_approx  (B, D) or None
            learned_guidance: g_ψ       (B, D)
            x:                x_t (detached), used to compute conflict score
        """
        if self.distribution is None or not self.targets:
            return learned_guidance if base_guidance is None else (base_guidance + learned_guidance)

        with torch.no_grad():
            conflict = self._compute_conflict_score(x.detach(), self.targets)

        if conflict is None:
            return learned_guidance if base_guidance is None else (base_guidance + learned_guidance)

        w = self._compute_gcar_weight(conflict)          # (B,)
        weighted = w.unsqueeze(-1) * learned_guidance
        return weighted if base_guidance is None else (base_guidance + weighted)
    
    def _compute_conflict_score(self, x: torch.Tensor, labels) -> torch.Tensor:
        """
        Compute conflict score between classifiers.
        
        Args:
            x: Input tensor, shape (B, D)
            labels: One or more target class labels. Can be:
                - int or 0-dim tensor: single label (applied to all classifiers)
                - list/tuple/1D tensor of ints: multiple labels (one per classifier)
            
        Returns:
            Conflict score tensor, shape (B,), range [0, 1]:
            - 0:   classifiers are perfectly aligned (same direction)
            - 0.5: classifiers are perpendicular
            - 1:   classifiers are opposite (max conflict)
        """
        # Normalize labels to a list[int]
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 0:
                label_list = [int(labels.item())]
            else:
                label_list = [int(l.item()) for l in labels.view(-1)]
        elif isinstance(labels, numbers.Integral):
            label_list = [int(labels)]
        elif isinstance(labels, (list, tuple)):
            label_list = [int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in labels]
        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

        classifier_indices = self._map_classifiers_to_indices()

        # With fewer than two classifiers, no conflict is possible.
        if len(classifier_indices) < 2:
            B = x.shape[0]
            return torch.zeros(B, device=x.device)

        # label_list[i] corresponds to self.classifiers[i]; classifier_indices[i] is
        # the index of self.classifiers[i] in _guidance_models. label_arg must be
        # reordered so label_arg[i] matches classifier_indices[i].
        if len(label_list) == 1:
            label_arg = label_list[0]
        else:
            clf_items = list(self.distribution._guidance_models.items())
            index_to_label = {}

            for i, clf in enumerate(self.classifiers):
                clf_idx = self._find_classifier_index(clf, clf_items)
                if clf_idx is not None:
                    if i < len(label_list):
                        index_to_label[clf_idx] = label_list[i]
                    elif i < len(self.targets):
                        index_to_label[clf_idx] = self.targets[i]

            label_arg = []
            for clf_idx in classifier_indices:
                if clf_idx in index_to_label:
                    label_arg.append(index_to_label[clf_idx])
                else:
                    raise ValueError(
                        f"Could not find label for classifier index {clf_idx} in _guidance_models. "
                        f"Available indices: {list(index_to_label.keys())}, "
                        f"classifier_indices: {classifier_indices}, "
                        f"label_list: {label_list}, "
                        f"self.classifiers guidance_names: {[getattr(c, 'guidance_name', None) for c in self.classifiers]}"
                    )

            if len(label_arg) != len(classifier_indices):
                raise ValueError(
                    f"Could not map all labels. Expected {len(classifier_indices)} labels, "
                    f"got {len(label_arg)}. Label list length: {len(label_list)}, "
                    f"Classifiers: {len(self.classifiers)}, "
                    f"classifier_indices: {classifier_indices}, "
                    f"index_to_label: {index_to_label}"
                )
        
        conflict = self.distribution.compute_direct_conflict_score(
            x,
            label=label_arg,
            classifier_indices=classifier_indices,
        )
        
        return conflict

    def train_model(self):
        raise NotImplementedError("Subclasses must implement train_model if learnable is True")

    def __call__(self, x, t, need_higher_order=False):
        x_in, t_ = _normalize_tensor_shapes(x, t)
        # IMPORTANT:
        # - Sampling/rollout usually calls with x.requires_grad=False -> no_grad is faster.
        # - Likelihood/divergence computation needs du/dx -> must keep grad when x.requires_grad=True.
        if x_in.requires_grad:
            v_uncond = self.flow_model(x_in, t_)
        else:
            with torch.no_grad():
                v_uncond = self.flow_model(x_in, t_)
        g = self.compute_guidance(x_in, t_, v_uncond, need_higher_order)
        start_th = float(getattr(self.cfg, "start_guidance_threshold", 0.0))
        cond = (t_ > start_th).expand_as(v_uncond)
        return torch.where(cond, v_uncond + g, v_uncond)

    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        raise NotImplementedError

    def _ckpt_path(self, kind="guidance"):
        suffix, prefix = ("z", f"{self.guidance_signature}__") if kind == "z" else ("guidance", f"{self.guidance_signature}__")
        path = os.path.join(self.guidance_ckpt_dir, f"{prefix}{suffix}.pth")
        if os.path.exists(path): return path

        # Fuzzy match for permuted signatures.
        _, combos = self._sig_components(self.guidance_signature)
        id_prefixes = [str(self.guidance_identifier)]
        for f in os.listdir(self.guidance_ckpt_dir):
            if not f.endswith(f"__{suffix}.pth"):
                continue
            if not any(f.startswith(prefix + "__") for prefix in id_prefixes):
                continue
            if self._sig_components(f.split(f"__{suffix}.pth")[0])[1] == combos:
                return os.path.join(self.guidance_ckpt_dir, f)
        return path

    def _pretrained_ckpt_path(self, kind="guidance"):
        """Look for a shipped pretrained model in PRETRAINED_GUIDANCE_DIR.

        Returns the matching checkpoint path (exact signature first, then a
        fuzzy match over permuted component order), or None if nothing matches.
        Used as a fallback so a pretrained guidance network is loaded instead of
        being retrained.
        """
        if not os.path.isdir(PRETRAINED_GUIDANCE_DIR):
            return None
        suffix = "z" if kind == "z" else "guidance"
        exact = os.path.join(PRETRAINED_GUIDANCE_DIR, f"{self.guidance_signature}__{suffix}.pth")
        if os.path.exists(exact):
            return exact
        _, combos = self._sig_components(self.guidance_signature)
        id_prefixes = [str(self.guidance_identifier)]
        for f in os.listdir(PRETRAINED_GUIDANCE_DIR):
            if not f.endswith(f"__{suffix}.pth"):
                continue
            if not any(f.startswith(prefix + "__") for prefix in id_prefixes):
                continue
            if self._sig_components(f.split(f"__{suffix}.pth")[0])[1] == combos:
                return os.path.join(PRETRAINED_GUIDANCE_DIR, f)
        return None

    def _sig_components(self, sig):
        parts = sig.split("__")
        if not parts: return None, ()
        combos = []
        for p in parts[1:]:
            if p == "unguided": combos.append(("unguided", None))
            elif "_y" in p:
                n, t = p.rsplit("_y", 1)
                try: combos.append((n, int(t)))
                except: combos.append((n, t))
            else: combos.append((p, None))
        return parts[0], tuple(sorted(combos, key=str))

    def _prepare_models(self, sample_fn, hidden_dim=256, output_dim_override=None):
        batch = sample_fn(2, self.device)
        inputs, target, _ = self._extract_batch(batch)
        dims = (inputs.shape[-1], output_dim_override if output_dim_override is not None else target.shape[-1])
        
        def build_mlp(out_d): 
            return nn.Sequential(nn.Linear(dims[0], hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_d)).to(self.device)
            
        if not self.learned_guidance_model: self.learned_guidance_model = build_mlp(dims[1])
        if getattr(self.cfg, "use_z_model", True) and not self.learned_z_model: self.learned_z_model = build_mlp(1)

    def _restore_models(self):
        """
        Restore models from checkpoints if they exist and are compatible.
        
        Returns:
            bool: True if ALL required models are successfully restored, False otherwise
        """
        guidance_restored = False
        z_restored = False

        # Search the run's guidance_ckpt_dir first, then fall back to the shipped
        # pretrained_guidance directory so a matching pretrained network is loaded
        # instead of being retrained.
        if self.learned_guidance_model:
            p = self._ckpt_path("guidance")
            if not os.path.exists(p):
                p = self._pretrained_ckpt_path("guidance") or p
            if os.path.exists(p):
                try:
                    print(f"[ComposedGuidance] Loading checkpoint from: {p}")
                    checkpoint_state = torch.load(p, map_location=self.device)
                    self.learned_guidance_model.load_state_dict(checkpoint_state)
                    self.learned_guidance_model.eval()
                    guidance_restored = True
                except RuntimeError as e:
                    # Architecture mismatch (e.g. scalar vs vector field): skip load and retrain.
                    if "size mismatch" in str(e) or "shape" in str(e).lower():
                        print(f"[ComposedGuidance] Guidance checkpoint architecture mismatch (likely scalar vs vector field).")
                        print(f"  Error: {e}")
                        print(f"  Skipping checkpoint load - will retrain with new architecture.")
                        guidance_restored = False
                    else:
                        raise
        else:
            guidance_restored = True

        # Restore z model (if any), with the same pretrained fallback.
        if self.learned_z_model:
            p = self._ckpt_path("z")
            if not os.path.exists(p):
                p = self._pretrained_ckpt_path("z") or p
            if os.path.exists(p):
                try:
                    print(f"[ComposedGuidance] Loading z model checkpoint from: {p}")
                    checkpoint_state = torch.load(p, map_location=self.device)
                    self.learned_z_model.load_state_dict(checkpoint_state)
                    self.learned_z_model.eval()
                    z_restored = True
                except RuntimeError as e:
                    if "size mismatch" in str(e) or "shape" in str(e).lower():
                        print(f"[ComposedGuidance] Z model checkpoint architecture mismatch.")
                        print(f"  Error: {e}")
                        print(f"  Skipping checkpoint load - will retrain with new architecture.")
                        z_restored = False
                    else:
                        raise
        else:
            z_restored = True

        fully_restored = guidance_restored and z_restored

        if fully_restored:
            print(f"[ComposedGuidance] Successfully restored all models for {self.guidance_signature}")
        else:
            if not guidance_restored:
                print(f"[ComposedGuidance] Guidance model not restored - will train")
            if not z_restored and self.learned_z_model:
                print(f"[ComposedGuidance] Z model not restored - will train")

        return fully_restored

    def save_learned_guidance(self):
        model = self.learned_guidance_model
        if hasattr(model, "module"): model = model.module
        if model: torch.save(model.state_dict(), self._ckpt_path("guidance"))
    def save_learned_z_model(self):
        model = self.learned_z_model
        if hasattr(model, "module"): model = model.module
        if model: torch.save(model.state_dict(), self._ckpt_path("z"))

    def _extract_batch(self, batch):
        """Unpack a training batch into (model input, regression target, weight).

            v_θ(xt, t): base flow velocity field (``v`` in the code).
            u_t(xt|x1): conditional velocity field (``ut`` in the code).
            Goal: learn g_φ(xt, t) ≈ u_t - v_θ (the ``target`` in the code).
        """
        if isinstance(batch, (tuple, list)):
            x, t = batch[0].to(self.device), batch[1].to(self.device)
            if t.dim() == 1: t = t.view(-1, 1)
            target = torch.zeros_like(x)  # placeholder target for the generic case
            return torch.cat([x, t], dim=-1), target, None
            
        inputs = batch.get("inputs")
        xt, t = batch.get("xt"), batch.get("t")
        if inputs is None: 
            t_ = t.view(-1, 1) if t.dim() == 1 else t
            inputs = torch.cat([xt, t_], dim=-1)
        elif xt is None: xt, t = inputs[:, :-1], inputs[:, -1:]
            
        xt, t = xt.to(self.device), t.to(self.device)
        if t.dim() == 1: t = t.view(-1, 1)

        target = batch.get("target")
        if target is None:
            v = batch.get("v") if batch.get("v") is not None else self.flow_model(xt, t).detach()
            ut = batch.get("ut")
            target = ut.to(self.device) - v
            
        w = batch.get("weight")
        return inputs.to(self.device), target.to(self.device), w.to(self.device) if w is not None else None

class GCovGGuidance(ComposedGuidance):
    @torch.enable_grad()
    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        if not self.classifiers: return torch.zeros_like(x)

        need_higher_order = bool(need_higher_order) if not isinstance(need_higher_order, bool) else need_higher_order

        x_req = x.detach().requires_grad_(True) if not need_higher_order else (x if x.requires_grad else x.clone().detach().requires_grad_(True))
        v_for_est = v_uncond.detach()
        
        energy = _compute_g_cov_g_energy(x_req, t, v_for_est, self.classifiers, self.targets, self.scales, self.cfg).sum()
        return torch.autograd.grad(energy, x_req, create_graph=need_higher_order, retain_graph=need_higher_order)[0]

    def train_model(self):
        pass  # not learnable

class GCovGGMGuidance(ComposedGuidance):
    def _prepare_models_for_subclass(self):
        batch = self.sample_batch_fn(2, self.device)
        inputs, _, _ = self._extract_batch(batch)
        space_dim = inputs.shape[-1] - 1  # subtract time dimension
        self._prepare_models(self.sample_batch_fn, output_dim_override=space_dim)
        print(f"[GCovGGMGuidance] output_dim={space_dim} (vector field)")

    def _compute_guidance_for_rollout(self, x, t, v_uncond):
        """Pure inference version of compute_guidance for rollout (no autograd graph).

        Ensures rollout stays truly no_grad even though compute_guidance carries the
        @torch.enable_grad() decorator, avoiding graph build-up during trajectory generation.
        """
        if not self.learned_guidance_model:
            return torch.zeros_like(x)

        x_in, t_ = _normalize_tensor_shapes(x, t)
        x_req = x_in.detach()

        learned_guidance_correction = self.learned_guidance_model(torch.cat([x_req, t_], dim=-1)).detach()

        with torch.enable_grad():
            base_guidance = GCovGGuidance.compute_guidance(self, x_req, t, v_uncond, need_higher_order=False).detach()

        return self._apply_gcar(base_guidance, learned_guidance_correction, x_req)

    @torch.enable_grad()
    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        if not self.learned_guidance_model:
            return torch.zeros_like(x)

        x_in, t_ = _normalize_tensor_shapes(x, t)
        x_req = _prepare_input_for_grad(x_in, need_higher_order)

        # Model directly outputs the learned residual ∇r.
        learned_guidance = self.learned_guidance_model(torch.cat([x_req, t_], dim=-1))
        base_guidance = GCovGGuidance.compute_guidance(self, x_req, t, v_uncond, need_higher_order)
        return self._apply_gcar(base_guidance, learned_guidance, x_req)

class GuidanceMatching(ComposedGuidance):
    def _prepare_models_for_subclass(self):
        # Output the vector field (guidance correction) directly.
        batch = self.sample_batch_fn(2, self.device)
        inputs, _, _ = self._extract_batch(batch)
        space_dim = inputs.shape[-1] - 1  # subtract 1 for the time dimension
        self._prepare_models(self.sample_batch_fn, output_dim_override=space_dim)
        print(f"[GuidanceMatching] output_dim={space_dim} (vector field)")
    
    def compute_guidance(self, x, t, v_uncond, need_higher_order=False):
        if not self.learned_guidance_model:
            return torch.zeros_like(x)
        
        x_in, t_ = _normalize_tensor_shapes(x, t)
        x_req = _prepare_input_for_grad(x_in, need_higher_order)
        return self.learned_guidance_model(torch.cat([x_req, t_], dim=-1))

    def train_model(self):
        import time
        
        steps = getattr(self.cfg, "guidance_train_steps", 1000)
        batch_size = getattr(self.cfg, "guidance_batch_size", 512)
        lr = getattr(self.cfg, "guidance_lr", 1e-3)
        log_interval = 100
        g3_mode = True

        training_start_time = time.time()

        if g3_mode and self.learned_z_model:
            opt = torch.optim.Adam(self.learned_z_model.parameters(), lr=lr)
            print(f"[GuidanceMatching] Training Z model ({steps} steps)")

            z_total_samples = 0
            z_samples_used = 0
            z_samples_skipped = 0
            
            for i in range(steps):
                inp, _, w = self._extract_batch(self.sample_batch_fn(batch_size, self.device))
                z_total_samples += batch_size
                if w is None:
                    z_samples_skipped += batch_size
                    continue
                z_samples_used += batch_size
                loss = F.mse_loss(self.learned_z_model(inp), w.to(self.device) + 1e-8)
                opt.zero_grad(); loss.backward(); opt.step()
                if log_interval and (i+1)%log_interval==0: print(f"Z-Step {i+1} loss={loss.item():.5f}")
            self.learned_z_model.eval()

            if z_total_samples > 0:
                z_efficiency = z_samples_used / z_total_samples
                print(f"[GuidanceMatching] Z Model Training Efficiency: {z_efficiency:.2%} ({z_samples_used}/{z_total_samples})")

        opt = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()
        print(f"[GuidanceMatching] Training guidance model ({steps} steps)")

        total_samples_generated = 0
        samples_used_for_training = 0
        samples_skipped = 0

        for i in range(steps):
            inp, target, w = self._extract_batch(self.sample_batch_fn(batch_size, self.device))
            total_samples_generated += batch_size
            
            pred = self.learned_guidance_model(inp)
            w_final = torch.ones_like(pred[:, :1]) if w is None else w.to(self.device)
            if g3_mode and self.learned_z_model:
                with torch.no_grad(): w_final /= (self.learned_z_model(inp).abs() + 1e-8)

            # Every sample is used (uniform weights when w is None), so count all.
            samples_used_for_training += batch_size

            loss = ((pred - target).pow(2) * w_final).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if log_interval and (i+1) % log_interval == 0:
                print(f"G-Step {i+1}/{steps} Loss: {loss.item():.6f}")

            loss_early_stop_threshold = getattr(self.cfg, "loss_early_stop_threshold", None)
            if loss_early_stop_threshold is not None:
                loss_value = loss.item()
                if loss_value < loss_early_stop_threshold:
                    print(f"[GuidanceMatching] Early stopping at step {i+1}/{steps}: "
                          f"loss={loss_value:.6f} < threshold={loss_early_stop_threshold:.6f}")
                    break

        self.learned_guidance_model.eval()
        self.save_learned_guidance()
        if self.learned_z_model: self.save_learned_z_model()

        training_time = time.time() - training_start_time
        self.training_time = training_time
        print(f"[GuidanceMatching] Training completed in {training_time:.2f}s")

        if total_samples_generated > 0:
            efficiency_ratio = samples_used_for_training / total_samples_generated
            print(f"\n[GuidanceMatching] Guidance Model Training Efficiency Statistics:")
            print(f"  - Total samples generated: {total_samples_generated}")
            print(f"  - Samples used for training: {samples_used_for_training}")
            print(f"  - Samples skipped: {samples_skipped}")
            print(f"  - Training efficiency: {efficiency_ratio:.2%} ({samples_used_for_training}/{total_samples_generated})")

class GCovGGMOnlineGuidance(GCovGGMGuidance):
    def _prepare_models_for_subclass(self):
        # The online residual g_psi is a vector field (like GuidanceMatching),
        # trained to regress u_t - v_θ on conflict regions.
        batch = self.sample_batch_fn(2, self.device)
        inputs, _, _ = self._extract_batch(batch)
        space_dim = inputs.shape[-1] - 1  # subtract 1 for the time dimension
        self._prepare_models(self.sample_batch_fn, output_dim_override=space_dim)
        print(f"[GCovGGMOnlineGuidance] output_dim={space_dim} (vector field for u_t - v_θ)")

        # Online guidance never trains, saves, or uses a separate z model, so we
        # drop the one created by _prepare_models. This way restoring only the
        # guidance network is enough to skip retraining (otherwise the missing
        # z-model checkpoint would force a retrain on every run).
        self.learned_z_model = None

        # Zero-init the last layer for a warm start (residual = 0 initially).
        if self.learned_guidance_model and isinstance(self.learned_guidance_model, nn.Sequential):
            nn.init.zeros_(self.learned_guidance_model[-1].weight)
            nn.init.zeros_(self.learned_guidance_model[-1].bias)
    
    def _compute_trajectory_conflict_mask(self, xs_stacked, num_steps, batch_size):
        """Compute the conflict weight for every trajectory point.

        Returns ``conflict_mask``: the degree of gradient conflict at each x_t
        along the trajectory.
        """
        if self.distribution is None:
            return None, 1.0

        with torch.no_grad():
            space_dim = xs_stacked.shape[-1]
            conflict = self._compute_conflict_score(xs_stacked.reshape(-1, space_dim), self.targets)

        if conflict is None:
            return None, 1.0

        conflict = conflict.view(num_steps, batch_size)
        mask = self._compute_gcar_weight(conflict)
        active_ratio = (mask > 0.5).float().mean().item()
        return mask, active_ratio

    def _compute_trajectory_weights_ground_truth(self, x1, batch_size):
        """Compute target label distribution p*(i) from terminal reward."""
        with torch.no_grad():
            # Build energy from the distribution's J (smaller is better), then reward = -energy.
            total_J = torch.zeros(batch_size, device=self.device)
            for clf, target, scale in zip(self.classifiers, self.targets, self.scales):
                J_i = self.distribution.get_J(x1, classifier=clf, label=target)  # [B]
                total_J = total_J + float(scale) * J_i
            r1 = -total_J  # [B]

            # Softmax over the batch gives the label distribution p*(i) ∝ exp(β r1(i)).
            beta = getattr(self.cfg, "energy_temperature", 1.0)
            logits = beta * r1
            logits = logits - logits.max()          # numerical stability
            w_eff = torch.softmax(logits, dim=0)    # [B]
            w_eff = w_eff.unsqueeze(-1)             # [B, 1] for broadcasting

        return w_eff, r1

    def _compute_online_loss_gradient(
        self,
        xs_stacked,      # (T, B, space_dim)
        ts_stacked,      # (T, B, 1)
        r1,              # (B,) or (B,1), terminal reward/label (not used for target computation)
        conflict_mask,   # (T, B)
        num_steps,
        batch_size,
        x1=None,         # (B, space_dim) - terminal point x1, required for computing u_t
    ):
        """
        Gradient regression loss for velocity field correction (like GuidanceMatching):
            For each (t, i), regress pred(t, i) to u_t - v_θ,
            where u_t is conditional velocity from OT path and v_θ is base flow model prediction.
            Weighted by conflict_mask only (no trajectory-level weights).
        
        Note: This is consistent with GuidanceMatching which learns g_φ(x_t, t) ≈ u_t - v_θ.
        
        For linear OT path (CondOT): x_t = (1-t) * x_0 + t * x_1
        Therefore: u_t = x_1 - x_0 = (x_1 - x_t) / (1 - t)
        """
        space_dim = xs_stacked.shape[-1]

        if x1 is None:
            raise ValueError("_compute_online_loss_gradient requires x1 parameter to compute u_t")

        xs_flat = xs_stacked.reshape(-1, space_dim)  # (T*B, space_dim)
        ts_flat = ts_stacked.reshape(-1, 1)  # (T*B, 1)
        ts_flat_expanded = ts_flat.squeeze(-1)  # (T*B,)
        inp = torch.cat([xs_flat, ts_flat], dim=-1)  # (T*B, space_dim+1)

        pred = self.learned_guidance_model(inp)  # (T*B, space_dim)
        pred = pred.view(num_steps, batch_size, space_dim)  # (T, B, space_dim)

        x1_expanded = x1.unsqueeze(0).expand(num_steps, -1, -1)  # (T, B, space_dim)

        t_expanded = ts_stacked.expand(-1, -1, space_dim)  # (T, B, space_dim)

        # u_t = (x_1 - x_t) / (1 - t); clamp the denominator to avoid div-by-zero at t ≈ 1.
        one_minus_t = (1.0 - ts_stacked).clamp_min(1e-6).expand(-1, -1, space_dim)  # (T, B, space_dim)
        u_t = (x1_expanded - xs_stacked) / one_minus_t  # (T, B, space_dim)
        u_t = torch.clamp(u_t, -100.0, 100.0)  # prevent extreme values from causing NaN loss

        with torch.no_grad():
            v_theta = self.flow_model(xs_flat, ts_flat_expanded)  # (T*B, space_dim)
            v_theta = v_theta.view(num_steps, batch_size, space_dim)  # (T, B, space_dim)

        # Velocity-field correction target: u_t - v_θ.
        target = (u_t - v_theta).detach()  # (T, B, space_dim)

        if conflict_mask is not None:
            weight = conflict_mask.to(pred.device).float().unsqueeze(-1).expand(-1, -1, space_dim)  # (T, B, space_dim)
        else:
            weight = torch.ones_like(pred)

        loss_unreduced = (pred - target).pow(2)  # (T, B, space_dim)

        loss = (loss_unreduced * weight).sum() / (weight.sum() + 1e-8)

        return loss
    
    def _log_online_training_progress_ground_truth(self, step, total_steps, loss, active_ratio, w_eff, r1):
        with torch.no_grad():
            w_eff_flat = w_eff.flatten()
            # Expect corr(w_eff, r1) to be positive: higher reward => higher weight
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

        print(
            f"Online-Step {step}/{total_steps} Loss: {loss.item():.6f} | "
            f"Active Conflict: {active_ratio:.1%} | "
            f"corr(w_eff, reward): {corr:.3f}"
        )
    
    def train_model(self):
        import time
        
        steps = getattr(self.cfg, "guidance_train_steps", 1000)
        batch_size = getattr(self.cfg, "guidance_batch_size", 512)
        lr = getattr(self.cfg, "guidance_lr", 1e-3)
        log_interval = 100

        # ODE solver params
        num_steps = 20
        dt = 1.0 / num_steps

        optimizer = torch.optim.Adam(self.learned_guidance_model.parameters(), lr=lr)
        self.learned_guidance_model.train()

        print(f"[GCovGGMOnlineGuidance] Training Online Residual model ({steps} steps)")

        training_start_time = time.time()

        total_samples_generated = 0
        samples_used_for_training = 0  # after conflict_mask filtering
        samples_skipped = 0

        for i in range(steps):
            # Generate online trajectories with the current guidance (residual ≈ 0 at first,
            # i.e. base guidance; it becomes base + learned as training proceeds).
            sample_batch = self.sample_batch_fn(1, self.device)
            sample_inputs, _, _ = self._extract_batch(sample_batch)
            space_dim = sample_inputs.shape[-1] - 1  # subtract 1 for the time dimension
            x = torch.randn(batch_size, space_dim, device=self.device)

            total_samples_generated += batch_size

            traj_xs = []
            traj_ts = []

            curr_x = x
            traj_diverged = False
            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((batch_size, 1), t_val, device=self.device)

                traj_xs.append(curr_x.clone())
                traj_ts.append(t_tensor)

                # Rollout-specific guidance under no_grad: velocity = v_uncond + g_total,
                # avoiding autograd graph build-up during trajectory generation.
                with torch.no_grad():
                    v_uncond = self.flow_model(curr_x, t_tensor)
                    g_total = self._compute_guidance_for_rollout(curr_x, t_tensor, v_uncond)
                    d_x = v_uncond + g_total

                # Euler step
                curr_x = curr_x + d_x * dt

                # Divergence check: abort if trajectory explodes (large values or NaN)
                if not torch.isfinite(curr_x).all() or curr_x.abs().max() > 1e4:
                    traj_diverged = True
                    break

            if traj_diverged:
                samples_skipped += batch_size
                continue

            x1 = curr_x

            # Conflict at terminal x1 drives the combined early-stopping flag.
            x1_conflict_threshold = getattr(self.cfg, "x1_conflict_threshold", None)
            x1_conflict_ratio = None
            x1_conflict_flag = False
            if self.distribution is not None:
                with torch.no_grad():
                    x1_conflict = self._compute_conflict_score(x1, self.targets)  # (B,)
                    if x1_conflict is not None:
                        threshold, _ = self._get_conflict_threshold_and_temperature()
                        x1_has_conflict = (x1_conflict > threshold).float()  # (B,)
                        x1_conflict_ratio = x1_has_conflict.mean().item()

                        print(f"[GCovGGMOnlineGuidance] Step {i+1}/{steps} | x1_conflict_ratio: {x1_conflict_ratio:.4f}")

                        if log_interval and (i+1) % log_interval == 0:
                            print(f"[GCovGGMOnlineGuidance] Step {i+1}/{steps} | x1_conflict_ratio: {x1_conflict_ratio:.4f} | "
                                  f"x1_conflict_mean: {x1_conflict.mean().item():.4f} | "
                                  f"x1_conflict_max: {x1_conflict.max().item():.4f} | "
                                  f"x1_conflict_min: {x1_conflict.min().item():.4f}")

                        if x1_conflict_threshold is not None:
                            if x1_conflict_ratio <= x1_conflict_threshold:
                                x1_conflict_flag = True
                                print(f"[GCovGGMOnlineGuidance] Step {i+1}/{steps} | x1_conflict_flag=True "
                                      f"(x1_conflict_ratio={x1_conflict_ratio:.4f} <= threshold={x1_conflict_threshold:.4f})")

            xs_stacked = torch.stack(traj_xs, dim=0)  # (T, B, space_dim)
            ts_stacked = torch.stack(traj_ts, dim=0)  # (T, B, 1)
            # xs_stacked[-1] is the last stored point x(t=0.95); the true terminal
            # x1 = x(t=1.00) is passed separately to the loss helpers.

            conflict_mask, active_ratio = self._compute_trajectory_conflict_mask(
                xs_stacked, num_steps, batch_size
            )

            # Early stop only if x1_conflict_flag is True AND active_ratio is below threshold.
            active_ratio_threshold = getattr(self.cfg, "active_ratio_threshold", None)
            print(f"[GCovGGMOnlineGuidance] Step {i+1}/{steps} | active_ratio: {active_ratio:.4f} | active_ratio_threshold: {active_ratio_threshold:.4f} | x1_conflict_flag: {x1_conflict_flag}")
            if active_ratio_threshold is not None and x1_conflict_flag and (conflict_mask is None or active_ratio <= active_ratio_threshold):
                print(f"[GCovGGMOnlineGuidance] Early stopping at step {i+1}/{steps}: "
                      f"x1_conflict_flag=True AND active_ratio={active_ratio:.6f} <= threshold={active_ratio_threshold:.6f} "
                      f"(no conflict regions along trajectory)")
                self.learned_guidance_model.eval()
                self.save_learned_guidance()
                break

            # Skip this batch when there is no usable conflict signal.
            if conflict_mask is None or active_ratio < 1e-6:
                samples_skipped += batch_size
                continue

            samples_used_for_training += batch_size

            # Skip batch if trajectory contains NaN (diverged rollout)
            if torch.isnan(xs_stacked).any() or torch.isnan(x1).any():
                print(f"[GCovGGMOnlineGuidance] Step {i+1}: NaN in trajectory, skipping")
                samples_skipped += batch_size
                continue

            w_eff, r1 = self._compute_trajectory_weights_ground_truth(x1, batch_size)

            # Gradient regression: g_psi predicts u_t - v_θ on conflict regions.
            loss = self._compute_online_loss_gradient(
                xs_stacked, ts_stacked, r1, conflict_mask,
                num_steps, batch_size, x1=x1
            )

            if log_interval and (i+1) % log_interval == 0:
                self._log_online_training_progress_ground_truth(i+1, steps, loss, active_ratio, w_eff, r1)

            loss_value = loss.item()
            print(f"[GCovGGMOnlineGuidance] Step {i+1}/{steps} | loss: {loss_value:.6f}")
            if not torch.isfinite(loss):
                # NaN/Inf loss: skip this update entirely to prevent weight corruption.
                # Corrupted weights make every subsequent inference return NaN and
                # cause ODE divergence at evaluation time.
                print(f"[GCovGGMOnlineGuidance] WARNING: loss={loss_value} at step {i+1}, skipping update")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.learned_guidance_model.parameters(), max_norm=10.0)
            optimizer.step()

            loss_early_stop_threshold = getattr(self.cfg, "loss_early_stop_threshold", None)
            if loss_early_stop_threshold is not None:
                if loss_value < loss_early_stop_threshold:
                    print(f"[GCovGGMOnlineGuidance] Early stopping at step {i+1}/{steps}: "
                          f"loss={loss_value:.6f} < threshold={loss_early_stop_threshold:.6f}")
                    break

        self.learned_guidance_model.eval()
        self.save_learned_guidance()

        training_time = time.time() - training_start_time
        self.training_time = training_time
        print(f"[GCovGGMOnlineGuidance] Training completed in {training_time:.2f}s")

        if total_samples_generated > 0:
            efficiency_ratio = samples_used_for_training / total_samples_generated
            print(f"\n[GCovGGMOnlineGuidance] Training Efficiency Statistics:")
            print(f"  - Total samples generated: {total_samples_generated}")
            print(f"  - Samples used for training: {samples_used_for_training}")
            print(f"  - Samples skipped (conflict_mask filtering): {samples_skipped}")
            print(f"  - Training efficiency: {efficiency_ratio:.2%} ({samples_used_for_training}/{total_samples_generated})")
            if samples_skipped > 0:
                reduction_ratio = (1.0 - efficiency_ratio) * 100
                print(f"  - Conflict score filtering reduced training samples by {reduction_ratio:.2f}%")

