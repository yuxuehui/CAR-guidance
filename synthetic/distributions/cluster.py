"""Target distribution for the synthetic experiment.

``ClusterDistribution`` is a 3-component 2D Gaussian mixture. It also hosts the
guidance machinery built on top of the pretrained reward classifiers: per-class
energies ``J(x) = -log p(y|x)``, their gradients, and the pairwise conflict
score used by CAR guidance.
"""

import math
import os
import sys
import numbers
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDistribution

GUIDANCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guidance"
)
if os.path.isdir(GUIDANCE_DIR) and GUIDANCE_DIR not in sys.path:
    sys.path.insert(0, GUIDANCE_DIR)

from classifier import Classifier

class ClusterDistribution(BaseDistribution):
    def __init__(
        self,
        centers: torch.Tensor = torch.tensor(
            [[8.0, 8.0], [8.0, -8.0], [0.0, 10.0]], dtype=torch.float32
        ),
        std: torch.Tensor = torch.ones(3, 2),
        device: Optional[torch.device] = None,
        guidance_dir: Optional[str] = None,
        classifier_targets: Optional[Dict[str, Tuple[str, int]]] = None,
        zero_gradient_threshold_direct: Optional[float] = None,
    ):
        super().__init__()
        self.centers = centers
        self.std = std
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        default_guidance_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "guidance",
            "pretrained_guidance",
        )
        self.guidance_dir = guidance_dir or default_guidance_dir
        self.classifier_targets = classifier_targets or {
            "j1": ("Classifier_0", 0),
            "j2": ("Classifier_1", 1),
        }
        self.zero_gradient_threshold_direct = (
            zero_gradient_threshold_direct if zero_gradient_threshold_direct is not None else 1e-2
        )
        self._guidance_models: Dict[str, Classifier] = {}
        self.guidance_func = self._build_guidance_func()

    def _build_guidance_func(self) -> Optional[Callable[[torch.Tensor, str, int], torch.Tensor]]:
        """
        Build a guidance function from pretrained classifiers for guided flow matching.
        
        In guided flow matching, the total energy is:
            E(x) = -log p_base(x) + Σ_j λ_j * (-log p(y_j|x))
        
        This function loads pretrained classifiers and returns a callable that computes
        the guidance energy term: -log p(y|x) for a given classifier and target label.
        
        Returns:
            A function that takes (x, classifier_name, label) and returns -log p(y=label|x)
        """
        if not os.path.isdir(self.guidance_dir):
            return None

        classifiers: Dict[str, Classifier] = {}
        for filename in os.listdir(self.guidance_dir):
            if not filename.endswith(".pth"):
                continue

            model_path = os.path.join(self.guidance_dir, filename)
            model = Classifier().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats (with or without "net." prefix)
            loaded = False
            candidate_dicts = [state_dict]
            remapped = {
                (key if key.startswith("net.") else f"net.{key}"): value
                for key, value in state_dict.items()
            }
            candidate_dicts.append(remapped)

            for cand in candidate_dicts:
                try:
                    model.load_state_dict(cand)
                    loaded = True
                    break
                except RuntimeError:
                    continue
            if not loaded:
                print(
                    f"[ClusterDistribution] Warning: could not load {filename}; skipping guidance weight."
                )
                continue

            # Map filename to naming: classifier_1.pth -> Classifier_0, classifier_2.pth -> Classifier_1
            base_name = os.path.splitext(filename)[0]
            if base_name == "classifier_1":
                key = "Classifier_0"
            elif base_name == "classifier_2":
                key = "Classifier_1"
            else:
                key = base_name

            model.eval()
            classifiers[key] = model

        if not classifiers:
            return None

        # Ensure _guidance_models keys are ordered as ['Classifier_0', 'Classifier_1', ...]
        ordered_keys = ['Classifier_0', 'Classifier_1']
        for key in classifiers.keys():
            if key not in ordered_keys:
                ordered_keys.append(key)

        self._guidance_models = {key: classifiers[key] for key in ordered_keys if key in classifiers}

        @torch.no_grad()
        def _guidance(x: torch.Tensor, classifier_name: str, label: int) -> torch.Tensor:
            """
            Compute the classifier guidance energy: -log p(y=label|x)
            
            Mathematical formulation:
                Given classifier output logits z = f(x), compute:
                    p(y|x) = softmax(z)_y = exp(z_y) / Σ_k exp(z_k)
                    -log p(y|x) = -z_y + log(Σ_k exp(z_k))
            
            Args:
                x: Input samples, shape (B, d)
                classifier_name: Name of the classifier to use (e.g., "Classifier_0")
                label: Target class label (int)
            
            Returns:
                Guidance energy -log p(y=label|x), shape (B,)
            """
            if classifier_name not in self._guidance_models:
                raise ValueError(
                    f"Unknown classifier '{classifier_name}'. "
                    f"Available: {list(self._guidance_models.keys())}"
                )

            logits = self._guidance_models[classifier_name](x.to(self.device))
            log_probs = F.log_softmax(logits, dim=-1)

            label = int(label)
            if label < 0 or label >= log_probs.shape[-1]:
                raise ValueError(f"Invalid label {label}")

            # Lower energy -> higher probability of belonging to the target class
            return -log_probs[:, label].to(x.device)

        return _guidance

    def _classifier_energy(self, x: torch.Tensor, tag: str) -> Optional[torch.Tensor]:
        """
        Compute classifier guidance energy using predefined tag mappings.
        
        This is a convenience wrapper that maps tags (e.g., "j1", "j2") to
        specific (classifier_name, label) pairs defined in self.classifier_targets.
        
        Example:
            classifier_targets = {
                "j1": ("Classifier_0", 0),  # Guide towards class 0 of Classifier_0
                "j2": ("Classifier_1", 1),  # Guide towards class 1 of Classifier_1
            }
        
        Args:
            x: Input samples, shape (B, d)
            tag: Predefined tag string (e.g., "j1", "j2")
        
        Returns:
            Guidance energy -log p(y=label|x), or None if tag not found
        """
        if self.guidance_func is None:
            return None
        if tag not in self.classifier_targets:
            return None

        classifier_name, label = self.classifier_targets[tag]
        return self.guidance_func(x, classifier_name, label)    # -log p(y=label|x)

    # ---------- Sampling ----------
    def sample_with_labels(self, batch_size: int, device: str = "cuda"):
        """
        Sample from the mixture of Gaussians with multiple clusters and return both samples and cluster indices.

        Args:
            batch_size: int, number of samples to generate
            device: str, device to use for sampling
        Returns:
            samples: torch.Tensor, samples from the mixture
            cluster_idx: torch.Tensor, cluster indices for each sample
        """
        device = torch.device(device)
        dtype = self.centers.dtype

        K = self.centers.shape[0]
        cluster_idx = torch.multinomial(
            torch.ones(K, device=device), batch_size, replacement=True
        )
        eps = torch.randn(batch_size, 2, device=device, dtype=dtype)
        centers = self.centers.to(device=device, dtype=dtype)[cluster_idx]
        stds = self.std.to(device=device, dtype=dtype)[cluster_idx]
        samples = centers + eps * stds
        return samples, cluster_idx

    def sample(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """Sample from the mixture of Gaussians with multiple clusters and return only the coordinates."""
        samples, _ = self.sample_with_labels(batch_size=batch_size, device=device)
        return samples

    # ---------- Probability ----------
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density of an equally weighted Gaussian mixture.
        
        Mathematical formulation:
        For a Gaussian Mixture Model with K equally-weighted components:
            p(x) = (1/K) * Σ_{k=1}^K N(x | μ_k, Σ_k)
        Where each Gaussian component is:
            N(x | μ_k, Σ_k) = 1 / ((2π)^(d/2) |Σ_k|^(1/2)) * exp(-1/2 * (x-μ_k)^T Σ_k^(-1) (x-μ_k))
        
        Args:
            x: torch.Tensor, samples from the mixture, shape (B, d)
        Returns:
            prob: torch.Tensor, probability density of the samples, shape (B,)
        """
        device, dtype = x.device, x.dtype
        centers = self.centers.to(device=device, dtype=dtype)  # (K,2)
        stds    = self.std.to(device=device, dtype=dtype)      # (K,2)

        # Quadratic (Mahalanobis) term: -1/2 * (x-μ_k)^T Σ_k^(-1) (x-μ_k)
        quad = -0.5 * ((x.unsqueeze(1) - centers.unsqueeze(0)) / stds.unsqueeze(0)).pow(2).sum(-1)

        # Log normalization constant: logZ_k = (d/2)*log(2π) + Σ_i log(σ_{k,i})
        d = x.size(-1)
        logZ = 0.5 * d * math.log(2 * math.pi) + torch.log(stds).sum(-1)  # (K,)

        logNk = quad - logZ.unsqueeze(0)                                   # (B,K)
        log_mix = torch.logsumexp(logNk, dim=1) - math.log(centers.size(0))# (B,)
        return torch.exp(log_mix)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.prob(x).clamp_min(1e-12).log()

    # ---------- Energy J ----------
    def get_J(
        self,
        x: torch.Tensor,
        mode: str = "product",
        alpha: float = 1.0,
        classifier: Optional[Union[str, nn.Module]] = None,
        label: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the energy for the synthetic clusters or reuse pretrained classifiers.

        If `classifier` and `label` are provided (and pretrained classifiers exist),
        the energy is derived from the negative log probability of the classifier.
        Otherwise the legacy geometric energy is returned, controlled by `mode`.
        """
        if classifier is not None:
            if label is None:
                raise ValueError("Label must be provided when using classifier-guided energy.")
            
            # Case 1: Classifier is a Module object (direct call)
            if isinstance(classifier, nn.Module):
                logits = classifier(x)
                log_probs = F.log_softmax(logits, dim=-1)
                label = int(label)
                if label < 0 or label >= log_probs.shape[-1]:
                    raise ValueError(f"Invalid label {label}")
                return -log_probs[:, label].to(x.device)

            # Case 2: Classifier is a string name (lookup in registry)
            if self.guidance_func is None:
                raise RuntimeError("No pretrained classifiers are available for guidance.")
            return self.guidance_func(x, classifier, label)

        J1 = self._classifier_energy(x, "j1")
        J2 = self._classifier_energy(x, "j2")

        if J1 is None or J2 is None:
            device, dtype = x.device, x.dtype
            tau = 0.15
            scale = 10.0

            target = x.new_tensor([0.8, 0.8])
            d2_j1 = (x - target).pow(2).sum(dim=-1)
            J1 = scale * torch.exp(-d2_j1 / (2 * tau * tau))

            centers2 = x.new_tensor([0.0, 1.0])
            d2_j2 = ((x.unsqueeze(1) - centers2.unsqueeze(0)) ** 2).sum(dim=-1)
            J2 = (scale * torch.exp(-d2_j2 / (2 * tau * tau))).max(dim=1).values

        if mode == "j1":
            J = J1
        elif mode == "j2":
            J = J2
        elif mode == "geo":
            J = torch.sqrt(J1 * J2 + 1e-12)
        elif mode == "min":
            J = torch.minimum(J1, J2)
        elif mode == "sum":
            J = J1 + J2
        elif mode == "product":
            J = J1 + J2
        elif mode == "negation":
            J = J1 - alpha * J2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if mode in ["sum", "product"]:
            return J.clamp(0.0, 100.0)
        if mode == "negation":
            return J.clamp(-100.0, 100.0)
        return J.clamp(0.0, 100.0)

    def get_classifier_gradients(
        self,
        x: torch.Tensor,
        label,
        classifier_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Compute gradients of log-probabilities w.r.t x for (classifier, label) combinations.

        This function is flexible to support:
          - Single label for ALL classifiers (backward compatible):
                get_classifier_gradients(x, label=1)
          - Per-classifier labels (same number of labels as classifiers):
                get_classifier_gradients(x, label=[1, 0])
            where label[i] is used for classifier i.
          - A subset of classifiers with their own labels:
                get_classifier_gradients(x, label=[1], classifier_indices=[0])

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            label:
                - int or 0-dim tensor: same label applied to all (or selected) classifiers.
                - list/tuple of ints/tensors: labels per classifier (or per index in
                  classifier_indices).
            classifier_indices:
                - None (default): iterate over all classifiers in self._guidance_models.
                - list of ints: indices into the ordered classifier list that should be used.

        Returns:
            A list of torch.Tensor, each containing the gradient of
            log p(y=label|x) w.r.t x for one (classifier, label) pair.
        """
        # Normalize classifiers into an ordered list for stable indexing
        clf_items = list(self._guidance_models.items())  # [(name, module), ...]
        num_clf = len(clf_items)

        # Normalize classifier_indices
        if classifier_indices is None:
            classifier_indices = list(range(num_clf))
        else:
            classifier_indices = list(classifier_indices)

        # Normalize label(s)
        if isinstance(label, torch.Tensor):
            if label.dim() == 0:
                label = int(label.item())
            else:
                label = [int(l.item()) for l in label.view(-1)]

        if isinstance(label, numbers.Integral):
            # Single label for all (selected) classifiers
            labels_per_clf = {idx: int(label) for idx in classifier_indices}
        elif isinstance(label, (list, tuple)):
            # Per-classifier labels
            flat_labels = [
                int(l.item()) if isinstance(l, torch.Tensor) else int(l) for l in label
            ]
            if len(flat_labels) == len(classifier_indices):
                # One label per selected classifier index
                labels_per_clf = {
                    idx: flat_labels[i] for i, idx in enumerate(classifier_indices)
                }
            elif len(flat_labels) == num_clf and classifier_indices == list(
                range(num_clf)
            ):
                # One label per classifier (full list, no subset)
                labels_per_clf = {i: flat_labels[i] for i in range(num_clf)}
            else:
                raise ValueError(
                    "Label list length must match either the number of selected "
                    "classifier_indices or the total number of classifiers."
                )
        else:
            raise TypeError(
                f"Unsupported label type: {type(label)}. "
                "Expected int, tensor, or list/tuple of ints/tensors."
            )

        grads = []
        x_ = x.detach().requires_grad_(True)

        with torch.enable_grad():
            # Order grads to match classifier_indices: grads[i] <-> classifier_indices[i]
            for idx in classifier_indices:
                if idx >= len(clf_items) or idx < 0:
                    raise ValueError(f"Invalid classifier index {idx}. Valid range: [0, {len(clf_items)-1}]")
                
                clf_name, classifier = clf_items[idx]
                lbl = labels_per_clf[idx]
                
                logits = classifier(x_)
                log_probs = F.log_softmax(logits, dim=-1)

                if lbl < 0 or lbl >= log_probs.shape[-1]:
                    raise ValueError(f"Invalid label {lbl} for classifier {clf_name}")

                # We want the gradient of the log-probability of the target class
                # Summing allows computing gradients for the whole batch at once
                target_log_prob = log_probs[:, lbl].sum()

                grad = torch.autograd.grad(target_log_prob, x_, retain_graph=True)[0]
                grads.append(grad)

        return grads

    def compute_direct_conflict_score(
        self,
        x: torch.Tensor,
        label,
        classifier_indices: list[int] | None = None,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        grads = self.get_classifier_gradients(x, label=label, classifier_indices=classifier_indices)
        return self.compute_conflict_score(grads, epsilon=epsilon)

    def compute_conflict_score(
        self,
        grads: list[torch.Tensor],
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the average pairwise conflict score (paper Eq. 15):

            w_raw = (1 / (G*(G-1))) * sum_{j<k} (1 - cos φ_jk)
                  = (1 - cos_φ) / 2,   in [0, 1]

        where cos_φ = (2 / (G*(G-1))) * sum_{j<k} cos(φ_jk) is the average
        pairwise cosine.

        Range [0, 1]: 0 = perfectly aligned, 0.5 = perpendicular, 1 = anti-parallel.
        The conflict gate uses conflict_threshold=0.495 (≈ perpendicular) on this scale.
        If either gradient in a pair has norm below zero_gradient_threshold_direct,
        that pair contributes 0 (no conflict).

        Args:
            grads:   list of G tensors, each of shape (B, D) or (D,).
            epsilon: small constant for numerical stability in unit-vector normalisation.

        Returns:
            Tensor of shape (B,) — per-point conflict score w_raw in [0, 1].
        """
        if not grads:
            raise ValueError("No gradients provided.")

        first = grads[0]
        B      = 1 if first.dim() == 1 else first.shape[0]
        device = first.device

        if len(grads) < 2:
            return torch.zeros(B, device=device)

        pairwise_scores = []
        zero_thr = self.zero_gradient_threshold_direct

        for i in range(len(grads)):
            gi = grads[i]
            gi = gi.view(1, -1).expand(B, -1) if gi.dim() == 1 else gi.view(B, -1)
            norm_i = gi.norm(dim=-1, keepdim=True)          # (B, 1)
            gi_unit = gi / (norm_i + epsilon)

            for j in range(i + 1, len(grads)):
                gj = grads[j]
                gj = gj.view(1, -1).expand(B, -1) if gj.dim() == 1 else gj.view(B, -1)
                norm_j = gj.norm(dim=-1, keepdim=True)      # (B, 1)
                gj_unit = gj / (norm_j + epsilon)

                cos      = (gi_unit * gj_unit).sum(dim=-1)  # (B,)
                conflict = (1.0 - cos) / 2.0               # (B,)  in [0, 1]

                near_zero = (norm_i.squeeze(-1) < zero_thr) | (norm_j.squeeze(-1) < zero_thr)
                conflict  = torch.where(near_zero, torch.zeros_like(conflict), conflict)
                pairwise_scores.append(conflict)

        return torch.stack(pairwise_scores, dim=0).mean(dim=0)  # (B,)  in [0, 1]

    def __name__(self):
        return "ClusterDistribution"