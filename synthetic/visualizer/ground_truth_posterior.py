import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

GuidanceComponent = Tuple[str, int, float]


class GroundTruthPosteriorVisualizer:
    """
    Visualize ground-truth posteriors p(x₁) · exp(-J(x₁)).

    The workflow is a simple rejection sampler:
        1. Draw candidate samples from the target mixture p(x₁) (ClusterDistribution).
        2. Evaluate the total energy J(x₁) induced by any set of classifier guidance signals.
        3. Accept each candidate with probability proportional to exp(-J(x₁)) to obtain
           samples from the reweighted posterior p(x₁) · exp(-J(x₁)).
        4. Plot the accepted samples to show the "ground-truth" posterior mass each
           guided field aims to track.
    This class wraps all steps in a reusable API for the main experiment script.
    """

    def __init__(
        self,
        cluster_distribution,
        device: str = "cpu",
        chunk_size: int = 4096,
    ):
        self.cluster_distribution = cluster_distribution
        self.device = torch.device(device)
        self.chunk_size = chunk_size

    def _compute_energy(
        self,
        samples: torch.Tensor,
        components: Sequence[GuidanceComponent],
    ) -> torch.Tensor:
        """
        Evaluate ∑ λ_i J_i(x) for a batch of samples.

        Each `components` entry is (classifier_name, label, scale). The ClusterDistribution
        exposes `get_J(..., classifier=..., label=...)` which internally maps to the
        pretrained classifier’s negative log-likelihood for that label. Scaling factors
        allow different guidance strengths per classifier.
        """
        if not components:
            return torch.zeros(samples.size(0), device=samples.device)

        total = torch.zeros(samples.size(0), device=samples.device)
        for classifier_name, label, scale in components:
            energy = self.cluster_distribution.get_J(
                samples, classifier=classifier_name, label=label
            )
            total = total + float(scale) * energy.to(samples.device)
        return total

    def _draw_candidates(self, batch_size: int) -> torch.Tensor:
        candidates = self.cluster_distribution.sample(
            batch_size=batch_size, device=self.device
        )
        return candidates.to(self.device)

    @torch.no_grad()
    def sample_posterior(
        self,
        components: Sequence[GuidanceComponent],
        num_samples: int,
        max_attempts: int = 10000,
    ) -> torch.Tensor:
        """
        Sample from the reweighted target p(x₁) · exp(-∑ λ_i J_i(x₁)) via rejection sampling.

        Algorithm:
            • Draw `chunk_size` candidates from the base ClusterDistribution.
            • Compute their total energy from `_compute_energy`.
            • Convert energies to normalized weights w ∝ exp(-J).
            • Accept each candidate with probability w / max(w); repeat until we collect
              `num_samples` accepted points or hit `max_attempts`.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if not components:
            samples = self.cluster_distribution.sample(
                batch_size=num_samples, device=self.device
            )
            return samples[:num_samples].cpu()

        accepted: List[torch.Tensor] = []
        attempts = 0
        while sum(chunk.size(0) for chunk in accepted) < num_samples:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Ground-truth posterior sampling did not converge. "
                    "Please reduce guidance strength or increase chunk_size."
                )
            attempts += 1

            candidates = self._draw_candidates(self.chunk_size)
            energies = self._compute_energy(candidates, components)
            weights = torch.exp(-energies)
            probs = weights / weights.max().clamp_min(1e-12)
            mask = torch.rand(candidates.size(0), device=self.device) < probs

            if mask.any():
                accepted.append(candidates[mask])

        samples = torch.cat(accepted, dim=0)[:num_samples]
        return samples.cpu()

    def visualize(
        self,
        guidance_specs: Dict[str, Sequence[GuidanceComponent]],
        save_dir: str,
        num_samples: int = 2000,
        scatter_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Generate scatter plots for each guided field's ground-truth posterior.

        Args:
            guidance_specs: mapping from field name to tuples describing which classifiers
                and labels contribute to its energy (see `_compute_energy`).
            save_dir: directory where plots will be written (`{name}_ground_truth.png`).
            num_samples: number of posterior samples to draw per field for visualization.
            scatter_kwargs: optional Matplotlib overrides (point size, color, alpha, …).

        The plots display accepted samples (blue dots by default) and overlay cluster
        means to make it clear how the guidance reshapes mass relative to the original
        mixture.
        """
        os.makedirs(save_dir, exist_ok=True)
        scatter_kwargs = scatter_kwargs or {}

        for name, components in guidance_specs.items():
            try:
                samples = self.sample_posterior(components, num_samples)
            except RuntimeError as exc:
                print(f"[GroundTruthPosterior] Skipped {name}: {exc}")
                continue

            if samples.numel() == 0:
                continue

            fig, ax = plt.subplots(figsize=(6, 6))
            data = samples.numpy()

            H = ax.hist2d(data[:, 0], data[:, 1], 300, range=((-5, 15), (-15, 15)))

            # Normalize colormap based on 99th percentile (matches visualizer.py)
            cmin, cmax = 0.0, torch.quantile(torch.from_numpy(H[0]), 0.99).item()
            norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

            ax.clear()
            ax.hist2d(data[:, 0], data[:, 1], 300, range=((-5, 15), (-15, 15)), norm=norm)

            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            ax.set_aspect('equal')

            ax.set_xlim(-5, 15)
            ax.set_ylim(-15, 15)
            ax.set_xlabel('x', color='black', fontsize=12)
            ax.set_ylabel('y', color='black', fontsize=12)
            ax.tick_params(colors='black', labelsize=10)
            ax.set_title(f"{name}: ground-truth posterior", color='black', fontsize=12, pad=10)

            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)

            save_path = os.path.join(save_dir, f"{name}_ground_truth.png")
            fig.tight_layout()
            fig.savefig(save_path, dpi=200, facecolor='white')
            plt.close(fig)
            print(f"[GroundTruthPosterior] Saved {save_path}")

