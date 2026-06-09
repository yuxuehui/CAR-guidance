"""Visualization utilities for synthetic experiments."""

import math
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.distributions as dist

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
try:
    from .energy import EnergyVisualizer, _smootherstep
except ImportError:  # pragma: no cover - fallback when package context missing
    from energy import EnergyVisualizer, _smootherstep

def _unit_ball_volume(d: int) -> float:
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)

def compute_mmd_rbf(X, Y, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) using RBF kernel.
    
    MMD²(P, Q) = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    where k is RBF kernel: k(x, y) = exp(-||x - y||² / (2σ²))
    
    Args:
        X: Samples from distribution P, shape (N, D)
        Y: Samples from distribution Q, shape (M, D)
        sigma: RBF kernel bandwidth (default: 1.0)
    
    Returns:
        MMD² value (scalar)
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    
    # Compute pairwise distances
    XX = torch.cdist(X, X) ** 2  # (N, N)
    YY = torch.cdist(Y, Y) ** 2  # (M, M)
    XY = torch.cdist(X, Y) ** 2  # (N, M)
    
    # RBF kernel: k(x, y) = exp(-||x - y||² / (2σ²))
    gamma = 1.0 / (2 * sigma ** 2)
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    
    # MMD² = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
    mmd_squared = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd_squared.item()

def compute_mode_coverage(samples, centers, stds, threshold=2.0):
    """
    Compute mode coverage: fraction of samples within threshold of each cluster center.
    
    Args:
        samples: Generated samples, shape (N, D)
        centers: Cluster centers, shape (K, D)
        stds: Standard deviations for each cluster, shape (K, D)
        threshold: Mahalanobis distance threshold (default: 2.0 for ~95% coverage)
    
    Returns:
        Dictionary with:
            - overall_coverage: fraction of samples within threshold of any center
            - per_mode_coverage: list of fractions for each mode
            - per_mode_counts: list of sample counts for each mode
    """
    samples = torch.as_tensor(samples, dtype=torch.float32)
    centers = torch.as_tensor(centers, dtype=torch.float32)
    stds = torch.as_tensor(stds, dtype=torch.float32)
    
    # Compute Mahalanobis distance from each sample to each center
    samples_expanded = samples.unsqueeze(1)  # (N, 1, D)
    centers_expanded = centers.unsqueeze(0)  # (1, K, D)
    stds_expanded = stds.unsqueeze(0)  # (1, K, D)
    
    normalized_diff = (samples_expanded - centers_expanded) / stds_expanded  # (N, K, D)
    mahalanobis_dist = torch.sqrt(normalized_diff.pow(2).sum(dim=-1))  # (N, K)
    
    # Find minimum distance to any center
    min_dist, closest_mode = mahalanobis_dist.min(dim=1)  # (N,)
    
    # Overall coverage: fraction within threshold
    overall_coverage = (min_dist <= threshold).float().mean().item()
    
    # Per-mode coverage
    K = centers.shape[0]
    per_mode_coverage = []
    per_mode_counts = []
    for k in range(K):
        # Samples assigned to mode k (closest to center k)
        mode_mask = (closest_mode == k)
        mode_samples = samples[mode_mask]
        
        if mode_samples.shape[0] > 0:
            # Distance from mode samples to their assigned center
            mode_dist = mahalanobis_dist[mode_mask, k]
            mode_coverage = (mode_dist <= threshold).float().mean().item()
        else:
            mode_coverage = 0.0
        
        per_mode_coverage.append(mode_coverage)
        per_mode_counts.append(mode_mask.sum().item())
    
    return {
        'overall_coverage': overall_coverage,
        'per_mode_coverage': per_mode_coverage,
        'per_mode_counts': per_mode_counts,
    }

def compute_diversity_pairwise(samples):
    """
    Compute diversity using pairwise distance metric.
    
    Returns mean Euclidean distance over all unique unordered pairs.
    """
    samples = torch.as_tensor(samples, dtype=torch.float32)
    dists = torch.cdist(samples, samples)  # (N, N)
    diversity = dists.triu(1).mean().item()  # Mean over upper triangle
    return diversity

def compute_gt_posterior_match(generated_samples, centers, stds, threshold=2.0):
    """
    Compute ground truth posterior fidelity using geometric centers (consistent with prior preservation).
    
    Uses Mahalanobis distance to check if generated samples fall within the valid modes
    of the ground truth posterior. This avoids the pitfalls of:
    1. Global std (inflated in multi-modal data)
    2. O(N*M) memory explosion
    3. Sparse GT sampling artifacts
    
    Args:
        generated_samples: Generated samples, shape (N, D)
        centers: Cluster centers from the distribution, shape (K, D)
        stds: Standard deviations for each cluster, shape (K, D)
        threshold: Mahalanobis distance threshold (default: 2.0 for ~95% coverage)
    
    Returns:
        Fraction of generated samples within threshold of any cluster center
    """
    generated_samples = torch.as_tensor(generated_samples, dtype=torch.float32)
    centers = torch.as_tensor(centers, dtype=torch.float32)
    stds = torch.as_tensor(stds, dtype=torch.float32)
    
    # Compute Mahalanobis distance from each sample to each cluster center
    samples_expanded = generated_samples.unsqueeze(1)  # (N, 1, D)
    centers_expanded = centers.unsqueeze(0)  # (1, K, D)
    stds_expanded = stds.unsqueeze(0)  # (1, K, D)
    
    # Normalized distance: (x - μ) / σ
    normalized_diff = (samples_expanded - centers_expanded) / stds_expanded  # (N, K, D)
    mahalanobis_dist = torch.sqrt(normalized_diff.pow(2).sum(dim=-1))  # (N, K)
    
    # Find minimum distance to any cluster center
    min_dist, _ = mahalanobis_dist.min(dim=1)  # (N,)
    
    # Consider sample "matching posterior" if within threshold of any center
    is_in_posterior = (min_dist <= threshold).float()
    match_ratio = is_in_posterior.mean().item()
    
    return match_ratio

@torch.no_grad()
def knn_entropy_torch_chunked(X, k=3, add_log2=False, jitter=0.0, device=None, query_chunk=2048, db_chunk=8192, max_points=None, same_set=True, eps=1e-12):
    """Kozachenko-Leonenko kNN entropy estimation with chunking."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)
    if device is not None:
        X = X.to(device)
    n, d = X.shape
    if n <= k:
        raise ValueError(f"Need n>k, got n={n}, k={k}")

    if max_points is not None and n > max_points:
        X = X[torch.randperm(n, device=device)[:max_points]]
        n = X.shape[0]

    if jitter > 0.0:
        X = X + torch.randn_like(X) * float(jitter)

    cd = _unit_ball_volume(d)
    kth_dists = torch.full((n, k), float("inf"), device=device, dtype=X.dtype)
    all_idx = torch.arange(n, device=device)

    for q_start in range(0, n, query_chunk):
        q_end = min(q_start + query_chunk, n)
        Q = X[q_start:q_end]
        Q_idx = all_idx[q_start:q_end]
        best = torch.full((Q.shape[0], k), float("inf"), device=device, dtype=X.dtype)

        for d_start in range(0, n, db_chunk):
            d_end = min(d_start + db_chunk, n)
            D = X[d_start:d_end]
            D_idx = all_idx[d_start:d_end]

            Dmat = torch.cdist(Q, D, p=2)

            if same_set:
                mask = Q_idx[:, None] == D_idx[None, :]
                if mask.any():
                    Dmat = Dmat.masked_fill(mask, float("inf"))

            new_topk_vals, _ = torch.topk(Dmat, k, dim=1, largest=False)
            merged = torch.cat([best, new_topk_vals], dim=1)
            best, _ = torch.topk(merged, k, dim=1, largest=False)

        kth_dists[q_start:q_end] = best

    rho_k = kth_dists.max(dim=1).values.clamp_min(float(eps))
    H = (torch.digamma(torch.tensor(float(n), device=device)) -
         torch.digamma(torch.tensor(float(k), device=device)) +
         math.log(cd) + (d / n) * torch.sum(torch.log(rho_k))).item()

    if add_log2:
        H += d * math.log(2.0)
    return H

def normalized_tightness_torch(X, k=3, R_null=3, tiny_scale=1e-3, add_log2=False, jitter=0.0,
                             seed=0, method="minmax", tau=0.5, eps=1e-12, device=None,
                             query_chunk=2048, db_chunk=8192, max_points=None, run_on="cpu"):
    """Compute normalized tightness score."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32)
    if device is None:
        device = torch.device(run_on)
    X = X.to(device)

    H_data = knn_entropy_torch_chunked(X, k=k, add_log2=add_log2, jitter=jitter, device=device,
                                     query_chunk=query_chunk, db_chunk=db_chunk, max_points=max_points)

    g = torch.Generator(device=device).manual_seed(seed)
    Xc = X - X.mean(dim=0, keepdim=True)
    avg_var = (Xc.square().sum(dim=0) / max(1, (X.shape[0] - 1))).mean().item()
    std = math.sqrt(max(avg_var, eps))

    H_null_list = []
    for _ in range(R_null):
        Z = torch.randn(X.shape, device=device, dtype=X.dtype, generator=g) * std
        H_null_list.append(knn_entropy_torch_chunked(Z, k=k, add_log2=add_log2, jitter=jitter,
                                                   device=device, query_chunk=query_chunk,
                                                   db_chunk=db_chunk, max_points=max_points))
    H_null = float(sum(H_null_list) / len(H_null_list))

    X_small = (X - X.mean(dim=0, keepdim=True)) * float(tiny_scale)
    H_min = knn_entropy_torch_chunked(X_small, k=k, add_log2=add_log2, jitter=jitter,
                                    device=device, query_chunk=query_chunk,
                                    db_chunk=db_chunk, max_points=max_points)

    if method == "minmax":
        denom = max(H_null - H_min, eps)
        score = (H_null - H_data) / denom
        score = float(max(0.0, min(1.0, score)))
    elif method == "sigmoid":
        score = 1.0 / (1.0 + math.exp(-(H_null - H_data) / max(tau, eps)))
    else:
        raise ValueError("method must be 'minmax' or 'sigmoid'")

    return float(score), float(H_data), float(H_null), float(H_min)

class Visualizer:
    """Handles all visualization logic."""
    def __init__(self, device, cfg, classifier_1=None, classifier_2=None, multi_guided_cls=None):
        self.device = device
        self.cfg = cfg
        self.classifier_1 = classifier_1
        self.classifier_2 = classifier_2
        self.multi_guided_cls = multi_guided_cls
        self.setup_plot_params()
        self.energy = EnergyVisualizer(device, cfg, lambda: self.num_points)

    def setup_plot_params(self):
        """Initialize visualization parameters."""
        self.num_points = 20
        self.t_steps = [0.0, 0.22, 0.44, 0.67, 0.89, 1.0]
        self.padding = 2.0

    @staticmethod
    def _is_guidance_matching_like(guided_field) -> bool:
        """Return True when guidance behaves like guidance-matching (single combined field)."""
        if guided_field is None:
            return False
        identifier = getattr(guided_field, "guidance_identifier", None)
        if isinstance(identifier, str):
            identifier = identifier.lower()
        return identifier in ("guidance_matching", "car_guidance")

    def _build_single_guided_field(self, vf, classifier, target, scale):
        if self.multi_guided_cls is None:
            raise ValueError(
                "multi_guided_cls must be provided to Visualizer when multi-guided "
                "field visualization is enabled."
            )
        return self.multi_guided_cls(
            vf,
            [classifier],
            [target],
            [scale],
            self.cfg,
        )
        
    def prepare_grid(self, p0, p1):
        """Prepare visualization grid."""
        all_points = torch.cat((p0, p1), dim=0)
        x_min, x_max = all_points[:, 0].min().item(), all_points[:, 0].max().item()
        y_min, y_max = all_points[:, 1].min().item(), all_points[:, 1].max().item()
        
        x_min -= self.padding
        x_max += self.padding
        y_min -= self.padding
        y_max += self.padding
        
        x_grid = torch.linspace(x_min, x_max, self.num_points).to(self.device)
        y_grid = torch.linspace(y_min, y_max, self.num_points).to(self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        positions = torch.stack([X_mesh.ravel(), Y_mesh.ravel()], dim=1)
        
        return positions, X_mesh, Y_mesh, (x_min, x_max, y_min, y_max)


    def plot_vector_field(
        self,
        ax,
        field_data,
        X_mesh,
        Y_mesh,
        p0,
        p1,
        y_orig,
        title=None,
        x_lim=None,
        y_lim=None,
        guidance_colors=None,
        trajectories=None,
        t_val=None,
    ):
        """Plot flow, guidance, and total vector fields.
        x_0: #9D9EA3 (light gray)
        x_t: #9B5C97 (purple)
        Classifier 1: #C03830 (red)
        Classifier 2: #317EC2 (blue)
        Total Field: #9B5C97 (purple)
        Args:
            trajectories: Optional trajectory data (num_traj, num_steps, dim) to plot x_0 and x_t
            t_val: Current time value to extract x_t from trajectories
        """

        clf1_color = '#C03830'  # Classifier 1
        clf2_color = '#317EC2'  # Classifier 2

        # x_0: starting points (gray)
        x_0 = trajectories[:, 0, :]  # (num_traj, dim)
        ax.scatter(x_0[:, 0], x_0[:, 1], color='#9D9EA3', s=15, alpha=0.5, label='$x_0$', zorder=2)

        # x_t: closest trajectory step to t_val (trajectories run over linspace(0, 1, ...))
        num_steps = trajectories.shape[1]
        t_idx = int(t_val * (num_steps - 1))
        t_idx = max(0, min(t_idx, num_steps - 1))
        x_t = trajectories[:, t_idx, :]  # (num_traj, dim)
        ax.scatter(x_t[:, 0], x_t[:, 1], color='#9B5C97', s=15, alpha=0.5, label='$x_t$', zorder=2)

        # Flow-only vectors
        ax.quiver(X_mesh.cpu(), Y_mesh.cpu(),
                 field_data['flow'][:, 0], field_data['flow'][:, 1],
                 color='#E29135', alpha=0.7, label='Base Velocity', minlength=0.1)
        
        if 'guidance' in field_data:
            if guidance_colors is None:
                guidance_colors = [clf1_color]
            
            if isinstance(field_data['guidance'], list):
                for idx, guidance in enumerate(field_data['guidance']):
                    ax.quiver(X_mesh.cpu(), Y_mesh.cpu(),
                            guidance[:, 0], guidance[:, 1],
                            color=guidance_colors[idx], alpha=0.7,
                            label=f'Guidance {idx+1}', minlength=0.1)
            else:
                ax.quiver(X_mesh.cpu(), Y_mesh.cpu(),
                         field_data['guidance'][:, 0], field_data['guidance'][:, 1],
                         color=guidance_colors[0], alpha=0.7,
                         label='Guidance', minlength=0.1)
            
            ax.quiver(X_mesh.cpu(), Y_mesh.cpu(),
                     field_data['total'][:, 0], field_data['total'][:, 1],
                     color='#9B5C97', alpha=0.7, label='Total Field', minlength=0.1)

        if title:
            ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal')
        if x_lim: ax.set_xlim(x_lim)
        if y_lim: ax.set_ylim(y_lim)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', framealpha=0.9)

    def visualize_sampling_process(self, solvers, x_init, T, titles, filename, step_size=0.05, 
                                   distribution=None, ground_truth_visualizer=None, guidance_components_list=None):
        """Visualize data points evolution during sampling.

        Note: x_init is sampled from the base Gaussian prior.

        Args:
            distribution: Optional ClusterDistribution instance. If provided, computes prior preservation
                         metrics (percentage of samples within target distribution).
            ground_truth_visualizer: Optional GroundTruthPosteriorVisualizer instance. If provided along
                                    with guidance_components_list, computes mode coverage and diversity metrics
                                    by comparing generated samples with ground truth posterior.
            guidance_components_list: Optional list of guidance components (one per solver). Each element is
                                     a list of tuples (classifier_name, label, scale) for that solver's guidance.
        """
        import time

        num_samples = x_init.shape[0]
        num_solvers = len(solvers)
        total_samples_to_generate = num_samples * num_solvers
        
        solutions = []
        inference_time_list = []
        solver_training_times = []
        prior_preservation_ratios = []

        mmd_scores = []
        mode_coverage_metrics = []
        diversity_scores = []
        gt_diversity_scores = []
        classifier_accuracies = []
        gt_posterior_match_ratios = []

        for i, solver in enumerate(solvers):
            solver_start = time.time()
            sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', 
                              step_size=step_size, return_intermediates=True)
            solver_end = time.time()
            solver_inference_time = solver_end - solver_start
            inference_time_list.append(solver_inference_time)
            solutions.append(sol.cpu().numpy())

            solver_training_time = 0.0
            if hasattr(solver, 'velocity_model') and hasattr(solver.velocity_model, 'model'):
                field = solver.velocity_model.model
                if hasattr(field, 'training_time') and field.training_time is not None:
                    solver_training_time = field.training_time
            solver_training_times.append(solver_training_time)

            prior_preservation_ratio = None
            if distribution is not None and hasattr(distribution, 'prob'):
                final_samples = sol[-1]  # (num_samples, 2)

                with torch.no_grad():
                    probs = distribution.prob(final_samples)  # (num_samples,)

                # Check if sample is within 2σ of any cluster center via Mahalanobis distance
                centers = distribution.centers.to(final_samples.device)  # (K, 2)
                stds = distribution.std.to(final_samples.device)  # (K, 2)

                # dist[i, k] = sqrt(sum(((x[i] - centers[k]) / stds[k])^2))
                samples_expanded = final_samples.unsqueeze(1)  # (num_samples, 1, 2)
                centers_expanded = centers.unsqueeze(0)  # (1, K, 2)
                stds_expanded = stds.unsqueeze(0)  # (1, K, 2)

                normalized_diff = (samples_expanded - centers_expanded) / stds_expanded  # (num_samples, K, 2)
                mahalanobis_dist = torch.sqrt(normalized_diff.pow(2).sum(dim=-1))  # (num_samples, K)

                min_dist, _ = mahalanobis_dist.min(dim=1)  # (num_samples,)

                # 2σ covers ~95% of Gaussian mass, 3σ covers ~99.7%
                threshold = 2.0
                in_distribution = (min_dist <= threshold).float()  # (num_samples,)
                prior_preservation_ratio = in_distribution.mean().item()
                prior_preservation_ratios.append(prior_preservation_ratio)
                
                print(f"[visualize_sampling_process] Solver {i+1}/{num_solvers} ({titles[i] if i < len(titles) else 'unknown'}): "
                      f"Generated {num_samples} samples in {solver_inference_time:.2f}s "
                      f"({num_samples/solver_inference_time:.0f} samples/sec) | "
                      f"Prior preservation: {prior_preservation_ratio*100:.1f}% ({in_distribution.sum().item()}/{num_samples} samples within 2σ)")
            else:
                print(f"[visualize_sampling_process] Solver {i+1}/{num_solvers} ({titles[i] if i < len(titles) else 'unknown'}): "
                      f"Generated {num_samples} samples in {solver_inference_time:.2f}s "
                      f"({num_samples/solver_inference_time:.0f} samples/sec)")
            
            # Average probability of the classifier predicting the target label
            classifier_accuracy = None
            if guidance_components_list is not None and i < len(guidance_components_list):
                components = guidance_components_list[i]
                if components:
                    final_samples = sol[-1]  # (num_samples, 2)

                    classifier_map = {}
                    if self.classifier_1 is not None:
                        clf1_name = getattr(self.classifier_1, 'guidance_name', 'Classifier_0')
                        classifier_map[clf1_name] = self.classifier_1
                    if self.classifier_2 is not None:
                        clf2_name = getattr(self.classifier_2, 'guidance_name', 'Classifier_1')
                        classifier_map[clf2_name] = self.classifier_2

                    total_prob = 0.0
                    num_components = 0

                    with torch.no_grad():
                        for clf_name, target_label, scale in components:
                            classifier = classifier_map.get(clf_name)
                            if classifier is not None:
                                samples_tensor = torch.as_tensor(final_samples, dtype=torch.float32).to(self.device)
                                logits = classifier(samples_tensor)  # (num_samples, num_classes)
                                probs = torch.softmax(logits, dim=-1)  # (num_samples, num_classes)

                                target_probs = probs[:, int(target_label)]  # (num_samples,)
                                avg_prob = target_probs.mean().item()
                                
                                total_prob += avg_prob
                                num_components += 1
                    
                    if num_components > 0:
                        classifier_accuracy = total_prob / num_components
                        classifier_accuracies.append(classifier_accuracy)
                        print(f"  [Classifier Accuracy] Average target label probability: {classifier_accuracy*100:.2f}%")
                    else:
                        classifier_accuracies.append(None)
                else:
                    classifier_accuracies.append(None)
            else:
                classifier_accuracies.append(None)
            
            # Compare with ground truth posterior if available
            if ground_truth_visualizer is not None and guidance_components_list is not None and i < len(guidance_components_list):
                final_samples = sol[-1]  # (num_samples, 2)
                components = guidance_components_list[i]

                if not components:
                    print(f"  [Ground Truth Comparison] Skipped for solver {i+1}/{num_solvers} ({titles[i] if i < len(titles) else 'unknown'}): "
                          f"components list is empty")
                    mmd_scores.append(None)
                    diversity_scores.append(None)
                    gt_diversity_scores.append(None)
                    gt_posterior_match_ratios.append(None)
                    if distribution is not None:
                        mode_coverage_metrics.append(None)
                    continue
                
                try:
                    gt_samples = ground_truth_visualizer.sample_posterior(
                        components, num_samples=num_samples
                    )  # (num_samples, 2)

                    # Both on CPU for metric computation to avoid device mismatch
                    final_samples_t = torch.as_tensor(final_samples, dtype=torch.float32).cpu()
                    gt_samples_t = torch.as_tensor(gt_samples, dtype=torch.float32).cpu()

                    mmd = compute_mmd_rbf(final_samples_t, gt_samples_t, sigma=1.0)
                    mmd_scores.append(mmd)

                    gen_diversity = compute_diversity_pairwise(final_samples_t)
                    gt_diversity = compute_diversity_pairwise(gt_samples_t)
                    diversity_scores.append(gen_diversity)
                    gt_diversity_scores.append(gt_diversity)

                    if distribution is not None:
                        centers = distribution.centers.cpu()  # (K, 2)
                        stds = distribution.std.cpu()  # (K, 2)

                        gt_posterior_match_ratio = compute_gt_posterior_match(final_samples_t, centers, stds, threshold=2.0)
                        gt_posterior_match_ratios.append(gt_posterior_match_ratio)

                        coverage_metrics = compute_mode_coverage(final_samples_t, centers, stds, threshold=2.0)
                        mode_coverage_metrics.append(coverage_metrics)
                        
                        print(f"  [Ground Truth Comparison] MMD: {mmd:.6f} | "
                              f"Diversity (gen/gt): {gen_diversity:.2f}/{gt_diversity:.2f} | "
                              f"GT Match: {gt_posterior_match_ratio*100:.1f}% | "
                              f"Mode coverage: {coverage_metrics['overall_coverage']*100:.1f}% | "
                              f"Per-mode: {[f'{c*100:.1f}%' for c in coverage_metrics['per_mode_coverage']]} | "
                              f"Counts: {coverage_metrics['per_mode_counts']}")
                    else:
                        gt_posterior_match_ratios.append(None)
                        print(f"  [Ground Truth Comparison] MMD: {mmd:.6f} | "
                              f"Diversity (gen/gt): {gen_diversity:.2f}/{gt_diversity:.2f}")
                except Exception as e:
                    print(f"  [Ground Truth Comparison] Failed: {e}")
                    mmd_scores.append(None)
                    diversity_scores.append(None)
                    gt_diversity_scores.append(None)
                    gt_posterior_match_ratios.append(None)
                    if distribution is not None:
                        mode_coverage_metrics.append(None)
            else:
                # Append None so list lengths stay aligned with num_solvers
                reason = []
                if ground_truth_visualizer is None:
                    reason.append("ground_truth_visualizer is None")
                if guidance_components_list is None:
                    reason.append("guidance_components_list is None")
                elif i >= len(guidance_components_list):
                    reason.append(f"solver index {i} >= guidance_components_list length {len(guidance_components_list)}")
                
                if reason:
                    print(f"  [Ground Truth Comparison] Skipped for solver {i+1}/{num_solvers} ({titles[i] if i < len(titles) else 'unknown'}): "
                          f"{'; '.join(reason)}")
                
                mmd_scores.append(None)
                diversity_scores.append(None)
                gt_diversity_scores.append(None)
                gt_posterior_match_ratios.append(None)
                if distribution is not None:
                    mode_coverage_metrics.append(None)
        
        print(f"\n[visualize_sampling_process] Summary Statistics:")
        print(f"  - Total samples generated: {total_samples_to_generate} ({num_samples} samples × {num_solvers} solvers)")

        print(f"  - Per-solver time and throughput:")
        for i, (solver_inference_time, training_time) in enumerate(zip(inference_time_list, solver_training_times)):
            solver_title = titles[i] if i < len(titles) else f"Solver {i+1}"
            total_time_per_solver = training_time + solver_inference_time
            time_per_sample = total_time_per_solver / num_samples
            throughput = num_samples / total_time_per_solver if total_time_per_solver > 0 else 0.0
            
            if training_time > 0:
                print(f"    * {solver_title}: training={training_time:.2f}s, inference={solver_inference_time:.2f}s, "
                      f"total={total_time_per_solver:.2f}s, time_per_sample={time_per_sample:.6f}s, "
                      f"throughput={throughput:.0f} samples/sec")
            else:
                print(f"    * {solver_title}: inference={solver_inference_time:.2f}s, "
                      f"time_per_sample={time_per_sample:.6f}s, throughput={throughput:.0f} samples/sec")
        
        if prior_preservation_ratios:
            avg_prior_preservation = sum(prior_preservation_ratios) / len(prior_preservation_ratios)
            print(f"  - Prior preservation (within 2σ): {avg_prior_preservation*100:.1f}% average")
            for i, ratio in enumerate(prior_preservation_ratios):
                solver_title = titles[i] if i < len(titles) else f"Solver {i+1}"
                print(f"    * {solver_title}: {ratio*100:.1f}%")
        
        if classifier_accuracies and any(acc is not None for acc in classifier_accuracies):
            valid_accs = [acc for acc in classifier_accuracies if acc is not None]
            if valid_accs:
                avg_classifier_acc = sum(valid_accs) / len(valid_accs)
                print(f"  - Classifier accuracy (avg target label probability): {avg_classifier_acc*100:.2f}% average")
                for i, acc in enumerate(classifier_accuracies):
                    if acc is not None:
                        solver_title = titles[i] if i < len(titles) else f"Solver {i+1}"
                        print(f"    * {solver_title}: {acc*100:.2f}%")
        
        if mmd_scores and any(m is not None for m in mmd_scores):
            valid_mmd = [m for m in mmd_scores if m is not None]
            if valid_mmd:
                avg_mmd = sum(valid_mmd) / len(valid_mmd)
                print(f"\n[Ground Truth Comparison] Summary:")
                print(f"  - Average MMD: {avg_mmd:.6f} (lower is better, measures distribution distance)")
                for i, mmd in enumerate(mmd_scores):
                    if mmd is not None:
                        solver_title = titles[i] if i < len(titles) else f"Solver {i+1}"
                        print(f"    * {solver_title}: MMD={mmd:.6f}")
                
                if diversity_scores and gt_diversity_scores:
                    valid_gen_div = [d for d in diversity_scores if d is not None]
                    valid_gt_div = [d for d in gt_diversity_scores if d is not None]
                    if valid_gen_div and valid_gt_div:
                        avg_gen_div = sum(valid_gen_div) / len(valid_gen_div)
                        avg_gt_div = sum(valid_gt_div) / len(valid_gt_div)
                        print(f"  - Average Diversity (generated): {avg_gen_div:.2f}")
                        print(f"  - Average Diversity (ground truth): {avg_gt_div:.2f}")
                        print(f"  - Diversity ratio (gen/gt): {avg_gen_div/avg_gt_div:.2f} (closer to 1.0 is better)")
                
                if gt_posterior_match_ratios and any(r is not None for r in gt_posterior_match_ratios):
                    valid_gt_match = [r for r in gt_posterior_match_ratios if r is not None]
                    if valid_gt_match:
                        avg_gt_match = sum(valid_gt_match) / len(valid_gt_match)
                        print(f"  - Average GT Posterior Match: {avg_gt_match*100:.1f}% (samples within 2σ of GT)")
                        for i, match_ratio in enumerate(gt_posterior_match_ratios):
                            if match_ratio is not None:
                                solver_title = titles[i] if i < len(titles) else f"Solver {i+1}"
                                print(f"    * {solver_title}: {match_ratio*100:.1f}%")
                
                if mode_coverage_metrics and any(m is not None for m in mode_coverage_metrics):
                    valid_coverage = [m for m in mode_coverage_metrics if m is not None]
                    if valid_coverage:
                        avg_overall = sum(m['overall_coverage'] for m in valid_coverage) / len(valid_coverage)
                        print(f"  - Average Mode Coverage: {avg_overall*100:.1f}%")
                        K = len(valid_coverage[0]['per_mode_coverage'])
                        for k in range(K):
                            avg_mode_cov = sum(m['per_mode_coverage'][k] for m in valid_coverage) / len(valid_coverage)
                            avg_mode_count = sum(m['per_mode_counts'][k] for m in valid_coverage) / len(valid_coverage)
                            print(f"    * Mode {k}: coverage={avg_mode_cov*100:.1f}%, avg_count={avg_mode_count:.0f}")
        else:
            if not mmd_scores:
                print(f"\n[Ground Truth Comparison] Summary: Not available (mmd_scores is empty)")
            elif not any(m is not None for m in mmd_scores):
                print(f"\n[Ground Truth Comparison] Summary: Not available (all MMD scores are None)")
                print(f"  - This may happen if:")
                print(f"    1. ground_truth_visualizer is None")
                print(f"    2. guidance_components_list is None or empty")
                print(f"    3. All ground truth comparisons failed with exceptions")
                print(f"    4. guidance_components_list length < number of solvers")
        
        print(f"  - Output file: {filename}\n")
        
        fig, axs = plt.subplots(len(solutions), 10, figsize=(20, 4*len(solutions)))
        if len(solutions) == 1:
            axs = axs[None, :]

        for row, (sol, title) in enumerate(zip(solutions, titles)):
            for i in range(10):
                H = axs[row,i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,15), (-15,15)))
                cmin, cmax = 0.0, torch.quantile(torch.from_numpy(H[0]), 0.99).item()
                norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
                axs[row,i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,15), (-15,15)), norm=norm)
                axs[row,i].set_aspect('equal')
                axs[row,i].axis('off')
                axs[row,i].set_title(f'{title} t={T.cpu()[i]:.2f}')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)

    def compute_trajectories(self, velocity_model, x_init, time_grid):
        """
        Compute trajectories by integrating the provided velocity model
        along the supplied time grid (explicit Euler for visualization).

        Note: x_init is sampled from the base Gaussian prior.
        """
        with torch.no_grad():
            positions = x_init.to(self.device)
            num_traj, dim = positions.shape
            num_steps = time_grid.shape[0]
            trajectories = torch.empty(
                num_traj, num_steps, dim, device=self.device, dtype=positions.dtype
            )
            trajectories[:, 0] = positions

            for idx in range(1, num_steps):
                t_prev = time_grid[idx - 1]
                dt = float(time_grid[idx] - t_prev)
                t_tensor = torch.full(
                    (num_traj, 1),
                    float(t_prev),
                    device=self.device,
                    dtype=positions.dtype,
                )
                velocities = velocity_model(positions, t_tensor)
                if velocities.dim() > 2:
                    velocities = velocities.mean(dim=1)
                elif velocities.dim() == 1:
                    velocities = velocities.reshape(num_traj, -1)
                positions = positions + velocities * dt
                trajectories[:, idx] = positions

        return trajectories.cpu().numpy()

    def visualize_field_evolution(self, vf, guided_fields, p0, p1, y_orig, prefix, titles=None, batch_size_plot_traj=100):
        """Visualize field evolution over time."""
        if not isinstance(guided_fields, list):
            guided_fields = [guided_fields]
        if titles is None:
            titles = [f"Field {i}" for i in range(len(guided_fields))]
        
        if batch_size_plot_traj > p0.shape[0]:
            indices_plot = torch.randperm(p0.shape[0])[:batch_size_plot_traj]
            x_init_plot = p0[indices_plot]
        else:
            x_init_plot = p0[torch.randperm(p0.shape[0])[:batch_size_plot_traj]]
        x_init_plot = x_init_plot.to(self.device)

        positions, X_mesh, Y_mesh, (x_min, x_max, y_min, y_max) = self.prepare_grid(p0, p1)

        # Two rows per field (vector field + energy landscape) plus 5 diagnostic columns:
        # logits, tightness, weighted_density, source_shift
        fig, axes = plt.subplots(
            len(guided_fields) * 2,
            len(self.t_steps) + 5,
            figsize=(40, 8 * len(guided_fields)),
        )
        if len(guided_fields) == 1:
            axes = axes.reshape(2, -1)
            
        for row, (guided_field, title) in enumerate(zip(guided_fields, titles)):
            velocity_model = guided_field if guided_field else vf
            T_plot = torch.linspace(0, 1, batch_size_plot_traj, device=self.device)
            print(
                "[Visualizer] computing trajectories with "
                f"{velocity_model.__class__.__name__} (guided={guided_field is not None})"
            )
            if guided_field is not None:
                targets = getattr(guided_field, "targets", None)
                classifiers = getattr(guided_field, "classifiers", None)
                signature = getattr(guided_field, "guidance_signature", None)
                cls_names = (
                    [getattr(clf, "guidance_name", clf.__class__.__name__) for clf in classifiers]
                    if classifiers
                    else []
                )
                print(
                    f"[Visualizer] guided_field={guided_field.__class__.__name__}, "
                    f"classifiers={cls_names}, targets={targets}, signature={signature}"
                )
            else:
                print("[Visualizer] guided_field=None (base flow)")
            
            x_init_for_field = x_init_plot.clone()

            trajectories = self.compute_trajectories(
                velocity_model, x_init_for_field, T_plot
            )
            
            for i, t_val in enumerate(self.t_steps):
                current_t = torch.full((positions.shape[0], 1), t_val, device=self.device)
                
                with torch.no_grad():
                    field_data = self.compute_field_data(vf, guided_field, positions, current_t)
                    
                    # Determine colors for classifier-specific guidance components
                    guidance_colors = None
                    if guided_field and hasattr(guided_field, "classifiers"):
                        is_guidance_matching = self._is_guidance_matching_like(guided_field)
                        
                        if is_guidance_matching:
                            guidance_colors = ['#8A2BE2']
                        elif len(guided_field.classifiers) > 1:
                            guidance_colors = []
                            for clf in guided_field.classifiers:
                                if clf is self.classifier_1:
                                    guidance_colors.append('#C03830')
                                else:
                                    guidance_colors.append('#317EC2')
                        else:
                            if guided_field.classifiers[0] is self.classifier_1:
                                guidance_colors = ['#C03830']
                            else:
                                guidance_colors = ['#317EC2']
                    
                ax_vector = axes[row*2, i]
                self.plot_vector_field(
                    ax_vector, field_data, X_mesh, Y_mesh, p0, p1, y_orig,
                    f"{title} t={t_val:.2f}", (-5, 15), (-15, 15),
                    guidance_colors=guidance_colors,
                    trajectories=trajectories,
                    t_val=t_val
                )

                ax_energy = axes[row*2 + 1, i]
                # Reward landscape (higher is better) if configured, else energy (lower is better)
                use_reward = getattr(self.cfg, "use_reward_landscape", False)
                if use_reward:
                    self.plot_reward_landscape(
                        ax_energy, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15
                    )
                else:
                    self.plot_energy_landscape(
                        ax_energy, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15
                    )

                current_t_idx = int(t_val * (len(T_plot) - 1))
                for traj in trajectories:
                    ax_vector.plot(traj[:current_t_idx + 1, 0], traj[:current_t_idx + 1, 1],
                                    color='k', alpha=0.1, linewidth=0.5)

                # Diagnostics for the final time slice
                if i == len(self.t_steps) - 1:
                    # Column -4: Classifier logits
                    ax_logits = axes[row*2, -4]
                    self.plot_classifier_logits(
                        ax_logits, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15, self.classifier_1, self.classifier_2
                    )
                    multi_non_matching = (
                        hasattr(guided_field, 'classifiers')
                        and len(guided_field.classifiers) > 1
                        and not self._is_guidance_matching_like(guided_field)
                    )
                    if multi_non_matching:
                        ax_logits.set_title(f"Combined Likelihood\nClf1({guided_field.targets[0]}) & Clf2({guided_field.targets[1]})")
                    else:
                        clf_type = "1" if guided_field and guided_field.classifiers[0] is self.classifier_1 else "2"
                        target = guided_field.targets[0] if guided_field else 0
                        ax_logits.set_title(f"Classifier {clf_type}\nTarget {target} Likelihood")

                    # Column -3: Tightness score
                    ax_tightness = axes[row*2, -3]
                    self.plot_tightness_score(
                        ax_tightness, positions, current_t, velocity_model, guided_field,
                        -5, 15, -15, 15, self.classifier_1, self.classifier_2
                    )

                    # Column -5 (bottom): Weighted density (base)
                    ax_weighted_density_base = axes[row*2 + 1, -5]
                    self.plot_weighted_density_ori(
                        ax_weighted_density_base, vf, guided_field,
                        -5, 15, -15, 15
                    )

                    # Column -1: Conflict score (if applicable)
                    ax_conflict = axes[row*2, -1]
                    if guided_field and hasattr(guided_field, 'compute_conflict_score'):
                        self.plot_conflict_score(
                            ax_conflict, positions, current_t, vf, guided_field,
                            -5, 15, -15, 15
                        )
                    else:
                        ax_conflict.axis('off')
                        ax_conflict.text(0.5, 0.5, 'N/A',
                                        ha='center', va='center', fontsize=12)

                    # Standalone energy figures for the final time step
                    new_fig = plt.figure(figsize=(10, 12))

                    ax_energy_3d_new = new_fig.add_subplot(211, projection='3d')
                    self.plot_energy_landscape_3d(
                        ax_energy_3d_new, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15
                    )

                    ax_contour_new = new_fig.add_subplot(212)
                    self.plot_energy_contour(
                        ax_contour_new, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15
                    )
                    
                    new_fig.tight_layout()
                    
                    if guided_field and hasattr(guided_field, 'classifiers'):
                        GUIDED_MULTI = len(guided_field.classifiers) > 1
                        if GUIDED_MULTI:
                            title_suffix = f'_guided_c{guided_field.targets[0]}c{guided_field.targets[1]}'
                        else:
                            title_suffix = f'_guided_c{guided_field.targets[0]}'
                    else:
                        title_suffix = '_base'
                    
                    out_path = os.path.join(os.path.dirname(prefix), 
                                            f"energy_landscape_t{t_val:.2f}{title_suffix}.png")
                    new_fig.savefig(out_path, dpi=300, bbox_inches='tight')
                    plt.close(new_fig)
                    print(f"[SAVED] {os.path.abspath(out_path)}")

                    try:
                        old_ax = axes[row*2 + 1, -4]
                        old_ax.remove()
                    except (KeyError, AttributeError):
                        pass
                    
                    ax_energy_3d = fig.add_subplot(len(guided_fields)*2, len(self.t_steps)+5,
                                                    (row*2+2)*(len(self.t_steps)+5) - 3, projection='3d')
                    self.plot_energy_landscape_3d(
                        ax_energy_3d, positions, current_t, vf, guided_field,
                        -5, 15, -15, 15
                    )

                    # Column -3 (bottom): Weighted density
                    ax_weighted_density_bottom = axes[row*2 + 1, -3]
                    self.plot_weighted_density(
                        ax_weighted_density_bottom, vf, guided_field,
                        -5, 15, -15, 15
                    )

                    # Column -1 (bottom): empty
                    ax_empty_bottom = axes[row*2 + 1, -1]
                    ax_empty_bottom.axis('off')

        plt.tight_layout(w_pad=3.0)  # extra horizontal spacing to avoid legend overlap
        out_path = f"{prefix}_field_evolution.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {os.path.abspath(out_path)}")
        plt.close(fig)

    def compute_field_data(self, vf, guided_field, positions, current_t):
        """Compute vector field data for visualization."""
        with torch.no_grad():
            v_uncond = vf(positions, current_t)
            # Collapse to (batch_size, 2) for plotting regardless of model output rank
            if v_uncond.dim() > 2:
                v_uncond = v_uncond.mean(dim=1)
            elif v_uncond.dim() == 1:
                v_uncond = v_uncond.reshape(positions.shape[0], -1)

            velocities_flow = v_uncond.cpu().numpy()
            
            if guided_field:
                identifier = getattr(guided_field, "guidance_identifier", None)
                multi_clf = (
                    hasattr(guided_field, "classifiers")
                    and len(guided_field.classifiers) > 1
                )
                if multi_clf and not self._is_guidance_matching_like(guided_field):
                    velocities_guided_total = guided_field(positions, current_t)
                    if velocities_guided_total.dim() > 2:
                        velocities_guided_total = velocities_guided_total.mean(dim=1)
                    elif velocities_guided_total.dim() == 1:
                        velocities_guided_total = velocities_guided_total.reshape(positions.shape[0], -1)

                    velocities_guidance = []
                    
                    scales = getattr(
                        guided_field,
                        "scales",
                        [1.0] * len(guided_field.classifiers),
                    )
                    for i, clf in enumerate(guided_field.classifiers):
                        single_clf = self._build_single_guided_field(
                            vf,
                            clf,
                            guided_field.targets[i],
                            scales[i],
                        )
                        v_guided = single_clf(positions, current_t)
                        if v_guided.dim() > 2:
                            v_guided = v_guided.mean(dim=1)
                        elif v_guided.dim() == 1:
                            v_guided = v_guided.reshape(positions.shape[0], -1)
                        velocities_guidance.append((v_guided - v_uncond).cpu().numpy())
                    
                    return {
                        'flow': velocities_flow,
                        'guidance': velocities_guidance,
                        'total': velocities_guided_total.cpu().numpy()
                    }
                else:
                    # Guidance matching: a single combined field
                    velocities_guided = guided_field(positions, current_t)
                    if velocities_guided.dim() > 2:
                        velocities_guided = velocities_guided.mean(dim=1)
                    elif velocities_guided.dim() == 1:
                        velocities_guided = velocities_guided.reshape(positions.shape[0], -1)
                    velocities_guidance = velocities_guided - v_uncond
                    return {
                        'flow': velocities_flow,
                        'guidance': velocities_guidance.cpu().numpy(),
                        'total': velocities_guided.cpu().numpy()
                    }
            return {'flow': velocities_flow}

    def compute_likelihood(self, solver, x_1_likelihood, gaussian_log_density, num_acc=10, step_size = 0.05):
        """Compute likelihood with both Hutchinson estimator and exact divergence."""
        # Unbiased Hutchinson estimator, averaged over num_acc draws
        log_p_acc = 0
        for _ in range(num_acc):
            _, log_p = solver.compute_likelihood(
                x_1=x_1_likelihood, method='midpoint', step_size=step_size,
                exact_divergence=False, log_p0=gaussian_log_density
            )
            log_p_acc += log_p
        log_p_acc /= num_acc

        # Exact divergence
        _, exact_log_p = solver.compute_likelihood(
            x_1=x_1_likelihood, method='midpoint', step_size=step_size,
            exact_divergence=True, log_p0=gaussian_log_density
        )
        
        return torch.exp(log_p_acc), torch.exp(exact_log_p)

    def visualize_likelihood_ori(self, vf, guided_field, prefix, step_size=0.05):
        """Visualize model likelihood."""
        grid_size = 200
        x_1_likelihood = torch.meshgrid(
            torch.linspace(-15, 15, grid_size),
            torch.linspace(-15, 15, grid_size),
            indexing='xy'
        )
        x_1_likelihood = torch.stack([x_1_likelihood[0].flatten(),
                                    x_1_likelihood[1].flatten()], dim=1).to(self.device)
        
        # Source distribution is an isotropic gaussian
        gaussian_log_density = dist.Independent(
            dist.Normal(torch.zeros(2, device=self.device),
                      torch.ones(2, device=self.device)), 1
        ).log_prob
        
        wrapped_vf = ModelWrapper(guided_field if guided_field else vf)
        solver = ODESolver(velocity_model=wrapped_vf)

        likelihood, exact_likelihood = self.compute_likelihood(
            solver, x_1_likelihood, gaussian_log_density, step_size=step_size
        )

        likelihood = likelihood.cpu().reshape(grid_size, grid_size).detach().numpy()
        exact_likelihood = exact_likelihood.cpu().reshape(grid_size, grid_size).detach().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        cmin, cmax = 0.0, 1/32  # 1/32 is the gt likelihood value
        norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
        
        im1 = ax1.imshow(likelihood, extent=(-15, 15, -15, 15),
                        origin='lower', cmap='viridis', norm=norm)
        ax1.set_title('Hutchinson Estimator Likelihood')
        fig.colorbar(im1, ax=ax1, orientation='horizontal', label='density')
        
        im2 = ax2.imshow(exact_likelihood, extent=(-15, 15, -15, 15),
                        origin='lower', cmap='viridis', norm=norm)
        ax2.set_title('Exact Likelihood (Jacobi)')
        fig.colorbar(im2, ax=ax2, orientation='horizontal', label='density')
        
        plt.tight_layout()
        plt.savefig(f"{prefix}_likelihood_ori.png")
        plt.close(fig)

    def visualize_residual_decomposition(self, vf, guided_field, out_path_prefix, t_val=1.0):
        if guided_field is None: return

        x_min, x_max, y_min, y_max = -5, 15, -13, 15
        bins = 200
        
        x_grid = np.linspace(x_min, x_max, bins)
        y_grid = np.linspace(y_min, y_max, bins)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        positions = torch.tensor(np.stack([X_mesh.ravel(), Y_mesh.ravel()], axis=1), dtype=torch.float32, device=self.device)
        t_tensor = torch.full((positions.shape[0], 1), float(t_val), device=self.device)
        
        comps = self.energy.compute_residual_components(positions, t_tensor, vf, guided_field)
        if not comps:
            return

        conflict_scores = None
        distribution = getattr(guided_field, "distribution", None)
        targets = list(getattr(guided_field, "targets", []))
        classifiers = getattr(guided_field, "classifiers", [])
        if distribution is not None and targets and len(classifiers) >= 2:
            try:
                with torch.enable_grad():
                    method = getattr(self.cfg, "conflict_score_method", "regional")

                    # Regional method is slow on large grids; fall back to direct
                    if method == "regional" and positions.shape[0] > 4096:
                        method = "direct"

                    if hasattr(distribution, "compute_direct_conflict_score"):
                        classifier_indices = self._map_classifiers_to_indices(classifiers, distribution)

                        if classifier_indices is not None and len(classifier_indices) >= 2:
                            sigma = getattr(self.cfg, "regional_conflict_sigma", 0.1)
                            num_samples = getattr(self.cfg, "regional_conflict_num_samples", 10)

                            # _map_classifiers_to_indices preserves the classifiers list order,
                            # so targets (same order) aligns with classifier_indices
                            if len(targets) == 1:
                                label_arg = int(targets[0])  # broadcasts to all classifiers
                            else:
                                label_arg = [int(t) for t in targets]

                            # Use x1_estimate for conflict score (consistent with composed_guidance.py)
                            with torch.no_grad():
                                v_uncond = vf(positions, t_tensor)
                                if v_uncond.dim() > 2:
                                    v_uncond = v_uncond.mean(dim=1)
                                elif v_uncond.dim() == 1:
                                    v_uncond = v_uncond.reshape(positions.shape[0], -1)
                            
                            estimate_x1 = bool(getattr(self.cfg, "estimate_x1", False))
                            x_for_conflict = positions if estimate_x1 else (positions + (1 - t_tensor) * v_uncond)
                            
                            print(f"[Visualizer] classifier_indices: {classifier_indices}")
                            print(f"[Visualizer] label_arg: {label_arg}")
                            print(f"[Visualizer] distribution._guidance_models: {list(distribution._guidance_models.keys())}")
                            print(f"[Visualizer] guided_field.classifiers: {[getattr(c, 'guidance_name', str(c)) for c in classifiers]}")
                            print(f"[Visualizer] estimate_x1: {estimate_x1}, x_for_conflict shape: {x_for_conflict.shape}")
                            
                            conflict_scores = distribution.compute_direct_conflict_score(
                                x_for_conflict, label=label_arg, classifier_indices=classifier_indices,
                                method=method, sigma=sigma, num_samples=num_samples
                            )
            except Exception as e:
                print(f"Warning: Could not compute conflict scores: {e}")

        has_scores = conflict_scores is not None

        # Plot count: base, learned, [conflict], [weight], total
        num_plots = 5 if has_scores else 3
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1: axes = [axes]

        base_e = comps["base_guidance"].reshape(bins, bins).cpu().numpy()
        learned_e = comps["learned_residual"].reshape(bins, bins).cpu().numpy()

        scores = None
        base_weight = None
        final_weight = None
        if has_scores:
            # conflict score in [0, 2]; kept as torch tensor for weight calc
            conflict_score_tensor = conflict_scores.reshape(bins, bins)

            threshold = getattr(self.cfg, "conflict_threshold", 0.9)
            temperature = getattr(self.cfg, "conflict_temperature", 0.01)
            blend_type = getattr(self.cfg, "blend_function", "smootherstep")

            # Blend function consistent with composed_guidance.py; threshold also in [0, 2]
            if blend_type == "smootherstep":
                conflict_normalized = (conflict_score_tensor - (threshold - temperature)) / (2 * temperature + 1e-8)
                weight = _smootherstep(conflict_normalized, edge0=0.0, edge1=1.0)
            else:
                weight = torch.sigmoid((conflict_score_tensor - threshold) / (temperature + 1e-8))

            conflict_score = conflict_score_tensor.detach().cpu().numpy()
            weight_np = weight.detach().cpu().numpy()

            scores = conflict_score

            # w(x) = blend_function((c(x) - τ) / T)
            final_weight = weight_np

            total_e = base_e + weight_np * learned_e
        else:
            total_e = base_e + learned_e

        # Plot 1: Base Guidance
        im0 = axes[0].imshow(base_e, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
        axes[0].set_title("Base Guidance Energy\n(g_cov_g)")
        plt.colorbar(im0, ax=axes[0])

        # Plot 2: Learned Residual
        im1 = axes[1].imshow(learned_e, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
        axes[1].set_title("Learned Residual Energy")
        plt.colorbar(im1, ax=axes[1])

        # Plot 3: Conflict Score (if it exists) OR Total
        current_idx = 2
        if has_scores:
            # scores is conflict score [0, 2] (0=aligned, 1=perpendicular, 2=opposite)
            im2 = axes[current_idx].imshow(scores, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', vmin=0, vmax=2)
            axes[current_idx].set_title("Conflict Score\n(0=aligned, 2=opposite)")
            plt.colorbar(im2, ax=axes[current_idx])
            current_idx += 1

            # Plot 4: Final Weight
            im_weight = axes[current_idx].imshow(final_weight, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', vmin=0, vmax=1)
            blend_type = getattr(self.cfg, "blend_function", "smootherstep")
            axes[current_idx].set_title(f"Final Weight w(x)\n({blend_type})")
            plt.colorbar(im_weight, ax=axes[current_idx])
            current_idx += 1

        # Plot 5 (or 3): Total Energy
        im_total = axes[current_idx].imshow(total_e, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
        axes[current_idx].set_title("Total Energy\n(Weighted Correction)" if has_scores else "Total Energy\n(Sum)")
        plt.colorbar(im_total, ax=axes[current_idx])

        sig = getattr(guided_field, "guidance_signature", "Unknown")
        fig.text(0.5, 0.02, f"Model: {sig}", ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f"{out_path_prefix}_residual_decomp.png")
        plt.close(fig)

    def visualize_likelihood(self, vf, guided_field, prefix,
                            x_min, x_max, y_min, y_max, bins=200, step_size=None):
        """
        Render likelihood on the same scale/resolution as plot_weighted_density.
        Uses exact divergence (Jacobian) only, log color scale with percentile clipping.
        """
        if step_size is None:
            step_size = float(self.cfg.step_size)

        # Grid identical to plot_weighted_density
        x_grid = torch.linspace(x_min, x_max, bins, device=self.device)
        y_grid = torch.linspace(y_min, y_max, bins, device=self.device)
        X_mesh, Y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x1 = torch.stack([X_mesh.ravel(), Y_mesh.ravel()], dim=1)

        wrapped_vf = ModelWrapper(guided_field if guided_field else vf)
        solver = ODESolver(velocity_model=wrapped_vf)

        gaussian_log_density = dist.Independent(
            dist.Normal(torch.zeros(x1.shape[-1], device=self.device),
                        torch.ones(x1.shape[-1],  device=self.device)), 1
        ).log_prob

        with torch.no_grad():
            _, exact_log_p = solver.compute_likelihood(
                x_1=x1, method='midpoint', step_size=step_size,
                exact_divergence=True, log_p0=gaussian_log_density
            )
            exact_like = torch.exp(exact_log_p).reshape(bins, bins).detach().cpu().numpy()

        # Log color scale with percentile clipping to match plot_weighted_density
        import numpy as np
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        title = 'Exact Likelihood (guided)' if guided_field else 'Exact Likelihood (base)'

        if np.all(np.isnan(exact_like)):
            ax.set_title(title + ' (unavailable: ODE diverged)')
            out_path = f"{prefix}_likelihood_exact.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return

        vmin = max(float(np.nanpercentile(exact_like, 1.0)), 1e-12)
        vmax = float(np.nanpercentile(exact_like, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-6

        im = ax.imshow(
            exact_like, extent=(x_min, x_max, y_min, y_max),
            origin='lower', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        ax.set_title(title)
        ax.grid(False)
        plt.colorbar(im, ax=ax, orientation='horizontal', label='density')

        out_path = f"{prefix}_likelihood_exact.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_conflict_score(self, ax, positions, t, vf, guided_field,
                            x_min, x_max, y_min, y_max):
        """Plot conflict score heatmap for multi-classifier guidance.
        
        Args:
            ax: Matplotlib axis
            positions: Grid positions (num_points*num_points, 2)
            t: Time tensor
            vf: Base velocity field
            guided_field: Guided field with compute_conflict_score method
            x_min, x_max, y_min, y_max: Plot boundaries
        """
        if not hasattr(guided_field, 'compute_conflict_score'):
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Conflict Score', 
                   ha='center', va='center', fontsize=10)
            return
            
        with torch.no_grad():
            v_uncond = vf(positions, t)
            if v_uncond.dim() > 2:
                v_uncond = v_uncond.mean(dim=1)
            elif v_uncond.dim() == 1:
                v_uncond = v_uncond.reshape(positions.shape[0], -1)
            
            if getattr(self.cfg, "estimate_x1", False):
                x_1_approx = positions
            else:
                x_1_approx = positions + (1 - t.view(-1, 1)) * v_uncond

            conflict_score = guided_field.compute_conflict_score(x_1_approx)

            conflict_map = conflict_score.cpu().reshape(self.num_points, self.num_points).numpy()

            im = ax.imshow(conflict_map, extent=(x_min, x_max, y_min, y_max),
                          origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1)
            ax.set_title('Conflict Score\n(Higher = More Conflict)')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.grid(False)

            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)


    def plot_classifier_logits(self, ax, positions, t, vf, guided_field,
                             x_min, x_max, y_min, y_max, classifier_1, classifier_2):
        """Plot classifier logits."""
        with torch.no_grad():
            v_uncond = vf(positions, t)
            if v_uncond.dim() > 2:
                v_uncond = v_uncond.mean(dim=1)
            elif v_uncond.dim() == 1:
                v_uncond = v_uncond.reshape(positions.shape[0], -1)

            # Respect cfg.estimate_x1 for consistency with other visualizations
            if getattr(self.cfg, "estimate_x1", False):
                x_1_approx = positions
            else:
                x_1_approx = positions + (1 - t.view(-1, 1)) * v_uncond
            
            if not guided_field:
                return
                
            if hasattr(guided_field, 'classifiers'):
                if len(guided_field.classifiers) > 1:
                    logits_clf1 = classifier_1(x_1_approx)
                    probs_clf1 = torch.softmax(logits_clf1, dim=1)[:, guided_field.targets[0]]
                    probs_clf1 = probs_clf1.cpu().reshape(self.num_points, self.num_points).numpy()
                    
                    logits_clf2 = classifier_2(x_1_approx)
                    probs_clf2 = torch.softmax(logits_clf2, dim=1)[:, guided_field.targets[1]]
                    probs_clf2 = probs_clf2.cpu().reshape(self.num_points, self.num_points).numpy()
                    
                    combined_probs = np.minimum(probs_clf1, probs_clf2)
                    
                    im = ax.imshow(combined_probs, extent=(x_min, x_max, y_min, y_max),
                            origin='lower', cmap='hot', vmin=0, vmax=1)
                    ax.set_title(f"Combined Clf1({guided_field.targets[0]}) & Clf2({guided_field.targets[1]})")
                    ax.grid(False)
                    
                else:
                    current_clf = guided_field.classifiers[0]
                    logits = current_clf(x_1_approx)
                    probs = torch.softmax(logits, dim=1)[:, guided_field.targets[0]]
                    probs = probs.cpu().reshape(self.num_points, self.num_points).numpy()
                    
                    ax.imshow(probs, extent=(x_min, x_max, y_min, y_max),
                              origin='lower', cmap='hot', vmin=0, vmax=1)
                    clf_type = "1" if current_clf is self.classifier_1 else "2"
                    ax.set_title(f"Clf{clf_type} Target {guided_field.targets[0]}")
                    ax.grid(False)

    def compute_target_energy(self, positions: torch.Tensor, t: torch.Tensor, vf, guided_field=None) -> torch.Tensor:
        """Delegate to EnergyVisualizer for energy computation."""
        return self.energy.compute_target_energy(positions, t, vf, guided_field)
    

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
        clip_percentiles=(1.0, 99.5),
        elev=30,
        azim=45,
    ):
        """Delegate 3D energy surface plotting."""
        self.energy.plot_energy_landscape_3d(
            ax,
            positions,
            t,
            vf,
            guided_field,
            x_min,
            x_max,
            y_min,
            y_max,
            clip_percentiles,
            elev,
            azim,
        )
    
    def plot_energy_landscape(
        self, ax, positions, t, vf, guided_field=None,
        x_min=None, x_max=None, y_min=None, y_max=None,
        clip_percentiles=(1.0, 99.5)
    ):
        """Delegate 2D energy heatmap plotting."""
        self.energy.plot_energy_landscape(
            ax, positions, t, vf, guided_field, x_min, x_max, y_min, y_max, clip_percentiles
        )

    def plot_reward_landscape(
        self, ax, positions, t, vf, guided_field=None,
        x_min=None, x_max=None, y_min=None, y_max=None,
        clip_percentiles=(1.0, 99.5)
    ):
        """2D reward heatmap: R = -E (higher is better).
        
        This simply negates the energy to get reward.
        Since energy = -log_p_base + Σλ(-log p) + (-learned_score),
        reward = log_p_base + Σλ(log p) + learned_score.
        """
        with torch.no_grad():
            E = self.energy.compute_target_energy(positions, t, vf, guided_field)  # energy (lower is better)
            R = (-E).reshape(self.num_points, self.num_points).detach().cpu().numpy()

            # robust clipping like energy plot
            if clip_percentiles is not None:
                p_low, p_high = clip_percentiles
                vmin = float(np.percentile(R, p_low))
                vmax = float(np.percentile(R, p_high))
            else:
                vmin = float(np.nanmin(R))
                vmax = float(np.nanmax(R))

            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = 1.0
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6

            norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
            im = ax.imshow(
                R,
                extent=(x_min, x_max, y_min, y_max),
                origin="lower",
                cmap="viridis",
                norm=norm,
            )
            title_suffix = ' (with guidance)' if (
                guided_field and hasattr(guided_field, 'classifiers')
            ) else ''
            ax.set_title("Reward (higher is better)" + title_suffix)
            ax.grid(False)
            plt.colorbar(im, ax=ax, orientation="horizontal", label="Reward")

            # streamlines: keep consistent with energy plot
            x = np.linspace(x_min, x_max, self.num_points)
            y = np.linspace(y_min, y_max, self.num_points)
            X, Y = np.meshgrid(x, y, indexing="xy")

            field_fn = guided_field if guided_field is not None else vf
            field = field_fn(positions, t)
            if field.dim() > 2:
                field = field.mean(dim=1)
            elif field.dim() == 1:
                field = field.reshape(positions.shape[0], -1)

            U = field[:, 0].reshape(self.num_points, self.num_points).cpu().numpy()
            V = field[:, 1].reshape(self.num_points, self.num_points).cpu().numpy()
            ax.streamplot(X, Y, U, V, color="white", linewidth=0.5, density=1.5, arrowsize=0.5)

    def plot_energy_contour(
        self, ax, positions, t, vf, guided_field=None,
        x_min=None, x_max=None, y_min=None, y_max=None,
        clip_percentiles=(1.0, 99.5)
    ):
        """Delegate contour plotting to EnergyVisualizer."""
        self.energy.plot_energy_contour(
            ax, positions, t, vf, guided_field, x_min, x_max, y_min, y_max, clip_percentiles
        )

    
    def plot_tightness_score(self, ax, positions, t, vf, guided_field,
                             x_min, x_max, y_min, y_max, classifier_1, classifier_2):
        """Plot tightness score."""
        with torch.no_grad():
            if not guided_field or not hasattr(guided_field, 'classifiers'):
                ax.set_title("No Guidance Field")
                ax.grid(False)
                return

            SDE_EPSILON, dt = 1.0, self.cfg.step_size
            num_samples = 30
            noise_std = torch.sqrt(torch.tensor(2. * SDE_EPSILON * dt))

            tightness_scores_grid = torch.zeros(positions.shape[0], device=self.device)

            v_uncond = vf(positions, t)
            if v_uncond.dim() > 2:
                v_uncond = v_uncond.mean(dim=1)
            elif v_uncond.dim() == 1:
                v_uncond = v_uncond.reshape(positions.shape[0], -1)

            if self.cfg.estimate_x1:
                x1_estimate = positions
            else:
                x1_estimate = positions + (1 - t.view(-1, 1)) * v_uncond

            all_logp = []
            for clf in guided_field.classifiers:
                for _ in range(num_samples):
                    noise = noise_std * torch.randn_like(x1_estimate)
                    out = clf(x1_estimate + noise)                    # (B, C)
                    logp = torch.log_softmax(out, dim=1)              # (B, C)
                    all_logp.append(logp)
            stacked_logp = torch.stack(all_logp)  # (S*C, B, C)

            B = positions.shape[0]
            Cnum = len(guided_field.classifiers)
            for j in range(B):
                scores_per_clf = []
                for i in range(Cnum):
                    target_class = guided_field.targets[i]
                    vals = stacked_logp[i*num_samples:(i+1)*num_samples, j, target_class]  # (S,)
                    X = vals.unsqueeze(1)  # (S, 1)
                    score_i, _, _, _ = normalized_tightness_torch(
                        X, k=5, R_null=3, tiny_scale=1e-3,
                        method="minmax", run_on="cpu", query_chunk=1024,
                        db_chunk=4096, max_points=None, device=self.device
                    )
                    scores_per_clf.append(score_i)
                # Aggregate with min to be conservative: unstable channels dominate
                tightness_scores_grid[j] = float(min(scores_per_clf))

            tightness_scores_grid = tightness_scores_grid.detach().cpu().reshape(self.num_points, self.num_points).numpy()
            im = ax.imshow(1-tightness_scores_grid, extent=(x_min, x_max, y_min, y_max),
                        origin='lower', cmap='viridis', vmin=0, vmax=1)
            ax.set_title("Tightness Score")
            ax.grid(False)
            plt.colorbar(im, ax=ax, orientation='horizontal', label='Tightness Score')

    def plot_weighted_density(self, ax, vf, guided_field, x_min, x_max, y_min, y_max, bins=200):
        """Delegate exp(-E) density visualization."""
        self.energy.plot_weighted_density(ax, vf, guided_field, x_min, x_max, y_min, y_max, bins)

    def plot_weighted_density_ori(self, ax, vf, guided_field, x_min, x_max, y_min, y_max, bins=200):
        """Delegate base-density-weighted visualization."""
        self.energy.plot_weighted_density_ori(ax, vf, guided_field, x_min, x_max, y_min, y_max, bins)

    def _map_classifiers_to_indices(self, classifiers, distribution):
        """Map classifiers to indices in distribution._guidance_models.
        
        This function maps classifiers from guided_field.classifiers to their
        corresponding indices in distribution._guidance_models, ensuring the
        order matches the classifiers list order.
        
        Args:
            classifiers: List of classifier objects from guided_field.classifiers
            distribution: ClusterDistribution instance with _guidance_models dict
            
        Returns:
            List of indices in distribution._guidance_models, or None if mapping fails
        """
        if not hasattr(distribution, "_guidance_models"):
            return None
        
        import re
        clf_items = list(distribution._guidance_models.items())
        classifier_indices = []

        clf_names_in_order = [getattr(c, 'guidance_name', None) for c in classifiers]
        model_keys_in_order = [name for name, _ in clf_items]

        for clf in classifiers:
            found = False
            clf_guidance_name = getattr(clf, 'guidance_name', None)

            # Strategy 1: exact match on guidance_name (primary path)
            if clf_guidance_name:
                for idx, (name, model) in enumerate(clf_items):
                    if name == clf_guidance_name:
                        classifier_indices.append(idx)
                        found = True
                        break

                # Strategy 1b: extract trailing index from guidance_name (e.g. "Classifier_0" -> 0)
                if not found:
                    match = re.search(r'(\d+)$', clf_guidance_name)
                    if match:
                        guidance_idx = int(match.group(1))
                        if 0 <= guidance_idx < len(clf_items):
                            expected_name = f"Classifier_{guidance_idx}"
                            if clf_items[guidance_idx][0] == expected_name:
                                classifier_indices.append(guidance_idx)
                                found = True

            # Strategy 2: match by object identity
            if not found:
                for idx, (name, model) in enumerate(clf_items):
                    if clf is model:
                        classifier_indices.append(idx)
                        found = True
                        break

            # Strategy 3: match by name or checkpoint_name attribute
            if not found:
                clf_name = getattr(clf, 'name', None) or getattr(clf, 'checkpoint_name', None)
                if clf_name:
                    for idx, (name, model) in enumerate(clf_items):
                        if name == clf_name:
                            classifier_indices.append(idx)
                            found = True
                            break

            if not found:
                # Return None to fall back to default behavior when mapping fails
                print(f"[_map_classifiers_to_indices] Warning: Could not map classifier "
                      f"with guidance_name={clf_guidance_name}. "
                      f"Available model keys: {model_keys_in_order}")
                return None

        if len(classifier_indices) != len(classifiers):
            print(f"[_map_classifiers_to_indices] Warning: Mapped {len(classifier_indices)} "
                  f"indices but expected {len(classifiers)}. "
                  f"Classifier names: {clf_names_in_order}, "
                  f"Model keys: {model_keys_in_order}, "
                  f"Indices: {classifier_indices}")
            return None
        
        return classifier_indices
