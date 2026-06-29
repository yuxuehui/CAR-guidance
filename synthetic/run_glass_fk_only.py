"""Standalone GLASS Flow FK-corrector baseline for the synthetic experiment.

Pipeline:
  1. load the pre-trained velocity field,
  2. load the pre-trained reward classifiers,
  3. run the GLASS Flow FK corrector for several particle counts K.

For each classifier combination and K, the script reports sampling metrics
(MMD, diversity, mode coverage, ...) and saves a figure with one panel per
backbone timestep showing the particle distribution evolving from t_0 to t_1.
"""

import os
import sys
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GUIDANCE_DIR  = os.path.join(SCRIPT_DIR, "guidance")
VISUALIZER_DIR = os.path.join(SCRIPT_DIR, "visualizer")
for _d in (GUIDANCE_DIR, VISUALIZER_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from backbone.velocity_fields import MLP, FiLMResNet, MiniUnetVelocityField
from config.config_loader import load_config
from distributions.cluster import ClusterDistribution
from distributions.prior import StandardGaussianPrior
from classifier import Classifier, train_classifier
from glass_flow_fk import GlassFlow as GlassFlowModel, GlassFlowFK
from visualizer import (
    compute_mmd_rbf,
    compute_diversity_pairwise,
    compute_gt_posterior_match,
    compute_mode_coverage,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def generate_data(n_samples_per_cluster=500, device="cpu"):
    prior        = StandardGaussianPrior(dim=2, device=device)
    cluster_dist = ClusterDistribution()
    n_samples    = n_samples_per_cluster * 3

    p0 = prior.sample(n_samples, device=device).cpu()
    p1, cluster_labels = cluster_dist.sample_with_labels(batch_size=n_samples, device=device)
    p1             = p1.cpu()
    cluster_labels = cluster_labels.cpu()

    zeros  = torch.zeros_like(cluster_labels)
    ones   = torch.ones_like(cluster_labels)
    y_clf1 = torch.where(cluster_labels == 0, zeros, ones)
    y_clf2 = torch.where(cluster_labels <= 1, zeros, ones)

    perm = torch.randperm(n_samples)
    return [x[perm] for x in [p0, p1, cluster_labels, y_clf1, y_clf2]]


def inf_train_gen(batch_size=200, device="cpu"):
    _, p1, *_ = generate_data(n_samples_per_cluster=batch_size // 3 + 1, device=device)
    return p1[:batch_size].to(device).float()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLASS Flow FK corrector (standalone)")
    default_config = os.path.join(SCRIPT_DIR, "config", "fm_config.yaml")
    parser.add_argument("--config",            type=str,   default=default_config)
    parser.add_argument("--fk_corr_rho",       type=float, default=0.5,
                        help="GLASS transition correlation ρ (0=max stochastic, 1=ODE)")
    parser.add_argument("--fk_n_backbone",     type=int,   default=6,
                        help="Number of backbone timesteps")
    parser.add_argument("--fk_n_inner_steps",  type=int,   default=20,
                        help="Inner Euler steps per GLASS transition")
    parser.add_argument("--result_output_dir", type=str,   default=None,
                        help="Root output directory")
    parser.add_argument("--fk_n_samples",      type=int,   default=10000,
                        help="Number of output samples to generate (one FK run each)")
    parser.add_argument("--seed",              type=int,   default=42,
                        help="Global random seed (default 42)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = cfg.device
    torch.manual_seed(args.seed)

    # ── Load pre-trained velocity field ──────────────────────────────────────
    vf_ckpt_dir  = os.path.join(SCRIPT_DIR, "pretrain_vf_model")
    vf_ckpt_path = os.path.join(
        vf_ckpt_dir,
        f"vf_{cfg.vf_model}_hidden{cfg.hidden_dim}_iters{cfg.iterations}.pth",
    )
    model_map = {"mlp": lambda: MLP(hidden_dim=cfg.hidden_dim), "filmresnet": FiLMResNet}
    vf = model_map.get(cfg.vf_model, MiniUnetVelocityField)().to(device)

    if os.path.exists(vf_ckpt_path):
        vf.load_state_dict(torch.load(vf_ckpt_path, map_location=device))
        print(f"[LOADED VF] {vf_ckpt_path}")
    else:
        print("[TRAIN VF] No checkpoint found, training from scratch …")
        path  = AffineProbPath(scheduler=CondOTScheduler())
        optim = torch.optim.Adam(vf.parameters(), lr=cfg.lr)
        for i in range(cfg.iterations):
            optim.zero_grad()
            x_1 = inf_train_gen(batch_size=cfg.batch_size, device=device)
            x_0 = torch.randn_like(x_1)
            t   = torch.rand(x_1.shape[0]).to(device)
            ps  = path.sample(t=t, x_0=x_0, x_1=x_1)
            loss = torch.pow(vf(ps.x_t, ps.t) - ps.dx_t, 2).mean()
            loss.backward(); optim.step()
            if (i + 1) % 2000 == 0:
                print(f"  iter {i+1}/{cfg.iterations}  loss={loss.item():.4f}")
        os.makedirs(vf_ckpt_dir, exist_ok=True)
        torch.save(vf.state_dict(), vf_ckpt_path)
        print(f"[SAVED VF] {vf_ckpt_path}")
    vf.eval()

    # ── Load pre-trained classifiers ─────────────────────────────────────────
    p0, p1, y_orig, y_clf1, y_clf2 = [t.to(device) for t in generate_data(500)]

    classifier_1 = Classifier(num_classes=2).to(device)
    classifier_2 = Classifier(num_classes=2).to(device)
    classifier_1.guidance_name = "Classifier_0"
    classifier_2.guidance_name = "Classifier_1"

    pretrained_dir = os.path.join(SCRIPT_DIR, "guidance", "pretrained_guidance")
    clf1_path = os.path.join(pretrained_dir, "classifier_1.pth")
    clf2_path = os.path.join(pretrained_dir, "classifier_2.pth")

    if os.path.exists(clf1_path) and os.path.exists(clf2_path):
        classifier_1.load_state_dict(torch.load(clf1_path, map_location=device))
        classifier_2.load_state_dict(torch.load(clf2_path, map_location=device))
        print(f"[LOADED] Classifiers from {pretrained_dir}")
    else:
        os.makedirs(pretrained_dir, exist_ok=True)
        for clf, labels, name in [
            (classifier_1, y_clf1, "Classifier 1"),
            (classifier_2, y_clf2, "Classifier 2"),
        ]:
            train_classifier(clf, p1, labels, name,
                             epochs=cfg.classifier_epochs, lr=cfg.classifier_lr,
                             save_dir=pretrained_dir, device=device)
    classifier_1.eval()
    classifier_2.eval()

    # ── Resolve output directory ──────────────────────────────────────────────
    result_dir = args.result_output_dir or getattr(cfg, "result_output_dir", "result_glass_fk")
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(SCRIPT_DIR, result_dir)
    output_dir = os.path.join(result_dir, "glass_fk")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[SAVE TO] {output_dir}")

    # ── GLASS Flow FK corrector ──────────────────────────────────────────────
    guidance_cluster = ClusterDistribution(device=device)

    class _VFAdapter(nn.Module):
        """Wraps vf(x_t, t) positional call into fm_model(x_t=…, t=…) keyword call."""
        def __init__(self, vf): super().__init__(); self.vf = vf
        def forward(self, x_t, t, **kwargs): return self.vf(x_t, t)

    def build_fk_reward(classifiers, targets):
        """r(x̂₁) = -Σ_j scale_j · J_j(x̂₁)  —  consistent with guidance_sample_fn weights."""
        def reward_fn(x1_hat):
            log_r = torch.zeros(x1_hat.shape[0], device=x1_hat.device, dtype=x1_hat.dtype)
            for clf, target in zip(classifiers, targets):
                energy = guidance_cluster.get_J(x1_hat, classifier=clf, label=int(target))
                log_r  = log_r - float(cfg.guidance_scale) * energy
            return log_r
        return reward_fn

    glass_flow            = GlassFlowModel(fm_model=_VFAdapter(vf).to(device))
    glass_flow.data_shape = (2,)
    fk_runner             = GlassFlowFK(glass_flow)

    n_backbone = args.fk_n_backbone
    n_inner    = args.fk_n_inner_steps
    corr_rho   = args.fk_corr_rho
    t_backbone = torch.linspace(0, 1, n_backbone, device=device, dtype=torch.float32)
    t_labels   = [f"t={t_backbone[i].item():.2f}" for i in range(n_backbone)]

    # Reference data for background scatter (down-sampled for speed)
    p1_np = p1.cpu().numpy()
    y_np  = y_orig.cpu().numpy()
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    gt_cols = [colors[int(l)] for l in y_np]

    # ── GT visualizer (needed for posterior sampling in metrics) ─────────────
    from ground_truth_posterior import GroundTruthPosteriorVisualizer
    gt_vis = GroundTruthPosteriorVisualizer(
        ClusterDistribution(device=device), device=device, chunk_size=50000)

    centers_cpu = guidance_cluster.centers.cpu()   # (n_modes, 2)
    stds_cpu    = guidance_cluster.std.cpu()        # (n_modes, 2)

    def compute_fk_metrics(samples_np, slug, K, clfs, targets):
        """Print per-run statistics matching visualize_sampling_process output."""
        samples_t = torch.as_tensor(samples_np, dtype=torch.float32).cpu()

        # Prior preservation (within 2σ of any cluster center)
        prior_pres = compute_gt_posterior_match(samples_t, centers_cpu, stds_cpu, threshold=2.0)

        # Classifier accuracy
        total_prob, n_comp = 0.0, 0
        with torch.no_grad():
            for clf, target in zip(clfs, targets):
                logits = clf(samples_t.to(device))
                probs  = torch.softmax(logits, dim=-1)
                total_prob += probs[:, int(target)].mean().item()
                n_comp += 1
        clf_acc = total_prob / n_comp if n_comp > 0 else float("nan")

        # Ground-truth posterior samples
        components = [
            (clf.guidance_name, int(tgt), float(cfg.guidance_scale))
            for clf, tgt in zip(clfs, targets)
        ]
        n_gt = min(len(samples_np), 10000)
        gt_samples = gt_vis.sample_posterior(components, num_samples=n_gt)
        gt_t = torch.as_tensor(gt_samples, dtype=torch.float32).cpu()

        mmd      = compute_mmd_rbf(samples_t, gt_t, sigma=1.0)
        gen_div  = compute_diversity_pairwise(samples_t)
        gt_div   = compute_diversity_pairwise(gt_t)
        gt_match = compute_gt_posterior_match(samples_t, centers_cpu, stds_cpu, threshold=2.0)
        cov      = compute_mode_coverage(samples_t, centers_cpu, stds_cpu, threshold=2.0)

        n = len(samples_np)
        label = f"GlassFlowFK {slug} K={K}"
        print(f"[{label}] Generated {n} samples | "
              f"Prior preservation: {prior_pres*100:.1f}% "
              f"({int(prior_pres*n)}/{n} samples within 2σ)")
        print(f"  [Classifier Accuracy] Average target label probability: {clf_acc*100:.2f}%")
        print(f"  [Ground Truth Comparison] "
              f"MMD: {mmd:.6f} | "
              f"Diversity (gen/gt): {gen_div:.2f}/{gt_div:.2f} | "
              f"GT Match: {gt_match*100:.1f}% | "
              f"Mode coverage: {cov['overall_coverage']*100:.1f}% | "
              f"Per-mode: {[f'{c*100:.1f}%' for c in cov['per_mode_coverage']]} | "
              f"Counts: {[int(c) for c in cov['per_mode_counts']]}")

    fk_combos = [
        ("c0",   [classifier_1],                [0]),
        ("c1",   [classifier_1],                [1]),
        ("c2_0", [classifier_2],                [0]),
        ("c2_1", [classifier_2],                [1]),
        ("c0c0", [classifier_1, classifier_2], [0, 0]),
        ("c1c0", [classifier_1, classifier_2], [1, 0]),
        ("c1c1", [classifier_1, classifier_2], [1, 1]),
    ]

    TIMING_REPEATS = 5
    TIMING_WARMUP  = 1
    _is_cuda = isinstance(device, torch.device) and device.type == "cuda"

    def _sync():
        if _is_cuda:
            torch.cuda.synchronize()

    def _batched_kwargs(n_groups, K, reward_fn, return_intermediates=False, verbose=False):
        return dict(
            n_groups=n_groups, K=K,
            reward_fn=reward_fn,
            t_backbone=t_backbone,
            n_inner_steps=n_inner, corr_rho=corr_rho,
            device=device, dtype=torch.float32,
            return_intermediates=return_intermediates,
            verbose=verbose,
        )

    N_SAMPLES = args.fk_n_samples

    for slug, clfs, targets in fk_combos:
        reward_fn = build_fk_reward(clfs, targets)

        for K in [4, 16]:
            print(f"[GlassFlowFK] {slug}  K={K}  backbone={n_backbone}  "
                  f"inner={n_inner}  rho={corr_rho}  n_samples={N_SAMPLES}")

            for _ in range(TIMING_WARMUP):
                with torch.no_grad():
                    fk_runner.sample_batched(**_batched_kwargs(1, K, reward_fn))
            _sync()

            # Timing: n_groups=1 → K particles internally → 1 output, so the
            # single-group wall time IS the cost per output sample (no /K).
            times_ms = []
            for _ in range(TIMING_REPEATS):
                _sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    fk_runner.sample_batched(**_batched_kwargs(1, K, reward_fn))
                _sync()
                times_ms.append((time.perf_counter() - t0) * 1000)

            ms_per_sample = sum(times_ms) / len(times_ms)
            print(f"[Timing] {slug} K={K}: "
                  f"{ms_per_sample:.1f} ms/sample  "
                  f"(over {TIMING_REPEATS} runs, {TIMING_WARMUP} warmup)")

            # Total batch = N_SAMPLES * K particles; resampling per group.
            # Each group → 1 output sample.  step_snaps[i]: (N_SAMPLES, 2).
            with torch.no_grad():
                final_t, step_snaps = fk_runner.sample_batched(
                    **_batched_kwargs(N_SAMPLES, K, reward_fn,
                                      return_intermediates=True))

            final_samples = final_t.cpu().numpy()                   # (N_SAMPLES, 2)
            step_arrays   = [s.numpy() for s in step_snaps]         # list of (N_SAMPLES, 2)

            compute_fk_metrics(final_samples, slug, K, clfs, targets)

            # ── Step-by-step scatter plot ─────────────────────────────
            n_show   = min(N_SAMPLES, 3000)
            idx_show = torch.randperm(N_SAMPLES)[:n_show].numpy()

            fig, axes = plt.subplots(1, n_backbone, figsize=(3 * n_backbone, 3),
                                     sharex=True, sharey=True)
            for ax, arr, label in zip(axes, step_arrays, t_labels):
                pts = arr[idx_show]
                ax.scatter(*p1_np.T, c=gt_cols, s=2, alpha=0.15, rasterized=True)
                ax.scatter(*pts.T, c="tab:red", s=4, alpha=0.5, rasterized=True)
                ax.set_title(label, fontsize=9)
                ax.set_xlim(-5, 15); ax.set_ylim(-13, 15)
                ax.set_aspect("equal")
                ax.axis("off")

            fig.suptitle(f"GLASS Flow FK  [{slug}]  K={K}  n={N_SAMPLES}",
                         fontsize=11, y=1.01)
            fig.tight_layout()
            save_path = os.path.join(output_dir, f"glass_fk_{slug}_K{K}_steps.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[GlassFlowFK] Saved → {save_path}")

    print("\n[Done] All FK experiments finished.")


if __name__ == "__main__":
    main()
