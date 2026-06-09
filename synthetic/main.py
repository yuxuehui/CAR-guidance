"""Entry point for the CAR-guidance synthetic 2D experiment.

Samples from a pretrained flow-matching velocity field while applying
compositional classifier guidance, comparing CAR guidance against the
baselines. For every classifier combination the script:

  1. loads (or trains) the velocity field and the two reward classifiers,
  2. builds the requested guidance field via ``ComposedGuidance``,
  3. renders ground-truth posteriors, guided trajectories and likelihoods.

The guidance method and all hyperparameters are selected through
``config/fm_config.yaml`` and the command-line overrides defined in ``main``.

Author: Xuehui Yu
"""

import argparse
import os
import sys
import warnings
from typing import Optional, Sequence

import torch
from torch import nn

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from backbone.velocity_fields import MLP, FiLMResNet, MiniUnetVelocityField
from config.config_loader import load_config
from distributions.prior import StandardGaussianPrior
from distributions.cluster import ClusterDistribution

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDANCE_DIR = os.path.join(SCRIPT_DIR, "guidance")
if GUIDANCE_DIR not in sys.path:
    sys.path.insert(0, GUIDANCE_DIR)

from classifier import Classifier, train_classifier
from composed_guidance import ComposedGuidance

VISUALIZER_DIR = os.path.join(SCRIPT_DIR, "visualizer")
if VISUALIZER_DIR not in sys.path:
    sys.path.insert(0, VISUALIZER_DIR)

from visualizer import Visualizer
from ground_truth_posterior import GroundTruthPosteriorVisualizer


def generate_data(n_samples_per_cluster=500, device: str = "cpu"):
    """Generate synthetic data using dedicated prior and cluster distributions."""
    n_samples = n_samples_per_cluster * 3
    prior = StandardGaussianPrior(dim=2, device=device)
    cluster_dist = ClusterDistribution()

    p0 = prior.sample(n_samples, device=device).cpu()
    p1, cluster_labels = cluster_dist.sample_with_labels(batch_size=n_samples, device=device)
    p1 = p1.cpu()
    cluster_labels = cluster_labels.cpu()

    y_orig = cluster_labels.clone()
    zeros = torch.zeros_like(cluster_labels)
    ones = torch.ones_like(cluster_labels)
    y_clf1 = torch.where(cluster_labels == 0, zeros, ones)
    y_clf2 = torch.where(cluster_labels <= 1, zeros, ones)

    perm = torch.randperm(n_samples)
    datasets = [p0, p1, y_orig, y_clf1, y_clf2]
    return [x[perm] for x in datasets]

def inf_train_gen(batch_size=200, device="cpu"):
    """Generate training data."""
    _, p1, _, _, _ = generate_data(n_samples_per_cluster=batch_size//3 + 1)
    return p1[:batch_size].to(device).float()


def build_output_dir(cfg, script_dir: str) -> str:
    """Resolve the result directory for the active config.

    The directory name encodes the key hyperparameters so that different runs
    do not overwrite each other. Relative ``result_output_dir`` values are
    resolved against ``script_dir``; absolute values are used as-is.
    """
    slug = f"{cfg.vf_model}_sgt{cfg.start_guidance_threshold}_ss{cfg.step_size}"
    if getattr(cfg, "estimate_x1", False):
        slug += "_estimatex1"
    slug += f"_cth{getattr(cfg, 'conflict_threshold', 0.9)}"
    slug += f"_csm{getattr(cfg, 'conflict_score_method', 'regional')}"
    slug += f"_ctemp{getattr(cfg, 'conflict_temperature', 0.15)}"

    result_dir = getattr(cfg, "result_output_dir", "result_new")
    if not os.path.isabs(result_dir):
        result_dir = os.path.join(script_dir, result_dir)
    return os.path.join(result_dir, getattr(cfg, "guidance_fn", "g_cov_g"), slug)


def main():
    # ----- Step 0: Parse CLI args and load configuration -----
    parser = argparse.ArgumentParser(description="Run Flow Matching with Entropy-based Guidance")
    # Default config path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "config", "fm_config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to the YAML configuration file")
    parser.add_argument("--guidance_fn", type=str, default=None, help="Override guidance function from config")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Override guidance_scale from config (scaling factor for guidance)")
    parser.add_argument("--conflict_threshold", type=float, default=None, help="Override conflict_threshold from config")
    parser.add_argument("--conflict_temperature", type=float, default=None, help="Override conflict_temperature from config")
    parser.add_argument("--blend_function", type=str, default=None, help="Override blend_function from config (sigmoid or smootherstep)")
    parser.add_argument("--conflict_score_method", type=str, default=None, help="Override conflict_score_method from config (regional or direct)")
    parser.add_argument("--regional_conflict_sigma", type=float, default=None, help="Override regional_conflict_sigma from config")
    parser.add_argument("--online_loss_type", type=str, default=None, help="Override online_loss_type from config (ground_truth, mse_simple, or gradient)")
    parser.add_argument("--guidance_ckpt_dir", type=str, default=None, help="Override guidance_ckpt_dir from config")
    parser.add_argument("--result_output_dir", type=str, default=None, help="Override result_output_dir from config")
    parser.add_argument("--x1_conflict_threshold", type=float, default=None, help="Override x1_conflict_threshold from config (early stopping threshold: stop training when x1_conflict_ratio < x1_conflict_threshold)")
    parser.add_argument("--active_ratio_threshold", type=float, default=None, help="Override active_ratio_threshold from config (early stopping threshold: stop training when active_ratio < active_ratio_threshold)")
    parser.add_argument("--loss_early_stop_threshold", type=float, default=None, help="Override loss_early_stop_threshold from config (early stopping threshold: stop training when loss < loss_early_stop_threshold)")
    parser.add_argument("--conflict_mask_type", type=str, default=None, help="Override conflict_mask_type from config (hard: binary gate, soft: continuous weight=conflict/2)")
    parser.add_argument("--disable_conflict_guidance_weight", type=lambda x: (str(x).lower() == 'true'), default=None, help="Override disable_conflict_guidance_weight from config (true: weight=1.0 ablation, false: use conflict-based weight)")
    parser.add_argument("--fixed_guidance_weight", type=float, default=None, help="Fix w_t to a constant value in [0,1] for all points (e.g. 0.5, 1.0). Overrides conflict-based weighting.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Every CLI flag mirrors a config field of the same name; apply the ones
    # the user explicitly set (i.e. left at their non-None value).
    override_keys = [
        "guidance_fn", "guidance_scale", "conflict_threshold", "conflict_temperature",
        "blend_function", "conflict_score_method", "regional_conflict_sigma",
        "online_loss_type", "guidance_ckpt_dir", "result_output_dir",
        "x1_conflict_threshold", "active_ratio_threshold",
        "loss_early_stop_threshold", "conflict_mask_type",
        "disable_conflict_guidance_weight", "fixed_guidance_weight",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    device = cfg.device
    torch.manual_seed(42)
    
    # ----- Step 1: Initialize velocity field (load from checkpoint or train) -----
    vf_ckpt_dir = os.path.join(script_dir, "pretrain_vf_model")
    os.makedirs(vf_ckpt_dir, exist_ok=True)
    vf_ckpt_path = os.path.join(
        vf_ckpt_dir,
        f"vf_{cfg.vf_model}_hidden{cfg.hidden_dim}_iters{cfg.iterations}.pth",
    )

    def build_vf():
        if cfg.vf_model == "mlp":
            return MLP(hidden_dim=cfg.hidden_dim).to(device)
        if cfg.vf_model == "filmresnet":
            return FiLMResNet().to(device)
        return MiniUnetVelocityField().to(device)

    vf = build_vf()
    if os.path.exists(vf_ckpt_path):
        vf.load_state_dict(torch.load(vf_ckpt_path, map_location=device))
        print(f"[LOADED VF] {vf_ckpt_path}")
    else:
        path = AffineProbPath(scheduler=CondOTScheduler())
        optim = torch.optim.Adam(vf.parameters(), lr=cfg.lr)

        for i in range(cfg.iterations):
            optim.zero_grad()
            x_1 = inf_train_gen(batch_size=cfg.batch_size, device=device)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0]).to(device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()

            loss.backward()
            optim.step()

            if (i + 1) % 2000 == 0:
                print(f"Iteration {i+1}, Loss: {loss.item():.3f}")

        torch.save(vf.state_dict(), vf_ckpt_path)
        print(f"[SAVED VF] {vf_ckpt_path}")

    # ----- Step 2: Prepare data and (pre)train classifiers -----
    p0, p1, y_orig, y_clf1, y_clf2 = [tensor.to(device) for tensor in generate_data(500)]
    classifier_1 = Classifier(num_classes=2).to(device)
    classifier_2 = Classifier(num_classes=2).to(device)
    # Assign stable guidance names so checkpoint resolution is deterministic
    classifier_1.guidance_name = "Classifier_0"
    classifier_2.guidance_name = "Classifier_1"
    
    pretrained_dir = os.path.join(script_dir, "guidance", "pretrained_guidance")

    clf1_path = os.path.join(pretrained_dir, "classifier_1.pth")
    clf2_path = os.path.join(pretrained_dir, "classifier_2.pth")

    if os.path.exists(clf1_path) and os.path.exists(clf2_path):
        classifier_1.load_state_dict(torch.load(clf1_path, map_location=device))
        classifier_2.load_state_dict(torch.load(clf2_path, map_location=device))
        print(f"[LOADED] Classifier weights from {pretrained_dir}")
    else:
        os.makedirs(pretrained_dir, exist_ok=True)
        train_classifier(
            classifier_1,
            p1,
            y_clf1,
            "Classifier 1",
            epochs=cfg.classifier_epochs,
            lr=cfg.classifier_lr,
            save_dir=pretrained_dir,
            device=device,
        )
        train_classifier(
            classifier_2,
            p1,
            y_clf2,
            "Classifier 2",
            epochs=cfg.classifier_epochs,
            lr=cfg.classifier_lr,
            save_dir=pretrained_dir,
            device=device,
        )

    guidance_prior = StandardGaussianPrior(dim=2)
    guidance_cluster = ClusterDistribution(device=device)
    guidance_path = AffineProbPath(scheduler=CondOTScheduler())

    def guidance_sample_fn(
        batch_size: int,
        device: str,
        targets: Optional[Sequence[int]] = None,
        classifiers: Optional[Sequence[nn.Module]] = None,
        scales: Optional[Sequence[float]] = None,
    ):
        """Monte-Carlo sampler of OT-path training batches for the learnable guidance models.

        Samples an OT path ``x_t = (1-t)·x_0 + t·x_1`` with conditional velocity
        ``u_t = x_1 - x_0``, evaluates the base field ``v_θ(x_t, t)``, and computes an
        importance weight ``exp(-scale·J(x_1))`` from the guidance energy J.
        ``guidance_matching`` regresses ``g_φ(x_t, t) ≈ u_t - v_θ`` on these batches.

        Args:
            batch_size: number of samples to generate.
            device: device to generate samples on.
            targets: per-classifier target labels (multi-classifier guidance).
            classifiers: pretrained classifiers (multi-classifier guidance).
            scales: per-classifier guidance scales λ_j.

        Returns:
            dict with xt (B,d), t (B,), ut (B,d), v (B,d), weight (B,1).
        """
        with torch.no_grad():
            # Sample OT-path endpoints x_0 ~ p_0 (prior) and x_1 ~ p_1 (data clusters)
            x0 = guidance_prior.sample(batch_size=batch_size, device=device)
            x1, _ = guidance_cluster.sample_with_labels(
                batch_size=batch_size, device=device
            )

            t = torch.rand(batch_size, device=device)
            path_sample = guidance_path.sample(t=t, x_0=x0, x_1=x1)
            xt = path_sample.x_t        # x_t = (1-t)·x_0 + t·x_1
            ut = path_sample.dx_t       # u_t = x_1 - x_0
            base_v = vf(xt, path_sample.t)  # v_θ(x_t, t)

            # Importance weight exp(-J(x_1)); lower energy -> higher weight
            if targets and classifiers:
                # Multi-classifier: J(x_1) = Σ_j λ_j · (-log p(y_j | x_1))
                total_energy = torch.zeros(batch_size, device=device)
                eff_scales = (
                    list(scales)
                    if scales is not None and len(scales) == len(targets)
                    else [cfg.guidance_scale] * len(targets)
                )
                for clf, target, lam in zip(classifiers, targets, eff_scales):
                    energy = guidance_cluster.get_J(x1, classifier=clf, label=int(target))
                    total_energy += float(lam) * energy
                weights = torch.exp(-total_energy).unsqueeze(-1)
            else:
                # Geometric energy from the cluster distribution
                weights = torch.exp(
                    -cfg.guidance_scale
                    * guidance_cluster.get_J(
                        x1, mode=getattr(cfg, "mode", "product"), alpha=getattr(cfg, "alpha", 1.0)
                    )
                ).unsqueeze(-1)

        return {
            "xt": xt.detach(),         # Input: intermediate state
            "t": path_sample.t.detach(),  # Input: time
            "ut": ut.detach(),         # Target: ground-truth velocity (for guidance_matching)
            "v": base_v.detach(),      # Base velocity (for computing u_t - v_t)
            "weight": weights.detach(),   # Importance weights (for contrastive loss)
        }

    cfg.guidance_sample_fn = guidance_sample_fn

    # ----- Step 3: Build guided vector fields for every classifier combination -----
    guidance_fn_val = getattr(cfg, "guidance_fn", None)
    learnable_guidance_fns = {"guidance_matching", "car_guidance"}
    is_learnable = getattr(cfg, "learnable", guidance_fn_val in learnable_guidance_fns)

    def make_guidance(classifiers, targets):
        scales = [cfg.guidance_scale] * len(classifiers)
        return ComposedGuidance(
            vf, classifiers, targets, scales, cfg, 
            guidance_fn=guidance_fn_val, learnable=is_learnable,
            distribution=guidance_cluster
        )

    # By default we only build the ``c1c0`` multi-classifier combination, i.e.
    # Classifier_0 -> label 1 AND Classifier_1 -> label 0. This is the
    # compositional-reward setting reported in the paper: the two reward signals
    # pull towards different regions, so their gradients conflict and CAR
    # guidance is needed. Each entry triggers training (or loading) of the
    # corresponding guidance network, so keeping a single combination keeps runs
    # fast. Uncomment any of the lines below to additionally build the
    # single-classifier (c0, c1, c2_0, c2_1) or other multi-classifier
    # (c0c0, c1c1) combinations.
    guided_fields = {
        # # Single-classifier combinations
        # 'c0':   make_guidance([classifier_1], [0]),                 # Classifier_0 -> label 0
        # 'c1':   make_guidance([classifier_1], [1]),                 # Classifier_0 -> label 1
        # 'c2_0': make_guidance([classifier_2], [0]),                 # Classifier_1 -> label 0
        # 'c2_1': make_guidance([classifier_2], [1]),                 # Classifier_1 -> label 1
        # # Other multi-classifier combinations
        # 'c0c0': make_guidance([classifier_1, classifier_2], [0, 0]),  # label 0 + label 0
        'c1c0': make_guidance([classifier_1, classifier_2], [1, 0]),    # label 1 + label 0
        # 'c1c1': make_guidance([classifier_1, classifier_2], [1, 1]),  # label 1 + label 1
    }

    # ===== Visualization is DISABLED by default =====
    # Everything below regenerates the ground-truth posteriors, guided
    # trajectories and likelihood / energy-landscape figures. It is skipped by
    # default so this script only trains (or loads) the guidance network(s),
    # which keeps runs fast and avoids the memory-heavy plotting code. Set
    # VISUALIZE = True to re-enable the full visualization pipeline below.
    VISUALIZE = False
    if not VISUALIZE:
        print(f"\n[DONE] Guidance network(s) ready for: {list(guided_fields.keys())}")
        print(f"[DONE] Checkpoints saved under: {getattr(cfg, 'guidance_ckpt_dir', '(default)')}")
        return

    # ----- Step 4: Configure visualization utilities and output directory -----
    visualizer = Visualizer(
        device,
        cfg,
        classifier_1,
        classifier_2,
        multi_guided_cls=ComposedGuidance,
    )

    output_base_path = build_output_dir(cfg, script_dir)
    os.makedirs(output_base_path, exist_ok=True)
    print(f"[SAVE TO] {output_base_path}")

    ground_truth_visualizer = GroundTruthPosteriorVisualizer(
        ClusterDistribution(device=device),
        device=device,
        chunk_size=max(50000, cfg.batch_size_plot_traj),
    )

    def build_ground_truth_components(field):
        """
        Build ground truth components from a guidance field.
        Uses the guidance_name attribute of classifiers instead of id() for reliability.
        """
        if field is None:
            return []
        comps = []
        scales = (
            field.scales
            if getattr(field, "scales", None) and len(field.scales) == len(field.classifiers)
            else [cfg.guidance_scale] * len(field.classifiers)
        )
        for clf, target, scale in zip(field.classifiers, field.targets, scales):
            # Use guidance_name attribute directly instead of id lookup
            clf_name = getattr(clf, 'guidance_name', None)
            if clf_name is None:
                print(f"[WARNING] Classifier {clf} does not have guidance_name attribute, skipping for ground truth")
                continue
            comps.append((clf_name, int(target), float(scale)))
        return comps

    # ----- Step 5: Sample ground-truth posteriors for all guided fields -----
    ground_truth_specs = {"vf": []}
    for name, field in guided_fields.items():
        components = build_ground_truth_components(field)
        ground_truth_specs[name] = components
        print(f"[Ground Truth Components] {name}: {components}")

    posterior_output_dir = os.path.join(output_base_path, "ground_truth_posterior")
    ground_truth_visualizer.visualize(
        ground_truth_specs,
        posterior_output_dir,
        num_samples=max(cfg.batch_size_plot_traj * 2, 2000),
    )

    # ----- Step 6: Render base flow and guided trajectories/likelihoods -----
    # Visualize whichever guided fields were built in Step 3 (c1c0 by default).

    # Base visualization
    visualizer.visualize_field_evolution(
        vf,
        None,
        p0,
        p1,
        y_orig,
        os.path.join(output_base_path, "vf"),
        ["Base Flow"],
        cfg.batch_size_plot_traj,
    )

    batch_size_sample = 10000  # generate 10000 samples
    T = torch.linspace(0, 1, 10).to(device)
    x_init = torch.randn((batch_size_sample, 2), dtype=torch.float32, device=device)

    def run_guidance_group(slug, fields_map):
        if not fields_map:
            return
        names = list(fields_map.keys())
        field_list = [fields_map[name] for name in names]
        solvers = [
            ODESolver(velocity_model=ModelWrapper(field)) for field in field_list
        ]
        titles = [f"Guidance {name}" for name in names]

        # Build guidance components list for ground truth comparison
        guidance_components_list = []
        for name in names:
            field = fields_map[name]
            components = build_ground_truth_components(field)
            guidance_components_list.append(components)
            print(f"[visualize_sampling_process] Building components for {name}: {components}")
        
        print(f"[visualize_sampling_process] Total guidance_components_list length: {len(guidance_components_list)}, num_solvers: {len(solvers)}")
        print(f"[visualize_sampling_process] ground_truth_visualizer is {'None' if ground_truth_visualizer is None else 'not None'}")

        visualizer.visualize_sampling_process(
            solvers,
            x_init,
            T,
            titles,
            os.path.join(output_base_path, f"{slug}_combined.png"),
            cfg.step_size,
            distribution=guidance_cluster,  # Pass distribution for prior preservation metrics
            ground_truth_visualizer=ground_truth_visualizer,  # Pass for ground truth comparison
            guidance_components_list=guidance_components_list,  # Pass guidance components for each solver
        )
        visualizer.visualize_field_evolution(
            vf,
            field_list,
            p0,
            p1,
            y_orig,
            os.path.join(output_base_path, slug),
            titles,
            cfg.batch_size_plot_traj,
        )
        for name, field in fields_map.items():
            prefix = os.path.join(output_base_path, name)
            visualizer.visualize_likelihood(
                vf,
                field,
                prefix,
                x_min=-5,
                x_max=15,
                y_min=-13,
                y_max=15,
                bins=200,
                step_size=cfg.step_size,
            )
            visualizer.visualize_likelihood_ori(
                vf, field, prefix, cfg.step_size
            )
            visualizer.visualize_residual_decomposition(
                vf, field, prefix
            )

    run_guidance_group("guided", guided_fields)

    print(f"\n[DONE] All results saved to {output_base_path}")


if __name__ == "__main__":
    main()
