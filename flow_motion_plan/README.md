# CAR Guidance — Maze2D Planning Experiment

Generative decision-making (planning) experiment for **Conflict-Aware Additive
Guidance for Flow Models under Compositional Rewards** (ICML 2026). A base
Conditional Flow Matching model is trained on collision-free Maze2D
demonstrations and acts as a learned planner; compositional energy rewards
(static obstacles, goals, dynamic agents) are injected at inference time.

Included methods:
- **base CFM** — unguided flow planner
- `g_cov_g` — non-learnable approximate guidance `g^approx` (baseline)
- `car_guidance` (**ours**) — `g^car = g^approx + w_t · g_ψ`, an online-trained
  residual `g_ψ` (vector field, velocity-correction target) gated by a
  conflict-aware weight `w_t` (smootherstep over the all-pairs `(1−cos)/2`
  conflict score)
- **MPPI** — optimization-based planner baseline
- **MPPI + g^car** (**ours**) — g^car corrects the base onto the data manifold,
  MPPI then refines it for hard constraints (paper Table 2)

## Setup

One shared conda env for both released experiments (this and `3d_pc_robot_manipulation`).
See `../INSTALL.md` and `../requirements.txt` (Python 3.10, PyTorch 2.1.2/cu121).
This experiment only needs `torch`, the vendored `flow_matching/`, `h5py`,
`numpy`, `matplotlib`.

## Checkpoint & data

- `checkpoints/checkpoint_epoch_20.pth` — pretrained base CFM (host externally;
  see download link below). Place it under `checkpoints/`.
- `data/randSmaze2d-…hdf5` — Maze2D training set (only needed to retrain the
  base; download link below).
- Test mazes for the guidance experiments live in
  `experiments/data/base_model_images/success_trajectories.json`
  (regenerable from saved MPPI cases or `experiments/scripts/collect_success_trajectories.py`).

> 📦 Checkpoint / dataset download: **<ADD LINK>** (Google Drive / HuggingFace).

## Run

```bash
# base (unguided) inference — single trajectory + figure
bash scripts/infer.sh

# inference with energy guidance / g^car
bash scripts/infer_guide.sh

# guidance experiments (compositional rewards): static / goal / dynamic / mixed
python experiments/scripts/run_exp1_static.py  --num-cases 100
python experiments/scripts/run_exp2_goal.py
python experiments/scripts/run_exp3_dynamic.py
python experiments/scripts/run_exp4_mixed.py

# MPPI baseline and MPPI + g^car (ours)
python exp_mppi/experiments/exp1_mppi_static.py
python exp_mppi/scripts/run_mppi_gcar.py --num-cases 100
```

Retrain the base CFM (optional; checkpoint shipped):
```bash
bash scripts/train.sh   # reads configs/train_flow_config.json
```

## Key hyperparameters

Set in `experiments/configs/*_gcov.yaml`:

| Key | Meaning | Value |
| --- | --- | --- |
| `online_loss_type` | residual training target | `gradient` (vector velocity-correction) |
| `conflict_threshold` | conflict gate center | `0.15` |
| `conflict_temperature` | gate transition width | `0.1` |
| `online_train_steps` | online residual training steps | `10` |
| `num_ode_steps` | ODE discretization | `20` |

MPPI hyperparameters: `exp_mppi/configs/mppi_exp*.yaml`.

## Layout

```
flow_motion_plan/
├── diffuser/                 # base CFM: model (flow_guide.py), training, datasets
│   └── models/diffusion_policy/  # ConditionalUnet1D velocity head (slimmed)
├── flow_matching/            # vendored Meta flow-matching library
├── scripts/                  # train / inference entry points
├── experiments/              # compositional-reward guidance experiments
│   ├── guidance/             # gcov_wrapper.py (g^car), static/goal/dynamic, energy_function
│   ├── experiments/          # exp1..4 (static/goal/dynamic/mixed)
│   ├── configs/ scripts/     # YAML configs + runners
│   └── core/ utils/          # experiment base, evaluator, inference, viz
├── exp_mppi/                 # MPPI planner + MPPI+g^car
│   ├── core/                 # mppi_flow_controller.py, energy_cost.py
│   ├── experiments/ scripts/ # MPPI runners (incl. run_mppi_gcar.py)
│   └── configs/
└── checkpoints/              # base CFM checkpoint (download)
```
