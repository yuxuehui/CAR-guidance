# CAR Guidance — ManiSkill Manipulation Experiment

Robot manipulation (policy-as-planner) experiment for **Conflict-Aware Additive
Guidance for Flow Models under Compositional Rewards** (ICML 2026). A base
Conditional Flow Matching (CFM) point-cloud policy is trained on ManiSkill2
demonstrations; compositional energy rewards (static obstacles, trajectory
smoothness, goals) are injected at inference time and corrected by **g^car**.

This code builds on **PointFlowMatch** (Chisari et al., CoRL 2024) for the base
CFM policy; here it is slimmed to the two tasks reported in the paper —
**PickCube** and **StackCube** — plus the g^car guidance experiments.

```
Base CFM policy: 4096-pt colored point cloud + robot state + goal  →  PointNet
encoder  →  Conditional 1D U-Net velocity field  →  action chunk a=[Δp,Δr,g]∈R^7
Guidance:  v' = v_base + g^car,   g^car = g^approx + w_t · g_ψ   (ours)
```

## Setup

One shared conda env for both released experiments. See `../INSTALL.md` and
`../requirements.txt` (Python 3.10, PyTorch 2.1.2/cu121, MosaicML Composer,
pytorch3d, `diffusion_policy`). The ManiSkill **simulator** (`mani_skill==3.0.0b21`
+ SAPIEN) is only needed for **evaluation rollouts**, not for training — see the
optional step in `../INSTALL.md`.

## Data & checkpoints

- Demonstrations (`data/`): `demo_data_pcd_from_three_cameras_small` (PickCube,
  100 train) and `stack_cube_demo_big_train` (StackCube, 100 train), plus
  validation sets. Host externally; download link below.
- Trained base policies are saved under `ckpt/<run_name>/`.

> 📦 Demonstration data / checkpoint download: **<ADD LINK>** (Google Drive / HuggingFace).

## Train the base CFM policy

```bash
python scripts/train_maniskill_pick_cube.py     # -> ckpt/maniskill_train_pcd_from_three_cameras_more_gripper_encoder_goal_pos
python scripts/train_maniskill_stack_cube.py    # -> ckpt/maniskill_train_stack_cube_concat_goal_pos_big
```
Data paths resolve to `<repo>/data/`; checkpoints autoresume. EMA weights are
saved and preferred at eval.

## Evaluate (needs the ManiSkill simulator)

```bash
python scripts/eval_maniskill.py --ckpt_name <run_name> --ckpt_episode latest
```

## g^car guidance experiments

```bash
# PickCube: static-only baseline / adaptive g^car / full g^car
python experiments/experiments/exp1_static_gcov.py
# StackCube: static-only / static-energy / full g^car
python experiments_stack_cube/experiments/task3_gcov_full.py
```
Guidance config in `experiments*/configs/*_gcov*.yaml`
(`online_loss_type: gradient`, smootherstep conflict gate, `(1−cos)/2` conflict
score — aligned with the synthetic reference implementation).

## Layout

```
3d_pc_robot_manipulation/
├── pfp/                      # base CFM policy library (slimmed to ManiSkill)
│   ├── policy/               # fm_policy_maniskill.py (FMPolicy), energy_guide.py
│   ├── backbones/            # pointnet.py, pointnet_concat_goal.py
│   ├── data/                 # dataset_maniskill*.py, dataset_pcd.py, replay_buffer.py
│   ├── common/ utils/        # fm/se3 utils, point-cloud utils, robot state
├── scripts/                  # train_maniskill_{pick,stack}_cube.py, eval_maniskill.py, tests
├── experiments/              # PickCube g^car experiments (guidance/, configs/, core/)
├── experiments_stack_cube/   # StackCube g^car experiments
├── data/                     # demonstrations (download; gitignored)
└── ckpt/                     # trained checkpoints (gitignored)
```

Original PointFlowMatch (base policy): Chisari et al., "Learning Robotic
Manipulation Policies from Point Clouds with Conditional Flow Matching", CoRL 2024.
