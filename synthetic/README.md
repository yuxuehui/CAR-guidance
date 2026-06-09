# CAR Guidance — Synthetic Experiment

Synthetic 2D experiment for **Conflict-Aware Additive Guidance for Flow Models
under Compositional Rewards** (ICML 2026). A pretrained flow-matching velocity
field samples a 3-cluster 2D distribution; two pretrained classifiers act as
compositional rewards. Methods compared:

- `g_cov_g` — non-learnable approximate guidance `g^approx` (baseline)
- `guidance_matching` — learnable guidance-matching baseline
- `glass_fk` — GLASS Flow FK corrector baseline
- `car_guidance` — **CAR guidance (ours)**: `g^car = (1 - w_t) g^approx + w_t · g_psi`, an online-trained residual `g_psi` gated by a conflict-aware weight `w_t`.

## Setup

```bash
conda env create -f environment.yml   # Python 3.10, PyTorch 2.6 (CUDA 12.4)
conda activate car_guidance
```

The pretrained velocity field (`pretrain_vf_model/`) and the two pretrained
classifiers (`guidance/pretrained_guidance/classifier_1.pth` and
`classifier_2.pth`) ship with the repo — no pretraining needed.
For CPU, set `device: cpu` in `config/fm_config.yaml`.

## Run

```bash
bash running.sh          # CAR runs enabled; baselines are commented blocks
```

Or a single config directly (CAR guidance, ours):

```bash
python -u ./main.py \
  --guidance_fn car_guidance \
  --guidance_ckpt_dir guidance/pretrained_guidance/online_ckpt \
  --result_output_dir result
```

`main.py` builds the `c1c0` combination and loads the residual `g_psi` from
`--guidance_ckpt_dir` if present (the command above uses the shipped
`guidance/pretrained_guidance/online_ckpt/`), otherwise it trains and saves one
there. Visualization is off by default; set `VISUALIZE = True` in `main.py` to
regenerate the figures.

## Key hyperparameters

Defaults in `config/fm_config.yaml`, overridable on the CLI (`python main.py --help`):

| Argument | Meaning | Value |
| --- | --- | --- |
| `--guidance_fn` | guidance method | `car_guidance` |
| `--conflict_threshold` | conflict gate (≈perpendicular on the `[0,1]` scale) | `0.495` |
| `--conflict_temperature` | gate transition width | `0.01` |
| `--active_ratio_threshold` | early-stop on conflict-region ratio | `0.00` / `0.05` |
| `--guidance_scale` | reward guidance scale | `7` |

## Layout

```
synthetic/
├── running.sh / environment.yml      # run configs, conda env
├── main.py / run_glass_fk_only.py    # entry points (ours+baselines / GLASS FK)
├── config/ backbone/ distributions/  # config, velocity fields, 2D target+prior
├── flow_matching/                    # flow-matching paths / solvers / utils
├── guidance/                         # classifier, composed_guidance, glass_flow_fk, pretrained/
├── visualizer/                       # plotting / energy / posterior
└── pretrain_vf_model/                # pretrained velocity field
```
