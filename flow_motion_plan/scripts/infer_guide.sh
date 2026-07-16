#!/bin/bash
set -e

# 项目根目录（脚本位于 <root>/scripts/）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"
cd "$ROOT"

# 带能量引导的推理示例（在 base flow 上叠加能量引导 / GCAR）。
# energy_center: 成对给出 (x, y)；energy_scale: 每个中心的尺度（正=吸引，负=排斥，0=关闭）。
CUDA_VISIBLE_DEVICES=0 python scripts/inference_guide.py \
    --checkpoint "$ROOT/checkpoints/checkpoint_epoch_20.pth" \
    --start 1.0 2.2 \
    --goal 4.0 1.5 \
    --walls 2.4387179417620333 4.150708066393468 2.509573022174967 2.0535250982772015 \
            3.614153125313898 0.7343649211124741 4.466125293722284 2.6177291678617265 \
            0.723634623409799 4.485568194025829 1.2044014703199601 2.870078036725651 \
    --save infer_traj.png \
    --energy_center 3.0 3.0 2.0 1.0 \
    --energy_scale 0.0 0.0 \
    --step_viz_save infer_step.png
