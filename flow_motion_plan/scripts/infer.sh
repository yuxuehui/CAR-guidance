#!/bin/bash
set -e

# 项目根目录（脚本位于 <root>/scripts/）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"
cd "$ROOT"

# 无引导推理示例（base flow model 采样一条轨迹）
python scripts/inference.py \
    --checkpoint "$ROOT/checkpoints/checkpoint_epoch_20.pth" \
    --start 2.027489736762842 2.6098391538459307 \
    --goal 4.3650726128237896 3.7675649583249604 \
    --walls 3.1005763729995 3.6249321984435796 4.298446930863955 1.2974427294501316 \
            0.5815200692813294 1.1095293831391615 1.004883899496825 3.1778353784796427 \
            2.7558783277200765 1.3718581636260145 1.8007417732746718 4.478511023443694 \
    --save infer_traj.png
