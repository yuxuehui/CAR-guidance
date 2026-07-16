#!/bin/bash
# Task 1: 在 pick-cube 上加能量场，正常执行，保存动作用于后续使用

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Task 1: 静态能量场引导 + 动作保存"
echo "=========================================="

cd "$PROJECT_ROOT"

# 运行实验
python experiments/scripts/01_run_task1.py \
    --config experiments/configs/01_task1_static_only.yaml \
    --visualize false \
    --save_videos true \
    --num_episodes 1

echo ""
echo "=========================================="
echo "Task 1 完成！"
echo "动作数据已保存至: experiments/outputs/01_task1_static_only/actions.json"
echo "=========================================="
