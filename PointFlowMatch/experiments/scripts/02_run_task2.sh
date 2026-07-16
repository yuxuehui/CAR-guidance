#!/bin/bash
# Task 2: 在 pick-cube 上加能量场和方法，第一次推理前在线训练
# 如果没完成任务，从第二次推理开始去掉能量场和方法，直接使用 base model

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Task 2: 自适应 G_Cov 引导 + 动作保存"
echo "=========================================="

cd "$PROJECT_ROOT"

# 运行实验（种子与 outputs_old/exp1_static_gcov_scale_1.0_sigma_0.2 一致）
python experiments/scripts/02_run_task2.py \
    --config experiments/configs/02_task2_gcov_adaptive.yaml \
    --visualize false \
    --save_videos false \
    --num_episodes 10 \
    --max_rounds 3 \
    --seeds 1,5,6,7,9,10,11,12,13,15

echo ""
echo "=========================================="
echo "Task 2 完成！"
echo "动作数据已保存至: experiments/outputs/02_task2_gcov_adaptive/actions.json"
echo "=========================================="
