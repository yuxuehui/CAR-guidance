#!/bin/bash
# 运行MPPI实验2：目标吸引场干扰

echo "======================================================================"
echo "运行MPPI实验2：目标吸引场干扰 Baseline"
echo "======================================================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行实验
python exp_mppi/scripts/run_mppi_exp2.py \
    --model checkpoints/flow_model/best_checkpoint.pth \
    --config exp_mppi/configs/mppi_exp2_goal.yaml \
    --seed 42

echo ""
echo "======================================================================"
echo "实验2完成！"
echo "结果保存在: exp_mppi/outputs/exp2_goal/"
echo "======================================================================"

