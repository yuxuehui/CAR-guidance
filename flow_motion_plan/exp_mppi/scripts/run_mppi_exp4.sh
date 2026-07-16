#!/bin/bash
# 运行MPPI实验4：混合场景

echo "======================================================================"
echo "运行MPPI实验4：混合场景 Baseline"
echo "======================================================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行实验
python exp_mppi/scripts/run_mppi_exp4.py \
    --model checkpoints/flow_model/best_checkpoint.pth \
    --config exp_mppi/configs/mppi_exp4_mixed.yaml \
    --seed 42

echo ""
echo "======================================================================"
echo "实验4完成！"
echo "结果保存在: exp_mppi/outputs/exp4_mixed/"
echo "======================================================================"

