#!/bin/bash
# 运行MPPI实验1：静态能量场

echo "======================================================================"
echo "运行MPPI实验1：静态能量场 Baseline"
echo "======================================================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行实验
python exp_mppi/scripts/run_mppi_exp1.py \
    --model checkpoints/flow_model/best_checkpoint.pth \
    --config exp_mppi/configs/mppi_exp1_static.yaml \
    --seed 42

echo ""
echo "======================================================================"
echo "实验1完成！"
echo "结果保存在: exp_mppi/outputs/exp1_static/"
echo "======================================================================"

