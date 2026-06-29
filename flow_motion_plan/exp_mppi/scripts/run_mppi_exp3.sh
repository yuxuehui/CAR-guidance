#!/bin/bash
# 运行MPPI实验3：动态障碍物

echo "======================================================================"
echo "运行MPPI实验3：动态障碍物 Baseline"
echo "======================================================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行实验
python exp_mppi/scripts/run_mppi_exp3.py \
    --model checkpoints/flow_model/best_checkpoint.pth \
    --config exp_mppi/configs/mppi_exp3_dynamic.yaml \
    --seed 42

echo ""
echo "======================================================================"
echo "实验3完成！"
echo "结果保存在: exp_mppi/outputs/exp3_dynamic/"
echo "======================================================================"

