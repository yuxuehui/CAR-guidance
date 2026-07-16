#!/bin/bash
# 运行所有MPPI实验

echo "======================================================================"
echo "运行所有MPPI Baseline实验"
echo "======================================================================"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 实验1：静态能量场
echo ""
echo ">>> 开始实验1：静态能量场"
bash exp_mppi/scripts/run_mppi_exp1.sh

# 实验2：目标吸引场干扰
echo ""
echo ">>> 开始实验2：目标吸引场干扰"
bash exp_mppi/scripts/run_mppi_exp2.sh

# 实验3：动态障碍物
echo ""
echo ">>> 开始实验3：动态障碍物"
bash exp_mppi/scripts/run_mppi_exp3.sh

# 实验4：混合场景
echo ""
echo ">>> 开始实验4：混合场景"
bash exp_mppi/scripts/run_mppi_exp4.sh

echo ""
echo "======================================================================"
echo "所有MPPI实验完成！"
echo "======================================================================"
echo "结果保存在:"
echo "  - exp_mppi/outputs/exp1_static/"
echo "  - exp_mppi/outputs/exp2_goal/"
echo "  - exp_mppi/outputs/exp3_dynamic/"
echo "  - exp_mppi/outputs/exp4_mixed/"
echo "======================================================================"

