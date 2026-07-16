#!/bin/bash
# 按顺序运行所有 Task

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "开始运行所有 Task (1 -> 2 -> 3)"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"

# Task 1: 静态能量场引导 + 动作保存
echo ">>> 运行 Task 1..."
bash 01_run_task1.sh
echo ""

# Task 2: 自适应 G_Cov 引导 + 动作保存
echo ">>> 运行 Task 2..."
bash 02_run_task2.sh
echo ""

# Task 3: 轨迹重放 + 可视化
echo ">>> 运行 Task 3..."
bash 03_run_task3.sh
echo ""

echo "=========================================="
echo "所有 Task 运行完成！"
echo ""
echo "输出目录:"
echo "  Task 1: $PROJECT_ROOT/experiments/outputs/01_task1_static_only/"
echo "  Task 2: $PROJECT_ROOT/experiments/outputs/02_task2_gcov_adaptive/"
echo "  Task 3: $PROJECT_ROOT/experiments/outputs/03_task3_replay_visualize/"
echo "=========================================="
