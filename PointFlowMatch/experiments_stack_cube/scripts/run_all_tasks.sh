#!/bin/bash

# 运行所有 Stack-Cube 实验任务

echo "=========================================="
echo "开始运行所有 Stack-Cube 实验任务"
echo "=========================================="

cd "$(dirname "$0")/../.."

# Task 1: 不使用能量场
echo ""
echo "=========================================="
echo "Task 1: 不使用能量场"
echo "=========================================="
python experiments_stack_cube/scripts/01_run_task1.py

# Task 2: 加能量场
echo ""
echo "=========================================="
echo "Task 2: 加能量场"
echo "=========================================="
python experiments_stack_cube/scripts/02_run_task2.py

# Task 3: 加能量场和方法
echo ""
echo "=========================================="
echo "Task 3: 加能量场和方法"
echo "=========================================="
python experiments_stack_cube/scripts/03_run_task3.py

echo ""
echo "=========================================="
echo "所有实验任务完成！"
echo "=========================================="

