#!/bin/bash

# 运行 Task 4 (baseline): PCGrad 梯度手术组合多reward梯度
# 复用 Task2 的 runner，仅切换 --config

cd "$(dirname "$0")/../.."

# 与 02_run_task2.sh 使用相同的测试种子，保证可比性
python experiments_stack_cube/scripts/02_run_task2.py \
    --config experiments_stack_cube/configs/04_task4_pcgrad.yaml \
    --seeds 10450,104388,1003617,1037483,121926,130450,145610,156671,166678,178577,186909,202743,223503,231021,259065,266804,292129,36987,300778,312477,324787,336911,354947,362553,370180,380159,399570,41933,418152,437770
