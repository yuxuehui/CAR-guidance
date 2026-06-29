#!/bin/bash
# Task 3: 使用保存的轨迹在相同环境下进行重放
# 把能量场中心可视化出来，同时把机器臂的运动轨迹也画出来，然后保存视频

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Task 3: 轨迹重放 + 可视化"
echo "=========================================="

cd "$PROJECT_ROOT"

# 默认使用 Task 2 的输出作为输入
# 如果想使用 Task 1 的输出，可以修改 --replay_actions_path
ACTIONS_PATH="experiments/outputs/01_task1_static_only/actions.json"

# 检查动作数据是否存在
if [ ! -f "$ACTIONS_PATH" ]; then
    echo "错误: 动作数据文件不存在: $ACTIONS_PATH"
    echo "请先运行 Task 1 或 Task 2 生成动作数据"
    exit 1
fi

echo "使用动作数据: $ACTIONS_PATH"

# 运行实验
python experiments/scripts/03_run_task3.py \
    --config experiments/configs/03_task3_replay_visualize.yaml \
    --replay_actions_path "$ACTIONS_PATH" \
    --num_episodes 10 \
    --video_fps 20

echo ""
echo "=========================================="
echo "Task 3 完成！"
echo "可视化视频已保存至: experiments/outputs/03_task3_replay_visualize/videos/"
echo "=========================================="
