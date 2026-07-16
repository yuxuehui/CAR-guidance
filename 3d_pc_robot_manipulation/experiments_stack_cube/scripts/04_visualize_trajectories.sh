#!/bin/bash
# Stack-Cube 可视化轨迹脚本：重放实验并可视化能量场和轨迹
#
# 用法:
#   ./experiments_stack_cube/scripts/04_visualize_trajectories.sh
#   ./experiments_stack_cube/scripts/04_visualize_trajectories.sh <results_path> [experiment_name] [num_demos]
#
# 示例:
#   ./experiments_stack_cube/scripts/04_visualize_trajectories.sh experiments_stack_cube/outputs/02_task2_static_energy/actions 02_task2_static_energy 10
#   ./experiments_stack_cube/scripts/04_visualize_trajectories.sh experiments_stack_cube/outputs/03_task3_gcov_full/actions 03_task3_gcov_full 5

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stack-Cube 可视化轨迹：重放实验并可视化能量场和轨迹${NC}"
echo -e "${BLUE}========================================${NC}"

# 默认参数：结果路径、实验名称、演示数量
RESULTS_PATH="${1:-experiments_stack_cube/outputs/02_task2_static_energy/actions}"
EXPERIMENT_NAME="${2:-02_task2_static_energy}"
NUM_DEMOS="${3:-10}"

# 若传入的是任务输出目录（如 experiments_stack_cube/outputs/02_task2_static_energy），且其下存在 actions，则使用 actions
if [ -d "${RESULTS_PATH}" ] && [ -d "${RESULTS_PATH}/actions" ]; then
    RESULTS_PATH="${RESULTS_PATH}/actions"
fi

echo -e "${YELLOW}参数:${NC}"
echo -e "  结果路径: ${RESULTS_PATH}"
echo -e "  实验名称: ${EXPERIMENT_NAME}"
echo -e "  演示数量: ${NUM_DEMOS}"
echo ""

# 在项目根目录下执行（与 01_run_task1.sh 等一致）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

python experiments_stack_cube/scripts/04_visualize_trajectories.py \
    --results_path "${RESULTS_PATH}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --num_demos "${NUM_DEMOS}" \
    --video_resolution 1920 1080 \
    --fps 20

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 可视化完成！${NC}"
    if [ -d "${RESULTS_PATH}" ]; then
        OUTPUT_DIR="${RESULTS_PATH}"
    else
        OUTPUT_DIR="$(dirname ${RESULTS_PATH})"
    fi
    echo -e "${BLUE}视频保存在: ${OUTPUT_DIR}/visualization_videos/${NC}"
    echo -e "${BLUE}图片保存在: ${OUTPUT_DIR}/visualization_images/${NC}"
else
    echo -e "\n${YELLOW}⚠️  可视化失败，请检查错误信息${NC}"
    exit 1
fi
