#!/bin/bash
# 可视化轨迹脚本：重放实验并可视化能量场和轨迹

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}可视化轨迹：重放实验并可视化能量场和轨迹${NC}"
echo -e "${BLUE}========================================${NC}"

# 默认参数
RESULTS_PATH="${1:-experiments/outputs/exp1_static/detailed_results.json}"
EXPERIMENT_NAME="${2:-exp1_static}"
NUM_DEMOS="${3:-10}"

# 如果路径是目录，自动使用该目录
if [ -d "${RESULTS_PATH}" ]; then
    echo -e "${YELLOW}检测到目录路径，将加载目录下所有demo文件${NC}"
fi

echo -e "${YELLOW}参数:${NC}"
echo -e "  结果文件: ${RESULTS_PATH}"
echo -e "  实验名称: ${EXPERIMENT_NAME}"
echo -e "  演示数量: ${NUM_DEMOS}"
echo ""

# 运行可视化脚本
python experiments/scripts/04_visualize_trajectories.py \
    --results_path "${RESULTS_PATH}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --num_demos "${NUM_DEMOS}" \
    --video_resolution 1920 1080 \
    --fps 20

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 可视化完成！${NC}"
    # 确定输出目录
    if [ -d "${RESULTS_PATH}" ]; then
        OUTPUT_DIR="${RESULTS_PATH}"
    else
        OUTPUT_DIR="$(dirname ${RESULTS_PATH})"
    fi
    echo -e "${BLUE}视频保存在: ${OUTPUT_DIR}/visualization_videos/${NC}"
else
    echo -e "\n${YELLOW}⚠️  可视化失败，请检查错误信息${NC}"
    exit 1
fi
