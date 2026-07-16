#!/bin/bash
# 运行实验1：静态能量场引导 + g_cov_a_gm_online 方法修正

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验1：静态能量场引导 + g_cov优化${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验脚本
# 参数说明：
#   --save_videos True/False: 是否保存视频（覆盖配置文件，不传递则使用配置文件中的值）
#   --visualize True/False: 是否可视化显示（覆盖配置文件，不传递则使用配置文件中的值）
# 注意：如果不传递这些参数，将使用配置文件 exp1_static_gcov.yaml 中的设置
python experiments/scripts/run_exp1_static_gcov.py \
    --config experiments/configs/exp1_static_gcov.yaml

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp1_static_gcov/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验失败，请检查错误信息${NC}"
    exit 1
fi

