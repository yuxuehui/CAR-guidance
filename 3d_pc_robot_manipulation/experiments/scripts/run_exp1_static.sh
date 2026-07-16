#!/bin/bash
# 运行实验1：静态能量场引导（基础版本）

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验1：静态能量场引导（基础版本）${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验脚本
# 参数说明：
#   --save_videos True/False: 是否保存视频（覆盖配置文件，可选）
#   --visualize True/False: 是否可视化显示（覆盖配置文件，与 save_videos 互斥，可选）
# 如果不指定这些参数，将使用配置文件中的设置
CUDA_VISIBLE_DEVICES=1 python experiments/scripts/run_exp1_static.py \
    --config experiments/configs/exp1_static.yaml

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp1_static/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验失败，请检查错误信息${NC}"
    exit 1
fi

