#!/bin/bash
# 实验1：静态障碍物Guidance（基础版本，只加guidance，不加g_cov_a_gm_online方法）

# 配置
SUCCESS_JSON="experiments/data/success_trajectories.json"
CONFIG="experiments/configs/exp1_static_base.yaml"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验1：静态障碍物Guidance（基础版本）${NC}"
echo -e "${BLUE}只加guidance，不加g_cov_a_gm_online方法${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验
python experiments/scripts/run_exp1_static.py \
    --config "$CONFIG" \
    --success-json "$SUCCESS_JSON"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验1基础版本完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp1_static_base/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验1基础版本失败，请检查错误信息${NC}"
    exit 1
fi

