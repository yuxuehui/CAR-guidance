#!/bin/bash
# 实验2：目标吸引Guidance（PCGrad梯度手术，Yu et al. 2020）
# 与基础版本(g_cov-G)同源的per-reward梯度，仅用PCGrad替代直接相加

# 配置
SUCCESS_JSON="experiments/data/success_trajectories.json"
CONFIG="experiments/configs/exp2_goal_pcgrad.yaml"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验2：目标吸引Guidance（PCGrad）${NC}"
echo -e "${BLUE}PCGrad梯度手术组合多reward梯度${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验
python experiments/scripts/run_exp2_goal.py \
    --config "$CONFIG" \
    --success-json "$SUCCESS_JSON" \
    --save-steps

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验2 PCGrad版本完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp2_goal_pcgrad/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验2 PCGrad版本失败，请检查错误信息${NC}"
    exit 1
fi
