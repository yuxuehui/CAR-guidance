#!/bin/bash
# 实验4：混合能量场Guidance（静态+动态）PCGrad梯度手术（Yu et al. 2020）
# 把静态+动态所有reward的per-reward梯度统一做PCGrad手术后相加

# 配置
SUCCESS_JSON="experiments/data/success_trajectories.json"
CONFIG="experiments/configs/exp4_mixed_pcgrad.yaml"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验4：混合能量场Guidance（静态+动态）${NC}"
echo -e "${BLUE}PCGrad梯度手术组合多reward梯度${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_exp4_mixed.py \
    --config "$CONFIG" \
    --success-json "$SUCCESS_JSON"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验4 PCGrad版本完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp4_mixed_pcgrad/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验4 PCGrad版本失败，请检查错误信息${NC}"
    exit 1
fi
