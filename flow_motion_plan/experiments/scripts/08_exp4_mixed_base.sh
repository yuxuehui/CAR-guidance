#!/bin/bash
# 实验4：混合能量场Guidance（静态+动态）基础版本（无优化）

# 配置
SUCCESS_JSON="experiments/data/success_trajectories.json"
CONFIG="experiments/configs/exp4_mixed_base.yaml"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验4：混合能量场Guidance（静态+动态）${NC}"
echo -e "${BLUE}基础版本（无优化）${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_exp4_mixed.py \
    --config "$CONFIG" \
    --success-json "$SUCCESS_JSON"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验4基础版本完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp4_mixed_base/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验4基础版本失败，请检查错误信息${NC}"
    exit 1
fi
