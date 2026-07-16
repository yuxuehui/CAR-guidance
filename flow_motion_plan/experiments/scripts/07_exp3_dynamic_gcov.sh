#!/bin/bash
# 实验3：动态障碍物Guidance + g_cov_a_gm_online（优化版本，加guidance加方法）

# 配置
SUCCESS_JSON="experiments/data/success_trajectories.json"
CONFIG="experiments/configs/exp3_dynamic_gcov.yaml"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}实验3：动态障碍物Guidance + g_cov_a_gm_online${NC}"
echo -e "${BLUE}加guidance加方法（优化版本）${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行实验
python experiments/scripts/run_exp3_dynamic.py \
    --config "$CONFIG" \
    --success-json "$SUCCESS_JSON"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 实验3优化版本完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/outputs/exp3_dynamic_gcov/${NC}"
else
    echo -e "\n${YELLOW}⚠️  实验3优化版本失败，请检查错误信息${NC}"
    exit 1
fi

