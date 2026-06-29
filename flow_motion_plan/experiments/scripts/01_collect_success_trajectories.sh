#!/bin/bash
# 步骤1：收集成功轨迹
# 从训练好的Flow模型中收集指定数量的成功轨迹（无碰撞且到达终点）

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}步骤1：收集成功轨迹${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行收集脚本
python experiments/scripts/collect_success_trajectories.py \
    --checkpoint checkpoints/flow_model/best_checkpoint.pth \
    --sample-json data/sample_maze.json \
    --num-success-per-maze 5 \
    --max-attempts-per-maze 200 \
    --goal-tol 0.1

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 成功轨迹收集完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/data/base_model_images/success_trajectories.json${NC}"
else
    echo -e "\n${YELLOW}⚠️  收集失败，请检查错误信息${NC}"
    exit 1
fi

