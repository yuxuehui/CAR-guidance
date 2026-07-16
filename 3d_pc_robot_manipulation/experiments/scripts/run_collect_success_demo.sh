#!/bin/bash
# 收集成功演示数据

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}收集成功演示数据${NC}"
echo -e "${BLUE}========================================${NC}"

# 运行收集脚本
# 参数说明：
#   --save_video: 保存视频（默认: False）
#   --visualize: 可视化显示（默认: False，与 save_video 互斥）
#   --num_actions_per_step: 每次推理执行多少个预测动作（默认: 32）
python experiments/scripts/collect_success_demo.py \
    --ckpt_name maniskill_train_pcd_from_three_cameras_more_gripper \
    --ckpt_episode ep1500-ba160500 \
    --num_success 10 \
    --max_attempts 500 \
    --max_steps 100 \
    --output_dir experiments/data/success_demos \
    --seed_start 0 \
    --save_video True \
    --visualize False \
    --num_actions_per_step 32

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ 成功演示收集完成！${NC}"
    echo -e "${BLUE}结果保存在: experiments/data/success_demos/success_demos.json${NC}"
else
    echo -e "\n${YELLOW}⚠️  收集失败，请检查错误信息${NC}"
    exit 1
fi

