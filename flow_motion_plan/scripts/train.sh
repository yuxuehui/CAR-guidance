#!/bin/bash
set -e

# 项目根目录（脚本位于 <root>/scripts/）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

CONFIG_PATH="$ROOT/configs/train_flow_config.json"

echo "🚀 开始训练流模型..."
echo "配置文件: $CONFIG_PATH"

cd "$ROOT"
python diffuser/train_flow_model.py --config "$CONFIG_PATH"

echo "✅ 训练完成！"
