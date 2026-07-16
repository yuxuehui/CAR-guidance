#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from experiments_stack_cube.experiments import Task3GcovFull

def main():
    parser = argparse.ArgumentParser(description="运行 Task 3: 加能量场和方法")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="指定测试使用的种子列表（逗号分隔）")
    args = parser.parse_args()

    config_path = project_root / "experiments_stack_cube/configs/03_task3_gcov_full.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.seeds is not None:
        fixed_seeds = [int(s.strip()) for s in args.seeds.split(',')]
        config.setdefault('data', {})['fixed_seeds'] = fixed_seeds

    print(f"加载配置: {config_path}")
    print(f"实验名称: {config['experiment_name']}")

    experiment = Task3GcovFull(config)
    metrics = experiment.run()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print(f"成功率: {metrics.get('success_rate', 0):.2%}")
    print(f"平均抖动: {metrics.get('average_jerk', 0):.4f}")
    print(f"最小障碍物距离: {metrics.get('min_obstacle_distance', 0):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
