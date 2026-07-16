#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
from experiments_stack_cube.experiments import Task1StaticOnly

def main():

    config_path = project_root / "experiments_stack_cube/configs/01_task1_static_only.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"加载配置: {config_path}")
    print(f"实验名称: {config['experiment_name']}")

    experiment = Task1StaticOnly(config)
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
