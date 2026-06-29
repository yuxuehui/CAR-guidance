#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.experiments.baseline import BaselineExperiment
from experiments.core.config_loader import load_config
from experiments.utils.data_loader import load_success_trajectories

def main():
    parser = argparse.ArgumentParser(description='运行Baseline实验（无guidance）')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/baseline_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--success-json',
        type=str,
        default='experiments/data/success_trajectories.json',
        help='成功轨迹JSON文件路径'
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=None,
        help='使用的测试用例数量（None表示全部）'
    )

    args = parser.parse_args()

    config = load_config(args.config)

    test_cases = load_success_trajectories(args.success_json)
    if args.num_cases is not None:
        test_cases = test_cases[:args.num_cases]

    exp = BaselineExperiment(
        model_checkpoint=config['model']['checkpoint_path'],
        config=config
    )

    results = exp.run(test_cases)

    print("\n✅ 实验完成！")
    return results

if __name__ == '__main__':
    main()
