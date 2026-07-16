#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config
from experiments.experiments.task3_replay_visualize import Task3ReplayVisualize

def main():
    parser = argparse.ArgumentParser(description="运行 Task 3: 轨迹重放 + 可视化")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/03_task3_replay_visualize.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--replay_actions_path",
        type=str,
        default=None,
        help="重放动作数据路径（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="重放的演示数量（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=None,
        help="视频帧率（覆盖配置文件中的设置）"
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.replay_actions_path is not None:
        config.setdefault('data', {})['replay_actions_path'] = args.replay_actions_path
    if args.num_episodes is not None:
        config.setdefault('data', {})['num_test_episodes'] = args.num_episodes
    if args.video_fps is not None:
        config.setdefault('evaluation', {})['video_fps'] = args.video_fps

    experiment = Task3ReplayVisualize(config=config)

    experiment.run()

if __name__ == "__main__":
    main()
