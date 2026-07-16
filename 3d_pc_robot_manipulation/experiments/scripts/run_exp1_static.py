#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config
from experiments.experiments.exp1_static import Exp1Static

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="运行实验1：静态能量场引导")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/exp1_static.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--save_videos",
        type=str2bool,
        default=None,
        help="是否保存视频（覆盖配置文件中的设置）"
    )
    parser.add_argument(
        "--visualize",
        type=str2bool,
        default=None,
        help="是否可视化显示（覆盖配置文件中的设置，与 save_videos 互斥）"
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.save_videos is not None:
        config.setdefault('evaluation', {})['save_videos'] = args.save_videos
    if args.visualize is not None:
        config.setdefault('evaluation', {})['visualize'] = args.visualize

    if config.get('evaluation', {}).get('save_videos', False) and config.get('evaluation', {}).get('visualize', False):
        print("⚠️  警告: save_videos 和 visualize 不能同时为 True，优先使用 visualize")
        config['evaluation']['save_videos'] = False

    experiment = Exp1Static(config=config)

    experiment.run()

if __name__ == "__main__":
    main()
