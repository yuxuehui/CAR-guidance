#!/usr/bin/env python3

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import random
import torch
from datetime import datetime

from experiments.utils.data_loader import load_success_trajectories
from exp_mppi.experiments.exp2_mppi_goal import Exp2MPPIGoal

def get_test_cases(success_json: str, num_cases: int = None):
    raw = load_success_trajectories(success_json)
    if num_cases is not None:
        raw = raw[:num_cases]
    test_cases = []
    for idx, case in enumerate(raw):
        test_cases.append({
            'case_id': idx,
            'start_pos': case['start'],
            'goal_pos': case['goal'],
            'walls': case['walls'],
        })
    return test_cases

def main():
    parser = argparse.ArgumentParser(description='运行MPPI实验2：目标吸引场干扰')
    parser.add_argument('--model', type=str,
                       default='checkpoints/flow_model/best_checkpoint.pth',
                       help='Flow模型检查点路径')
    parser.add_argument('--config', type=str,
                       default='exp_mppi/configs/mppi_exp2_goal.yaml',
                       help='MPPI配置文件路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--success-json', type=str,
                       default='experiments/data/success_trajectories.json',
                       help='成功轨迹 JSON（与 experiments/run_exp2_goal 一致，保证同一 demo）')
    parser.add_argument('--num-cases', type=int, default=None,
                       help='使用的 demo 数量，None 表示全部')

    args = parser.parse_args()

    log_dir = Path(project_root) / 'exp_mppi' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'mppi_exp2_goal_{timestamp}.txt'

    start_time = datetime.now()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("="*70)
    print("MPPI实验2：目标吸引场干扰 Baseline")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"配置: {args.config}")
    print(f"随机种子: {args.seed}")
    print(f"Demo 数据: {args.success_json}")
    if args.num_cases is not None:
        print(f"使用 demo 数: {args.num_cases}")
    print("="*70)

    experiment = Exp2MPPIGoal(
        model_checkpoint=args.model,
        config_path=args.config
    )

    test_cases = get_test_cases(args.success_json, args.num_cases)

    results = experiment.run(test_cases)

    end_time = datetime.now()
    duration = end_time - start_time

    total_count = len(results)
    if total_count > 0:
        success_count = sum(1 for r in results if r['metrics']['success'])
        collision_count = sum(1 for r in results if r['metrics']['collision'])
        no_collision_count = total_count - collision_count
        perfect_count = sum(1 for r in results if r['metrics']['success'] and not r['metrics']['collision'])
        avg_path_length = np.mean([r['metrics']['path_length'] for r in results])
        avg_smoothness = np.mean([r['metrics']['smoothness'] for r in results])
    else:
        success_count = no_collision_count = perfect_count = 0
        avg_path_length = avg_smoothness = 0.0

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MPPI实验2：目标吸引场干扰 Baseline\n")
        f.write("="*70 + "\n\n")

        f.write("实验配置:\n")
        f.write(f"  模型路径: {args.model}\n")
        f.write(f"  配置文件: {args.config}\n")
        f.write(f"  随机种子: {args.seed}\n")
        f.write(f"  Demo数据: {args.success_json}\n")
        f.write(f"  测试用例数: {args.num_cases if args.num_cases else '全部'}\n\n")

        f.write("实验时间:\n")
        f.write(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  运行时长: {duration}\n\n")

        f.write("实验结果统计:\n")
        f.write(f"  总Demo数量: {total_count}\n")
        f.write(f"  未碰撞Demo数量: {no_collision_count}/{total_count} ({no_collision_count/total_count*100:.1f}%)\n")
        f.write(f"  到达终点Demo数量: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)\n")
        f.write(f"  完美轨迹数量 (无碰撞且到达终点): {perfect_count}/{total_count} ({perfect_count/total_count*100:.1f}%)\n")
        f.write(f"  平均路径长度: {avg_path_length:.4f}\n")
        f.write(f"  平均平滑度: {avg_smoothness:.4f}\n\n")

        f.write(f"结果保存路径: {experiment.output_dir}\n")
        f.write(f"日志保存路径: {log_file}\n")
        f.write("="*70 + "\n")

    print("\n✅ 实验2完成！")
    print(f"结果保存在: {experiment.output_dir}")
    print(f"日志保存在: {log_file}")

if __name__ == '__main__':
    main()
