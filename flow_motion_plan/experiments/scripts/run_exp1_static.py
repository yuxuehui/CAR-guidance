#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.experiments.exp1_static import Exp1Static
from experiments.core.config_loader import load_config
from experiments.utils.data_loader import load_success_trajectories
from experiments.utils.visualization import visualize_trajectory, visualize_step_by_step
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='运行实验1：静态障碍物Guidance + g_cov_a_gm_online')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/exp1_static_gcov.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--success-json',
        type=str,
        default='experiments/data/base_model_images/success_trajectories.json',
        help='成功轨迹JSON文件路径'
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=None,
        help='使用的测试用例数量（None表示全部）'
    )
    parser.add_argument(
        '--save-steps',
        action='store_true',
        default=False,
        help='是否保存步骤可视化图（step-by-step图）'
    )

    args = parser.parse_args()

    config = load_config(args.config)

    test_cases = load_success_trajectories(args.success_json)
    if args.num_cases is not None:
        test_cases = test_cases[:args.num_cases]

    exp = Exp1Static(
        model_checkpoint=config['model']['checkpoint_path'],
        config=config
    )

    original_generate = exp.generate_trajectory
    def generate_with_steps(*args, **kwargs):
        kwargs['record_steps'] = True
        return original_generate(*args, **kwargs)
    exp.generate_trajectory = generate_with_steps

    output_dir = Path(exp.output_dir) / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(exp.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    log_fp = open(log_file, 'w', encoding='utf-8')
    log_fp.write(f"实验日志 - {exp.__class__.__name__}\n")
    log_fp.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fp.write(f"配置文件: {args.config}\n")
    log_fp.write(f"测试用例数量: {len(test_cases)}\n")
    log_fp.write("=" * 80 + "\n\n")

    stats = {
        'total': 0,
        'reached_goal': 0,
        'no_collision': 0,
        'perfect': 0
    }

    original_run = exp.run
    def run_with_immediate_save(test_cases):
        all_results = []

        for idx, case in enumerate(test_cases):
            print(f"\n处理测试用例 {idx + 1}/{len(test_cases)}")
            log_fp.write(f"测试用例 {idx + 1}/{len(test_cases)}\n")
            log_fp.write(f"  起点: {case['start']}\n")
            log_fp.write(f"  终点: {case['goal']}\n")

            start_pos = case['start']
            goal_pos = case['goal']
            walls = case['walls']

            try:

                trajectories = exp.generate_trajectory(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    wall_positions=walls,
                    num_samples=1,
                    **case.get('extra_params', {})
                )

                traj = trajectories[0]

                traj = np.asarray(traj, dtype=np.float64)

                goal_array = np.array(goal_pos, dtype=np.float64)
                walls_array = np.array(walls, dtype=np.float64)
                eval_result = exp.evaluator.evaluate_trajectory(
                    traj, goal_array, walls_array
                )

                goal_tol = config.get('evaluation', {}).get('goal_tolerance', 0.3)
                manual_reached = False
                for point in traj:
                    dist = np.linalg.norm(point - goal_array)
                    if dist <= goal_tol:
                        manual_reached = True
                        break

                stats['total'] += 1
                if eval_result['reached_goal']:
                    stats['reached_goal'] += 1
                if not eval_result['has_collision']:
                    stats['no_collision'] += 1
                if eval_result['is_perfect']:
                    stats['perfect'] += 1

                log_fp.write(f"  到达终点: {'是' if eval_result['reached_goal'] else '否'}\n")
                log_fp.write(f"  碰撞检测: {'有碰撞' if eval_result['has_collision'] else '无碰撞'}\n")
                log_fp.write(f"  完美轨迹: {'是' if eval_result['is_perfect'] else '否'}\n")
                if eval_result['reached_goal'] != manual_reached:
                    log_fp.write(f"  ⚠️ 警告: 评估结果与手动检查不一致 (评估: {eval_result['reached_goal']}, 手动: {manual_reached})\n")
                log_fp.write("\n")
                log_fp.flush()

                result = {
                    'case_idx': idx,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'walls': walls,
                    'trajectory': traj.tolist(),
                    **eval_result
                }

                if hasattr(exp, '_last_energy_centers') and exp._last_energy_centers is not None:
                    result['energy_centers'] = exp._last_energy_centers

                all_results.append(result)

                if 'trajectory' in result and 'error' not in result:
                    image_path = output_dir / f"demo_{idx+1:04d}.png"

                    energy_centers = None
                    if 'energy_centers' in result:
                        energy_centers = result['energy_centers']

                    visualize_trajectory(
                        trajectory=traj,
                        start_pos=start_pos,
                        goal_pos=goal_pos,
                        wall_positions=walls,
                        energy_centers=energy_centers,
                        save_path=str(image_path),
                        show=False,
                        title=f"Exp1: Static Guidance (Demo {idx+1})",
                        goal_tol=config.get('evaluation', {}).get('goal_tolerance', 0.3)
                    )

                    if args.save_steps:
                        if hasattr(exp.model, 'guide_model') and hasattr(exp.model.guide_model, 'step_data'):
                            step_data = exp.model.guide_model.step_data
                            if step_data and hasattr(exp, '_last_traj_normalizer'):
                                steps_image_path = output_dir / f"demo_{idx+1:04d}_steps.png"
                                visualize_step_by_step(
                                    start_pos=start_pos,
                                    goal_pos=goal_pos,
                                    wall_positions=walls,
                                    step_data=step_data,
                                    traj_normalizer=exp._last_traj_normalizer,
                                    energy_centers=energy_centers,
                                    save_path=str(steps_image_path),
                                    show=False,
                                    title=f"Exp1: Static Guidance - Step-by-Step (Demo {idx+1})"
                                )

                                exp.model.guide_model.step_data = []
                                print(f"✅ 已保存步骤图: {steps_image_path}")

                    print(f"✅ 已保存: {image_path}")

            except Exception as e:
                import traceback
                print(f"❌ 处理测试用例 {idx + 1} 时出错: {e}")
                traceback.print_exc()
                log_fp.write(f"  错误: {str(e)}\n")
                log_fp.write(f"  错误类型: {type(e)}\n")
                log_fp.write(f"  详细堆栈:\n{traceback.format_exc()}\n\n")
                log_fp.flush()
                result = {
                    'case_idx': idx,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'walls': walls,
                    'error': str(e)
                }
                all_results.append(result)

        log_fp.write("=" * 80 + "\n")
        log_fp.write("实验统计\n")
        log_fp.write("=" * 80 + "\n")
        log_fp.write(f"总测试用例数: {stats['total']}\n")
        if stats['total'] > 0:
            log_fp.write(f"到达终点数量: {stats['reached_goal']} ({stats['reached_goal']/stats['total']*100:.2f}%)\n")
            log_fp.write(f"无碰撞数量: {stats['no_collision']} ({stats['no_collision']/stats['total']*100:.2f}%)\n")
            log_fp.write(f"完美轨迹数量: {stats['perfect']} ({stats['perfect']/stats['total']*100:.2f}%)\n")
        else:
            log_fp.write("到达终点数量: 0 (0.00%)\n")
            log_fp.write("无碰撞数量: 0 (0.00%)\n")
            log_fp.write("完美轨迹数量: 0 (0.00%)\n")
        log_fp.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_fp.close()

        summary = {
            'experiment_name': exp.__class__.__name__,
            'model_checkpoint': exp.model_checkpoint,
            'config': exp.config,
            'results': all_results,
            'statistics': stats
        }
        return summary

    exp.run = run_with_immediate_save
    results = exp.run(test_cases)

    print("\n" + "=" * 80)
    print("实验统计")
    print("=" * 80)
    print(f"总测试用例数: {stats['total']}")
    if stats['total'] > 0:
        print(f"到达终点数量: {stats['reached_goal']} ({stats['reached_goal']/stats['total']*100:.2f}%)")
        print(f"无碰撞数量: {stats['no_collision']} ({stats['no_collision']/stats['total']*100:.2f}%)")
        print(f"完美轨迹数量: {stats['perfect']} ({stats['perfect']/stats['total']*100:.2f}%)")
    else:
        print("到达终点数量: 0 (0.00%)")
        print("无碰撞数量: 0 (0.00%)")
        print("完美轨迹数量: 0 (0.00%)")
    print(f"\n✅ 实验完成！")
    print(f"   图片保存在: {output_dir}")
    print(f"   日志保存在: {log_file}")
    return results

if __name__ == '__main__':
    main()
