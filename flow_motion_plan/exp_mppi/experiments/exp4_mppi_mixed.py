from pathlib import Path
import numpy as np
import torch
import yaml
import json
from typing import Dict, List
import matplotlib.pyplot as plt

from ..core.mppi_flow_controller import MPPIFlowController
from experiments.core.evaluator import Evaluator
from experiments.experiments.exp1_static import select_energy_centers_from_base_traj
from experiments.experiments.exp4_mixed import create_dynamic_paths_random
from experiments.utils.inference import FlowModelInference
from experiments.utils.visualization import visualize_trajectory

class Exp4MPPIMixed:

    def __init__(self, model_checkpoint: str, config_path: str):

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("📦 加载Flow Matching模型...")
        self.inferencer = FlowModelInference(model_checkpoint, None)
        self.flow_model = self.inferencer.model

        print("🎯 创建MPPI控制器...")
        self.controller = MPPIFlowController(
            flow_model=self.flow_model,
            config=self.config['mppi']
        )

        self.evaluator = Evaluator(self.config['evaluation'])

        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ 实验4初始化完成")
        print(f"   输出目录: {self.output_dir}")

    def run(self, test_cases: List[Dict]):
        results = []

        print(f"\n{'='*70}")
        print(f"开始运行实验4：混合场景 (MPPI Baseline)")
        print(f"测试用例数: {len(test_cases)}")
        print(f"{'='*70}\n")

        for idx, case in enumerate(test_cases):
            start_pos = case['start_pos']
            goal_pos = case['goal_pos']
            walls = case['walls']
            case_id = case.get('case_id', idx)

            print(f"\n{'='*70}")
            print(f"测试用例 {case_id + 1}/{len(test_cases)}")
            print(f"起点: {start_pos}, 终点: {goal_pos}")
            print(f"{'='*70}")

            print("  [Step 1] 生成base轨迹用于选择静态能量中心...")
            base_seed = self.config.get('seed', 42)
            base_traj_seed = base_seed
            energy_seed = base_seed + 1

            torch.manual_seed(base_traj_seed)
            torch.cuda.manual_seed_all(base_traj_seed)
            np.random.seed(base_traj_seed)
            import random
            random.seed(base_traj_seed)

            base_traj, _ = self.inferencer.generate_trajectory(
                start_pos=start_pos,
                goal_pos=goal_pos,
                wall_positions=walls,
                num_steps=20,
                dt=0.05,
                num_samples=1,
                record_steps=False
            )
            base_traj = base_traj[0]

            print("  [Step 2] 选择静态能量中心...")
            num_static_centers = self.config.get('num_static_energy_centers', 2)

            random.seed(energy_seed)
            static_energy_centers = select_energy_centers_from_base_traj(
                base_traj=base_traj,
                walls=walls,
                num_centers=num_static_centers,
                seed=energy_seed
            )
            static_energy_scales = self.config.get('static_energy_scales',
                                                   [-1.0] * num_static_centers)

            print(f"      静态能量中心数量: {len(static_energy_centers)}")
            for i, center in enumerate(static_energy_centers):
                print(f"      中心{i+1}: {center}, 缩放: {static_energy_scales[i]}")

            print("  [Step 3] 创建动态路径...")
            use_random_paths = self.config.get('use_random_dynamic_paths', True)
            num_dynamic_paths = self.config.get('num_dynamic_paths', 1)

            if use_random_paths:

                path_seed = base_seed + 100
                offset_range = self.config.get('dynamic_path_offset_range', (0.3, 0.8))
                num_control_points = self.config.get('num_dynamic_control_points', 3)
                fully_random = self.config.get('fully_random_dynamic_paths', True)

                dynamic_paths = create_dynamic_paths_random(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    base_traj=base_traj,
                    walls=walls,
                    num_paths=num_dynamic_paths,
                    offset_range=offset_range,
                    num_control_points=num_control_points,
                    maze_size=(5.0, 5.0),
                    seed=path_seed,
                    fully_random=fully_random
                )
            else:

                from experiments.experiments.exp4_mixed import create_dynamic_paths_linear
                path_offset = self.config.get('dynamic_path_offset', 0.5)
                dynamic_paths = create_dynamic_paths_linear(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    offset=path_offset,
                    num_paths=num_dynamic_paths,
                    maze_size=(5.0, 5.0)
                )

            print(f"      动态路径数量: {len(dynamic_paths)}")
            for i, path_fn in enumerate(dynamic_paths):
                path_start = path_fn(0.0)
                path_end = path_fn(1.0)
                print(f"      路径{i+1}: 起点{path_start} -> 终点{path_end}")

            print("  [Step 4] MPPI优化轨迹（静态+动态）...")
            trajectory = self.controller.generate_trajectory(
                start_pos=start_pos,
                goal_pos=goal_pos,
                walls=walls,
                energy_centers=static_energy_centers,
                energy_scales=static_energy_scales,
                dynamic_paths=dynamic_paths,
                verbose=True
            )

            print("  [Step 5] 评估轨迹...")
            metrics = self.evaluator.evaluate(
                trajectory=trajectory,
                goal_pos=goal_pos,
                walls=walls
            )

            print(f"  📊 评估结果:")
            print(f"      成功: {metrics['success']}")
            print(f"      到达目标距离: {metrics['goal_distance']:.4f}")
            print(f"      路径长度: {metrics['path_length']:.4f}")
            print(f"      平滑度: {metrics['smoothness']:.4f}")
            print(f"      碰撞: {metrics['collision']}")

            print("  [Step 6] 可视化...")
            fig = visualize_trajectory(
                trajectory=trajectory,
                start_pos=start_pos,
                goal_pos=goal_pos,
                wall_positions=walls,
                energy_centers=static_energy_centers,
                path_functions=dynamic_paths,
                title=f"MPPI Exp4 - Case {case_id}"
            )
            save_path = self.output_dir / f"case_{case_id}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"      保存到: {save_path}")

            results.append({
                'case_id': case_id,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'walls': walls,
                'static_energy_centers': static_energy_centers,
                'static_energy_scales': static_energy_scales,
                'num_dynamic_paths': len(dynamic_paths),
                'trajectory': trajectory.tolist(),
                'metrics': metrics
            })

        self._save_results(results)

        self._print_summary(results)

        return results

    def _save_results(self, results: List[Dict]):
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        total = len(results)
        if total > 0:
            success_count = sum(1 for r in results if r['metrics']['success'])
            collision_count = sum(1 for r in results if r['metrics']['collision'])
            no_collision_count = total - collision_count
            perfect_count = sum(1 for r in results if r['metrics']['success'] and not r['metrics']['collision'])
            horizon = self.config.get('mppi', {}).get('horizon', 40)
            summary = {
                'total': total,
                'reached_goal_count': success_count,
                'no_collision_count': no_collision_count,
                'perfect_count': perfect_count,
                'trajectory_num_points': horizon,
            }
        else:
            summary = {'total': 0, 'reached_goal_count': 0, 'no_collision_count': 0, 'perfect_count': 0, 'trajectory_num_points': self.config.get('mppi', {}).get('horizon', 40)}

        results_converted = convert_types(results)
        out = {'summary': summary, 'results': results_converted}

        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n💾 结果已保存到: {results_file}")

    def _print_summary(self, results: List[Dict]):
        total_count = len(results)
        if total_count == 0:
            print("\n📊 无结果可统计。\n")
            return

        success_count = sum(1 for r in results if r['metrics']['success'])
        collision_count = sum(1 for r in results if r['metrics']['collision'])
        no_collision_count = total_count - collision_count
        perfect_count = sum(1 for r in results if r['metrics']['success'] and not r['metrics']['collision'])
        horizon = self.config.get('mppi', {}).get('horizon', 40)

        print(f"\n{'='*70}")
        print("📊 实验4汇总统计 (MPPI Baseline)")
        print(f"{'='*70}")
        print(f"总轨迹数: {total_count}")
        print(f"到达终点: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"无碰撞:   {no_collision_count}/{total_count} ({no_collision_count/total_count*100:.1f}%)")
        print(f"完美轨迹 (无碰撞且到达终点): {perfect_count}/{total_count} ({perfect_count/total_count*100:.1f}%)")
        print(f"{'='*70}")
        print(f"每条轨迹点数: {horizon} (由配置 mppi.horizon 决定)")
        print(f"{'='*70}")

        avg_path_length = np.mean([r['metrics']['path_length'] for r in results])
        avg_smoothness = np.mean([r['metrics']['smoothness'] for r in results])
        print(f"平均路径长度: {avg_path_length:.4f}")
        print(f"平均平滑度: {avg_smoothness:.4f}")
        print(f"{'='*70}\n")
