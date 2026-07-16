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
from experiments.utils.inference import FlowModelInference
from experiments.utils.visualization import visualize_trajectory
import random

class Exp1MPPIStatic:

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

        print(f"✅ 实验1初始化完成")
        print(f"   输出目录: {self.output_dir}")

    def run(self, test_cases: List[Dict]):
        results = []

        print(f"\n{'='*70}")
        print(f"开始运行实验1：静态能量场 (MPPI Baseline)")
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

            print("  [Step 1] 生成base轨迹用于选择能量中心...")
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

            print("  [Step 2] 选择能量中心...")
            num_centers = self.config.get('num_energy_centers', 2)

            random.seed(energy_seed)
            energy_centers = select_energy_centers_from_base_traj(
                base_traj=base_traj,
                walls=walls,
                num_centers=num_centers,
                seed=energy_seed
            )
            energy_scales = self.config.get('energy_scales', [-1.0] * num_centers)

            print(f"      能量中心数量: {len(energy_centers)}")
            for i, center in enumerate(energy_centers):
                print(f"      中心{i+1}: {center}, 缩放: {energy_scales[i]}")

            print("  [Step 3] MPPI优化轨迹...")
            trajectory = self.controller.generate_trajectory(
                start_pos=start_pos,
                goal_pos=goal_pos,
                walls=walls,
                energy_centers=energy_centers,
                energy_scales=energy_scales,
                dynamic_paths=None,
                verbose=True
            )

            print("  [Step 4] 评估轨迹...")
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

            print("  [Step 5] 可视化...")
            fig = visualize_trajectory(
                trajectory=trajectory,
                start_pos=start_pos,
                goal_pos=goal_pos,
                wall_positions=walls,
                energy_centers=energy_centers,
                title=f"MPPI Exp1 - Case {case_id}"
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
                'energy_centers': energy_centers,
                'energy_scales': energy_scales,
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
        print("📊 实验1汇总统计 (MPPI Baseline)")
        print(f"{'='*70}")
        print(f"总轨迹数: {total_count}")
        print(f"到达终点: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"无碰撞:   {no_collision_count}/{total_count} ({no_collision_count/total_count*100:.1f}%)")
        print(f"完美轨迹 (无碰撞且到达终点): {perfect_count}/{total_count} ({perfect_count/total_count*100:.1f}%)")
        print(f"{'='*70}")
        print(f"每条轨迹点数: {horizon} (由配置 mppi.horizon 决定)")
        print(f"  图中看起来点少的原因: 轨迹是 {horizon} 个点依次连成的折线，")
        print(f"  可视化只画了连线、未单独画每个点的标记，所以看起来像一条连续线。")
        print(f"{'='*70}")

        avg_path_length = np.mean([r['metrics']['path_length'] for r in results])
        avg_smoothness = np.mean([r['metrics']['smoothness'] for r in results])
        print(f"平均路径长度: {avg_path_length:.4f}")
        print(f"平均平滑度: {avg_smoothness:.4f}")
        print(f"{'='*70}\n")
