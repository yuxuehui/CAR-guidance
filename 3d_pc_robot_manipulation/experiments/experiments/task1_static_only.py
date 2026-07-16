import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import time

from .exp1_static import Exp1Static

class Task1StaticOnly(Exp1Static):

    def __init__(self, config: Dict[str, Any]):

        config.setdefault('guidance', {})['enable_dynamic_energy_field'] = False

        config.setdefault('guidance', {})['energy_scales'] = [0.0, 0.0]
        super().__init__(config)

        self.all_actions = []

    def run_inference(self, demo: Dict[str, Any]) -> Dict[str, Any]:

        result = super().run_inference(demo)

        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_list(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_list(value) for key, value in obj.items()}
            else:
                return obj

        demo_id = result['demo_id']
        seed = result['seed']

        demo_actions = {
            'demo_id': demo_id,
            'seed': seed,
            'success': result['success'],
            'step_count': result.get('steps', result.get('step_count', 0)),
            'execution_time': result['execution_time'],
            'trajectory': convert_to_list(result.get('trajectory', [])),
            'energy_centers': convert_to_list(result.get('energy_centers', [])),
            'energy_scales': convert_to_list(result.get('energy_scales', [])),
            'actions_sequence': convert_to_list(result.get('actions_sequence', [])),
            'robot_state_history': convert_to_list(result.get('robot_state_history', [])),
        }

        if 'video_path' in result:
            demo_actions['video_path'] = result['video_path']

        actions_output_dir = Path(self.config.get('output_dir', 'experiments/outputs/01_task1_static_only')) / "actions"
        actions_output_dir.mkdir(parents=True, exist_ok=True)
        action_file = actions_output_dir / f"demo_{demo_id:04d}_seed_{seed}.json"

        with open(action_file, 'w', encoding='utf-8') as f:
            json.dump(demo_actions, f, indent=2, ensure_ascii=False)

        self.logger.info(f"演示 {demo_id} 的动作序列已保存至: {action_file}")

        self.all_actions.append(demo_actions)

        return result

    def save_results(self, metrics: Dict[str, float]):

        super().save_results(metrics)

        actions_output_path = self.output_dir / 'actions_summary.json'

        actions_data = {
            'experiment': 'Task1StaticOnly',
            'num_demos': len(self.all_actions),
            'actions': self.all_actions
        }

        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_list(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_list(value) for key, value in obj.items()}
            else:
                return obj

        actions_data = convert_to_list(actions_data)

        with open(actions_output_path, 'w', encoding='utf-8') as f:
            json.dump(actions_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"动作序列汇总已保存至: {actions_output_path}")
        self.logger.info(f"每个demo的单独动作文件保存在: {self.output_dir / 'actions'}")

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始 Task 1: 静态能量场引导 + 动作保存")
        self.logger.info("=" * 60)

        self.logger.info("步骤 1/5: 设置实验...")
        self.setup()

        self.logger.info("步骤 2/5: 加载测试数据...")
        test_data = self.load_data()

        self.logger.info("步骤 3/5: 运行推理...")
        num_episodes = self.config.get('data', {}).get('num_test_episodes', len(test_data))

        results = []
        for i, demo in enumerate(test_data[:num_episodes]):
            self.logger.info(f"\n处理演示 {i+1}/{num_episodes} (demo_id={demo['demo_id']}, seed={demo['seed']})")

            result = self.run_inference(demo)
            results.append(result)

            trajectory = result['trajectory']
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            self.trajectories.append(trajectory)
            self.success_flags.append(result['success'])
            self.execution_times.append(result['execution_time'])

        self.logger.info("\n步骤 4/5: 评估结果...")
        metrics = self.evaluate()

        self.logger.info("步骤 5/5: 保存结果...")
        self.save_results(metrics)

        self.logger.info("=" * 60)
        self.logger.info("Task 1 完成！")
        self.logger.info(f"成功率: {metrics.get('success_rate', 0):.2%}")
        self.logger.info(f"保存了 {len(self.all_actions)} 个演示的动作序列")
        self.logger.info("=" * 60)

        return metrics
