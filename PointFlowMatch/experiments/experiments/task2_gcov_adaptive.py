import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import time

from .exp1_static_gcov import Exp1StaticGcov

class Task2GcovAdaptive(Exp1StaticGcov):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.all_actions = []

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始 Task 2: 自适应 G_Cov 引导 + 动作保存")
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

            demo_actions = self._result_to_serializable(result)
            self.all_actions.append(demo_actions)

            self.trajectories.append(result['trajectory'])
            self.success_flags.append(result['success'])
            self.execution_times.append(result['execution_time'])

        self.logger.info("\n步骤 4/5: 评估结果...")
        metrics = self.evaluate()

        self.logger.info("步骤 5/5: 保存结果...")
        self.save_results(metrics)

        if self.config.get('save_actions', False):
            actions_output_path = self.config.get('actions_output_path',
                                                   str(self.output_dir / 'actions.json'))
            actions_output_path = Path(actions_output_path)
            actions_output_path.parent.mkdir(parents=True, exist_ok=True)

            actions_data = {
                'experiment': 'Task2GcovAdaptive',
                'num_demos': len(self.all_actions),
                'actions': self.all_actions
            }

            with open(actions_output_path, 'w', encoding='utf-8') as f:
                json.dump(actions_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"动作序列已保存至: {actions_output_path}")

        self.logger.info("=" * 60)
        self.logger.info("Task 2 完成！")
        self.logger.info(f"成功率: {metrics.get('success_rate', 0):.2%}")
        self.logger.info(f"保存了 {len(self.all_actions)} 个演示的动作序列")
        self.logger.info("=" * 60)

        return metrics
