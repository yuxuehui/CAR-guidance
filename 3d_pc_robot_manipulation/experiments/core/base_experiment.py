from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import numpy as np
import json
from datetime import datetime
import logging

from .evaluator import Evaluator
from .config_loader import load_config

class BaseExperiment(ABC):

    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        experiment_name = self.config.get('experiment_name', 'experiment')
        self.output_dir = Path(self.config.get('output', {}).get('base_dir', 'experiments/outputs')) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        eval_config = self.config.get('evaluation', {})
        self.evaluator = Evaluator(eval_config)

        self.trajectories = []
        self.success_flags = []
        self.execution_times = []

        self.policy = None
        self.guidance = None

    def _setup_logging(self):
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"实验输出目录: {self.output_dir}")

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    def run_inference(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        import time
        start_time = time.time()

        if 'goal_pos' not in obs_dict and 'goal' in obs_dict:

            goal = obs_dict['goal']
            if goal.dim() == 2:
                n_obs_steps = obs_dict['robot_state'].shape[1]
                goal_pos = goal.unsqueeze(1).repeat(1, n_obs_steps, 1)
                obs_dict['goal_pos'] = goal_pos

        action = self.policy.predict(obs_dict)

        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)

        return action

    def evaluate(self) -> Dict[str, float]:
        if len(self.trajectories) == 0:
            self.logger.warning("没有轨迹数据，无法评估")
            return {}

        metrics = self.evaluator.compute_metrics(
            trajectories=self.trajectories,
            success_flags=self.success_flags,
            execution_times=self.execution_times if self.execution_times else None
        )

        self.logger.info("评估指标:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

        return metrics

    def save_results(self, metrics: Dict[str, float]):

        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if self.config.get('evaluation', {}).get('save_trajectories', False):
            traj_file = self.output_dir / "trajectories.npz"
            np.savez_compressed(
                traj_file,
                trajectories=np.array(self.trajectories, dtype=object),
                success_flags=np.array(self.success_flags),
                execution_times=np.array(self.execution_times) if self.execution_times else None
            )

        self.logger.info(f"结果已保存到: {self.output_dir}")

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info(f"开始实验: {self.__class__.__name__}")
        self.logger.info("=" * 60)

        self.logger.info("步骤 1/5: 设置实验...")
        self.setup()

        self.logger.info("步骤 2/5: 加载测试数据...")
        test_data = self.load_data()

        self.logger.info("步骤 3/5: 运行推理...")
        num_episodes = self.config.get('data', {}).get('num_test_episodes', 10)
        for i, episode_data in enumerate(test_data[:num_episodes]):
            self.logger.info(f"处理 Episode {i+1}/{num_episodes}")

            action = self.run_inference(episode_data['obs'])

            self.trajectories.append(action)
            self.success_flags.append(episode_data.get('success', True))

        self.logger.info("步骤 4/5: 评估结果...")
        metrics = self.evaluate()

        self.logger.info("步骤 5/5: 保存结果...")
        self.save_results(metrics)

        self.logger.info("=" * 60)
        self.logger.info("实验完成！")
        self.logger.info("=" * 60)

        return metrics
