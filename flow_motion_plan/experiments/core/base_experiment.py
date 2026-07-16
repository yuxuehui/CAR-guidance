from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import numpy as np
import json
from datetime import datetime
import logging

from .evaluator import Evaluator
from .config_loader import load_config

class BaseExperiment(ABC):

    def __init__(self,
                 model_checkpoint: str,
                 config: Dict[str, Any]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model_checkpoint = model_checkpoint

        self.inferencer = self._load_model(model_checkpoint)
        self.model = self.inferencer.model

        self.guidance = self._create_guidance()

        eval_config = self.config.get('evaluation', {})
        self.evaluator = Evaluator(eval_config)

        self.output_dir = Path(self.config.get('output_dir', 'experiments/results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

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

    def _load_model(self, checkpoint_path: str):
        from ..utils.inference import FlowModelInference

        config_path = self.config.get('model_config')

        inferencer = FlowModelInference(checkpoint_path, config_path)
        return inferencer

    @abstractmethod
    def _create_guidance(self):
        pass

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _normalize_inputs(self, start_pos: List[float], goal_pos: List[float],
                         wall_positions: List[List[float]]) -> Tuple:
        from ..utils.normalization import normalize_trajectory_inputs
        return normalize_trajectory_inputs(start_pos, goal_pos, wall_positions)

    def generate_trajectory(self,
                           start_pos: List[float],
                           goal_pos: List[float],
                           wall_positions: List[List[float]],
                           num_samples: int = 1,
                           **kwargs) -> np.ndarray:

        norm_start, norm_goal, norm_walls, traj_normalizer = self._normalize_inputs(
            start_pos, goal_pos, wall_positions
        )

        self._last_traj_normalizer = traj_normalizer

        if len(wall_positions) > 6:
            wall_positions = wall_positions[:6]
        elif len(wall_positions) < 6:
            wall_positions = wall_positions + [[0, 0]] * (6 - len(wall_positions))

        start_tensor = torch.FloatTensor(norm_start).to(self.device)
        goal_tensor = torch.FloatTensor(norm_goal).to(self.device)
        walls_tensor = torch.FloatTensor(norm_walls).to(self.device)

        horizon = self.model.horizon if hasattr(self.model, 'horizon') else 40
        conditions = {
            0: start_tensor.unsqueeze(0).repeat(num_samples, 1),
            horizon - 1: goal_tensor.unsqueeze(0).repeat(num_samples, 1)
        }
        wall_locations = walls_tensor.unsqueeze(0).repeat(num_samples, 1, 1)

        use_energy_guide = (self.guidance is not None) or (hasattr(self.model, 'guide_model') and self.model.guide_model is not None)

        num_steps = self.config.get('inference', {}).get('num_steps', 20)
        record_steps = kwargs.get('record_steps', False)
        with torch.no_grad():
            trajectories = self.model.sample_trajectory(
                conditions=conditions,
                wall_locations=wall_locations,
                num_steps=num_steps,
                energy_guide=use_energy_guide,
                energy_function=None,
                energy_scale=None,
                record_steps=record_steps
            )

        trajectories_np = trajectories.cpu().numpy()

        if len(trajectories_np.shape) == 4:

            trajectories_np = trajectories_np.squeeze(0)

        unnorm_trajectories = []

        for i in range(num_samples):
            traj = trajectories_np[i]

            if len(traj.shape) == 3:

                if traj.shape[0] == 1:
                    traj = traj.squeeze(0)
                else:
                    raise ValueError(f"Unexpected 3D trajectory shape: {traj.shape}, expected [1, horizon, state_dim]")

            if len(traj.shape) == 2:

                pos_traj = traj[:, :2]
            else:
                raise ValueError(f"Unexpected trajectory shape after processing: {traj.shape}, expected [horizon, state_dim]")
            unnorm_traj = traj_normalizer.unnormalize(pos_traj)

            if len(unnorm_traj.shape) == 3 and unnorm_traj.shape[0] == 1:
                unnorm_traj = unnorm_traj.squeeze(0)
            elif len(unnorm_traj.shape) == 1:

                if unnorm_traj.shape[0] == 2:
                    unnorm_traj = unnorm_traj.reshape(1, 2)

            unnorm_trajectories.append(unnorm_traj)

        return np.array(unnorm_trajectories)

    def run(self, test_cases: List[Dict]) -> Dict[str, Any]:
        self.logger.info("=" * 60)
        self.logger.info(f"开始运行实验: {self.__class__.__name__}")
        self.logger.info(f"测试用例数量: {len(test_cases)}")
        self.logger.info("=" * 60)

        all_results = []
        trajectories_list = []
        goals_list = []
        walls_list = []

        for idx, case in enumerate(test_cases):
            self.logger.info(f"\n处理测试用例 {idx + 1}/{len(test_cases)}")

            start_pos = case['start']
            goal_pos = case['goal']
            walls = case['walls']

            try:

                trajectories = self.generate_trajectory(
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    wall_positions=walls,
                    num_samples=1,
                    **case.get('extra_params', {})
                )

                traj = trajectories[0]

                eval_result = self.evaluator.evaluate_trajectory(
                    traj, np.array(goal_pos), np.array(walls)
                )

                result = {
                    'case_idx': idx,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'walls': walls,
                    'trajectory': traj.tolist(),
                    **eval_result
                }

                if hasattr(self, '_last_energy_centers') and self._last_energy_centers is not None:
                    result['energy_centers'] = self._last_energy_centers

                all_results.append(result)

                trajectories_list.append(traj)
                goals_list.append(np.array(goal_pos))
                walls_list.append(np.array(walls))

                status = []
                if eval_result['reached_goal']:
                    status.append("到达终点")
                else:
                    status.append("未到达终点")
                if eval_result['has_collision']:
                    status.append("有碰撞")
                else:
                    status.append("无碰撞")

                self.logger.info(f"  {' | '.join(status)}")

            except Exception as e:
                self.logger.error(f"  用例 {idx + 1} 失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

                all_results.append({
                    'case_idx': idx,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'walls': walls,
                    'error': str(e)
                })

        batch_stats = self.evaluator.evaluate_batch(
            trajectories_list, goals_list, walls_list
        )

        summary = {
            'experiment_name': self.__class__.__name__,
            'model_checkpoint': self.model_checkpoint,
            'config': self.config,
            'batch_stats': batch_stats,
            'results': all_results
        }

        summary = self._convert_numpy_types(summary)

        summary_path = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("实验完成")
        self.logger.info("=" * 60)
        self.logger.info(f"总用例数: {batch_stats['total']}")
        self.logger.info(f"到达终点: {batch_stats['reached_goal_count']} ({batch_stats['reached_goal_rate']:.2%})")
        self.logger.info(f"无碰撞: {batch_stats['collision_free_count']} ({batch_stats['collision_free_rate']:.2%})")
        self.logger.info(f"完美轨迹: {batch_stats['perfect_count']} ({batch_stats['perfect_rate']:.2%})")
        self.logger.info(f"汇总结果: {summary_path}")

        return summary
