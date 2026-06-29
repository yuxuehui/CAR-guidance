import torch
import numpy as np
import json
from typing import List, Optional, Tuple, Dict

from diffuser.models.flow_guide import TrajFlowModel
from diffuser.datasets.normalize import WallLocLimitsNormalizer, TrajectoryLimitsNormalizer

class FlowModelInference:

    def __init__(self, checkpoint_path: Optional[str], config_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if checkpoint_path is None:
            self.model = None
            self.horizon = None
            self.maze_size = (5, 5)
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config['model']
        else:
            model_config = checkpoint.get('config', {}).get('model', {})
            if not model_config:

                model_config = {
                    'position_dim': 2,
                    'horizon': 40,
                    'hidden_dim': 512,
                    'num_layers': 8,
                    'time_embedding_dim': 256,
                    'condition_dim': 4,
                    'max_walls': 6,
                    'wall_feature_dim': 512,
                    'num_attention_heads': 8,
                    'dropout': 0.15
                }

        self.model = TrajFlowModel(**model_config)

        model_keys = set(self.model.state_dict().keys())
        loaded_state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {
            k: v for k, v in loaded_state_dict.items()
            if k in model_keys
        }
        ignored_keys = set(loaded_state_dict.keys()) - model_keys
        if ignored_keys:
            print(f"⚠️  忽略了检查点中的以下多余参数 ({len(ignored_keys)} 个):")
            for k in sorted(list(ignored_keys)):
                print(f"  - {k}")

        self.model.load_state_dict(filtered_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.horizon = model_config['horizon']
        self.maze_size = (5, 5)

        print(f"✅ 模型加载成功")
        print(f"   - 设备: {self.device}")
        print(f"   - 轨迹长度: {self.horizon}")
        print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def _sample_trajectory_direct(self,
                                  conditions: Dict[int, torch.Tensor],
                                  wall_locations: torch.Tensor,
                                  num_steps: int = 20,
                                  dt: float = 0.05) -> torch.Tensor:
        batch_size = list(conditions.values())[0].shape[0]
        device = list(conditions.values())[0].device

        position_dim = self.model.position_dim
        state_dim = position_dim * 2

        x_0 = torch.randn(batch_size, self.horizon, state_dim, device=device)

        start_pos = conditions[0][..., :position_dim]
        goal_pos = conditions[self.horizon - 1][..., :position_dim]

        x_0[:, 0, :position_dim] = start_pos
        x_0[:, -1, :position_dim] = goal_pos
        x_0[:, 0, position_dim:] = 0.0
        x_0[:, -1, position_dim:] = 0.0

        x = x_0.clone()

        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size,), t_val, device=device)

            velocity_field = self.model.velocity_field(x, t_tensor, conditions, wall_locations)

            x = x + velocity_field * dt

            x[:, 0, :position_dim] = start_pos
            x[:, -1, :position_dim] = goal_pos
            x[:, 0, position_dim:] = 0.0
            x[:, -1, position_dim:] = 0.0

        return x

    def normalize_inputs(self,
                        start_pos: List[float],
                        goal_pos: List[float],
                        wall_positions: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TrajectoryLimitsNormalizer]:
        walls_np = np.array(wall_positions, dtype=np.float32).reshape(-1, 2)
        wall_normalizer = WallLocLimitsNormalizer(walls_np, self.maze_size)

        dummy_traj = np.array([start_pos, goal_pos], dtype=np.float32)
        traj_normalizer = TrajectoryLimitsNormalizer(dummy_traj, self.maze_size)

        norm_walls = wall_normalizer.normalize(walls_np)
        norm_start = traj_normalizer.normalize(np.array([start_pos], dtype=np.float32))[0]
        norm_goal = traj_normalizer.normalize(np.array([goal_pos], dtype=np.float32))[0]

        return norm_start, norm_goal, norm_walls, traj_normalizer

    def generate_trajectory(self,
                          start_pos: List[float],
                          goal_pos: List[float],
                          wall_positions: List[List[float]],
                          num_steps: int = 20,
                          dt: float = 0.01,
                          num_samples: int = 1,
                          record_steps: bool = False) -> Tuple[np.ndarray, TrajectoryLimitsNormalizer]:

        if len(wall_positions) > 6:
            wall_positions = wall_positions[:6]
        elif len(wall_positions) < 6:

            wall_positions = wall_positions + [[0, 0]] * (6 - len(wall_positions))

        norm_start, norm_goal, norm_walls, traj_normalizer = self.normalize_inputs(
            start_pos, goal_pos, wall_positions
        )

        start_tensor = torch.FloatTensor(norm_start).to(self.device)
        goal_tensor = torch.FloatTensor(norm_goal).to(self.device)
        walls_tensor = torch.FloatTensor(norm_walls).to(self.device)

        conditions = {
            0: start_tensor.unsqueeze(0).repeat(num_samples, 1),
            self.horizon - 1: goal_tensor.unsqueeze(0).repeat(num_samples, 1)
        }
        wall_locations = walls_tensor.unsqueeze(0).repeat(num_samples, 1, 1)

        with torch.no_grad():
            trajectories = self._sample_trajectory_direct(
                conditions=conditions,
                wall_locations=wall_locations,
                num_steps=num_steps,
                dt=dt
            )

        trajectories_np = trajectories.cpu().numpy()
        position_dim = self.model.position_dim
        unnorm_trajectories = []

        for i in range(num_samples):
            traj = trajectories_np[i]

            pos_traj = traj[:, :position_dim]

            unnorm_traj = traj_normalizer.unnormalize(pos_traj)
            unnorm_trajectories.append(unnorm_traj)

        return np.array(unnorm_trajectories), traj_normalizer
