import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Tuple, Any
import os

class SequenceDataset(Dataset):

    def __init__(self,
                 env: str,
                 horizon: int = 48,
                 normalizer: str = 'LimitsNormalizer',
                 preprocess_fns: List = None,
                 use_padding: bool = True,
                 max_path_length: int = 1000,
                 include_returns: bool = False,
                 include_cond_returns: bool = False,
                 discount: float = 0.99,
                 max_walls: int = 10,
                 dataset_path: Optional[str] = None):
        self.env = env
        self.horizon = horizon
        self.normalizer = normalizer
        self.preprocess_fns = preprocess_fns or []
        self.use_padding = use_padding
        self.max_path_length = max_path_length
        self.include_returns = include_returns
        self.include_cond_returns = include_cond_returns
        self.discount = discount
        self.max_walls = max_walls

        if dataset_path is not None:
            self.dataset_path = dataset_path
        else:

            self.dataset_path = f'./datasets/{env}.hdf5'

        print(f"🔍 加载数据集: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.dataset_path}")

        self._load_data()

        print(f"✅ 数据集加载完成")
        print(f"   - 样本数量: {len(self)}")
        print(f"   - 轨迹长度: {self.horizon}")
        print(f"   - 障碍物数量: {self.wall_locations.shape[1] if self.wall_locations is not None else '无'}")

    def _load_data(self):
        with h5py.File(self.dataset_path, 'r') as f:
            print(f"📊 HDF5文件包含的字段: {list(f.keys())}")

            self.observations = np.array(f['observations'])
            print(f"   - observations: {self.observations.shape}")

            if 'actions' in f:
                self.actions = np.array(f['actions'])
            else:

                self.actions = np.zeros((len(self.observations), self.horizon, 2))

            if 'terminals' in f:
                self.terminals = np.array(f['terminals'])
            else:
                self.terminals = np.zeros((len(self.observations), self.horizon), dtype=bool)

            if 'rewards' in f:
                self.rewards = np.array(f['rewards'])
            else:
                self.rewards = np.zeros((len(self.observations), self.horizon))

            self.wall_locations = None
            if 'infos/wall_locations' in f:
                wall_data = f['infos/wall_locations']
                print(f"   - wall_locations原始形状: {wall_data.shape}")

                wall_locations_list = []
                for i in range(len(wall_data)):
                    walls = np.array(wall_data[i])

                    if walls.shape[1] >= 2:
                        walls = walls[:, :2]

                    if len(walls) > self.max_walls:
                        walls = walls[:self.max_walls]
                    elif len(walls) < self.max_walls:

                        padding = np.zeros((self.max_walls - len(walls), 2))
                        walls = np.vstack([walls, padding])

                    wall_locations_list.append(walls)

                self.wall_locations = np.array(wall_locations_list)
                print(f"   - 处理后wall_locations: {self.wall_locations.shape}")

            if 'infos/goal' in f:
                self.goals = np.array(f['infos/goal'])
                print(f"   - goals: {self.goals.shape}")
            else:

                self.goals = self.observations[:, -1, :2]

            if 'maze_idx' in f:
                self.maze_indices = np.array(f['maze_idx'])
                print(f"   - maze_indices: {self.maze_indices.shape}")
            else:
                self.maze_indices = np.zeros(len(self.observations), dtype=int)

        if self.observations.shape[2] >= 2:
            self.trajectories = self.observations[:, :, :2]
        else:
            raise ValueError(f"观察维度不足，期望至少2维，实际: {self.observations.shape[2]}")

        self._apply_preprocessing()

    def _apply_preprocessing(self):

        if self.normalizer == 'LimitsNormalizer':

            pass

        for preprocess_fn in self.preprocess_fns:
            if callable(preprocess_fn):
                self.trajectories = preprocess_fn(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        trajectory = torch.tensor(self.trajectories[idx], dtype=torch.float32)

        start_pos = trajectory[0]
        goal_pos = self.goals[idx] if self.goals is not None else trajectory[-1]
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)

        walls = None
        if self.wall_locations is not None:
            walls = torch.tensor(self.wall_locations[idx], dtype=torch.float32)

        sample = {
            'trajectories': trajectory,
            'start_pos': start_pos,
            'goal_pos': goal_pos,
            'actions': torch.tensor(self.actions[idx], dtype=torch.float32),
            'terminals': torch.tensor(self.terminals[idx], dtype=torch.bool),
            'rewards': torch.tensor(self.rewards[idx], dtype=torch.float32),
        }

        if walls is not None:
            sample['wall_locations'] = walls

        if hasattr(self, 'maze_indices'):
            sample['maze_idx'] = torch.tensor(self.maze_indices[idx], dtype=torch.long)

        return sample

    def get_batch_conditions(self, batch_indices: List[int]) -> Dict[int, torch.Tensor]:
        batch_size = len(batch_indices)

        start_positions = []
        goal_positions = []

        for idx in batch_indices:
            start_positions.append(self.trajectories[idx][0])

            if self.goals is not None:
                goal_positions.append(self.goals[idx])
            else:
                goal_positions.append(self.trajectories[idx][-1])

        conditions = {
            0: torch.tensor(np.array(start_positions), dtype=torch.float32),
            self.horizon - 1: torch.tensor(np.array(goal_positions), dtype=torch.float32)
        }

        return conditions

    def get_batch_walls(self, batch_indices: List[int]) -> Optional[torch.Tensor]:
        if self.wall_locations is None:
            return None

        batch_walls = []
        for idx in batch_indices:
            batch_walls.append(self.wall_locations[idx])

        return torch.tensor(np.array(batch_walls), dtype=torch.float32)

def collate_sequence_batch(batch: List[Dict[str, torch.Tensor]]) -> Any:
    class BatchData:
        def __init__(self):
            pass

    batch_data = BatchData()

    batch_data.trajectories = torch.stack([sample['trajectories'] for sample in batch])
    batch_data.start_pos = torch.stack([sample['start_pos'] for sample in batch])
    batch_data.goal_pos = torch.stack([sample['goal_pos'] for sample in batch])
    batch_data.actions = torch.stack([sample['actions'] for sample in batch])
    batch_data.terminals = torch.stack([sample['terminals'] for sample in batch])
    batch_data.rewards = torch.stack([sample['rewards'] for sample in batch])

    if 'wall_locations' in batch[0]:
        batch_data.wall_locations = torch.stack([sample['wall_locations'] for sample in batch])

    if 'maze_idx' in batch[0]:
        batch_data.maze_idx = torch.stack([sample['maze_idx'] for sample in batch])

    return batch_data
