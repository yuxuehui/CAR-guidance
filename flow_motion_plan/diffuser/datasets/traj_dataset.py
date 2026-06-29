import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .normalize import TrajectoryLimitsNormalizer, WallLocLimitsNormalizer, GoalLimitsNormalizer

MAZE_SIZE = (5, 5)
NUM_MAZE = 3000
NUM_POINTS_PER_MAZE = 25000

def sequence_dataset(hdf5_path, horizon=48):
    with h5py.File(hdf5_path, 'r') as f:
        for maze_idx in tqdm(range(NUM_MAZE), desc='正在处理迷宫数据'):

            wall_locations = f['infos/wall_locations'][maze_idx * NUM_POINTS_PER_MAZE]

            wall_locations = WallLocLimitsNormalizer(wall_locations, MAZE_SIZE).normalize(wall_locations)

            wall_locations = torch.FloatTensor(wall_locations)

            start_idx = maze_idx * NUM_POINTS_PER_MAZE
            end_idx = start_idx + NUM_POINTS_PER_MAZE
            maze_points = f['observations'][start_idx: end_idx]
            maze_goals = f['infos/goal'][start_idx: end_idx]

            goal_changes = np.where(np.any(maze_goals[1:] != maze_goals[:-1], axis=1))[0]
            traj_start_indices = np.concatenate([[0], goal_changes + 1])
            traj_end_indices = np.concatenate([goal_changes, [len(maze_points) - 1]])

            for start_indice, end_indice in zip(traj_start_indices, traj_end_indices):
                traj_points = maze_points[start_indice: end_indice + 1]

                traj_goal = maze_goals[start_indice]

                traj_points = TrajectoryLimitsNormalizer(traj_points, MAZE_SIZE).normalize(traj_points)
                traj_goal = GoalLimitsNormalizer(traj_goal, MAZE_SIZE).normalize(traj_goal)

                traj_points, mask = adjust_traj_length(traj_points, horizon)

                traj_points = torch.FloatTensor(traj_points)
                traj_goal = torch.FloatTensor(traj_goal)
                mask = torch.BoolTensor(mask)

                conditions = {
                    0: traj_points[0],
                    horizon - 1: traj_goal
                }

                traj_data = {
                    'observations': traj_points,
                    'conditions': conditions,

                    'wall_locations': wall_locations,
                    'mask': mask,
                    'maze_idx': maze_idx,

                }

                yield traj_data

def adjust_traj_length(traj_points, horizon=48):
    original_length = traj_points.shape[0]
    mask = np.ones(horizon, dtype=bool)

    if original_length < horizon:

        mask[original_length:] = False
        traj_points = np.concatenate([
            traj_points,
            np.zeros((horizon - original_length, traj_points.shape[1]))
        ], axis=0)
    elif original_length > horizon:
        traj_points = traj_points[:horizon]

    return traj_points, mask

class TrajDataset(Dataset):
    def __init__(self, hdf5_path, horizon=48):
        self.horizon = horizon

        self.traj_data_list = []
        for traj_data in tqdm(sequence_dataset(hdf5_path, horizon), desc='正在处理轨迹数据'):
            self.traj_data_list.append(traj_data)

        print(f"共有{len(self.traj_data_list)}条轨迹")

    def __len__(self):
        return len(self.traj_data_list)

    def __getitem__(self, idx):
        return self.traj_data_list[idx]

def collate_fn(batch):

    observations = torch.stack([item['observations'] for item in batch])
    wall_locations = torch.stack([item['wall_locations'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    maze_idx = torch.tensor([item['maze_idx'] for item in batch])

    conditions = {
        k: torch.stack([item['conditions'][k] for item in batch])
        for k in batch[0]['conditions'].keys()
    }

    return {
        'observations': observations,
        'conditions': conditions,
        'wall_locations': wall_locations,
        'mask': mask,
        'maze_idx': maze_idx
    }

def get_traj_dataloader(hdf5_path, horizon=48, batch_size=32, num_workers=4, shuffle=True):
    print("开始")
    traj_dataset = TrajDataset(hdf5_path, horizon)
    print("结束")
    return DataLoader(traj_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=shuffle)

if __name__ == '__main__':
    hdf5_path = '/C/cfc/flow_motion_plan/pb_diff_envs/pb_diff_envs/datasets/randSmaze2d-ng3ks25k-ms55nw6-hExt05-v0.hdf5'
    horizon = 48
    batch_size = 4
    num_workers = 4
    traj_dataloader = get_traj_dataloader(hdf5_path, horizon, batch_size, num_workers, shuffle=True)
    for traj_data in traj_dataloader:
        print(traj_data)
        break
