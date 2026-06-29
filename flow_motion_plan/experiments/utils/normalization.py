import numpy as np
from typing import List, Tuple
from diffuser.datasets.normalize import (
    WallLocLimitsNormalizer,
    TrajectoryLimitsNormalizer
)

class TrajNormalizer:

    def __init__(self, normalizer: TrajectoryLimitsNormalizer):
        self.normalizer = normalizer
        self.mins = normalizer.mins
        self.maxs = normalizer.maxs

    def normalize(self, traj: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize(traj)

    def unnormalize(self, traj: np.ndarray) -> np.ndarray:
        return self.normalizer.unnormalize(traj)

def normalize_trajectory_inputs(
    start_pos: List[float],
    goal_pos: List[float],
    wall_positions: List[List[float]],
    maze_size: Tuple[float, float] = (5.0, 5.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TrajNormalizer]:
    walls_np = np.array(wall_positions, dtype=np.float32).reshape(-1, 2)
    wall_normalizer = WallLocLimitsNormalizer(walls_np, maze_size)

    dummy_traj = np.array([start_pos, goal_pos], dtype=np.float32)
    traj_normalizer = TrajectoryLimitsNormalizer(dummy_traj, maze_size)

    norm_walls = wall_normalizer.normalize(walls_np)
    norm_start = traj_normalizer.normalize(np.array([start_pos], dtype=np.float32))[0]
    norm_goal = traj_normalizer.normalize(np.array([goal_pos], dtype=np.float32))[0]

    return norm_start, norm_goal, norm_walls, TrajNormalizer(traj_normalizer)
