import numpy as np
import random
from typing import List, Tuple

def select_energy_centers_from_trajectory(
    trajectory: np.ndarray,
    num_centers: int = 2,
    min_dist_from_traj: float = 0.05,
    max_dist_from_traj: float = 0.15,
    seed: int = 0,
) -> List[List[float]]:
    rng = random.Random(seed)
    n_steps = trajectory.shape[0]

    energy_centers = []

    start_ratio = 0.4
    end_ratio = 0.9

    start_idx = int(n_steps * start_ratio)
    end_idx = int(n_steps * end_ratio)

    if end_idx <= start_idx:

        start_idx, end_idx = 0, n_steps - 1

    chosen_indices = np.linspace(start_idx, end_idx, num_centers + 2)[1:-1].astype(int)

    for idx in chosen_indices:
        traj_point = trajectory[idx]

        theta = rng.uniform(0, 2 * np.pi)

        dist = rng.uniform(min_dist_from_traj, max_dist_from_traj)

        z_offset = rng.uniform(-0.05, 0.05)

        xy_dist = np.sqrt(max(0, dist**2 - z_offset**2))

        offset = np.array([
            xy_dist * np.cos(theta),
            xy_dist * np.sin(theta),
            z_offset
        ])

        center = traj_point + offset

        center[2] = np.clip(center[2], 0.05, 0.10)

        energy_centers.append(center.tolist())

    return energy_centers

def get_energy_centers_for_demo(
    demo_id: int,
    trajectory: np.ndarray,
    num_centers: int = 2,
    seed_base: int = 42,
) -> List[List[float]]:

    seed = seed_base + demo_id * 1000

    return select_energy_centers_from_trajectory(
        trajectory=trajectory,
        num_centers=num_centers,
        min_dist_from_traj=0.05,
        max_dist_from_traj=0.15,
        seed=seed,
    )
