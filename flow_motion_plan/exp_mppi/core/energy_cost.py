import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Callable, Dict

class EnergyCostFunction:

    def __init__(self,
                 goal_pos: torch.Tensor,
                 walls: torch.Tensor,
                 energy_centers: Optional[List[List[float]]] = None,
                 energy_scales: Optional[List[float]] = None,
                 dynamic_paths: Optional[List[Callable]] = None,
                 horizon: int = 40,
                 weights: Optional[dict] = None,
                 sigma: float = 2.0,
                 maze_min: float = 0.0,
                 maze_max: float = 5.0):
        self.goal_pos = goal_pos
        self.walls = walls
        self.energy_centers = energy_centers
        self.energy_scales = energy_scales or []
        self.dynamic_paths = dynamic_paths
        self.horizon = horizon
        self.sigma = sigma
        self.maze_min = maze_min
        self.maze_max = maze_max

        default_weights = {
            'goal': 50.0,
            'static_energy': 30.0,
            'dynamic_energy': 30.0,
            'smoothness': 0.1,
            'wall_collision': 80.0,
            'boundary': 200.0,
        }
        self.weights = {**default_weights, **(weights or {})}

    def goal_cost(self, trajs: torch.Tensor) -> torch.Tensor:

        dists = torch.norm(trajs - self.goal_pos.view(1, 1, 2), dim=-1)
        final_dist = dists[:, -1]
        min_dist, _ = torch.min(dists, dim=1)
        return final_dist ** 2 + 0.1 * (min_dist ** 2)

    def static_energy_cost(self, trajs: torch.Tensor) -> torch.Tensor:
        if self.energy_centers is None or len(self.energy_centers) == 0:
            return torch.zeros(trajs.shape[0], device=trajs.device)

        K = trajs.shape[0]
        total_cost = torch.zeros(K, device=trajs.device)

        for i, center in enumerate(self.energy_centers):
            center_tensor = torch.tensor(center, dtype=torch.float32, device=trajs.device)
            scale = self.energy_scales[i] if i < len(self.energy_scales) else -1.0

            dist_sq = torch.sum((trajs - center_tensor.view(1, 1, 2)) ** 2, dim=-1)

            energy = torch.exp(-dist_sq / (2 * self.sigma ** 2))

            if scale < 0:

                total_cost += torch.sum(energy, dim=-1) * abs(scale)
            else:

                total_cost += torch.sum(1.0 - energy, dim=-1) * scale

        return total_cost

    def dynamic_energy_cost(self, trajs: torch.Tensor) -> torch.Tensor:
        if self.dynamic_paths is None or len(self.dynamic_paths) == 0:
            return torch.zeros(trajs.shape[0], device=trajs.device)

        K = trajs.shape[0]
        total_cost = torch.zeros(K, device=trajs.device)

        for path_fn in self.dynamic_paths:
            for h in range(self.horizon):
                t_h = h / (self.horizon - 1) if self.horizon > 1 else 0.0
                obs_pos = path_fn(t_h)
                obs_tensor = torch.tensor(obs_pos, dtype=torch.float32, device=trajs.device)

                dist_sq = torch.sum((trajs[:, h, :] - obs_tensor) ** 2, dim=-1)

                energy = torch.exp(-dist_sq / (2 * self.sigma ** 2))
                total_cost += energy

        return total_cost

    def smoothness_cost(self, trajs: torch.Tensor) -> torch.Tensor:
        if self.horizon < 3:
            return torch.zeros(trajs.shape[0], device=trajs.device)

        accel = trajs[:, 2:, :] - 2 * trajs[:, 1:-1, :] + trajs[:, :-2, :]
        smoothness = torch.sum(accel ** 2, dim=(1, 2))

        return smoothness

    def wall_collision_cost(self, trajs: torch.Tensor) -> torch.Tensor:
        K = trajs.shape[0]
        total_cost = torch.zeros(K, device=trajs.device)

        wall_size = 1.0

        collision_threshold_sq = (wall_size / 2.0 + 0.2) ** 2

        for wall in self.walls:

            if torch.abs(wall[0]) < 1e-6 and torch.abs(wall[1]) < 1e-6:
                continue

            dist_sq = torch.sum((trajs - wall.view(1, 1, 2)) ** 2, dim=-1)

            energy = torch.exp(-dist_sq / (2 * 0.5 ** 2))

            total_cost += torch.sum(energy, dim=-1)

        return total_cost

    def boundary_cost(self, trajs: torch.Tensor) -> torch.Tensor:

        lower_violation = F.relu(self.maze_min - trajs)

        upper_violation = F.relu(trajs - self.maze_max)

        violation = lower_violation + upper_violation

        cost = torch.sum(violation ** 2, dim=(1, 2))

        return cost

    def compute_total_cost(self, trajs: torch.Tensor) -> torch.Tensor:
        cost = torch.zeros(trajs.shape[0], device=trajs.device)

        if self.weights['goal'] > 0:
            cost += self.weights['goal'] * self.goal_cost(trajs)

        if self.weights['static_energy'] > 0:
            cost += self.weights['static_energy'] * self.static_energy_cost(trajs)

        if self.weights['dynamic_energy'] > 0:
            cost += self.weights['dynamic_energy'] * self.dynamic_energy_cost(trajs)

        if self.weights['smoothness'] > 0:
            cost += self.weights['smoothness'] * self.smoothness_cost(trajs)

        if self.weights['wall_collision'] > 0:
            cost += self.weights['wall_collision'] * self.wall_collision_cost(trajs)

        if self.weights.get('boundary', 0) > 0:
            cost += self.weights['boundary'] * self.boundary_cost(trajs)

        return cost
