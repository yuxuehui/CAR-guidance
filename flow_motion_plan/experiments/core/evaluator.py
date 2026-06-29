import numpy as np
from typing import Dict, List, Tuple, Optional

class Evaluator:

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        goal_tol_raw = self.config.get('goal_tolerance', 0.3)
        self.goal_tolerance = float(goal_tol_raw)

        collision_margin_raw = self.config.get('collision_margin', 0.0)
        self.collision_margin = float(collision_margin_raw)

        wall_size_raw = self.config.get('wall_size', 1.0)
        self.wall_size = float(wall_size_raw)

    def point_in_box(self, point: np.ndarray, center: np.ndarray,
                     box_size: float = None, margin: float = None) -> bool:
        box_size = float(box_size) if box_size is not None else self.wall_size
        margin = float(margin) if margin is not None else self.collision_margin

        point = np.asarray(point, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)

        point = np.atleast_1d(point)
        center = np.atleast_1d(center)

        px, py = float(point[0]), float(point[1])
        cx, cy = float(center[0]), float(center[1])
        half = float(box_size) / 2.0 + float(margin)
        result = (abs(px - cx) <= half) and (abs(py - cy) <= half)
        return result

    def trajectory_collides(self, traj: np.ndarray, walls: np.ndarray) -> bool:

        traj = np.asarray(traj, dtype=np.float64)
        walls = np.asarray(walls, dtype=np.float64)

        if traj.ndim == 1:

            if traj.size % 2 == 0:
                traj = traj.reshape(-1, 2)
            else:
                raise ValueError(f"traj has shape {traj.shape}, cannot reshape to [H, 2]")
        elif traj.ndim == 0:
            raise ValueError(f"traj is a scalar, expected 2D array [H, 2]")
        elif traj.ndim > 2:
            raise ValueError(f"traj has {traj.ndim} dimensions, expected 2D array [H, 2]")

        valid_walls = []
        for i, w in enumerate(walls):
            w = np.asarray(w, dtype=np.float64)
            w0_val = float(w[0])
            w1_val = float(w[1])
            if not (abs(w0_val) < 1e-6 and abs(w1_val) < 1e-6):
                valid_walls.append(w)

        if len(valid_walls) == 0:
            return False

        for p_idx, p in enumerate(traj):
            p = np.asarray(p, dtype=np.float64)

            p = np.atleast_1d(p)

            if p.ndim != 1 or len(p) != 2:
                continue
            for w_idx, w in enumerate(valid_walls):
                if self.point_in_box(p, w):
                    return True
        return False

    def reached_goal(self, traj: np.ndarray, goal: np.ndarray,
                    tol: Optional[float] = None) -> bool:
        tol = float(tol) if tol is not None else self.goal_tolerance

        traj = np.asarray(traj, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        for point in traj:
            dist = np.linalg.norm(point - goal)
            if float(dist) <= float(tol):
                return True
        return False

    def evaluate_trajectory(self, traj: np.ndarray, goal: np.ndarray,
                           walls: np.ndarray) -> Dict[str, bool]:

        traj = np.asarray(traj, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        walls = np.asarray(walls, dtype=np.float64)

        if traj.ndim == 1:
            if traj.size % 2 == 0:
                traj = traj.reshape(-1, 2)
            else:
                raise ValueError(f"traj has shape {traj.shape}, cannot reshape to [H, 2]")
        elif traj.ndim == 0:
            raise ValueError(f"traj is a scalar, expected 2D array [H, 2]")
        elif traj.ndim > 2:
            raise ValueError(f"traj has {traj.ndim} dimensions, expected 2D array [H, 2]")

        reached = self.reached_goal(traj, goal)
        has_collision = self.trajectory_collides(traj, walls)
        is_perfect = reached and not has_collision

        return {
            'reached_goal': reached,
            'has_collision': has_collision,
            'is_perfect': is_perfect
        }

    def evaluate(self, trajectory: np.ndarray, goal_pos: np.ndarray,
                 walls: np.ndarray) -> Dict[str, float]:
        trajectory = np.asarray(trajectory, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        walls = np.asarray(walls, dtype=np.float64)
        if trajectory.ndim == 1 and trajectory.size % 2 == 0:
            trajectory = trajectory.reshape(-1, 2)

        base = self.evaluate_trajectory(trajectory, goal_pos, walls)
        goal_distance = float(np.linalg.norm(trajectory[-1] - goal_pos))
        path_length = 0.0
        if len(trajectory) > 1:
            path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
        smoothness = 1.0
        if len(trajectory) >= 3:
            segs = np.diff(trajectory, axis=0)
            norms = np.linalg.norm(segs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-8, norms, 1e-8)
            dirs = segs / norms
            cos = np.sum(dirs[:-1] * dirs[1:], axis=1)
            cos = np.clip(cos, -1.0, 1.0)
            smoothness = float(np.mean(cos))
        return {
            'success': base['reached_goal'],
            'goal_distance': goal_distance,
            'path_length': path_length,
            'smoothness': smoothness,
            'collision': base['has_collision'],
        }

    def evaluate_batch(self, trajectories: List[np.ndarray],
                      goals: List[np.ndarray],
                      walls_list: List[np.ndarray]) -> Dict[str, float]:
        total = len(trajectories)
        reached_count = 0
        collision_free_count = 0
        perfect_count = 0

        for traj, goal, walls in zip(trajectories, goals, walls_list):
            result = self.evaluate_trajectory(traj, goal, walls)
            if result['reached_goal']:
                reached_count += 1
            if not result['has_collision']:
                collision_free_count += 1
            if result['is_perfect']:
                perfect_count += 1

        return {
            'total': total,
            'reached_goal_count': reached_count,
            'collision_free_count': collision_free_count,
            'perfect_count': perfect_count,
            'reached_goal_rate': reached_count / total if total > 0 else 0.0,
            'collision_free_rate': collision_free_count / total if total > 0 else 0.0,
            'perfect_rate': perfect_count / total if total > 0 else 0.0,
        }
