import random
import numpy as np
import torch
from typing import Dict, List, Optional, Callable

from ..core.base_experiment import BaseExperiment
from ..core.guidance_factory import GuidanceFactory
from ..guidance.composed_guidance import ComposedGuidance

def get_trajectory_tangent(traj_point_idx: int, traj_points: np.ndarray):
    if len(traj_points) < 2:
        return None

    if traj_point_idx == 0:
        tangent = traj_points[1] - traj_points[0]
    elif traj_point_idx == len(traj_points) - 1:
        tangent = traj_points[-1] - traj_points[-2]
    else:
        tangent = traj_points[traj_point_idx + 1] - traj_points[traj_point_idx - 1]

    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        return None
    return tangent / tangent_norm

def select_energy_centers_from_base_traj(
    base_traj: np.ndarray,
    walls: List[List[float]],
    num_centers: int = 2,
    min_dist_from_traj: float = 0.3,
    max_dist_from_traj: float = 0.8,
    seed: int = 0,
) -> List[List[float]]:
    traj = np.asarray(base_traj, dtype=np.float32)
    H = traj.shape[0]
    maze_w, maze_h = 5.0, 5.0

    def point_in_wall(point, walls, wall_size=1.0, margin=0.2):
        for wall in walls:
            if abs(wall[0]) < 1e-6 and abs(wall[1]) < 1e-6:
                continue
            dist = np.sqrt((point[0] - wall[0])**2 + (point[1] - wall[1])**2)
            if dist < wall_size / 2.0 + margin:
                return True
        return False

    rng = random.Random(seed)
    mid_start = H // 3
    mid_end = 2 * H // 3
    if mid_end <= mid_start:
        mid_start, mid_end = 1, max(2, H - 1)

    energy_centers = []
    for center_idx in range(num_centers):
        found = False
        for _ in range(200):
            traj_idx = rng.randint(mid_start, mid_end - 1)
            base_point = traj[traj_idx]
            tangent = get_trajectory_tangent(traj_idx, traj)
            if tangent is None:
                continue

            tx, ty = tangent[0], tangent[1]
            if center_idx % 2 == 0:
                n = np.array([ty, -tx], dtype=np.float32)
            else:
                n = -np.array([ty, -tx], dtype=np.float32)

            d = rng.uniform(min_dist_from_traj, max_dist_from_traj)
            c = base_point + d * n
            c = [float(max(0.5, min(maze_w - 0.5, c[0]))), float(max(0.5, min(maze_h - 0.5, c[1])))]

            if point_in_wall(c, walls):
                continue

            too_close = False
            for center in energy_centers:
                dist_to_center = np.sqrt((c[0] - center[0])**2 + (c[1] - center[1])**2)
                if dist_to_center < 0.8:
                    too_close = True
                    break

            if not too_close:
                energy_centers.append(c)
                found = True
                break

        if not found:

            x = rng.uniform(1.0, maze_w - 1.0)
            y = rng.uniform(1.0, maze_h - 1.0)
            energy_centers.append([x, y])

    return energy_centers

def create_dynamic_paths_linear(
    start_pos: List[float],
    goal_pos: List[float],
    offset: float = 0.5,
    num_paths: int = 1,
    maze_size: tuple = (5.0, 5.0),
) -> List[Callable[[float], List[float]]]:
    maze_w, maze_h = maze_size
    start_pos = np.array(start_pos, dtype=np.float32)
    goal_pos = np.array(goal_pos, dtype=np.float32)

    direction = goal_pos - start_pos
    direction_norm = np.linalg.norm(direction)

    if direction_norm > 1e-6:
        direction_unit = direction / direction_norm
        perpendicular = np.array([-direction_unit[1], direction_unit[0]], dtype=np.float32)
    else:
        perpendicular = np.array([0.0, 1.0], dtype=np.float32)

    def clamp_point(point):
        x = max(0.5, min(maze_w - 0.5, point[0]))
        y = max(0.5, min(maze_h - 0.5, point[1]))
        return np.array([x, y], dtype=np.float32)

    paths = []

    if num_paths == 1:

        path_start = start_pos + offset * perpendicular
        path_end = goal_pos + offset * perpendicular
        path_start = clamp_point(path_start)
        path_end = clamp_point(path_end)

        def path_func(t: float) -> List[float]:
            t = max(0.0, min(1.0, t))
            point = path_start + t * (path_end - path_start)
            return [float(point[0]), float(point[1])]

        paths.append(path_func)
    else:

        path1_start = start_pos + offset * perpendicular
        path1_end = goal_pos + offset * perpendicular
        path2_start = start_pos - offset * perpendicular
        path2_end = goal_pos - offset * perpendicular

        path1_start = clamp_point(path1_start)
        path1_end = clamp_point(path1_end)
        path2_start = clamp_point(path2_start)
        path2_end = clamp_point(path2_end)

        def path_1(t: float) -> List[float]:
            t = max(0.0, min(1.0, t))
            point = path1_start + t * (path1_end - path1_start)
            return [float(point[0]), float(point[1])]

        def path_2(t: float) -> List[float]:
            t = max(0.0, min(1.0, t))
            point = path2_start + t * (path2_end - path2_start)
            return [float(point[0]), float(point[1])]

        paths.append(path_1)
        paths.append(path_2)

    return paths

def create_dynamic_paths_random(
    start_pos: List[float],
    goal_pos: List[float],
    base_traj: Optional[np.ndarray] = None,
    walls: Optional[List[List[float]]] = None,
    num_paths: int = 1,
    offset_range: tuple = (0.3, 0.8),
    num_control_points: int = 3,
    maze_size: tuple = (5.0, 5.0),
    seed: int = 0,
    fully_random: bool = True,
) -> List[Callable[[float], List[float]]]:
    maze_w, maze_h = maze_size

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    def clamp_point(point):
        x = max(0.5, min(maze_w - 0.5, point[0]))
        y = max(0.5, min(maze_h - 0.5, point[1]))
        return np.array([x, y], dtype=np.float32)

    def point_in_wall(point, walls, wall_size=1.0, margin=0.2):
        if walls is None:
            return False
        for wall in walls:
            if abs(wall[0]) < 1e-6 and abs(wall[1]) < 1e-6:
                continue
            dist = np.sqrt((point[0] - wall[0])**2 + (point[1] - wall[1])**2)
            if dist < wall_size / 2.0 + margin:
                return True
        return False

    def generate_random_point(walls, max_attempts=50):
        for _ in range(max_attempts):
            x = rng.uniform(0.8, maze_w - 0.8)
            y = rng.uniform(0.8, maze_h - 0.8)
            point = np.array([x, y], dtype=np.float32)
            if not point_in_wall(point, walls):
                return point

        x = rng.uniform(0.8, maze_w - 0.8)
        y = rng.uniform(0.8, maze_h - 0.8)
        return np.array([x, y], dtype=np.float32)

    def bezier_curve(control_points: np.ndarray, t: float) -> np.ndarray:
        points = control_points.copy()
        n = len(points) - 1

        for i in range(n):
            for j in range(n - i):
                points[j] = (1 - t) * points[j] + t * points[j + 1]

        return points[0]

    paths = []

    if fully_random:

        for path_idx in range(num_paths):

            path_start = generate_random_point(walls)
            path_end = generate_random_point(walls)

            attempts = 0
            while np.linalg.norm(path_end - path_start) < 1.5 and attempts < 10:
                path_end = generate_random_point(walls)
                attempts += 1

            def make_linear_path_func(start, end):
                def path_func(t: float) -> List[float]:
                    t = max(0.0, min(1.0, t))

                    point = start + t * (end - start)
                    point = clamp_point(point)
                    return [float(point[0]), float(point[1])]
                return path_func

            paths.append(make_linear_path_func(path_start, path_end))

    else:

        start_pos = np.array(start_pos, dtype=np.float32)
        goal_pos = np.array(goal_pos, dtype=np.float32)

        direction = goal_pos - start_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 1e-6:
            direction_unit = direction / direction_norm
            perpendicular = np.array([-direction_unit[1], direction_unit[0]], dtype=np.float32)
        else:
            direction_unit = np.array([1.0, 0.0], dtype=np.float32)
            perpendicular = np.array([0.0, 1.0], dtype=np.float32)

        for path_idx in range(num_paths):
            control_points = [start_pos]
            side_multiplier = 1.0 if path_idx % 2 == 0 else -1.0

            for cp_idx in range(num_control_points):
                t_along = (cp_idx + 1) / (num_control_points + 1)
                base_point = start_pos + t_along * direction

                offset_dist = rng.uniform(offset_range[0], offset_range[1])
                angle_variation = rng.uniform(-0.3, 0.3)
                rotation_matrix = np.array([
                    [np.cos(angle_variation), -np.sin(angle_variation)],
                    [np.sin(angle_variation), np.cos(angle_variation)]
                ], dtype=np.float32)
                offset_direction = rotation_matrix @ (side_multiplier * perpendicular)

                control_point = base_point + offset_dist * offset_direction
                control_point = clamp_point(control_point)

                attempts = 0
                while point_in_wall(control_point, walls) and attempts < 10:
                    offset_dist = rng.uniform(offset_range[0], offset_range[1])
                    angle_variation = rng.uniform(-0.5, 0.5)
                    rotation_matrix = np.array([
                        [np.cos(angle_variation), -np.sin(angle_variation)],
                        [np.sin(angle_variation), np.cos(angle_variation)]
                    ], dtype=np.float32)
                    offset_direction = rotation_matrix @ (side_multiplier * perpendicular)
                    control_point = base_point + offset_dist * offset_direction
                    control_point = clamp_point(control_point)
                    attempts += 1

                control_points.append(control_point)

            control_points.append(goal_pos)
            control_points = np.array(control_points, dtype=np.float32)

            def make_path_func(ctrl_pts):
                def path_func(t: float) -> List[float]:
                    t = max(0.0, min(1.0, t))
                    point = bezier_curve(ctrl_pts, t)
                    point = clamp_point(point)
                    return [float(point[0]), float(point[1])]
                return path_func

            paths.append(make_path_func(control_points))

    return paths

class Exp4Mixed(BaseExperiment):

    def __init__(self, model_checkpoint: str, config: Dict):

        self.guidance_config = config.get('guidance', {})
        self.auto_select_static_centers = self.guidance_config.get('static_energy_centers') is None

        super().__init__(model_checkpoint, config)

        if self.auto_select_static_centers:
            from ..utils.inference import FlowModelInference
            self.base_inferencer = FlowModelInference(model_checkpoint, config.get('model_config'))
        else:
            self.base_inferencer = None

    def _create_guidance(self):
        guidance_config = self.guidance_config.copy()
        use_gcov = guidance_config.get('use_gcov_optimization', False)

        if use_gcov or self.auto_select_static_centers:
            return None

        return None

    def generate_trajectory(self,
                           start_pos: List[float],
                           goal_pos: List[float],
                           wall_positions: List[List[float]],
                           num_samples: int = 1,
                           **kwargs) -> np.ndarray:
        guidance_config = self.guidance_config.copy()
        use_gcov = guidance_config.get('use_gcov_optimization', False)

        static_energy_centers = None
        base_traj = None
        if self.auto_select_static_centers:

            base_seed = self.guidance_config.get('seed', kwargs.get('seed', 42))
            base_traj_seed = base_seed
            energy_seed = base_seed + 1

            torch.manual_seed(base_traj_seed)
            torch.cuda.manual_seed_all(base_traj_seed)
            np.random.seed(base_traj_seed)
            random.seed(base_traj_seed)

            base_trajs, _ = self.base_inferencer.generate_trajectory(
                start_pos=start_pos,
                goal_pos=goal_pos,
                wall_positions=wall_positions,
                num_steps=self.config.get('inference', {}).get('num_steps', 20),
                dt=self.config.get('inference', {}).get('dt', 0.05),
                num_samples=1,
                record_steps=False,
            )
            base_traj = base_trajs[0]

            random.seed(energy_seed)
            num_static_centers = self.guidance_config.get('num_static_energy_centers', 2)
            static_energy_centers = select_energy_centers_from_base_traj(
                base_traj=base_traj,
                walls=wall_positions,
                num_centers=num_static_centers,
                seed=energy_seed,
            )
        else:
            static_energy_centers = self.guidance_config.get('static_energy_centers')

        use_random_paths = guidance_config.get('use_random_dynamic_paths', True)
        num_dynamic_paths = guidance_config.get('num_dynamic_paths', 1)

        if use_random_paths:

            offset_range = guidance_config.get('dynamic_path_offset_range', (0.3, 0.8))
            num_control_points = guidance_config.get('num_dynamic_control_points', 3)
            fully_random = guidance_config.get('fully_random_dynamic_paths', True)

            base_seed = self.guidance_config.get('seed', kwargs.get('seed', 42))
            path_seed = base_seed + 100

            path_functions = create_dynamic_paths_random(
                start_pos=start_pos,
                goal_pos=goal_pos,
                base_traj=base_traj,
                walls=wall_positions,
                num_paths=num_dynamic_paths,
                offset_range=offset_range,
                num_control_points=num_control_points,
                maze_size=(5.0, 5.0),
                seed=path_seed,
                fully_random=fully_random,
            )
        else:

            path_offset = guidance_config.get('dynamic_path_offset', 0.5)
            path_functions = create_dynamic_paths_linear(
                start_pos=start_pos,
                goal_pos=goal_pos,
                offset=path_offset,
                num_paths=num_dynamic_paths,
                maze_size=(5.0, 5.0),
            )

        self._last_path_functions = path_functions

        _, _, _, traj_normalizer = self._normalize_inputs(start_pos, goal_pos, wall_positions)

        guidance_config['static_energy_centers'] = static_energy_centers
        guidance_config['path_functions'] = path_functions

        if 'output_dir' not in guidance_config:
            guidance_config['output_dir'] = self.config.get('output_dir', 'experiments/outputs')

        eval_config = self.config.get('evaluation', {})
        if 'wall_size' not in guidance_config:
            guidance_config['wall_size'] = eval_config.get('wall_size', 1.0)
        if 'collision_margin' not in guidance_config:
            guidance_config['collision_margin'] = eval_config.get('collision_margin', 0.0)

        if use_gcov:

            from ..guidance.composed_learned_guidance import ComposedLearnedGuidance

            static_config = guidance_config.copy()
            static_config['type'] = 'static'
            static_config['energy_centers'] = static_energy_centers
            static_config['energy_scales'] = guidance_config.get('static_energy_scales', [-1.0] * len(static_energy_centers))
            static_base_guidance = GuidanceFactory.create(
                static_config,
                flow_model=self.model,
                traj_normalizer=traj_normalizer
            )

            dynamic_config = guidance_config.copy()
            dynamic_config['type'] = 'dynamic'
            dynamic_config['path_functions'] = path_functions
            dynamic_config['energy_scales'] = guidance_config.get('dynamic_energy_scales', [-1.0] * len(path_functions))
            dynamic_base_guidance = GuidanceFactory.create(
                dynamic_config,
                flow_model=self.model,
                traj_normalizer=traj_normalizer
            )

            static_weight = guidance_config.get('static_weight', 1.0)
            dynamic_weight = guidance_config.get('dynamic_weight', 1.0)
            combined_base_guidance = ComposedGuidance(
                guidances=[static_base_guidance, dynamic_base_guidance],
                weights=[static_weight, dynamic_weight],
                config=guidance_config,
                normalize=False
            )

            all_energy_centers = static_energy_centers.copy() if static_energy_centers else []
            all_energy_scales = guidance_config.get('static_energy_scales', [-1.0] * len(static_energy_centers)).copy()

            self.guidance = ComposedLearnedGuidance(
                flow_model=self.model,
                base_guidance=combined_base_guidance,
                traj_normalizer=traj_normalizer,
                config=guidance_config,
                use_legacy_mode=False,
                energy_centers=all_energy_centers if all_energy_centers else None,
                energy_scales=all_energy_scales if all_energy_scales else None,
            )

            if guidance_config.get('train_online', True):
                norm_start, norm_goal, norm_walls, _ = self._normalize_inputs(
                    start_pos, goal_pos, wall_positions
                )
                start_tensor = torch.FloatTensor(norm_start).to(self.device)
                goal_tensor = torch.FloatTensor(norm_goal).to(self.device)
                walls_tensor = torch.FloatTensor(norm_walls).to(self.device)

                horizon = getattr(self.model, 'horizon', 40)
                conditions = {
                    0: start_tensor.unsqueeze(0).repeat(num_samples, 1),
                    horizon - 1: goal_tensor.unsqueeze(0).repeat(num_samples, 1)
                }
                wall_locations = walls_tensor.unsqueeze(0).repeat(num_samples, 1, 1)

                self.guidance.train_online(conditions, wall_locations)

            self.model.guide_model = self.guidance.get_online_guidance()
        else:

            static_config = guidance_config.copy()
            static_config['type'] = 'static'
            static_config['energy_centers'] = static_energy_centers
            static_config['energy_scales'] = guidance_config.get('static_energy_scales', [-1.0] * len(static_energy_centers))
            static_config['traj_normalizer'] = traj_normalizer
            static_guidance = GuidanceFactory.create(
                static_config,
                flow_model=self.model,
                traj_normalizer=traj_normalizer
            )

            dynamic_config = guidance_config.copy()
            dynamic_config['type'] = 'dynamic'
            dynamic_config['path_functions'] = path_functions
            dynamic_config['energy_scales'] = guidance_config.get('dynamic_energy_scales', [-1.0] * len(path_functions))
            dynamic_config['traj_normalizer'] = traj_normalizer
            dynamic_guidance = GuidanceFactory.create(
                dynamic_config,
                flow_model=self.model,
                traj_normalizer=traj_normalizer
            )

            static_weight = guidance_config.get('static_weight', 1.0)
            dynamic_weight = guidance_config.get('dynamic_weight', 1.0)
            combined_guidance = ComposedGuidance(
                guidances=[static_guidance, dynamic_guidance],
                weights=[static_weight, dynamic_weight],
                config=guidance_config,
                normalize=False
            )

            class GuidanceWrapper:
                def __init__(self, guidance, flow_model):
                    self.guidance = guidance
                    self.flow_model = flow_model
                    self.step_data = []

                def __call__(self, x, t, conditions, wall_locations, record_step=False):
                    v_uncond = self.flow_model.velocity_field(x, t, conditions, wall_locations)
                    if hasattr(self.guidance, 'config'):
                        self.guidance.config['record_step'] = record_step
                    v_guided = self.guidance.forward(x, t, v_uncond, conditions, wall_locations)
                    if record_step:
                        individual_grads = []
                        if hasattr(self.guidance, '_last_individual_grads'):
                            individual_grads = self.guidance._last_individual_grads

                        step_info = {
                            'trajectory': x.detach().cpu().numpy(),
                            'v_uncond': v_uncond.detach().cpu().numpy(),
                            'guidance_grad': (v_guided - v_uncond).detach().cpu().numpy(),
                            't': t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t,
                            'individual_grads': individual_grads,
                        }
                        self.step_data.append(step_info)
                    return v_guided

            self.model.guide_model = GuidanceWrapper(combined_guidance, self.model)

        self._last_static_energy_centers = static_energy_centers

        trajectories = super().generate_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            wall_positions=wall_positions,
            num_samples=num_samples,
            **kwargs
        )

        return trajectories
