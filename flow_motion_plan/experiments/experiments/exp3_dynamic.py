import numpy as np
import torch
from typing import Dict, List, Callable

from ..core.base_experiment import BaseExperiment
from ..core.guidance_factory import GuidanceFactory

def create_dynamic_paths_linear(
    start_pos: List[float],
    goal_pos: List[float],
    offset: float = 0.5,
    maze_size: tuple = (5.0, 5.0),
    seed: int = None,
) -> List[Callable[[float], List[float]]]:
    import random

    maze_w, maze_h = maze_size

    if seed is None:

        seed = int(abs(hash((tuple(start_pos), tuple(goal_pos)))) % 10000)
    rng = random.Random(seed)

    def clamp_point(point):
        x = max(0.5, min(maze_w - 0.5, point[0]))
        y = max(0.5, min(maze_h - 0.5, point[1]))
        return np.array([x, y], dtype=np.float32)

    def generate_random_path():

        path_start = np.array([
            rng.uniform(0.8, maze_w - 0.8),
            rng.uniform(0.8, maze_h - 0.8)
        ], dtype=np.float32)

        path_end = np.array([
            rng.uniform(0.8, maze_w - 0.8),
            rng.uniform(0.8, maze_h - 0.8)
        ], dtype=np.float32)

        while np.linalg.norm(path_end - path_start) < 1.0:
            path_end = np.array([
                rng.uniform(0.8, maze_w - 0.8),
                rng.uniform(0.8, maze_h - 0.8)
            ], dtype=np.float32)

        path_start = clamp_point(path_start)
        path_end = clamp_point(path_end)

        def path_func(t: float) -> List[float]:
            t = max(0.0, min(1.0, t))
            point = path_start + t * (path_end - path_start)
            return [float(point[0]), float(point[1])]

        return path_func

    num_paths = 2
    paths = [generate_random_path() for _ in range(num_paths)]

    return paths

class Exp3Dynamic(BaseExperiment):

    def __init__(self, model_checkpoint: str, config: Dict):
        self.guidance_config = config.get('guidance', {})
        super().__init__(model_checkpoint, config)

    def _create_guidance(self):
        guidance_config = self.guidance_config.copy()
        use_gcov = guidance_config.get('use_gcov_optimization', False)
        guidance_type = guidance_config.get('type')

        if guidance_type == 'learned' or guidance_type == 'dynamic' or use_gcov:
            return None

        return GuidanceFactory.create(
            guidance_config,
            flow_model=self.model,
            traj_normalizer=None
        )

    def generate_trajectory(self,
                           start_pos: List[float],
                           goal_pos: List[float],
                           wall_positions: List[List[float]],
                           num_samples: int = 1,
                           **kwargs) -> np.ndarray:
        guidance_config = self.guidance_config.copy()
        use_gcov = guidance_config.get('use_gcov_optimization', False)

        path_offset = guidance_config.get('path_offset', 0.5)
        path_seed = guidance_config.get('path_seed', None)

        path_functions = create_dynamic_paths_linear(
            start_pos=start_pos,
            goal_pos=goal_pos,
            offset=path_offset,
            maze_size=(5.0, 5.0),
            seed=path_seed,
        )

        self._last_path_functions = path_functions

        _, _, _, traj_normalizer = self._normalize_inputs(start_pos, goal_pos, wall_positions)

        guidance_config['path_functions'] = path_functions

        if use_gcov:

            from ..guidance.composed_learned_guidance import ComposedLearnedGuidance

            base_config = guidance_config.copy()
            base_config['type'] = 'dynamic'
            base_config['path_functions'] = path_functions
            base_guidance = GuidanceFactory.create(
                base_config,
                flow_model=self.model,
                traj_normalizer=traj_normalizer
            )

            self.guidance = ComposedLearnedGuidance(
                flow_model=self.model,
                base_guidance=base_guidance,
                traj_normalizer=traj_normalizer,
                config=guidance_config,
                use_legacy_mode=False,
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

                guidance_config['traj_normalizer'] = traj_normalizer
                self.guidance = GuidanceFactory.create(
                    guidance_config,
                    flow_model=self.model,
                    traj_normalizer=traj_normalizer
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

                self.model.guide_model = GuidanceWrapper(self.guidance, self.model)

        return super().generate_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            wall_positions=wall_positions,
            num_samples=num_samples,
            **kwargs
        )
