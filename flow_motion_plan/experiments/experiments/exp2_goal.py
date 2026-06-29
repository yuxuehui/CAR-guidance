import random
import numpy as np
import torch
from typing import Dict, List

from ..core.base_experiment import BaseExperiment
from ..core.guidance_factory import GuidanceFactory
from .exp1_static import get_trajectory_tangent

def select_interfering_energy_centers_from_base_traj(
    base_traj: np.ndarray,
    walls: List[List[float]],
    num_centers: int = 2,
    min_dist_from_traj: float = 0.4,
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

class Exp2Goal(BaseExperiment):

    def __init__(self, model_checkpoint: str, config: Dict):
        self.guidance_config = config.get('guidance', {})
        self.auto_select_centers = self.guidance_config.get('energy_centers') is None

        super().__init__(model_checkpoint, config)

        if self.auto_select_centers:
            from ..utils.inference import FlowModelInference
            self.base_inferencer = FlowModelInference(model_checkpoint, config.get('model_config'))
        else:
            self.base_inferencer = None

    def _create_guidance(self):
        guidance_config = self.guidance_config.copy()
        guidance_type = guidance_config.get('type', 'goal')
        use_gcov = guidance_config.get('use_gcov_optimization', False)

        if use_gcov:

            base_config = guidance_config.copy()
            base_config['type'] = 'goal'
            base_guidance = GuidanceFactory.create(
                base_config,
                flow_model=self.model,
                traj_normalizer=None
            )

            if self.auto_select_centers:
                self.base_guidance = base_guidance
                return None

            from ..guidance.composed_learned_guidance import ComposedLearnedGuidance
            _, _, _, traj_normalizer = self._normalize_inputs([0, 0], [1, 1], [[0, 0]])

            return ComposedLearnedGuidance(
                flow_model=self.model,
                base_guidance=base_guidance,
                energy_centers=guidance_config.get('energy_centers'),
                energy_scales=guidance_config.get('energy_scales', [1.0]),
                traj_normalizer=traj_normalizer,
                config=guidance_config
            )
        else:

            if guidance_type == 'learned' and self.auto_select_centers:
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

        energy_centers = None

        if self.auto_select_centers:

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
            num_centers = self.guidance_config.get('num_energy_centers', 2)
            energy_centers = select_interfering_energy_centers_from_base_traj(
                base_traj=base_traj,
                walls=wall_positions,
                num_centers=num_centers,
                seed=energy_seed,
            )

            _, _, _, traj_normalizer = self._normalize_inputs(start_pos, goal_pos, wall_positions)

            guidance_config = self.guidance_config.copy()
            guidance_config['energy_centers'] = energy_centers

            if use_gcov:

                from ..guidance.composed_learned_guidance import ComposedLearnedGuidance

                base_config = guidance_config.copy()
                base_config['type'] = 'goal'
                base_guidance = GuidanceFactory.create(
                    base_config,
                    flow_model=self.model,
                    traj_normalizer=traj_normalizer
                )

                energy_scales = guidance_config.get('energy_scales', [1.0] * len(energy_centers))

                self.guidance = ComposedLearnedGuidance(
                    flow_model=self.model,
                    base_guidance=base_guidance,
                    traj_normalizer=traj_normalizer,
                    config=guidance_config,
                    use_legacy_mode=False,
                    energy_centers=energy_centers,
                    energy_scales=energy_scales,
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
        else:

            energy_centers = self.guidance_config.get('energy_centers')

        if energy_centers is not None:

            self._last_energy_centers = energy_centers
        elif hasattr(self, 'guidance') and hasattr(self.guidance, 'energy_centers') and self.guidance.energy_centers:

            self._last_energy_centers = self.guidance.energy_centers
        else:

            self._last_energy_centers = self.guidance_config.get('energy_centers')

        trajectories = super().generate_trajectory(
            start_pos=start_pos,
            goal_pos=goal_pos,
            wall_positions=wall_positions,
            num_samples=num_samples,
            **kwargs
        )

        if not hasattr(self, '_last_energy_centers') or self._last_energy_centers is None:
            if hasattr(self, 'guidance') and hasattr(self.guidance, 'energy_centers') and self.guidance.energy_centers:
                self._last_energy_centers = self.guidance.energy_centers

        return trajectories
