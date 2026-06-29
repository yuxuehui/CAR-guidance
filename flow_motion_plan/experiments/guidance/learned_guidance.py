import torch
import sys
from pathlib import Path
from typing import Dict, Optional, List, Callable, Any

from .base_guidance import BaseGuidance

from .energy_function import EnergyFunction

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from old_experiments.trajectory_g_cov_a_gm_online import TrajectoryGCovAGMOnlineGuidance
    from old_experiments.trajectory_dynamic_g_cov_a_gm_online import TrajectoryDynamicGCovAGMOnlineGuidance
except ImportError as e:
    print(f"⚠️  无法导入g_cov_a_gm_online相关类: {e}")
    print(f"   注意：LearnedGuidance类依赖于old_experiments中的实现")
    print(f"   建议使用ComposedLearnedGuidance + GCovWrapper（新模式，不依赖old_experiments）")
    TrajectoryGCovAGMOnlineGuidance = None
    TrajectoryDynamicGCovAGMOnlineGuidance = None

class LearnedGuidance(BaseGuidance):

    def __init__(self,
                 flow_model,
                 energy_centers: Optional[List[List[float]]] = None,
                 energy_scales: Optional[List[float]] = None,
                 path_functions: Optional[List[Callable[[float], List[float]]]] = None,
                 traj_normalizer=None,
                 config: Optional[Dict] = None):
        super().__init__(config)

        if TrajectoryGCovAGMOnlineGuidance is None:
            raise ImportError("无法导入TrajectoryGCovAGMOnlineGuidance，请检查old_experiments目录")

        self.flow_model = flow_model
        self.traj_normalizer = traj_normalizer
        self.energy_centers = energy_centers or []
        self.energy_scales = energy_scales or []
        self.path_functions = path_functions

        self.is_dynamic = path_functions is not None and len(path_functions) > 0

        self.online_config = {
            'online_train_steps': self.config.get('online_train_steps', 1000),
            'online_batch_size': self.config.get('online_batch_size', 4),
            'online_lr': self.config.get('online_lr', 1e-4),
            'num_ode_steps': self.config.get('num_ode_steps', 20),
            'conflict_threshold': self.config.get('conflict_threshold', 0.5),
            'conflict_temperature': self.config.get('conflict_temperature', 0.1),
            'online_loss_type': self.config.get('online_loss_type', 'mse_simple'),
            'residual_unet_channels': self.config.get('residual_unet_channels', 64),
            'residual_unet_down_dims': self.config.get('residual_unet_down_dims', [256, 512, 1024]),
            'use_learned_correction': self.config.get('train_online', True),
        }

        self._create_online_guidance()

        self.trained = False

    def _create_online_guidance(self):
        horizon = getattr(self.flow_model, 'horizon', 40)
        device = str(next(self.flow_model.parameters()).device)

        if self.is_dynamic:

            if TrajectoryDynamicGCovAGMOnlineGuidance is None:
                raise ImportError("无法导入TrajectoryDynamicGCovAGMOnlineGuidance")

            self.online_guidance = TrajectoryDynamicGCovAGMOnlineGuidance(
                flow_model=self.flow_model,
                path_functions=self.path_functions,
                energy_scales=self.energy_scales,
                horizon=horizon,
                device=device,
                config=self.online_config,
                traj_normalizer=self.traj_normalizer,
            )
        else:

            if len(self.energy_centers) == 0:
                raise ValueError("静态能量场需要提供energy_centers")

            energy_functions = []
            for center in self.energy_centers:
                energy_functions.append(EnergyFunction(center))

            self.online_guidance = TrajectoryGCovAGMOnlineGuidance(
                flow_model=self.flow_model,
                energy_functions=energy_functions,
                energy_scales=self.energy_scales,
                horizon=horizon,
                device=device,
                config=self.online_config,
                traj_normalizer=self.traj_normalizer,
            )

    def train_online(self, conditions: Dict, wall_locations: torch.Tensor):
        if self.online_config.get('use_learned_correction', True):
            self.online_guidance.train_model(
                conditions=conditions,
                wall_locations=wall_locations
            )
            self.trained = True

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict,
                wall_locations: Optional[torch.Tensor] = None) -> torch.Tensor:

        return base_velocity

    def get_online_guidance(self):
        return self.online_guidance

class LearnedGuidanceFactory:

    @staticmethod
    def create_static(flow_model,
                     energy_centers: List[List[float]],
                     energy_scales: List[float],
                     traj_normalizer=None,
                     config: Optional[Dict] = None) -> LearnedGuidance:
        return LearnedGuidance(
            flow_model=flow_model,
            energy_centers=energy_centers,
            energy_scales=energy_scales,
            traj_normalizer=traj_normalizer,
            config=config
        )

    @staticmethod
    def create_dynamic(flow_model,
                      path_functions: List[Callable[[float], List[float]]],
                      energy_scales: List[float],
                      traj_normalizer=None,
                      config: Optional[Dict] = None) -> LearnedGuidance:
        return LearnedGuidance(
            flow_model=flow_model,
            path_functions=path_functions,
            energy_scales=energy_scales,
            traj_normalizer=traj_normalizer,
            config=config
        )
