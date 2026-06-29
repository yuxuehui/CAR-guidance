from typing import Dict, Any, Optional

from ..guidance.base_guidance import BaseGuidance
from ..guidance.none_guidance import NoneGuidance
from ..guidance.static_guidance import StaticGuidance
from ..guidance.goal_guidance import GoalGuidance
from ..guidance.dynamic_guidance import DynamicGuidance
from ..guidance.learned_guidance import LearnedGuidance, LearnedGuidanceFactory
from ..guidance.composed_guidance import ComposedGuidance

class GuidanceFactory:

    @staticmethod
    def create(guidance_config: Dict[str, Any],
              flow_model=None,
              traj_normalizer=None) -> BaseGuidance:
        guidance_type = guidance_config.get('type', 'none')

        if guidance_type == 'none':
            return NoneGuidance(guidance_config)

        elif guidance_type == 'static':
            return StaticGuidance(guidance_config)

        elif guidance_type == 'goal':
            return GoalGuidance(guidance_config)

        elif guidance_type == 'dynamic':
            path_functions = guidance_config.get('path_functions')
            if path_functions is None:
                raise ValueError("动态guidance需要提供path_functions")
            return DynamicGuidance(path_functions, guidance_config)

        elif guidance_type == 'learned':
            if flow_model is None:
                raise ValueError("学习型guidance需要提供flow_model")

            if 'path_functions' in guidance_config:

                return LearnedGuidanceFactory.create_dynamic(
                    flow_model=flow_model,
                    path_functions=guidance_config['path_functions'],
                    energy_scales=guidance_config.get('energy_scales', [-1.0]),
                    traj_normalizer=traj_normalizer,
                    config=guidance_config
                )
            else:

                energy_centers = guidance_config.get('energy_centers')
                if energy_centers is None:
                    raise ValueError("静态学习型guidance需要提供energy_centers")

                return LearnedGuidanceFactory.create_static(
                    flow_model=flow_model,
                    energy_centers=energy_centers,
                    energy_scales=guidance_config.get('energy_scales', [-1.0]),
                    traj_normalizer=traj_normalizer,
                    config=guidance_config
                )

        elif guidance_type == 'composed':
            guidances_config = guidance_config.get('guidances', [])
            weights = guidance_config.get('weights')

            guidances = [
                GuidanceFactory.create(gc, flow_model, traj_normalizer)
                for gc in guidances_config
            ]

            return ComposedGuidance(guidances, weights, guidance_config)

        else:
            raise ValueError(f"未知的guidance类型: {guidance_type}")
