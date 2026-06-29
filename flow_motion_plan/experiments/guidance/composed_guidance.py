import torch
from typing import Dict, Optional, List

from .base_guidance import BaseGuidance
from .pcgrad import pcgrad_combine

class ComposedGuidance(BaseGuidance):

    def __init__(self,
                 guidances: List[BaseGuidance],
                 weights: Optional[List[float]] = None,
                 config: Optional[Dict] = None,
                 normalize: bool = False):
        super().__init__(config)
        self.guidances = guidances
        self.normalize = normalize

        if weights is None:
            weights = [1.0] * len(guidances)

        assert len(weights) == len(guidances), \
            f"权重数量 ({len(weights)}) 与guidance数量 ({len(guidances)}) 不匹配"

        self.weights = weights

        self.combine_method = (config or {}).get('combine_method', 'sum')

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict,
                wall_locations: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(self.guidances) == 0:
            return base_velocity

        if self.combine_method == 'pcgrad':

            per_reward_grads = []
            for guidance, weight in zip(self.guidances, self.weights):
                if hasattr(guidance, 'compute_reward_grads'):
                    sub_grads = guidance.compute_reward_grads(
                        x, t, base_velocity, conditions, wall_locations
                    )
                    for g in sub_grads:
                        per_reward_grads.append(weight * g)
                else:

                    guided_v = guidance(x, t, base_velocity, conditions, wall_locations)
                    per_reward_grads.append(weight * (guided_v - base_velocity))

            if len(per_reward_grads) == 0:
                return base_velocity

            if len(per_reward_grads) > 1:
                total_grad = pcgrad_combine(per_reward_grads)
            else:
                total_grad = per_reward_grads[0]

            if self.normalize:
                total_weight = sum(self.weights)
                if total_weight > 0:
                    total_grad = total_grad / total_weight

            return base_velocity + total_grad

        guidance_grads = []
        for guidance, weight in zip(self.guidances, self.weights):
            guided_v = guidance(x, t, base_velocity, conditions, wall_locations)

            grad = guided_v - base_velocity
            guidance_grads.append(weight * grad)

        total_grad = sum(guidance_grads)

        if self.normalize:
            total_weight = sum(self.weights)
            if total_weight > 0:
                total_grad = total_grad / total_weight

        return base_velocity + total_grad
