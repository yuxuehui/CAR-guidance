import torch
from typing import Dict

from .base_guidance import BaseGuidance

class NoneGuidance(BaseGuidance):

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict) -> torch.Tensor:
        return base_velocity
