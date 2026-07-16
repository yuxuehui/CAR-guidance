import torch
from typing import Dict, Optional

from .base_guidance import BaseGuidance

class NoneGuidance(BaseGuidance):

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict,
                wall_locations: Optional[torch.Tensor] = None) -> torch.Tensor:
        return base_velocity
