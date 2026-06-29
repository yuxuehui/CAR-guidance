from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch

class BaseGuidance(ABC):

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                base_velocity: torch.Tensor,
                conditions: Dict) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
