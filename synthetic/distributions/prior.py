"""Standard isotropic Gaussian prior p0 = N(0, I) for the synthetic experiment."""

import math
import torch

from .base import BaseDistribution


class StandardGaussianPrior(BaseDistribution):
    """Standard isotropic Gaussian prior p0 = N(0, I)."""

    def __init__(self, dim: int = 2, device: str = "cuda"):
        super().__init__()
        self.dim = dim
        self.device = device

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = self.device
        return torch.randn(batch_size, self.dim, device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        d = x.size(-1)
        return -0.5 * (x.pow(2).sum(dim=-1) + d * math.log(2 * math.pi))

    def get_J(self, x: torch.Tensor) -> torch.Tensor:
        # Energy corresponding to standard Gaussian: J(x) = 0.5 * ||x||^2 + const
        return 0.5 * x.pow(2).sum(dim=-1)

    def __name__(self):
        return "StandardGaussianPrior"

