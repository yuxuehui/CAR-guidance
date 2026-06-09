"""Abstract base class for the 2D distributions used in the synthetic experiment."""

import torch
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Common interface for prior and target distributions (all 2D)."""

    @abstractmethod
    def sample(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """Draw ``batch_size`` samples, returning a tensor of shape (B, 2)."""
        ...

    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-density of ``x``."""
        ...

    @abstractmethod
    def get_J(self, x1) -> torch.Tensor:
        """Return the guidance energy J(x) = -log p(x) (up to a constant)."""
        ...

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_prob(x))

    def __str__(self):
        return self.__class__.__name__
