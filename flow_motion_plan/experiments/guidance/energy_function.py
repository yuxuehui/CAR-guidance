import torch
import torch.nn as nn

class EnergyFunction(nn.Module):

    def __init__(self, center):
        super().__init__()
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
