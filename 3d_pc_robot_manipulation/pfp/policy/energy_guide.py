import torch
import torch.nn as nn
import numpy as np

class EnergyFunction(nn.Module):
    def __init__(self, center, radius=0.3):
        super().__init__()
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.radius = float(radius)

class EnergyGuideVectorField(nn.Module):
    def __init__(self, flow_model, energy_functions, energy_scales):
        super().__init__()
        self.flow_model = flow_model
        self.energy_functions = energy_functions
        self.energy_scales = energy_scales

        assert len(self.energy_functions) == len(self.energy_scales), \
            f"能量函数数量 ({len(self.energy_functions)}) 与缩放系数数量 ({len(self.energy_scales)}) 不匹配"

        self.step_data = []

    def forward(self, x, t, conditions=None, record_step=False):

        raise NotImplementedError(
            "EnergyGuideVectorField 需要在 FMPolicy.infer_y 中集成使用，"
            "不能直接作为独立的模块调用"
        )

    def compute_energy_gradient(self, x_pos, energy_fn, scale):
        center = energy_fn.center.to(x_pos.device)
        radius = energy_fn.radius

        diff = x_pos - center
        dist = torch.norm(diff, dim=-1, keepdim=True)

        weight = torch.clamp((radius - dist) / radius, min=0.0)

        dir_to_center = -diff / (dist + 1e-8)

        grad_pos = scale * weight * dir_to_center

        return grad_pos
