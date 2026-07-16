import torch
import torch.nn as nn
from typing import List, Callable, Optional, Dict, Any

class EnergyFunction(nn.Module):

    def __init__(self, center):
        super().__init__()

        if isinstance(center, torch.Tensor):
            self.register_buffer("center", center)
        else:
            self.register_buffer("center", torch.tensor(center, dtype=torch.float32))

class EnergyGuideVectorField(nn.Module):

    def __init__(self, flow_model, energy_function, energy_scale):
        super().__init__()
        self.flow_model = flow_model
        self.energy_function = energy_function
        self.energy_scale = energy_scale
        self.step_data = []

        from .guidance.static_guidance import StaticGuidance

        if isinstance(energy_function, EnergyFunction):
            center = energy_function.center.cpu().numpy().tolist()
            energy_centers = [center]
            energy_scales = [energy_scale]
        else:

            if hasattr(energy_function, 'center'):
                center = energy_function.center.cpu().numpy().tolist()
                energy_centers = [center]
                energy_scales = [energy_scale]
            else:
                raise ValueError(f"不支持的 energy_function 类型: {type(energy_function)}")

        config = {
            'energy_centers': energy_centers,
            'energy_scales': energy_scales,
            'sigma': 1.0,
            'repulsion_radius': 3.0,
        }
        self.static_guidance = StaticGuidance(config)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                conditions: Optional[Dict] = None,
                wall_locations: Optional[torch.Tensor] = None,
                record_step: bool = False) -> torch.Tensor:

        v_uncond = self.flow_model.velocity_field(x, t, conditions, wall_locations)

        v_guided = self.static_guidance.forward(
            x=x,
            t=t,
            base_velocity=v_uncond,
            conditions=conditions or {},
            wall_locations=wall_locations
        )

        if record_step:
            step_info = {
                'trajectory': x.detach().cpu().numpy(),
                'v_uncond': v_uncond.detach().cpu().numpy(),
                'guidance_grad': (v_guided - v_uncond).detach().cpu().numpy(),
                't': t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t,
            }
            self.step_data.append(step_info)

        return v_guided

class DynamicEnergyGuideVectorField(nn.Module):

    def __init__(self, flow_model, path_functions: List[Callable[[float], List[float]]], energy_scales: List[float]):
        super().__init__()
        self.flow_model = flow_model
        self.path_functions = path_functions
        self.energy_scales = energy_scales
        self.step_data = []

        from .guidance.dynamic_guidance import DynamicGuidance

        config = {
            'energy_scales': energy_scales,
            'sigma': 1.0,
            'repulsion_radius': 3.0,
        }
        self.dynamic_guidance = DynamicGuidance(path_functions, config)

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                conditions: Optional[Dict] = None,
                wall_locations: Optional[torch.Tensor] = None,
                record_step: bool = False) -> torch.Tensor:

        v_uncond = self.flow_model.velocity_field(x, t, conditions, wall_locations)

        v_guided = self.dynamic_guidance.forward(
            x=x,
            t=t,
            base_velocity=v_uncond,
            conditions=conditions or {},
            wall_locations=wall_locations
        )

        if record_step:
            step_info = {
                'trajectory': x.detach().cpu().numpy(),
                'v_uncond': v_uncond.detach().cpu().numpy(),
                'guidance_grad': (v_guided - v_uncond).detach().cpu().numpy(),
                't': t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t,
            }
            self.step_data.append(step_info)

        return v_guided

__all__ = ['EnergyFunction', 'EnergyGuideVectorField', 'DynamicEnergyGuideVectorField']
