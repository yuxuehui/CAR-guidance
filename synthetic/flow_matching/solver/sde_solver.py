# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from diffusers.utils.torch_utils import randn_tensor

from flow_matching.solver.solver import Solver
from flow_matching.utils import gradient, ModelWrapper


class SDESolver(Solver):
    """A class to solve stochastic differential equations (SDEs) using a specified velocity model.

    This class utilizes a velocity field model to solve SDEs over a given time grid using numerical SDE solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",  # ignored for SDE (kept for API parity)
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        noise_level: float = 0.7,
        generator: Optional[torch.Generator] = None,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:

        # --- prepare ---
        x = x_init.float()
        time_grid = time_grid.to(x.device).flatten()
        assert time_grid.numel() >= 2, "time_grid must have at least 2 points."

        # 单调性检查（允许升序或降序，但不能乱序）
        diffs = time_grid[1:] - time_grid[:-1]
        assert torch.all(diffs > 0) or torch.all(diffs < 0), \
            "time_grid must be strictly monotonic (ascending or descending)."

        if return_intermediates:
            sol = [x]

        # 选择扩散系数 g(t)：最简单和稳定的是常数
        def diffusion_coeff(t: Tensor) -> Tensor:
            # 也可以改为随时间变化：例如 noise_level * sqrt(clamp(t*(1-t), eps, 1))
            return torch.as_tensor(noise_level, device=x.device, dtype=x.dtype)

        # 采样循环：Euler–Maruyama
        # dx = u(x,t) * dt + g(t) * dW,  dW ~ N(0, |dt|)
        ctx_grad = torch.set_grad_enabled(enable_grad)
        with ctx_grad:
            for i in range(time_grid.numel() - 1):
                t_i = time_grid[i]
                t_ip1 = time_grid[i + 1]
                dt = (t_ip1 - t_i).to(x.dtype)

                # 漂移（速度场）
                u = self.velocity_model(x=x, t=t_i, **model_extras).float()
                # 数值兜底
                u = torch.nan_to_num(u, nan=0.0, posinf=1e3, neginf=-1e3)

                # 扩散系数 g(t)
                g = diffusion_coeff(t_i)
                if isinstance(g, torch.Tensor):
                    g = torch.nan_to_num(g, nan=1e-3, posinf=1.0, neginf=1e-3)
                    g = torch.clamp(g, min=1e-6)   # 防止完全零扩散
                else:
                    # 标量 -> 张量
                    g = torch.as_tensor(max(float(g), 1e-6), device=x.device, dtype=x.dtype)

                # 生成噪声增量：√|dt| * N(0,1)
                # 伊藤型 SDE 的 Euler–Maruyama 离散必须用 √|dt| 来缩放噪声增量（dW ~ N(0, |dt|)）
                dW = randn_tensor(
                    x.shape, generator=generator, device=x.device, dtype=x.dtype
                ) * torch.sqrt(torch.abs(dt).clamp_min(torch.finfo(x.dtype).eps))

                # Euler–Maruyama 更新
                x = x + u * dt + g * dW

                # 结果再兜底，避免后续可视化 NaN/Inf
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

                if return_intermediates:
                    sol.append(x)

        if return_intermediates:
            return torch.stack(sol)
        else:
            return x

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        x_1.requires_grad_()
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]
