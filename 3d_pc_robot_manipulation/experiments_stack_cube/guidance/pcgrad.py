from typing import List, Optional

import torch

def pcgrad_combine(
    grads: List[torch.Tensor],
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if len(grads) == 0:
        raise ValueError("pcgrad_combine requires at least one gradient")
    if len(grads) == 1:
        return grads[0]

    stack = torch.stack(grads, dim=0)
    num_tasks = stack.shape[0]

    proj = stack.clone()

    device = stack.device
    if generator is not None:
        order = torch.randperm(num_tasks, generator=generator, device=device)
    else:
        order = torch.randperm(num_tasks, device=device)
    order = order.tolist()

    for i in range(num_tasks):
        g_i = proj[i]
        for k in order:
            if k == i:
                continue
            g_k = stack[k]
            inner = (g_i * g_k).sum(dim=-1, keepdim=True)
            sq_norm = (g_k * g_k).sum(dim=-1, keepdim=True)

            coeff = torch.clamp(inner / (sq_norm + eps), max=0.0)

            coeff = torch.where(sq_norm > eps, coeff, torch.zeros_like(coeff))
            g_i = g_i - coeff * g_k
        proj[i] = g_i

    return proj.sum(dim=0)
