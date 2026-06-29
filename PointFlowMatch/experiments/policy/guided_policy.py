import torch
import numpy as np
from typing import Dict, Optional
from pfp.common.fm_utils import get_timesteps
from pfp import DEVICE

class GuidedFMPolicy:

    def __init__(self, base_policy, guidance):
        self.base_policy = base_policy
        self.guidance = guidance
        self.device = DEVICE

    def predict(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        pcd = obs_dict['pcd']
        robot_state_obs = obs_dict['robot_state']

        goal_pos = None
        if 'goal_pos' in obs_dict:
            goal_pos = obs_dict['goal_pos']
        elif 'goal' in obs_dict:

            goal = obs_dict['goal']
            if goal.dim() == 2:
                n_obs_steps = robot_state_obs.shape[1]
                goal_pos = goal.unsqueeze(1).repeat(1, n_obs_steps, 1)
            else:
                goal_pos = goal

        result = self.infer_y_guided(pcd, robot_state_obs, goal_pos)
        return result.detach().cpu().numpy()

    def infer_y_guided(self,
                      pcd: torch.Tensor,
                      robot_state_obs: torch.Tensor,
                      goal_pos: Optional[torch.Tensor] = None,
                      noise: Optional[torch.Tensor] = None,
                      return_traj: bool = False) -> torch.Tensor:
        batch_size = pcd.shape[0]

        if goal_pos is not None:

            obs_enc = self.base_policy.obs_encoder(pcd, robot_state_obs, goal_pos)
        else:

            obs_enc = self.base_policy.obs_encoder(pcd, robot_state_obs)

        if noise is None:
            z = self.base_policy._init_noise(batch_size)
        else:
            z = noise

        traj = [z]
        t0, dt = get_timesteps(
            self.base_policy.flow_schedule,
            self.base_policy.num_k_infer,
            exp_scale=self.base_policy.exp_scale
        )

        t0 = t0.to(device=self.device)
        dt = dt.to(device=self.device)

        for i in range(self.base_policy.num_k_infer):

            timesteps = torch.ones((batch_size,), device=self.device) * t0[i]
            timesteps *= self.base_policy.pos_emb_scale

            vel_base = self.base_policy.diffusion_net(z, timesteps, global_cond=obs_enc)

            t_normalized = t0[i]

            conditions = {
                'obs_enc': obs_enc,
                'goal_pos': goal_pos if goal_pos is not None else None
            }

            vel_guided = self.guidance.forward(
                x=z,
                t=t_normalized,
                base_velocity=vel_base,
                conditions=conditions
            )

            z = z.detach().clone() + vel_guided * dt[i]
            traj.append(z)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    def infer_y(self,
                pcd: torch.Tensor,
                robot_state_obs: torch.Tensor,
                goal_pos: Optional[torch.Tensor] = None,
                noise: Optional[torch.Tensor] = None,
                return_traj: bool = False) -> torch.Tensor:
        return self.infer_y_guided(pcd, robot_state_obs, goal_pos, noise, return_traj)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):

            return self.predict(args[0])
        elif len(args) >= 2:

            pcd = args[0]
            robot_state_obs = args[1]
            goal_pos = args[2] if len(args) > 2 else kwargs.get('goal_pos', None)
            return self.infer_y(pcd, robot_state_obs, goal_pos)
        else:
            raise ValueError("不支持的调用方式")
