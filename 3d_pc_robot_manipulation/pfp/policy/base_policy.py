import torch
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from pfp import DEVICE

class BasePolicy(ABC):

    def __init__(self, n_obs_steps: int, subs_factor: int = 1) -> None:
        maxlen = n_obs_steps * subs_factor - (subs_factor - 1)
        self.obs_list = deque(maxlen=maxlen)
        self.robot_state_list = deque(maxlen=maxlen)
        self.subs_factor = subs_factor
        return

    def reset_obs(self):
        self.obs_list.clear()
        self.robot_state_list.clear()
        return

    def update_obs_lists(self, obs: np.ndarray, robot_state: np.ndarray):
        self.obs_list.append(obs)
        if len(self.obs_list) < self.obs_list.maxlen:
            self.obs_list.extendleft(
                [self.obs_list[0]] * (self.obs_list.maxlen - len(self.obs_list))
            )
        self.robot_state_list.append(robot_state)
        if len(self.robot_state_list) < self.robot_state_list.maxlen:
            n = self.robot_state_list.maxlen - len(self.robot_state_list)
            self.robot_state_list.extendleft([self.robot_state_list[0]] * n)
        return

    def sample_stacked_obs(self) -> tuple[np.ndarray, ...]:
        obs_stacked = np.stack(self.obs_list, axis=0)[:: self.subs_factor]
        robot_state_stacked = np.stack(self.robot_state_list, axis=0)[:: self.subs_factor]
        return obs_stacked, robot_state_stacked

    def predict_action(self, obs: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        self.update_obs_lists(obs, robot_state)
        obs_stacked, robot_state_stacked = self.sample_stacked_obs()
        action = self.infer_from_np(obs_stacked, robot_state_stacked)
        return action

    def infer_from_np(self, obs: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        obs_th = torch.tensor(obs, device=DEVICE).unsqueeze(0)
        robot_state_th = torch.tensor(robot_state, device=DEVICE).unsqueeze(0)
        obs_th = self._norm_obs(obs_th)
        robot_state_th = self._norm_robot_state(robot_state_th)
        ny = self.infer_y(
            obs_th,
            robot_state_th,
            return_traj=True,
        )
        ny = self._denorm_robot_state(ny)
        ny = ny.squeeze().detach().cpu().numpy()

        return ny

    @abstractmethod
    def _norm_obs(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def infer_y(
        self, obs: torch.Tensor, robot_state: torch.Tensor, return_traj: bool
    ) -> torch.Tensor:
        pass
