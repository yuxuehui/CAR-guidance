from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import init_random_traj_th
from pfp.common.fm_utils import get_timesteps
from pfp.data.dataset_pcd import augment_pcd_data

class FMPolicyAction(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        time_conditioning: bool,
        obs_encoder: nn.Module,
        diffusion_net: nn.Module,
        augment_data: bool = False,
        loss_weights: dict[int] = None,
        pos_emb_scale: int = 20,
        norm_pcd_center: list = None,
        noise_type: str = "gaussian",
        noise_scale: float = 1.0,
        loss_type: str = "l2",
        flow_schedule: str = "linear",
        exp_scale: float = None,
        snr_sampler: str = "uniform",
        subs_factor: int = 1,
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps, subs_factor)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer

        self.time_conditioning = time_conditioning
        self.obs_encoder = obs_encoder
        self.diffusion_net = diffusion_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        self.snr_sampler = snr_sampler
        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError
        return

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:

        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:

        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)

        robot_state[..., 9] -= torch.tensor(0.0, device=DEVICE)
        return robot_state

    def _norm_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)

        robot_state[..., 9] += torch.tensor(0.0, device=DEVICE)
        return robot_state

    def _norm_goal_pos(self, goal_pos: torch.Tensor) -> torch.Tensor:
        goal_pos[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return goal_pos

    def _denorm_goal_pos(self, goal_pos: torch.Tensor) -> torch.Tensor:
        goal_pos[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        return goal_pos

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        pcd, robot_state_obs, y_target, goal_pos = batch
        pcd = self._norm_obs(pcd)
        robot_state_obs = self._norm_robot_state(robot_state_obs)

        if y_target.shape[-1] == 10:
            y_target = self._norm_robot_state(y_target)
        else:
            y_target = self._norm_action(y_target)
        goal_pos = self._norm_goal_pos(goal_pos)
        return pcd, robot_state_obs, y_target, goal_pos

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return augment_pcd_data(batch)

    def _init_noise(self, batch_size: int) -> torch.Tensor:
        B = batch_size
        T = self.n_pred_steps
        if self.noise_type == "gaussian":
            noise = torch.randn((batch_size, *self.ny_shape), device=DEVICE)
            return noise * self.noise_scale
        elif self.noise_type == "trajectory":
            return init_random_traj_th(batch_size, self.n_pred_steps, self.noise_scale)
        elif self.noise_type == "igso3":
            noise_pos = torch.randn((B, T, 3), device=DEVICE)
            noise_rot = pp.randn_SO3((B, T), device=DEVICE).matrix()
            noise_gripper = torch.randn((B, T, 1), device=DEVICE)
            noise = torch.cat(
                [noise_pos, noise_rot[..., :3, 0], noise_rot[..., :3, 1], noise_gripper], dim=-1
            )
            return noise
        else:
            raise NotImplementedError

    def _sample_snr(self, batch_size: int) -> torch.Tensor:
        if self.snr_sampler == "uniform":
            return torch.rand((batch_size, 1, 1), device=DEVICE)
        elif self.snr_sampler == "logit_normal":
            return torch.sigmoid(torch.randn((batch_size, 1, 1), device=DEVICE))
        else:
            raise NotImplementedError

    def forward(self, batch):
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        pcd, robot_state_obs, y_target, goal_pos = batch

        loss_xyz, loss_rot, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, y_target, goal_pos
        )

        rot_key = "rot" if (self.l_w is not None and "rot" in self.l_w) else "rot6d"
        loss = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w[rot_key] * loss_rot
            + self.l_w["grip"] * loss_grip
        )

        metrics = {
            "loss/train/xyz": loss_xyz.item(),
            "loss/train/rot": loss_rot.item(),
            "loss/train/grip": loss_grip.item(),
        }
        self.logger.log_metrics(metrics)
        return loss

    def calculate_loss(
        self, pcd: torch.Tensor, robot_state_obs: torch.Tensor, y_target: torch.Tensor, goal_pos: torch.Tensor
    ):
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs, goal_pos)
        ny: torch.Tensor = y_target

        B = ny.shape[0]
        t = self._sample_snr(B)
        z0 = self._init_noise(ny.shape[0])
        z1 = ny
        z_t = t * z1 + (1.0 - t) * z0
        target_vel = z1 - z0
        timesteps = t.squeeze() * self.pos_emb_scale if self.time_conditioning else None
        pred_vel = self.diffusion_net(z_t, timesteps, global_cond=nx)

        loss_xyz = self.loss_fun(pred_vel[..., :3], target_vel[..., :3])

        if ny.shape[-1] == 10:
            loss_rot = self.loss_fun(pred_vel[..., 3:9], target_vel[..., 3:9])
            loss_grip = self.loss_fun(pred_vel[..., 9], target_vel[..., 9])
        else:
            loss_rot = self.loss_fun(pred_vel[..., 3:6], target_vel[..., 3:6])
            loss_grip = self.loss_fun(pred_vel[..., 6], target_vel[..., 6])
        return loss_xyz, loss_rot, loss_grip

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        batch = self._norm_data(batch)
        pcd, robot_state_obs, y_target, goal_pos = batch

        loss_xyz, loss_rot, loss_grip = self.calculate_loss(
            pcd, robot_state_obs, y_target, goal_pos
        )
        rot_key = "rot" if (self.l_w is not None and "rot" in self.l_w) else "rot6d"
        loss_total = (
            self.l_w["xyz"] * loss_xyz
            + self.l_w[rot_key] * loss_rot
            + self.l_w["grip"] * loss_grip
        )
        self.logger.log_metrics(
            {
                "loss/eval/xyz": loss_xyz.item(),
                "loss/eval/rot": loss_rot.item(),
                "loss/eval/grip": loss_grip.item(),
                "loss/eval/total": loss_total.item(),
            }
        )

        pred_y = self.infer_y(pcd, robot_state_obs, goal_pos)
        mse_xyz = nn.functional.mse_loss(pred_y[..., :3], y_target[..., :3])
        if y_target.shape[-1] == 10:
            mse_rot = nn.functional.mse_loss(pred_y[..., 3:9], y_target[..., 3:9])
            mse_grip = nn.functional.mse_loss(pred_y[..., 9], y_target[..., 9])
        else:
            mse_rot = nn.functional.mse_loss(pred_y[..., 3:6], y_target[..., 3:6])
            mse_grip = nn.functional.mse_loss(pred_y[..., 6], y_target[..., 6])
        self.logger.log_metrics(
            {
                "metrics/eval/mse_xyz": mse_xyz.item(),
                "metrics/eval/mse_rot": mse_rot.item(),
                "metrics/eval/mse_grip": mse_grip.item(),
            }
        )
        return pred_y

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        goal_pos: torch.Tensor,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        nx: torch.Tensor = self.obs_encoder(pcd, robot_state_obs, goal_pos)
        B = nx.shape[0]
        z = self._init_noise(B) if noise is None else noise
        traj = [z]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            timesteps = torch.ones((B), device=DEVICE) * t0[i]
            timesteps *= self.pos_emb_scale
            vel_pred = self.diffusion_net(z, timesteps, global_cond=nx)
            z = z.detach().clone() + vel_pred * dt[i]
            traj.append(z)

        if return_traj:
            return torch.stack(traj)
        return traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int,
        flow_schedule: str = None,
        exp_scale: float = None,
        subs_factor: int = 1,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")

        cfg.model.subs_factor = subs_factor
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: FMPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model

class FMPolicyImageAction(FMPolicyAction):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        return image

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError
