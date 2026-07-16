import os
import json
import argparse
import numpy as np
import torch
import gymnasium as gym

import mani_skill.envs
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)

from pathlib import Path
from pfp import DEVICE, REPO_DIRS, set_seeds
from pfp.backbones.pointnet import PointNetBackbone
from pfp.policy.fm_policy_maniskill import FMPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

def load_model_from_checkpoint(
    ckpt_name: str,
    ckpt_episode: str = "latest",
    num_k_infer: int = None,
    flow_schedule: str = None,
    exp_scale: float = None,
    use_ema: bool = True,
) -> FMPolicy:
    ckpt_dir = REPO_DIRS.CKPT / ckpt_name

    ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
    assert len(ckpt_path_list) > 0, f"未找到checkpoint: {ckpt_dir}/{ckpt_episode}*"
    assert len(ckpt_path_list) < 2, f"找到多个匹配的checkpoint: {ckpt_path_list}"
    ckpt_fpath = ckpt_path_list[0]
    print(f"加载checkpoint: {ckpt_fpath}")

    config_path = ckpt_dir / "config.json"
    assert config_path.exists(), f"配置文件不存在: {config_path}"

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"加载配置: {config_path}")

    backbone_config = config["backbone_config"]
    obs_encoder = PointNetBackbone(**backbone_config)

    diffusion_net_config = config["diffusion_net_config"]
    diffusion_net = ConditionalUnet1D(**diffusion_net_config)

    model_config = config["model_config"]
    model = FMPolicy(
        x_dim=config["x_dim"],
        y_dim=config["y_dim"],
        n_obs_steps=config["n_obs_steps"],
        n_pred_steps=config["n_pred_steps"],
        obs_encoder=obs_encoder,
        diffusion_net=diffusion_net,
        **model_config,
    )

    state_dict = torch.load(ckpt_fpath, map_location=DEVICE)

    if use_ema:

        ema_model_state = None
        if "state" in state_dict and "algorithms" in state_dict["state"]:
            algorithms = state_dict["state"]["algorithms"]
            if "EMA" in algorithms and "model" in algorithms["EMA"]:
                ema_model_state = algorithms["EMA"]["model"]

        if ema_model_state is not None:
            model.load_state_dict(ema_model_state)
            print("✓ 使用EMA权重加载模型（推荐用于推理）")
        else:

            model.load_state_dict(state_dict["state"]["model"])
            print("⚠ EMA权重不存在，使用普通模型权重")
    else:

        model.load_state_dict(state_dict["state"]["model"])
        print("使用普通模型权重（未使用EMA）")

    model.to(DEVICE)
    model.eval()
    print(f"模型已设置为eval模式，设备: {DEVICE}")

    if flow_schedule is not None:
        model.set_flow_schedule(flow_schedule, exp_scale)
        print(f"覆盖flow_schedule: {flow_schedule}, exp_scale: {exp_scale}")

    if num_k_infer is not None:
        model.set_num_k_infer(num_k_infer)
        print(f"覆盖num_k_infer: {num_k_infer}")

    return model, config

def get_ground_ids(env) -> list:
    ground_ids = []
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if "ground" in name.lower():
            ground_ids.append(seg_id)
    return ground_ids

def filter_ground_from_pointcloud(obs: dict, ground_ids: list) -> dict:
    xyzw = obs["pointcloud"]["xyzw"][0]
    rgb = obs["pointcloud"]["rgb"][0]
    seg = obs["pointcloud"]["segmentation"][0].squeeze()

    mask = torch.ones(len(seg), dtype=torch.bool, device=seg.device)
    for ground_id in ground_ids:
        mask = mask & (seg != ground_id)

    return {
        "xyz": xyzw[mask, :3],
        "rgb": rgb[mask],
        "segmentation": seg[mask]
    }

def sample_pointcloud(xyz: np.ndarray, rgb: np.ndarray, num_points: int) -> tuple:
    n = len(xyz)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32), np.zeros((num_points, 3), dtype=np.float32)

    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
    else:
        indices = np.random.choice(n, num_points, replace=True)

    return xyz[indices], rgb[indices]

def extract_pointcloud(filtered_pcd: dict, num_points: int = 150000) -> dict:
    xyz = filtered_pcd["xyz"].cpu().numpy().astype(np.float32)
    rgb = filtered_pcd["rgb"].cpu().numpy().astype(np.float32)

    xyz_sampled, rgb_sampled = sample_pointcloud(xyz, rgb, num_points)

    return {
        "xyz": xyz_sampled,
        "rgb": rgb_sampled,
    }

def get_robot_state(env) -> torch.Tensor:
    base_env = env.unwrapped

    tcp_pose = base_env.agent.tcp_pose
    ee_pos = tcp_pose.p[0]
    ee_quat = tcp_pose.q[0]

    ee_rot_matrix = quaternion_to_matrix(ee_quat)
    ee_rot_6d = matrix_to_rotation_6d(ee_rot_matrix)

    qpos = base_env.agent.robot.get_qpos()
    gripper = qpos[0, -1:]

    robot_state = torch.cat([ee_pos, ee_rot_6d, gripper], dim=-1)
    return robot_state

def compute_delta_rot_from_rot6d(
    current_rot6d: torch.Tensor,
    target_rot6d: torch.Tensor
) -> np.ndarray:

    R_current = rotation_6d_to_matrix(current_rot6d.unsqueeze(0))
    R_target = rotation_6d_to_matrix(target_rot6d.unsqueeze(0))

    R_delta = torch.bmm(R_target, R_current.transpose(1, 2))

    delta_rot = matrix_to_axis_angle(R_delta)

    return delta_rot.squeeze(0).cpu().numpy()

def prepare_observation(
    obs: dict,
    robot_state_history: list,
    ground_ids: list,
    n_obs_steps: int,
    n_points: int = 150000,
    use_pc_color: bool = True,
) -> tuple:

    filtered_pcd = filter_ground_from_pointcloud(obs, ground_ids)
    pcd_data = extract_pointcloud(filtered_pcd, num_points=n_points)

    if use_pc_color:
        pcd = np.concatenate([pcd_data["xyz"], pcd_data["rgb"]], axis=-1)
    else:
        pcd = pcd_data["xyz"]

    while len(robot_state_history) < n_obs_steps:
        robot_state_history.insert(0, robot_state_history[0])

    robot_state_history = robot_state_history[-n_obs_steps:]

    pcd_tensor = torch.from_numpy(pcd).float().unsqueeze(0).unsqueeze(0)
    pcd_tensor = pcd_tensor.repeat(1, n_obs_steps, 1, 1)

    robot_state_obs = torch.stack(robot_state_history, dim=0).unsqueeze(0)

    return pcd_tensor.to(DEVICE), robot_state_obs.to(DEVICE)

def main():
    parser = argparse.ArgumentParser(description="ManiSkill环境推理测试")
    parser.add_argument("--ckpt_name", type=str, default="maniskill_train", help="checkpoint目录名称")
    parser.add_argument("--ckpt_episode", type=str, default="latest", help="checkpoint的episode标识")
    parser.add_argument("--num_k_infer", type=int, default=None, help="推理步数")
    parser.add_argument("--flow_schedule", type=str, default=None, help="流调度类型")
    parser.add_argument("--exp_scale", type=float, default=4.0, help="exp调度scale参数")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--num_episodes", type=int, default=1000, help="测试episode数量")
    parser.add_argument("--max_steps", type=int, default=50, help="每个episode最大步数")
    parser.add_argument("--render", action="store_true", help="是否渲染可视化")
    parser.add_argument("--use_ema", action="store_true", default=True, help="是否使用EMA权重 (默认: True，推理时推荐使用)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false", help="不使用EMA权重，使用普通模型权重")

    args = parser.parse_args()

    set_seeds(args.seed)

    print("=" * 60)
    print("ManiSkill 推理测试")
    print("=" * 60)

    print("\n>>> 加载模型...")
    model, config = load_model_from_checkpoint(
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
        num_k_infer=args.num_k_infer,
        flow_schedule=args.flow_schedule,
        exp_scale=args.exp_scale,
        use_ema=args.use_ema,
    )

    n_obs_steps = config["n_obs_steps"]
    n_pred_steps = config["n_pred_steps"]
    n_points = config["dataset_config"]["n_points"]
    use_pc_color = config["dataset_config"]["use_pc_color"]
    norm_pcd_center = config["model_config"]["norm_pcd_center"]

    print(f"\n模型配置:")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  n_pred_steps: {n_pred_steps}")
    print(f"  n_points: {n_points}")
    print(f"  use_pc_color: {use_pc_color}")
    print(f"  norm_pcd_center: {norm_pcd_center}")
    print(f"  num_k_infer: {model.num_k_infer}")
    print(f"  flow_schedule: {model.flow_schedule}")

    print("\n>>> 创建ManiSkill环境...")
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="pointcloud",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda_wristcam",
        render_mode="human"
    )

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    print(f"Ground IDs: {ground_ids}")

    print("\n>>> 模型加载完成，准备测试!")
    print("=" * 60)

    success_count = 0

    GRIPPER_LOWER = 0.0
    GRIPPER_UPPER = 0.04

    for ep_idx in range(args.num_episodes):
        print(f"\n--- Episode {ep_idx + 1}/{args.num_episodes} ---")

        obs, _ = env.reset(seed=ep_idx)
        robot_state_history = []

        done = False
        step_count = 0

        while not done and step_count < args.max_steps:

            robot_state = get_robot_state(env).cpu()
            robot_state_history.append(robot_state)

            pcd_tensor, robot_state_obs = prepare_observation(
                obs=obs,
                robot_state_history=robot_state_history.copy(),
                ground_ids=ground_ids,
                n_obs_steps=n_obs_steps,
                n_points=n_points,
                use_pc_color=use_pc_color,
            )

            with torch.no_grad():

                pcd_tensor[..., :3] -= torch.tensor(norm_pcd_center, device=DEVICE)
                robot_state_obs[..., :3] -= torch.tensor(norm_pcd_center, device=DEVICE)

                pred_y = model.infer_y(pcd_tensor, robot_state_obs)

                pred_y[..., :3] += torch.tensor(norm_pcd_center, device=DEVICE)

            pred_action = pred_y[0, 0]

            current_state = robot_state.to(DEVICE)
            delta_pos = (pred_action[:3] - current_state[:3]).cpu().numpy()

            POS_LIMIT = 0.1
            delta_pos_normalized = np.clip(delta_pos / POS_LIMIT, -1.0, 1.0)

            delta_rot = np.zeros(3)

            pred_gripper_qpos = pred_action[9].item()
            gripper = (pred_gripper_qpos - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
            gripper = np.clip(gripper, -1.0, 1.0)

            action = np.concatenate([delta_pos_normalized, delta_rot, [gripper]])

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()

            done = info["success"].item()
            step_count += 1

        success = info["success"].item()
        if success:
            success_count += 1
        print(f"  结果: {'成功' if success else '失败'}, 步数: {step_count}")

    env.close()

    print("\n" + "=" * 60)
    print(f"测试完成！成功率: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
