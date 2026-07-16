import numpy as np
import torch
import gymnasium as gym
import os
import time
from pathlib import Path
import sys
from datetime import datetime
import json

REPO_ROOT = Path(__file__).parent.parent
MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MANISKILL_ROOT))

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

sys.path.insert(0, str(Path(__file__).parent))
from eval_maniskill import get_robot_state, prepare_observation
from utils.utils_tool import load_model_from_checkpoint_concat_goal
from pfp import DEVICE, set_seeds
from mani_skill.utils.wrappers.record import RecordEpisode
from pfp.utils.pointcloud_utils import (
    get_pointcloud_from_multi_cameras,
    get_ground_ids,
    get_cube_ids,
    get_robot_ids,
)

from pfp.policy.energy_guide import EnergyFunction

GRIPPER_LOWER = 0.0
GRIPPER_UPPER = 0.04
POS_LIMIT = 0.1

@register_env("PickCubeDiverse-v1", max_episode_steps=50)
class PickCubeDiverseEnv(PickCubeEnv):

    @property
    def _default_sensor_configs(self):
        return []

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            spawn_range = 0.15
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * spawn_range * 2 - spawn_range
            xyz[:, 1] = torch.rand((b,)) * spawn_range * 2 - spawn_range
            xyz[:, 2] = self.cube_half_size

            from mani_skill.envs.utils import randomization
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * spawn_range * 2 - spawn_range
            goal_xyz[:, 1] = torch.rand((b,)) * spawn_range * 2 - spawn_range
            goal_xyz[:, 2] = torch.rand((b,)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

def test_pick_cube_with_energy(
    model,
    config,
    seed=1,
    max_steps=200,
    energy_centers=None,
    energy_scales=None,
    energy_radius=0.3,
    save_video=True,
    video_output_dir=None,
):
    set_seeds(seed)

    print("=" * 60)
    print(f"ManiSkill 能量场测试 - seed={seed}")
    if energy_centers:
        print(f"能量场数量: {len(energy_centers)}")
        for i, center in enumerate(energy_centers):
            scale = energy_scales[i] if energy_scales else -1.0
            print(f"  能量场 {i+1}: 中心={center}, scale={scale}, radius={energy_radius}")
    print("=" * 60)

    n_obs_steps = config["n_obs_steps"]
    n_pred_steps = config["n_pred_steps"]
    n_points = config["dataset_config"]["n_points"]
    use_pc_color = config["dataset_config"]["use_pc_color"]
    norm_pcd_center = config["model_config"]["norm_pcd_center"]

    print("\n>>> 创建ManiSkill环境...")
    render_mode = "rgb_array" if save_video else "human"

    env_kwargs = {
        "num_envs": 1,
        "obs_mode": "sensor_data",
        "control_mode": "pd_ee_delta_pose",
        "robot_uids": "panda_wristcam",
        "render_mode": render_mode,
    }

    if save_video:
        env_kwargs["human_render_camera_configs"] = {
            "width": 1920,
            "height": 1080,
        }

    env = gym.make("PickCubeDiverse-v1", **env_kwargs)
    print(">>> 使用改进的PickCubeDiverse-v1 环境")

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    base_env = env.unwrapped
    cube_ids = get_cube_ids(env)
    robot_ids = get_robot_ids(env)

    selected_cameras = [
        "right_shoulder_camera",
        "left_shoulder_camera",
        "hand_camera",
    ]

    voxel_size = 0.003
    pcd_n_points = n_points

    existing_video_files = set()
    if save_video:
        if video_output_dir is None:
            video_output_dir = "outputs/videos/test_energy"
        os.makedirs(video_output_dir, exist_ok=True)
        video_output_dir_path = Path(video_output_dir)
        existing_video_files = set(video_output_dir_path.glob("*.mp4"))
        print(f"\n>>> 启用视频录制，保存目录: {video_output_dir}")
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=30,
            save_on_reset=True,
        )

    print(f"\n>>> 重置环境 (seed={seed})...")
    obs, _ = env.reset(seed=seed)
    robot_state_history = []

    goal_pos = base_env.goal_site.pose.p[0].cpu().numpy().astype(np.float32)
    print(f"Goal位置: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")

    energy_functions = None
    if energy_centers is not None and len(energy_centers) > 0:
        energy_functions = []
        for center in energy_centers:

            norm_center = np.array(center) - np.array(norm_pcd_center)
            energy_functions.append(EnergyFunction(norm_center, radius=energy_radius))
        print(f"\n>>> 创建了 {len(energy_functions)} 个能量函数")

    print("\n>>> 开始测试...")
    print("=" * 60)

    done = False
    step_count = 0

    ws_aabb = None

    while not done and step_count < max_steps:

        robot_state = get_robot_state(env).cpu()
        robot_state_history.append(robot_state)

        pointcloud_data = get_pointcloud_from_multi_cameras(
            obs=obs,
            ground_ids=ground_ids,
            voxel_size=voxel_size,
            n_points=pcd_n_points,
            ws_aabb=ws_aabb,
            cube_ids=cube_ids,
            robot_ids=robot_ids,
            selected_cameras=selected_cameras,
        )

        if use_pc_color:
            pcd = np.concatenate([pointcloud_data["xyz"], pointcloud_data["rgb"]], axis=-1)
        else:
            pcd = pointcloud_data["xyz"]

        robot_state_history_copy = robot_state_history.copy()
        while len(robot_state_history_copy) < n_obs_steps:
            robot_state_history_copy.insert(0, robot_state_history_copy[0])
        robot_state_history_copy = robot_state_history_copy[-n_obs_steps:]

        pcd_tensor = torch.from_numpy(pcd).float().unsqueeze(0).unsqueeze(0)
        pcd_tensor = pcd_tensor.repeat(1, n_obs_steps, 1, 1)

        robot_state_obs = torch.stack(robot_state_history_copy, dim=0).unsqueeze(0)
        pcd_tensor = pcd_tensor.to(DEVICE)
        robot_state_obs = robot_state_obs.to(DEVICE)

        goal_pos_tensor = torch.from_numpy(goal_pos).float().unsqueeze(0).unsqueeze(0)
        goal_pos_tensor = goal_pos_tensor.repeat(1, n_obs_steps, 1)
        goal_pos_tensor = goal_pos_tensor.to(DEVICE)

        with torch.no_grad():

            pcd_tensor[..., :3] -= torch.tensor(norm_pcd_center, device=DEVICE)
            robot_state_obs[..., :3] -= torch.tensor(norm_pcd_center, device=DEVICE)
            goal_pos_tensor[..., :3] -= torch.tensor(norm_pcd_center, device=DEVICE)

            pred_y = model.infer_y(
                pcd_tensor,
                robot_state_obs,
                goal_pos_tensor,
                energy_functions=energy_functions,
                energy_scales=energy_scales,
                energy_radius=energy_radius,
            )

            pred_y[..., :3] += torch.tensor(norm_pcd_center, device=DEVICE)

        pred_action = pred_y[0, 0]

        current_state = robot_state.to(DEVICE)
        delta_pos = (pred_action[:3] - current_state[:3]).cpu().numpy()
        delta_pos_normalized = np.clip(delta_pos / POS_LIMIT, -1.0, 1.0)

        delta_rot = np.zeros(3)

        pred_gripper_qpos = pred_action[9].item()
        gripper = (pred_gripper_qpos - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
        gripper = np.clip(gripper, -1.0, 1.0)

        GRIPPER_CLOSE_THRESHOLD = 0.025
        if pred_gripper_qpos < GRIPPER_CLOSE_THRESHOLD:
            gripper = -0.9

        action = np.concatenate([delta_pos_normalized, delta_rot, [gripper]])

        obs, reward, terminated, truncated, info = env.step(action)

        if not save_video:
            env.render()

        done = info["success"].item()
        step_count += 1

    success = info["success"].item()

    if save_video:
        print("\n>>> 准备保存视频...")

    env.close()

    if save_video:
        time.sleep(0.5)
        video_output_dir_path = Path(video_output_dir)
        all_video_files = set(video_output_dir_path.glob("*.mp4"))
        new_video_files = all_video_files - existing_video_files
        if new_video_files:
            latest_video = max(new_video_files, key=lambda x: x.stat().st_mtime)
            new_video_name = video_output_dir_path / f"seed_{seed}_energy.mp4"
            if new_video_name.exists():
                new_video_name.unlink()
            latest_video.rename(new_video_name)
            print(f"  视频已保存: {new_video_name}")

    print("\n" + "=" * 60)
    print(f"测试完成！")
    print(f"  seed: {seed}")
    print(f"  结果: {'成功' if success else '失败'}")
    print(f"  步数: {step_count}/{max_steps}")
    if energy_centers:
        print(f"  能量场: {len(energy_centers)} 个")
    print("=" * 60)

    return {
        "success": success,
        "step_count": step_count,
        "seed": seed,
        "max_steps": max_steps,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="在 pick-cube 模型上测试能量场引导"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="maniskill_train_pcd_from_three_cameras_more_gripper",
        help="checkpoint目录名称"
    )
    parser.add_argument(
        "--ckpt_episode",
        type=str,
        default="ep1500-ba160500",
        help="checkpoint的episode标识"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="环境seed"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="最大步数"
    )
    parser.add_argument(
        "--energy_centers",
        type=float,
        nargs="+",
        default=None,
        help="能量场中心坐标列表，格式: x1 y1 z1 x2 y2 z2 ... (每个能量场3个坐标)"
    )
    parser.add_argument(
        "--energy_scales",
        type=float,
        nargs="+",
        default=[-1.0],
        help="能量场缩放系数列表（负数表示排斥，正数表示吸引）"
    )
    parser.add_argument(
        "--energy_radius",
        type=float,
        default=0.3,
        help="能量场作用半径"
    )
    parser.add_argument(
        "--save_video",
        type=lambda x: x.lower() == 'true',
        nargs='?',
        const=True,
        default=True,
        help="是否保存视频"
    )
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default=None,
        help="视频保存目录"
    )

    args = parser.parse_args()

    energy_centers = None
    if args.energy_centers is not None:
        if len(args.energy_centers) % 3 != 0:
            raise ValueError("能量中心坐标必须是3的倍数（每个能量场需要x, y, z三个坐标）")
        energy_centers = [
            [args.energy_centers[i], args.energy_centers[i+1], args.energy_centers[i+2]]
            for i in range(0, len(args.energy_centers), 3)
        ]
        print(f"能量场中心: {energy_centers}")

    print(">>> 加载模型...")
    model, config = load_model_from_checkpoint_concat_goal(
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
    )

    test_pick_cube_with_energy(
        model=model,
        config=config,
        seed=args.seed,
        max_steps=args.max_steps,
        energy_centers=energy_centers,
        energy_scales=args.energy_scales,
        energy_radius=args.energy_radius,
        save_video=args.save_video,
        video_output_dir=args.video_output_dir,
    )
