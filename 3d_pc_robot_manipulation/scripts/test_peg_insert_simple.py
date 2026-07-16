import numpy as np
import torch
import gymnasium as gym
import os
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MANISKILL_ROOT))

import mani_skill.envs

sys.path.insert(0, str(Path(__file__).parent))

from pfp.utils.get_robot_state import get_robot_state, get_robot_state_no_transform

from utils.utils_tool import load_action_model_from_checkpoint_concat_goal, load_action_model_from_checkpoint_concat_goal_rot_no_transform
from pfp import DEVICE, set_seeds
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env

from pfp.utils.pointcloud_utils_peg_insertion import (
    get_pointcloud_from_multi_cameras_peg_insert,
    get_ground_ids,
    get_peg_ids,
    get_box_ids,
    get_robot_ids,
)

import open3d as o3d

def visualize_pointcloud(xyz: np.ndarray, rgb: np.ndarray, title: str = "Point Cloud"):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    print(f"\n>>> 可视化点云: {title}")
    print(f"  点数: {len(xyz)}")

    valid_mask = np.any(xyz != 0, axis=1)
    num_valid_points = np.sum(valid_mask)
    print(f"  有效点数: {num_valid_points}")
    print(f"  无效点数（零填充）: {len(xyz) - num_valid_points}")

    if num_valid_points > 0:
        print(f"  XYZ范围: X[{xyz[valid_mask, 0].min():.3f}, {xyz[valid_mask, 0].max():.3f}], "
              f"Y[{xyz[valid_mask, 1].min():.3f}, {xyz[valid_mask, 1].max():.3f}], "
              f"Z[{xyz[valid_mask, 2].min():.3f}, {xyz[valid_mask, 2].max():.3f}]")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.5, -0.5, -0.7])
    view_ctl.set_lookat([0.0, 0.0, 0.2])
    view_ctl.set_up([0.0, 0.0, 1.0])
    view_ctl.set_zoom(0.7)

    print("  提示: 按 'Q' 或关闭窗口退出可视化")
    print("  坐标轴: 红色=X轴, 绿色=Y轴, 蓝色=Z轴")
    vis.run()
    vis.destroy_window()

@register_env("PegInsertionSideNoBaseCam-v1", max_episode_steps=100)
class PegInsertionSideNoBaseCamEnv(PegInsertionSideEnv):

    @property
    def _default_sensor_configs(self):
        return []

def load_seed_from_episode(episode_dir: str) -> int:
    episode_path = Path(episode_dir)
    info_path = episode_path / "info.npz"

    if not info_path.exists():
        raise FileNotFoundError(f"未找到info.npz文件: {info_path}")

    info = np.load(info_path)
    seed = info.get("seed", None)

    if seed is None:
        raise ValueError(f"info.npz中未找到seed字段: {info_path}")

    return int(seed)

def test_single_episode(seed=1, ckpt_name="maniskill_train_peg_insertion_action", ckpt_episode="latest", max_steps=200, episode_dir=None, save_video=True, video_output_dir=None, video_resolution=(1920, 1080)):

    if episode_dir is not None:
        actual_seed = load_seed_from_episode(episode_dir)
        print(f"\n>>> 从episode文件夹加载seed: {actual_seed}")
        print(f"  Episode目录: {episode_dir}")
    else:
        actual_seed = seed
        print(f"\n>>> 使用指定的seed: {actual_seed}")

    set_seeds(actual_seed)

    print("=" * 60)
    print(f"ManiSkill PegInsertionSide 简单测试 - seed={actual_seed}")
    if episode_dir is not None:
        print(f"Episode目录: {episode_dir}")
    print("=" * 60)

    print("\n>>> 加载模型...")

    model, config = load_action_model_from_checkpoint_concat_goal_rot_no_transform(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
    )

    n_obs_steps = config["n_obs_steps"]
    n_pred_steps = config["n_pred_steps"]
    n_points = config["dataset_config"]["n_points"]
    print(f"n_points: {n_points}")
    use_pc_color = config["dataset_config"]["use_pc_color"]
    norm_pcd_center = config["model_config"]["norm_pcd_center"]
    y_dim = config["y_dim"]

    print(f"\n模型配置:")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  n_pred_steps: {n_pred_steps}")
    print(f"  n_points: {n_points}")
    print(f"  use_pc_color: {use_pc_color}")
    print(f"  y_dim: {y_dim} (action模式)")
    print(f"  num_k_infer: {model.num_k_infer}")
    print(f"  flow_schedule: {model.flow_schedule}")

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
        video_width, video_height = video_resolution
        env_kwargs["human_render_camera_configs"] = {
            "width": video_width,
            "height": video_height,
        }
        print(f"  视频分辨率: {video_width}x{video_height}")
        print(f"  渲染模式: rgb_array (用于视频录制)")
    else:
        print(f"  渲染模式: human (显示窗口)")

    env = gym.make("PegInsertionSideNoBaseCam-v1", **env_kwargs)
    print(">>> 使用PegInsertionSideNoBaseCam-v1 环境")

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    print(f"Ground IDs: {ground_ids}")

    base_env = env.unwrapped
    peg_ids = get_peg_ids(env)
    box_ids = get_box_ids(env)
    robot_ids = get_robot_ids(env)

    selected_cameras = [
        "right_shoulder_camera",
        "left_shoulder_camera",
        "right_rear_camera"
    ]

    voxel_size = 0.003
    pcd_n_points = n_points

    existing_video_files = set()
    if save_video:
        if video_output_dir is None:

            video_output_dir = "outputs/videos/test_peg_insert_simple"
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

    print(f"\n>>> 重置环境 (seed={actual_seed})...")
    obs, _ = env.reset(seed=actual_seed)
    robot_state_history = []
    predicted_actions = []

    base_env = env.unwrapped

    goal_pos = base_env.box_hole_pose.raw_pose[0].cpu().numpy()[:3].astype(np.float32)
    print(f"Goal位置 (box位置): [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")

    print("\n>>> 开始测试...")
    print("=" * 60)

    done = False
    step_count = 0

    action_queue = []

    ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.5, -0.5, 0.0),
        max_bound=(0.5, 0.5, 0.5),
    )

    while not done and step_count < max_steps:

        if len(action_queue) == 0:

            robot_state = get_robot_state_no_transform(env).cpu()
            robot_state_history.append(robot_state)

            pointcloud_data = get_pointcloud_from_multi_cameras_peg_insert(
                obs=obs,
                ground_ids=ground_ids,
                voxel_size=voxel_size,
                n_points=pcd_n_points,
                ws_aabb=ws_aabb,
                robot_ids=robot_ids,
                peg_ids=peg_ids,
                box_ids=box_ids,
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

                pred_y = model.infer_y(pcd_tensor, robot_state_obs, goal_pos_tensor)

            print(f"  Step {step_count}: 推理得到 {n_pred_steps} 个预测动作，开始处理...")

            for i in range(4):
                pred_action = pred_y[0, i].cpu().numpy()

                action = pred_action.copy()

                action[:6] = np.clip(action[:6], -1.0, 1.0)
                action[6] = np.clip(action[6], -1.0, 1.0)

                action_queue.append(action)

            print(f"  Step {step_count}: 已将 {len(action_queue)} 个动作加入队列")

        action = action_queue.pop(0)

        predicted_actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)

        if not save_video:
            env.render()

        done = info["success"].item()
        step_count += 1

    success = info["success"].item()

    if save_video:
        print("\n>>> 准备保存视频...")

    env.close()

    if episode_dir is not None:
        episode_name = Path(episode_dir).name
        file_suffix = episode_name
    else:
        file_suffix = f"seed_{actual_seed}"

    if save_video:
        import time
        video_output_dir_path = Path(video_output_dir)

        time.sleep(0.5)

        all_video_files = set(video_output_dir_path.glob("*.mp4"))
        new_video_files = all_video_files - existing_video_files
        if new_video_files:

            latest_video = max(new_video_files, key=lambda x: x.stat().st_mtime)
            new_video_name = video_output_dir_path / f"{file_suffix}.mp4"

            if new_video_name.exists():
                new_video_name.unlink()

            latest_video.rename(new_video_name)
            print(f"  视频已保存: {new_video_name}")
        else:
            print(f"  警告: 未找到新生成的视频文件，请检查目录: {video_output_dir_path}")

    if save_video and predicted_actions:
        action_file = Path(video_output_dir) / f"{file_suffix}_predicted_actions.txt"
        with open(action_file, 'w') as f:
            f.write(f"# Predicted Actions for {file_suffix}\n")
            if episode_dir is not None:
                f.write(f"# Episode目录: {episode_dir}\n")
            f.write(f"# Seed: {actual_seed}\n")
            f.write(f"# Format: [delta_pos(3), delta_rot(3), gripper(1)] = 7 dimensions\n")
            f.write(f"# Total steps: {len(predicted_actions)}\n")
            f.write(f"# Success: {success}\n")
            f.write(f"# Episode length: {step_count}\n")
            f.write("#\n")
            f.write("# Step | delta_pos_x | delta_pos_y | delta_pos_z | delta_rot_x | delta_rot_y | delta_rot_z | gripper\n")
            for i, action in enumerate(predicted_actions):
                f.write(f"{i:4d} | {action[0]:.6f} | {action[1]:.6f} | {action[2]:.6f} | "
                       f"{action[3]:.6f} | {action[4]:.6f} | {action[5]:.6f} | {action[6]:.6f}\n")
        print(f"  预测action已保存: {action_file}")

    print("\n" + "=" * 60)
    print(f"测试完成！")
    if episode_dir is not None:
        print(f"  episode目录: {episode_dir}")
    print(f"  seed: {actual_seed}")
    print(f"  结果: {'成功' if success else '失败'}")
    print(f"  步数: {step_count}/{max_steps}")
    if save_video:
        print(f"  视频保存目录: {video_output_dir}")
    print("=" * 60)

    return success, step_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="简单的ManiSkill PegInsertionSide测试脚本",
        epilog="""
使用示例:
  python scripts/test_peg_insert_simple.py
  python scripts/test_peg_insert_simple.py --ckpt_name maniskill_train_peg_insertion_action --ckpt_episode latest
  python scripts/test_peg_insert_simple.py --episode_dir ./data/peg_demo_train/episode0
        """
    )
    parser.add_argument("--ckpt_name", type=str, default="maniskill_train_peg_insertion_action_concat_goal_action_length_1", help="checkpoint目录名称")
    parser.add_argument("--ckpt_episode", type=str, default="ep1500-ba160500", help="checkpoint的episode标识")
    parser.add_argument("--seed", type=int, default=1, help="测试seed（如果未指定episode_dir时使用，默认: 1）")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大步数")
    parser.add_argument("--episode_dir", type=str, default=None, help="episode文件夹路径（如果提供，则从info.npz加载seed）")
    parser.add_argument("--save_video", type=lambda x: x.lower() == 'true', nargs='?', const=True, default=True, help="是否保存视频 (默认: True, 使用 --save_video False 来禁用)")
    parser.add_argument("--video_output_dir", type=str, default=None, help="视频保存目录 (默认: outputs/videos/test_peg_insert_simple/)")
    parser.add_argument("--video_width", type=int, default=1920, help="视频宽度 (默认: 1920)")
    parser.add_argument("--video_height", type=int, default=1080, help="视频高度 (默认: 1080)")

    args = parser.parse_args()

    test_single_episode(
        seed=args.seed,
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
        max_steps=args.max_steps,
        episode_dir=args.episode_dir,
        save_video=args.save_video,
        video_output_dir=args.video_output_dir,
        video_resolution=(args.video_width, args.video_height),
    )
