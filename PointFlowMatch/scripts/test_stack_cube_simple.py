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

from eval_maniskill import (
    get_robot_state,
    prepare_observation,
)
from utils.utils_tool import load_model_from_checkpoint_concat_goal
from pfp import DEVICE, set_seeds
from mani_skill.utils.wrappers.record import RecordEpisode

from pfp.utils.pointcloud_utils import (
    get_pointcloud_from_multi_cameras,
    get_ground_ids,
    get_cube_ids,
    get_robot_ids,
)

import open3d as o3d

GRIPPER_LOWER = 0.0
GRIPPER_UPPER = 0.04
POS_LIMIT = 0.1

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

def test_single_episode(seed=1, ckpt_name="maniskill_train_stack_cube", ckpt_episode="latest", max_steps=200, episode_dir=None, save_video=True, video_output_dir=None, video_resolution=(1920, 1080)):

    if episode_dir is not None:
        actual_seed = load_seed_from_episode(episode_dir)
        print(f"\n>>> 从episode文件夹加载seed: {actual_seed}")
        print(f"  Episode目录: {episode_dir}")
    else:
        actual_seed = seed
        print(f"\n>>> 使用指定的seed: {actual_seed}")

    set_seeds(actual_seed)

    print("=" * 60)
    print(f"ManiSkill StackCube 测试 - seed={actual_seed}")
    if episode_dir is not None:
        print(f"Episode目录: {episode_dir}")
    print("=" * 60)

    print("\n>>> 加载模型...")
    model, config = load_model_from_checkpoint_concat_goal(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
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

    env = gym.make("StackCube-v1", **env_kwargs)
    print(">>> 使用 StackCube-v1 环境")

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    print(f"Ground IDs: {ground_ids}")

    base_env = env.unwrapped
    cube_ids = get_cube_ids(env)
    robot_ids = get_robot_ids(env)
    print(f"Cube IDs: {cube_ids}")
    print(f"Robot IDs: {robot_ids}")

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

            video_output_dir = "outputs/videos/test_stack_cube_simple"
        os.makedirs(video_output_dir, exist_ok=True)

        video_output_dir_path = Path(video_output_dir)
        existing_video_files = set(video_output_dir_path.glob("*.mp4"))
        print(f"\n>>> 启用视频录制，保存目录: {video_output_dir}")
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=10,
            save_on_reset=True,
        )

    print(f"\n>>> 重置环境 (seed={actual_seed})...")
    obs, _ = env.reset(seed=actual_seed)
    robot_state_history = []
    predicted_robot_states = []

    base_env = env.unwrapped
    cubeB_pos = base_env.cubeB.pose.p[0].cpu().numpy().astype(np.float32)
    cube_half_size = base_env.cube_half_size[2].item()
    goal_pos = cubeB_pos
    print(f"CubeA位置: [{base_env.cubeA.pose.p[0, 0]:.4f}, {base_env.cubeA.pose.p[0, 1]:.4f}, {base_env.cubeA.pose.p[0, 2]:.4f}]")
    print(f"CubeB位置: [{cubeB_pos[0]:.4f}, {cubeB_pos[1]:.4f}, {cubeB_pos[2]:.4f}]")
    print(f"Goal位置（cubeB上方）: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")

    print("\n>>> 开始测试...")
    print("=" * 60)

    done = False
    step_count = 0

    ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.3, -0.3, 0.0),
        max_bound=(0.3, 0.3, 0.5),
    )

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

        visualize_pointcloud(pointcloud_data["xyz"], pointcloud_data["rgb"], title=f"Step")

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

            pred_y[..., :3] += torch.tensor(norm_pcd_center, device=DEVICE)

        print(f"  Step {step_count}: 推理得到 {n_pred_steps} 个预测动作，开始执行前32个...")

        for i in range(32):

            pred_action = pred_y[0, i]
            print(f"pred_action: {pred_action}")

            pred_robot_state = pred_action.cpu().numpy()
            predicted_robot_states.append(pred_robot_state.copy())

            current_robot_state = get_robot_state(env).cpu()

            if i == 0:
                current_pos = current_robot_state[:3].numpy()
                pred_pos = pred_action[:3].cpu().numpy()
                dist_to_goal = np.linalg.norm(current_pos - goal_pos)
                dist_pred_to_goal = np.linalg.norm(pred_pos - goal_pos)

            current_state = current_robot_state.to(DEVICE)
            delta_pos = (pred_action[:3] - current_state[:3]).cpu().numpy()
            delta_pos_normalized = np.clip(delta_pos / POS_LIMIT, -1.0, 1.0)

            delta_rot = np.zeros(3)

            pred_gripper_qpos = pred_action[9].item()
            gripper = (pred_gripper_qpos - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
            gripper = np.clip(gripper, -1.0, 1.0)

            GRIPPER_CLOSE_THRESHOLD = 0.025
            if pred_gripper_qpos < GRIPPER_CLOSE_THRESHOLD:

                gripper = -0.9
                if i == 0:
                    print(f"  Step {step_count}, Action {i}: 手动夹紧gripper (pred_qpos={pred_gripper_qpos:.4f} -> action={gripper:.2f})")

            action = np.concatenate([delta_pos_normalized, delta_rot, [gripper]])

            obs, reward, terminated, truncated, info = env.step(action)

            if not save_video:
                env.render()

            new_robot_state = get_robot_state(env).cpu()
            robot_state_history.append(new_robot_state)

            done = info["success"].item()
            step_count += 1

            if done:
                print(f"  Step {step_count}: 任务完成！")
                break

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

    if save_video and predicted_robot_states:
        robot_state_file = Path(video_output_dir) / f"{file_suffix}_predicted_robot_states.txt"
        with open(robot_state_file, 'w') as f:
            f.write(f"# Predicted Robot States for {file_suffix}\n")
            if episode_dir is not None:
                f.write(f"# Episode目录: {episode_dir}\n")
            f.write(f"# Seed: {actual_seed}\n")
            f.write(f"# Format: [ee_pos(3), ee_rot_6d(6), gripper(1)] = 10 dimensions\n")
            f.write(f"# Total steps: {len(predicted_robot_states)}\n")
            f.write(f"# Success: {success}\n")
            f.write(f"# Episode length: {step_count}\n")
            f.write("#\n")
            f.write("# Step | ee_pos_x | ee_pos_y | ee_pos_z | rot_6d_0 | rot_6d_1 | rot_6d_2 | rot_6d_3 | rot_6d_4 | rot_6d_5 | gripper\n")
            for i, state in enumerate(predicted_robot_states):
                f.write(f"{i:4d} | {state[0]:.6f} | {state[1]:.6f} | {state[2]:.6f} | "
                       f"{state[3]:.6f} | {state[4]:.6f} | {state[5]:.6f} | "
                       f"{state[6]:.6f} | {state[7]:.6f} | {state[8]:.6f} | {state[9]:.6f}\n")
        print(f"  预测robot_state已保存: {robot_state_file}")

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
        description="简单的ManiSkill StackCube测试脚本",
        epilog="""
使用示例:
  python scripts/test_stack_cube_simple.py
  python scripts/test_stack_cube_simple.py --ckpt_name maniskill_train_stack_cube --ckpt_episode latest
  python scripts/test_stack_cube_simple.py --episode_dir ./data/stack_cube_demo/episode1
        """
    )
    parser.add_argument("--ckpt_name", type=str, default="maniskill_train_stack_cube_concat_goal_pos_big", help="checkpoint目录名称")
    parser.add_argument("--ckpt_episode", type=str, default="ep1500-ba43500", help="checkpoint的episode标识")
    parser.add_argument("--seed", type=int, default=1, help="测试seed（如果未指定episode_dir时使用，默认: 1）")
    parser.add_argument("--max_steps", type=int, default=500, help="最大步数")
    parser.add_argument("--episode_dir", type=str, default=None, help="episode文件夹路径（如果提供，则从info.npz加载seed）")
    parser.add_argument("--save_video", type=lambda x: x.lower() == 'true', nargs='?', const=True, default=True, help="是否保存视频 (默认: True, 使用 --save_video False 来禁用)")
    parser.add_argument("--video_output_dir", type=str, default=None, help="视频保存目录 (默认: outputs/videos/test_stack_cube_simple/)")
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
