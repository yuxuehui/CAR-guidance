#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import gymnasium as gym
import json
import argparse
from datetime import datetime

import mani_skill.envs
from scripts.eval_maniskill import get_robot_state
from pfp.utils.pointcloud_utils import (
    get_pointcloud_from_multi_cameras,
    get_ground_ids,
    get_cube_ids,
    get_robot_ids,
)
from pfp import set_seeds, DEVICE
from utils.utils_tool import load_model_from_checkpoint_concat_goal
import open3d as o3d
import imageio

def load_seed_from_episode(episode_dir: Path) -> int:
    info_path = episode_dir / "info.npz"

    if not info_path.exists():
        raise FileNotFoundError(f"未找到 info.npz 文件: {info_path}")

    info = np.load(info_path)
    seed = info.get("seed", None)

    if seed is None:
        raise ValueError(f"info.npz 中未找到 seed 字段: {info_path}")

    return int(seed)

def run_episode_with_seed(
    seed: int,
    model,
    config: dict,
    save_video: bool = False,
    videos_path: Path = None,
) -> dict:

    GRIPPER_LOWER = 0.0
    GRIPPER_UPPER = 0.04
    POS_LIMIT = 0.1
    GRIPPER_CLOSE_THRESHOLD = 0.025

    set_seeds(seed)

    n_obs_steps = config["n_obs_steps"]
    n_pred_steps = config["n_pred_steps"]
    n_points = config["dataset_config"]["n_points"]
    use_pc_color = config["dataset_config"]["use_pc_color"]
    norm_pcd_center = config["model_config"]["norm_pcd_center"]

    print(f"    [DEBUG] 模型配置: n_obs_steps={n_obs_steps}, n_pred_steps={n_pred_steps}, n_points={n_points}")
    print(f"    [DEBUG] use_pc_color={use_pc_color}, norm_pcd_center={norm_pcd_center}")
    print(f"    [DEBUG] model.num_k_infer={model.num_k_infer}, model.flow_schedule={model.flow_schedule}")

    render_mode = "rgb_array" if save_video else "human"

    env_kwargs = {
        "num_envs": 1,
        "obs_mode": "sensor_data",
        "control_mode": "pd_ee_delta_pose",
        "robot_uids": "panda_wristcam",
        "render_mode": render_mode,
    }

    if save_video:
        video_width, video_height = 1920, 1080
        env_kwargs["human_render_camera_configs"] = {
            "width": video_width,
            "height": video_height,
        }

    env = gym.make("StackCube-v1", **env_kwargs)

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    cube_ids = get_cube_ids(env)
    robot_ids = get_robot_ids(env)

    print(f"    [DEBUG] Ground IDs: {ground_ids}")
    print(f"    [DEBUG] Cube IDs: {cube_ids}")
    print(f"    [DEBUG] Robot IDs: {robot_ids}")

    selected_cameras = ["right_shoulder_camera", "left_shoulder_camera", "hand_camera"]
    voxel_size = 0.003
    pcd_n_points = n_points
    ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.3, -0.3, 0.0),
        max_bound=(0.3, 0.3, 0.5),
    )

    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped

    cubeB_pos = base_env.cubeB.pose.p[0].cpu().numpy().astype(np.float32)
    goal_pos = cubeB_pos

    print(f"    [DEBUG] CubeA位置: [{base_env.cubeA.pose.p[0, 0]:.4f}, {base_env.cubeA.pose.p[0, 1]:.4f}, {base_env.cubeA.pose.p[0, 2]:.4f}]")
    print(f"    [DEBUG] CubeB位置: [{cubeB_pos[0]:.4f}, {cubeB_pos[1]:.4f}, {cubeB_pos[2]:.4f}]")
    print(f"    [DEBUG] Goal位置: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")

    robot_state_history = []
    trajectory = []
    episode_frames = []
    done = False
    step_count = 0
    max_steps = 300

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

            pred_y = model.infer_y(pcd_tensor, robot_state_obs, goal_pos_tensor)

            pred_y[..., :3] += torch.tensor(norm_pcd_center, device=DEVICE)

        if step_count == 0:
            print(f"    [DEBUG] Step 0: pred_y shape={pred_y.shape}")
            print(f"    [DEBUG] Step 0: pred_y[0,0,:3]={pred_y[0,0,:3].cpu().numpy()}")
            print(f"    [DEBUG] Step 0: pred_y[0,0,9]={pred_y[0,0,9].item():.4f}")

        for i in range(32):

            pred_action = pred_y[0, i]

            current_robot_state = get_robot_state(env).cpu()

            current_state = current_robot_state.to(DEVICE)
            delta_pos = (pred_action[:3] - current_state[:3]).cpu().numpy()
            delta_pos_normalized = np.clip(delta_pos / POS_LIMIT, -1.0, 1.0)

            delta_rot = np.zeros(3)

            pred_gripper_qpos = pred_action[9].item()
            gripper = (pred_gripper_qpos - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
            gripper = np.clip(gripper, -1.0, 1.0)

            if pred_gripper_qpos < GRIPPER_CLOSE_THRESHOLD:
                gripper = -0.9

            action = np.concatenate([delta_pos_normalized, delta_rot, [gripper]])

            if step_count == 0 and i == 0:
                print(f"    [DEBUG] Action 0: delta_pos={delta_pos}, delta_pos_normalized={delta_pos_normalized}")
                print(f"    [DEBUG] Action 0: pred_gripper_qpos={pred_gripper_qpos:.4f}, gripper={gripper:.2f}")
                print(f"    [DEBUG] Action 0: action={action}")

            obs, reward, terminated, truncated, info = env.step(action)

            if save_video:
                frame = env.render()
                if torch.is_tensor(frame):
                    frame = frame.detach().cpu().numpy()
                if isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                episode_frames.append(np.array(frame))

            new_robot_state = get_robot_state(env).cpu()
            robot_state_history.append(new_robot_state)

            done = info["success"].item()
            step_count += 1

            if done:
                break

        if done:
            break

    success = info["success"].item()

    video_path = None
    if save_video and len(episode_frames) > 0 and videos_path is not None:
        video_name = f"demo_seed_{seed}.mp4"
        video_path = str(videos_path / video_name)
        imageio.mimsave(video_path, episode_frames, fps=20)

    env.close()

    trajectory = [state[:3].numpy().tolist() for state in robot_state_history]

    return {
        'success': success,
        'steps': step_count,
        'trajectory': trajectory,
        'goal_pos': goal_pos.tolist(),
        'video_path': video_path,
    }

def collect_success_demos_from_episodes(
    data_dir: str = "data/stack_cube_demo_big_train",
    num_demos: int = 10,
    ckpt_name: str = "maniskill_train_stack_cube_concat_goal_pos_big",
    ckpt_episode: str = "ep1500-ba43500",
    output_dir: str = "experiments_stack_cube/data/success_demos",
    save_videos: bool = True,
) -> None:
    print("=" * 60)
    print("从 episode 数据中读取 seed 并使用模型推理收集成功演示")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"目标数量: {num_demos}")
    print(f"模型: {ckpt_name}/{ckpt_episode}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    videos_path = output_path / "videos"
    if save_videos:
        videos_path.mkdir(parents=True, exist_ok=True)

    print("\n>>> 加载模型...")
    model, config = load_model_from_checkpoint_concat_goal(
        ckpt_name=ckpt_name,
        ckpt_episode=ckpt_episode,
    )
    print("模型加载完成")

    episode_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if len(episode_dirs) == 0:
        raise ValueError(f"在 {data_path} 中未找到任何 episode 目录")

    print(f"\n>>> 找到 {len(episode_dirs)} 个 episode 目录")

    success_demos = []
    processed = 0

    for episode_dir in episode_dirs:
        if len(success_demos) >= num_demos:
            break

        processed += 1
        episode_name = episode_dir.name

        try:

            seed = load_seed_from_episode(episode_dir)

            print(f"\n  [{processed}/{len(episode_dirs)}] {episode_name}: seed={seed}, 开始推理...")

            result = run_episode_with_seed(
                seed=seed,
                model=model,
                config=config,
                save_video=save_videos,
                videos_path=videos_path if save_videos else None,
            )

            if result['success']:

                demo_data = {
                    'demo_id': len(success_demos) + 1,
                    'seed': int(seed),
                    'steps': result['steps'],
                    'trajectory': result['trajectory'],
                    'goal_pos': result['goal_pos'],
                    'video_path': result['video_path'],
                    'episode_dir': str(episode_dir),
                    'timestamp': datetime.now().isoformat(),
                }

                success_demos.append(demo_data)
                print(f"  ✓ 成功！步数: {result['steps']}")
            else:
                print(f"  ✗ 失败，步数: {result['steps']}")

        except Exception as e:
            print(f"  [{processed}/{len(episode_dirs)}] {episode_name}: 跳过（错误: {e}）")
            continue

    output_file = output_path / "success_demos.json"
    output_data = {
        'num_demos': len(success_demos),
        'collection_date': datetime.now().isoformat(),
        'data_source': str(data_path),
        'model': f"{ckpt_name}/{ckpt_episode}",
        'demos': success_demos,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("收集完成！")
    print("=" * 60)
    print(f"成功收集: {len(success_demos)}/{num_demos}")
    print(f"处理的 episode: {processed}/{len(episode_dirs)}")
    if processed > 0:
        print(f"成功率: {len(success_demos)/processed:.2%}")
    print(f"数据已保存至: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从已有 episode 数据中读取 seed 并使用模型推理收集成功演示",
        epilog="""
使用示例:
  python experiments_stack_cube/scripts/collect_success_demo.py --num_demos 10
  python experiments_stack_cube/scripts/collect_success_demo.py --data_dir data/stack_cube_demo_big_train --num_demos 15
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/stack_cube_demo_big_train",
        help="episode 数据目录（默认: data/stack_cube_demo_big_train）"
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=10,
        help="需要收集的成功演示数量（默认: 10）"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="maniskill_train_stack_cube_concat_goal_pos_big",
        help="checkpoint 名称（默认: maniskill_train_stack_cube_concat_goal_pos_big）"
    )
    parser.add_argument(
        "--ckpt_episode",
        type=str,
        default="ep1500-ba43500",
        help="checkpoint episode（默认: ep1500-ba43500）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_stack_cube/data/success_demos",
        help="输出目录（默认: experiments_stack_cube/data/success_demos）"
    )
    parser.add_argument(
        "--save_videos",
        type=lambda x: x.lower() == 'true',
        nargs='?',
        const=True,
        default=True,
        help="是否保存视频（默认: True）"
    )

    args = parser.parse_args()

    collect_success_demos_from_episodes(
        data_dir=args.data_dir,
        num_demos=args.num_demos,
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
        output_dir=args.output_dir,
        save_videos=args.save_videos,
    )
