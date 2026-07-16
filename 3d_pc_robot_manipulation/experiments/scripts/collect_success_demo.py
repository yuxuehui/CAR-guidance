#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys
from typing import Dict, List, Any
import os
import time
import traceback
import imageio

REPO_ROOT = Path(__file__).parent.parent.parent
MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MANISKILL_ROOT))

from scripts.test_pick_cube_simple import PickCubeDiverseEnv
from pfp import set_seeds, DEVICE
import gymnasium as gym
import mani_skill.envs
from scripts.eval_maniskill import get_robot_state
from pfp.utils.pointcloud_utils import (
    get_pointcloud_from_multi_cameras,
    get_ground_ids,
    get_cube_ids,
    get_robot_ids,
)
from utils.utils_tool import load_model_from_checkpoint_concat_goal
import open3d as o3d

GRIPPER_LOWER = 0.0
GRIPPER_UPPER = 0.04
POS_LIMIT = 0.1

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def to_numpy(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    if isinstance(data, np.ndarray) and data.ndim == 4:
        if data.shape[0] == 1:
            data = data[0]

    return np.array(data)

def collect_success_demos(
    ckpt_name: str,
    ckpt_episode: str,
    num_success: int = 50,
    max_attempts: int = 500,
    max_steps: int = 500,
    output_dir: str = "experiments/data/success_demos",
    seed_start: int = 1000,
    save_video: bool = False,
    visualize: bool = False,
    num_actions_per_step: int = 32,
    min_start_dist: float = 0.05,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_dir = output_path / "videos"
    if save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(">>> 加载模型...")
    model, config = load_model_from_checkpoint_concat_goal(ckpt_name, ckpt_episode)

    n_obs_steps = config["n_obs_steps"]
    n_pred_steps = config["n_pred_steps"]
    n_points = config["dataset_config"]["n_points"]
    use_pc_color = config.get("dataset_config", {}).get("use_pc_color", False)
    norm_pcd_center = config["model_config"]["norm_pcd_center"]

    print(">>> 创建环境...")

    render_mode = "human" if visualize else "rgb_array"

    env_kwargs = {
        "num_envs": 1,
        "obs_mode": "sensor_data",
        "control_mode": "pd_ee_delta_pose",
        "robot_uids": "panda_wristcam",
        "render_mode": render_mode,
    }

    if save_video:
        env_kwargs["human_render_camera_configs"] = {"width": 1280, "height": 720}

    env = gym.make("PickCubeDiverse-v1", **env_kwargs)

    env.reset(seed=0)
    ground_ids = to_numpy(get_ground_ids(env))
    cube_ids = to_numpy(get_cube_ids(env))
    robot_ids = to_numpy(get_robot_ids(env))

    selected_cameras = ["right_shoulder_camera", "left_shoulder_camera", "hand_camera"]
    ws_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.3, -0.3, 0.0), max_bound=(0.3, 0.3, 0.5))

    success_demos = []
    success_count = 0
    attempts = 0
    current_seed = seed_start

    print("=" * 60)
    print(f"数据采集开始 | 目标成功数: {num_success} | 起始种子: {seed_start}")
    print("=" * 60)

    try:
        while success_count < num_success and attempts < max_attempts:
            attempts += 1
            current_seed += 1

            print(f"\n[尝试 {attempts}/{max_attempts}] 正在测试 Seed: {current_seed}")

            try:
                set_seeds(current_seed)
                obs, _ = env.reset(seed=current_seed)
                base_env = env.unwrapped

                goal_pos = base_env.goal_site.pose.p[0].detach().cpu().numpy().astype(np.float32)

                cube_pos = base_env.cube.pose.p[0].detach().cpu().numpy().astype(np.float32)
                dist_to_goal = np.linalg.norm(cube_pos - goal_pos)
                print(f"  └─ 物体与目标的初始距离: {dist_to_goal:.4f}m")

                if dist_to_goal < min_start_dist:
                    print(f"  └─ ⏩ 跳过: 物体离目标太近 (距离: {dist_to_goal:.4f}m < {min_start_dist}m)")
                    continue

                robot_state_history = []
                trajectory = []
                episode_frames = []
                done = False
                step_count = 0

                if save_video:
                    frame = env.render()
                    episode_frames.append(to_numpy(frame))

                while not done and step_count < max_steps:

                    cur_state = get_robot_state(env).detach().cpu()
                    robot_state_history.append(cur_state)
                    trajectory.append(cur_state[:3].numpy().tolist())

                    pcd_data = get_pointcloud_from_multi_cameras(
                        obs, ground_ids, 0.003, n_points, ws_aabb, cube_ids, robot_ids, selected_cameras
                    )

                    xyz = to_numpy(pcd_data["xyz"])
                    if use_pc_color:
                        rgb = to_numpy(pcd_data["rgb"])
                        pcd = np.concatenate([xyz, rgb], axis=-1)
                    else:
                        pcd = xyz

                    hist = robot_state_history.copy()
                    while len(hist) < n_obs_steps: hist.insert(0, hist[0])
                    hist = hist[-n_obs_steps:]

                    pcd_t = torch.from_numpy(pcd).float().unsqueeze(0).repeat(1, n_obs_steps, 1, 1).to(DEVICE)
                    state_t = torch.stack(hist, dim=0).unsqueeze(0).to(DEVICE)
                    goal_t = torch.from_numpy(goal_pos).float().unsqueeze(0).repeat(1, n_obs_steps, 1).to(DEVICE)

                    with torch.no_grad():
                        off = torch.tensor(norm_pcd_center, device=DEVICE)
                        pcd_t[..., :3] -= off
                        state_t[..., :3] -= off
                        goal_t[..., :3] -= off
                        pred_y = model.infer_y(pcd_t, state_t, goal_t)
                        pred_y[..., :3] += off

                    for i in range(min(num_actions_per_step, n_pred_steps)):
                        target = pred_y[0, i]
                        now_state = get_robot_state(env).detach().cpu()

                        d_pos = (target[:3] - now_state[:3].to(DEVICE)).detach().cpu().numpy()
                        d_pos_norm = np.clip(d_pos / POS_LIMIT, -1.0, 1.0)

                        q_grip = target[9].item()
                        g_val = (q_grip - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
                        if q_grip < 0.025: g_val = -0.9

                        action = np.concatenate([d_pos_norm, np.zeros(3), [np.clip(g_val, -1.0, 1.0)]])

                        obs, _, _, _, info = env.step(action)

                        if save_video:
                            frame = env.render()
                            episode_frames.append(to_numpy(frame))
                        elif visualize:
                            env.render()

                        done = bool(info["success"].item())
                        step_count += 1
                        if done: break

                if done:
                    success_count += 1
                    video_path = None

                    if save_video and len(episode_frames) > 0:
                        video_name = f"demo_seed_{current_seed}.mp4"
                        video_path = str(video_dir / video_name)

                        imageio.mimsave(video_path, episode_frames, fps=20)

                    print(f"  └─ ✅ 成功 {success_count}/{num_success} | 步数: {step_count}")

                    success_demos.append({
                        "demo_id": success_count,
                        "seed": int(current_seed),
                        "step_count": int(step_count),
                        "goal_pos": goal_pos.tolist(),
                        "trajectory": trajectory,
                        "video_path": video_path
                    })
                else:
                    print(f"  └─ ❌ 失败: 未能在 {max_steps} 步内完成任务")

            except Exception as e:
                print(f"  └─ ⚠️ 异常 (Seed {current_seed}): {e}")
                continue

    finally:

        print("\n>>> 正在关闭环境并清理资源...")
        if 'env' in locals():
            env.close()

    output_file = output_path / "success_demos.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "ckpt": f"{ckpt_name}/{ckpt_episode}",
            "num_success": len(success_demos),
            "total_attempts": attempts,
            "demos": success_demos,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✨ 采集完成！结果已保存至: {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ManiSkill 数据采集脚本 (手动视频保存版)")
    parser.add_argument("--ckpt_name", type=str, default="maniskill_train_pcd_from_three_cameras_more_gripper")
    parser.add_argument("--ckpt_episode", type=str, default="ep1500-ba160500")
    parser.add_argument("--num_success", type=int, default=50)
    parser.add_argument("--max_attempts", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="experiments/data/success_demos")
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--save_video", type=str2bool, default=False)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--num_actions_per_step", type=int, default=32)
    parser.add_argument("--min_start_dist", type=float, default=0.05, help="物体与目标的最小初始距离，太近则跳过")

    args = parser.parse_args()

    collect_success_demos(
        ckpt_name=args.ckpt_name,
        ckpt_episode=args.ckpt_episode,
        num_success=args.num_success,
        max_attempts=args.max_attempts,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        save_video=args.save_video,
        visualize=args.visualize,
        num_actions_per_step=args.num_actions_per_step,
        min_start_dist=args.min_start_dist,
    )
