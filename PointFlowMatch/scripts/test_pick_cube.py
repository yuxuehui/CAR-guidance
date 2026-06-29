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

sys.path.insert(0, str(Path(__file__).parent))

from eval_maniskill import (
    get_robot_state,
    prepare_observation,
)
from utils.utils_tool import load_model_from_checkpoint_concat_goal
from pfp import DEVICE, set_seeds
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env

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

def test_single_episode(model, config, seed=1, max_steps=200, episode_dir=None, save_video=True, video_output_dir=None, video_resolution=(1920, 1080)):

    if episode_dir is not None:
        actual_seed = load_seed_from_episode(episode_dir)
        print(f"\n>>> 从episode文件夹加载seed: {actual_seed}")
        print(f"  Episode目录: {episode_dir}")
    else:
        actual_seed = seed
        print(f"\n>>> 使用指定的seed: {actual_seed}")

    set_seeds(actual_seed)

    print("=" * 60)
    print(f"ManiSkill 简单测试 - seed={actual_seed}")
    if episode_dir is not None:
        print(f"Episode目录: {episode_dir}")
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
        video_width, video_height = video_resolution
        env_kwargs["human_render_camera_configs"] = {
            "width": video_width,
            "height": video_height,
        }
        print(f"  视频分辨率: {video_width}x{video_height}")
        print(f"  渲染模式: rgb_array (用于视频录制)")
    else:
        print(f"  渲染模式: human (显示窗口)")

    env = gym.make("PickCubeDiverse-v1", **env_kwargs)
    print(">>> 使用改进的PickCubeDiverse-v1 环境（随机范围：[-0.15, 0.15]）")

    env.reset(seed=0)
    ground_ids = get_ground_ids(env)
    print(f"Ground IDs: {ground_ids}")

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

            video_output_dir = "outputs/videos/test_simple"
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
    predicted_robot_states = []

    base_env = env.unwrapped
    goal_pos = base_env.goal_site.pose.p[0].cpu().numpy().astype(np.float32)
    print(f"Goal位置: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]")

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

        pred_action = pred_y[0, 0]

        pred_robot_state = pred_action.cpu().numpy()

        predicted_robot_states.append(pred_robot_state.copy())

        current_pos = robot_state[:3].cpu().numpy()
        pred_pos = pred_action[:3].cpu().numpy()
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        dist_pred_to_goal = np.linalg.norm(pred_pos - goal_pos)

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

        current_gripper_qpos = robot_state[9].item()

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

    return {
        "success": success,
        "step_count": step_count,
        "seed": actual_seed,
        "episode_dir": episode_dir,
        "max_steps": max_steps,
    }

def test_batch(
    episode_dirs=None,
    num_tests=None,
    ckpt_name="maniskill_train",
    ckpt_episode="latest",
    max_steps=200,
    save_video=True,
    video_output_dir=None,
    video_resolution=(1920, 1080),
    log_file=None,
):

    if episode_dirs is not None:
        test_list = [("episode_dir", ep_dir) for ep_dir in episode_dirs]
    elif num_tests is not None:

        seed_min, seed_max = 1, 10000

        seeds = np.random.choice(range(seed_min, seed_max + 1), size=num_tests, replace=False).tolist()
        seeds.sort()
        test_list = [("seed", seed) for seed in seeds]
        print(f"随机生成 {num_tests} 个测试，seed范围: [{seed_min}, {seed_max}]")
        print(f"选中的seed: {seeds[:10]}{'...' if len(seeds) > 10 else ''} (共{len(seeds)}个)")
    else:
        raise ValueError("必须提供 episode_dirs 或 num_tests 之一")

    total_tests = len(test_list)
    print(f"\n{'='*60}")
    print(f"开始批量测试 - 共 {total_tests} 个测试")
    print(f"{'='*60}\n")

    print(">>> 加载模型...")
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
    print()

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"outputs/logs/test_batch_{timestamp}.log"

    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    if save_video and video_output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_output_dir = f"outputs/videos/test_batch_{timestamp}"

    results = []

    for idx, (test_type, test_value) in enumerate(test_list, 1):
        print(f"\n{'='*60}")
        print(f"测试 {idx}/{total_tests}")
        print(f"{'='*60}")

        test_video_dir = None
        if save_video:
            if test_type == "episode_dir":
                episode_name = Path(test_value).name
                test_video_dir = str(Path(video_output_dir) / episode_name)
            else:
                test_video_dir = str(Path(video_output_dir) / f"seed_{test_value}")

        if test_type == "episode_dir":
            result = test_single_episode(
                model=model,
                config=config,
                seed=None,
                max_steps=max_steps,
                episode_dir=test_value,
                save_video=save_video,
                video_output_dir=test_video_dir,
                video_resolution=video_resolution,
            )
        else:
            result = test_single_episode(
                model=model,
                config=config,
                seed=test_value,
                max_steps=max_steps,
                episode_dir=None,
                save_video=save_video,
                video_output_dir=test_video_dir,
                video_resolution=video_resolution,
            )

        results.append(result)

        success_count = sum(1 for r in results if r["success"])
        print(f"\n当前进度: {idx}/{total_tests} | 成功: {success_count}/{idx} ({success_count/idx*100:.1f}%)")

    success_count = sum(1 for r in results if r["success"])
    total_success_rate = success_count / total_tests * 100 if total_tests > 0 else 0

    avg_steps = np.mean([r["step_count"] for r in results]) if results else 0
    success_avg_steps = np.mean([r["step_count"] for r in results if r["success"]]) if success_count > 0 else 0

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "ckpt_name": ckpt_name,
        "ckpt_episode": ckpt_episode,
        "max_steps": max_steps,
        "total_tests": total_tests,
        "success_count": success_count,
        "failure_count": total_tests - success_count,
        "success_rate": total_success_rate,
        "avg_steps": float(avg_steps),
        "success_avg_steps": float(success_avg_steps),
        "results": results,
    }

    json_log_file = str(Path(log_file).with_suffix('.json'))
    with open(json_log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"ManiSkill 批量测试日志\n")
        f.write(f"{'='*60}\n")
        f.write(f"时间: {log_data['timestamp']}\n")
        f.write(f"Checkpoint: {ckpt_name} / {ckpt_episode}\n")
        f.write(f"最大步数: {max_steps}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"测试结果:\n")
        f.write(f"{'='*60}\n")
        for idx, result in enumerate(results, 1):
            test_id = result.get("episode_dir") or f"seed_{result['seed']}"
            f.write(f"{idx:3d}. {test_id:50s} | "
                   f"成功: {'是' if result['success'] else '否':3s} | "
                   f"步数: {result['step_count']:4d}/{result['max_steps']}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"统计信息:\n")
        f.write(f"{'='*60}\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"成功数: {success_count}\n")
        f.write(f"失败数: {total_tests - success_count}\n")
        f.write(f"成功率: {total_success_rate:.2f}%\n")
        f.write(f"平均步数: {avg_steps:.2f}\n")
        if success_count > 0:
            f.write(f"成功案例平均步数: {success_avg_steps:.2f}\n")
        f.write(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"批量测试完成！")
    print(f"{'='*60}")
    print(f"总测试数: {total_tests}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_tests - success_count}")
    print(f"成功率: {total_success_rate:.2f}%")
    print(f"平均步数: {avg_steps:.2f}")
    if success_count > 0:
        print(f"成功案例平均步数: {success_avg_steps:.2f}")
    print(f"\n日志文件: {log_file}")
    print(f"JSON日志: {json_log_file}")
    if save_video:
        print(f"视频目录: {video_output_dir}")
    print(f"{'='*60}\n")

    return results, log_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ManiSkill批量测试脚本",
        epilog="""
使用示例:
  # 批量测试指定数量（随机生成seed，范围: 1-10000）
  python scripts/test_pick_cube.py --num_tests 10

  # 批量测试多个episode目录
  python scripts/test_pick_cube.py --episode_dirs ./data/demo_data_diverse_small/episode3 ./data/demo_data_diverse_small/episode4

  # 单个测试（向后兼容）
  python scripts/test_pick_cube.py --seed 13

  # 禁用视频保存以加快测试
  python scripts/test_pick_cube.py --num_tests 10 --save_video False
        """
    )
    parser.add_argument("--ckpt_name", type=str, default="maniskill_train_pcd_from_three_cameras_more_gripper", help="checkpoint目录名称")
    parser.add_argument("--ckpt_episode", type=str, default="ep1500-ba160500", help="checkpoint的episode标识")
    parser.add_argument("--seed", type=int, default=None, help="单个测试seed")
    parser.add_argument("--num_tests", type=int, default=None, help="批量测试的数量（随机生成num_tests个seed，范围: 1-10000）")
    parser.add_argument("--episode_dirs", type=str, nargs='+', default=None, help="批量测试的episode目录列表")
    parser.add_argument("--max_steps", type=int, default=300, help="最大步数")
    parser.add_argument("--episode_dir", type=str, default=None, help="单个测试的episode文件夹路径（向后兼容）")
    parser.add_argument("--save_video", type=lambda x: x.lower() == 'true', nargs='?', const=True, default=True, help="是否保存视频 (默认: True, 使用 --save_video False 来禁用)")
    parser.add_argument("--video_output_dir", type=str, default=None, help="视频保存目录 (默认: outputs/videos/test_batch_TIMESTAMP/)")
    parser.add_argument("--video_width", type=int, default=1920, help="视频宽度 (默认: 1920)")
    parser.add_argument("--video_height", type=int, default=1080, help="视频高度 (默认: 1080)")
    parser.add_argument("--log_file", type=str, default=None, help="日志文件路径 (默认: outputs/logs/test_batch_TIMESTAMP.log)")

    args = parser.parse_args()

    if args.num_tests is not None or args.episode_dirs is not None:

        test_batch(
            episode_dirs=args.episode_dirs,
            num_tests=args.num_tests,
            ckpt_name=args.ckpt_name,
            ckpt_episode=args.ckpt_episode,
            max_steps=args.max_steps,
            save_video=args.save_video,
            video_output_dir=args.video_output_dir,
            video_resolution=(args.video_width, args.video_height),
            log_file=args.log_file,
        )
    else:

        print(">>> 加载模型...")
        model, config = load_model_from_checkpoint_concat_goal(
            ckpt_name=args.ckpt_name,
            ckpt_episode=args.ckpt_episode,
        )
        test_single_episode(
            model=model,
            config=config,
            seed=args.seed if args.seed is not None else 13,
            max_steps=args.max_steps,
            episode_dir=args.episode_dir,
            save_video=args.save_video,
            video_output_dir=args.video_output_dir,
            video_resolution=(args.video_width, args.video_height),
        )
