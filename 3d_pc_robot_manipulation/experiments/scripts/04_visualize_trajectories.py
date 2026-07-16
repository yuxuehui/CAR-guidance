import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy import interpolate

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.test_pick_cube_simple import PickCubeDiverseEnv
import gymnasium as gym
from scripts.eval_maniskill import get_robot_state
from pfp import set_seeds
from mani_skill.utils.building.actors.common import build_sphere, build_cylinder
from mani_skill.utils.structs.pose import Pose

def to_numpy(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray) and data.ndim == 4:
        if data.shape[0] == 1:
            data = data[0]
    return np.array(data)

def load_results(results_path: Path):
    if results_path.is_dir():
        demo_files = sorted(results_path.glob("demo_*.json"))
        if len(demo_files) == 0:
            raise ValueError(f"目录中没有找到demo文件: {results_path}")
        results = []
        for demo_file in demo_files:
            with open(demo_file, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        return results

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'actions' in data:
        return data['actions']
    elif isinstance(data, dict) and 'demo_id' in data:
        return [data]
    else:
        raise ValueError(f"未知的结果文件格式: {results_path}")

def create_energy_center_markers(scene, energy_centers, energy_scales, marker_size=0.025):
    markers = []
    for i, (center, scale) in enumerate(zip(energy_centers, energy_scales)):

        if abs(scale) < 1e-6:
            continue

        if scale < 0:

            color = [1.0, 0.0, 1.0, 1.0]
        else:

            color = [0.0, 1.0, 1.0, 1.0]

        marker = build_sphere(
            scene=scene,
            radius=marker_size,
            color=color,
            name=f"energy_center_{i}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create_from_pq(np.array(center), np.array([1, 0, 0, 0])),
        )
        markers.append(marker)
    return markers

def smooth_trajectory(trajectory, num_points=1000, s=0.0):
    trajectory = np.array(trajectory)
    if len(trajectory) < 4:
        return trajectory

    dist = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
    mask = np.concatenate(([True], dist > 1e-6))
    clean_traj = trajectory[mask]

    if len(clean_traj) < 4:
        return trajectory

    try:

        x = clean_traj[:, 0]
        y = clean_traj[:, 1]
        z = clean_traj[:, 2]

        tck, u = interpolate.splprep([x, y, z], s=s)

        u_new = np.linspace(0, 1, num_points)

        new_points = interpolate.splev(u_new, tck)
        return np.array(new_points).T
    except Exception as e:
        print(f"平滑失败，使用原始轨迹: {e}")
        return trajectory

def create_trajectory_line(scene, trajectory, cmap_name='turbo', line_width=0.003, alpha=1.0):
    objects = []

    smoothed_traj = smooth_trajectory(trajectory, num_points=max(len(trajectory)*2, 500))
    total_points = len(smoothed_traj)

    colormap = plt.get_cmap(cmap_name)
    x_axis = np.array([1, 0, 0])

    for i in range(total_points - 1):
        p1 = smoothed_traj[i]
        p2 = smoothed_traj[i + 1]

        progress = i / max(total_points - 1, 1)
        color_rgba = colormap(progress)
        final_color = [color_rgba[0], color_rgba[1], color_rgba[2], alpha]

        sphere = build_sphere(
            scene=scene,
            radius=line_width,
            color=final_color,
            name=f"traj_joint_{i}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create_from_pq(p1, np.array([1, 0, 0, 0])),
        )
        objects.append(sphere)

        if i == total_points - 2:
            sphere_end = build_sphere(
                scene=scene,
                radius=line_width,
                color=colormap(1.0),
                name=f"traj_joint_end",
                body_type="kinematic",
                add_collision=False,
                initial_pose=Pose.create_from_pq(p2, np.array([1, 0, 0, 0])),
            )
            objects.append(sphere_end)

        mid_point = (p1 + p2) / 2
        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            continue

        direction = direction / length

        if np.abs(np.dot(direction, x_axis)) > 0.99:
            if np.dot(direction, x_axis) > 0:
                rotation = np.array([1, 0, 0, 0])
            else:
                rotation = np.array([0, 0, 1, 0])
        else:
            axis = np.cross(x_axis, direction)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = np.arccos(np.clip(np.dot(x_axis, direction), -1, 1))
            rotation = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
            ])

        segment = build_cylinder(
            scene=scene,
            radius=line_width,
            half_length=length / 2,
            color=final_color,
            name=f"traj_seg_{i}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create_from_pq(mid_point, rotation),
        )
        objects.append(segment)

    return objects

def visualize_trajectory(result_data, output_dir: Path, experiment_name: str,
                         video_resolution=[1920, 1080], fps=20):
    demo_id = result_data['demo_id']
    seed = result_data['seed']
    actions_sequence = result_data.get('actions_sequence', [])
    energy_centers = result_data.get('energy_centers', [])

    if 'energy_scales' in result_data:
        energy_scales = result_data['energy_scales']
        if isinstance(energy_scales, list):
            if len(energy_scales) != len(energy_centers):
                energy_scales = energy_scales[:len(energy_centers)] if len(energy_scales) > len(energy_centers) else energy_scales + [-1.0] * (len(energy_centers) - len(energy_scales))
        else:
            energy_scales = [-1.0] * len(energy_centers)
    else:
        energy_scales = [-1.0] * len(energy_centers) if len(energy_centers) > 0 else []

    if len(actions_sequence) == 0:
        print(f"警告: 演示 {demo_id} 没有动作序列，跳过")
        return

    print(f"可视化演示 {demo_id} (seed={seed})")

    set_seeds(seed)

    camera_pos = [0.4, 0.8, 0.5]
    camera_look_at = [0.0, 0.0, 0.15]

    from mani_skill.utils import sapien_utils
    camera_pose = sapien_utils.look_at(eye=camera_pos, target=camera_look_at)

    env_kwargs = {
        "num_envs": 1,
        "obs_mode": "sensor_data",
        "control_mode": "pd_ee_delta_pose",
        "robot_uids": "panda_wristcam",
        "render_mode": "rgb_array",
        "human_render_camera_configs": {
            "width": video_resolution[0],
            "height": video_resolution[1],
            "pose": camera_pose,
        }
    }

    env = gym.make("PickCubeDiverse-v1", **env_kwargs)

    print(f"  相机视角已设置: pos={camera_pos}, look_at={camera_look_at}")

    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped

    video_dir = output_dir / "visualization_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{experiment_name}_demo_{demo_id:04d}_seed_{seed}.mp4"

    image_dir = output_dir / "visualization_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{experiment_name}_demo_{demo_id:04d}_seed_{seed}.png"

    if len(energy_centers) > 0:
        create_energy_center_markers(
            base_env.scene,
            energy_centers,
            energy_scales,
            marker_size=0.025
        )

    episode_frames = []
    trajectory = []

    initial_state = get_robot_state(env).detach().cpu()
    trajectory.append(initial_state[:3].numpy().tolist())

    frame = env.render()
    episode_frames.append(to_numpy(frame))

    for i, action in enumerate(actions_sequence):
        action = np.array(action)
        obs, _, _, _, info = env.step(action)

        after_action_state = get_robot_state(env).detach().cpu()
        trajectory.append(after_action_state[:3].numpy().tolist())

        frame = env.render()
        episode_frames.append(to_numpy(frame))

        if info.get("success", False):
            print(f"  任务在第 {i+1} 步完成")
            break

    if len(trajectory) > 1:
        trajectory_objects = create_trajectory_line(
            base_env.scene,
            trajectory,
            cmap_name='hot',
            line_width=0.005,
            alpha=1.0
        )
        print(f"  已生成高画质轨迹，包含 {len(trajectory_objects)} 个几何体")

        for _ in range(3):
            zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9])
            env.step(zero_action)

        final_frame = env.render()
        final_frame_np = to_numpy(final_frame)

        if len(episode_frames) > 0:
            episode_frames[-1] = final_frame_np

        imageio.imwrite(str(image_path), final_frame_np)
        print(f"  高清轨迹图已保存: {image_path}")

    else:
        print(f"  警告: 轨迹长度不足，跳过轨迹可视化")

    if len(episode_frames) > 0:
        imageio.mimsave(str(video_path), episode_frames, fps=fps)
        print(f"  视频已保存: {video_path}")

    env.close()

def main():
    parser = argparse.ArgumentParser(description="可视化轨迹：重放实验并可视化能量场和轨迹")
    parser.add_argument("--results_path", type=str, required=True, help="结果路径")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--video_resolution", type=int, nargs=2, default=[1920, 1080])
    parser.add_argument("--fps", type=int, default=20)

    args = parser.parse_args()

    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"错误: 结果文件不存在: {results_path}")
        return

    results = load_results(results_path)

    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = results_path.parent.name if results_path.name == "detailed_results.json" else results_path.name

    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent

    if args.num_demos:
        results = results[:args.num_demos]

    for i, result in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] ", end="")
        try:
            visualize_trajectory(
                result, output_dir, experiment_name,
                video_resolution=args.video_resolution, fps=args.fps
            )
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n可视化完成！")

if __name__ == "__main__":
    main()
