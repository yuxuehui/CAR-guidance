import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import time
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image

from ..core.base_experiment import BaseExperiment
from ..utils.energy_center_utils import get_energy_centers_for_demo

def to_numpy(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray) and data.ndim == 4:
        if data.shape[0] == 1:
            data = data[0]
    return np.array(data)

class Task3ReplayVisualize(BaseExperiment):

    def setup(self):
        self.logger.info("设置实验3：轨迹重放 + 可视化...")

        data_config = self.config.get('data', {})
        replay_actions_path = data_config.get('replay_actions_path',
                                               'experiments/outputs/02_task2_gcov_adaptive/actions.json')
        self.replay_actions_path = Path(replay_actions_path)

        if not self.replay_actions_path.exists():
            raise FileNotFoundError(f"动作数据文件不存在: {self.replay_actions_path}")

        with open(self.replay_actions_path, 'r', encoding='utf-8') as f:
            self.actions_data = json.load(f)

        self.logger.info(f"加载了 {self.actions_data['num_demos']} 个演示的动作数据")

        demos_path = data_config.get('demos_path', 'experiments/data/success_demos/success_demos.json')
        self.demos_path = Path(demos_path)

        if not self.demos_path.exists():
            raise FileNotFoundError(f"演示数据文件不存在: {self.demos_path}")

        with open(self.demos_path, 'r', encoding='utf-8') as f:
            self.demos_data = json.load(f)

        self.demo_dict = {demo['demo_id']: demo for demo in self.demos_data['demos']}

        self.logger.info("实验设置完成")

    def load_data(self):
        actions = self.actions_data['actions']

        num_test = self.config.get('data', {}).get('num_test_episodes', len(actions))
        test_actions = actions[:num_test]

        self.logger.info(f"将重放 {len(test_actions)} 个演示")

        return test_actions

    def _create_visualization_overlay(self, trajectory: np.ndarray, energy_centers: List[List[float]],
                                       energy_scales: List[float], sigma: float,
                                       current_step: int, fig_size=(6, 6)) -> np.ndarray:

        fig = plt.figure(figsize=fig_size, dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(elev=20, azim=45)

        if self.config.get('visualization', {}).get('show_energy_centers', True):
            for i, (center, scale) in enumerate(zip(energy_centers, energy_scales)):
                center_np = np.array(center)
                color = 'red' if scale < 0 else 'blue'
                size = self.config.get('visualization', {}).get('energy_center_size', 200)
                ax.scatter(center_np[0], center_np[1], center_np[2],
                          c=color, s=size, alpha=0.6, marker='o',
                          label=f'Center {i+1} ({"repulsive" if scale < 0 else "attractive"})')

        if self.config.get('visualization', {}).get('show_trajectory', True) and current_step > 0:
            traj_executed = trajectory[:current_step + 1]
            if len(traj_executed) > 1:
                color = self.config.get('visualization', {}).get('trajectory_color', 'blue')
                linewidth = self.config.get('visualization', {}).get('trajectory_linewidth', 2)
                ax.plot(traj_executed[:, 0], traj_executed[:, 1], traj_executed[:, 2],
                       c=color, linewidth=linewidth, alpha=0.8, label='Trajectory')

                ax.scatter(traj_executed[-1, 0], traj_executed[-1, 1], traj_executed[-1, 2],
                          c='green', s=100, marker='*', label='Current Position')

        if self.config.get('visualization', {}).get('show_energy_field', False):
            resolution = self.config.get('visualization', {}).get('energy_field_resolution', 50)

            if len(trajectory) > 0:
                x_min, x_max = trajectory[:, 0].min() - 0.1, trajectory[:, 0].max() + 0.1
                y_min, y_max = trajectory[:, 1].min() - 0.1, trajectory[:, 1].max() + 0.1
                z_mean = trajectory[:, 2].mean()
            else:
                x_min, x_max = -0.3, 0.3
                y_min, y_max = -0.3, 0.3
                z_mean = 0.2

            x_grid = np.linspace(x_min, x_max, resolution)
            y_grid = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.full_like(X, z_mean)

            Energy = np.zeros_like(X)
            for center, scale in zip(energy_centers, energy_scales):
                center_np = np.array(center)
                dist_sq = (X - center_np[0])**2 + (Y - center_np[1])**2 + (Z - center_np[2])**2
                energy = np.exp(-dist_sq / (sigma**2 + 1e-8))
                Energy += scale * energy

            ax.contour(X, Y, Energy, levels=10, zdir='z', offset=z_mean, alpha=0.3, cmap='coolwarm')

        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.set_title(f'Step {current_step}', fontsize=10)

        if len(trajectory) > 0:
            ax.set_xlim([trajectory[:, 0].min() - 0.1, trajectory[:, 0].max() + 0.1])
            ax.set_ylim([trajectory[:, 1].min() - 0.1, trajectory[:, 1].max() + 0.1])
            ax.set_zlim([trajectory[:, 2].min() - 0.1, trajectory[:, 2].max() + 0.1])
        else:
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])
            ax.set_zlim([0.0, 0.5])

        ax.legend(fontsize=6, loc='upper left')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        return img

    def run_inference(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        from scripts.test_pick_cube_simple import PickCubeDiverseEnv
        import gymnasium as gym
        import mani_skill.envs
        from scripts.eval_maniskill import get_robot_state
        from pfp import set_seeds

        start_time = time.time()
        demo_id = action_data['demo_id']
        seed = action_data['seed']
        trajectory = np.array(action_data['trajectory'])

        if 'robot_state_history' in action_data:
            robot_state_history = np.array(action_data['robot_state_history'])
        else:

            self.logger.warning(f"演示 {demo_id}: 没有 robot_state_history，将从轨迹构造简化版本")

            robot_state_history = np.zeros((len(trajectory), 10), dtype=np.float32)
            robot_state_history[:, :3] = trajectory

        self.logger.info(f"重放演示 {demo_id} (seed={seed}), 轨迹长度: {len(trajectory)}")

        if demo_id not in self.demo_dict:
            self.logger.error(f"演示 {demo_id} 不存在于 success_demos 中")
            return {'demo_id': demo_id, 'success': False, 'error': 'Demo not found'}

        demo = self.demo_dict[demo_id]

        guidance_config = self.config.get('guidance', {})
        num_centers = guidance_config.get('num_energy_centers', 2)
        seed_base = guidance_config.get('seed', 42)

        energy_centers = get_energy_centers_for_demo(
            demo_id=demo_id,
            trajectory=trajectory,
            num_centers=num_centers,
            seed_base=seed_base,
        )

        energy_scales = guidance_config.get('energy_scales', [-1.0] * num_centers)
        sigma = guidance_config.get('sigma', 0.2)

        self.logger.info(f"  能量中心: {energy_centers}")
        self.logger.info(f"  能量强度: {energy_scales}")

        set_seeds(seed)

        video_resolution = self.config.get('evaluation', {}).get('video_resolution', [1920, 1080])
        env_kwargs = {
            "num_envs": 1,
            "obs_mode": "sensor_data",
            "control_mode": "pd_ee_delta_pose",
            "robot_uids": "panda_wristcam",
            "render_mode": "rgb_array",
            "human_render_camera_configs": {
                "width": video_resolution[0],
                "height": video_resolution[1],
            }
        }

        env = gym.make("PickCubeDiverse-v1", **env_kwargs)

        obs, _ = env.reset(seed=seed)

        video_output_dir = self.config.get('video_output_dir', None)
        if video_output_dir:
            video_dir = Path(video_output_dir)
        else:
            video_dir = Path(self.config.get('output_dir', 'experiments/outputs/03_task3_replay_visualize')) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        GRIPPER_LOWER = 0.0
        GRIPPER_UPPER = 0.04
        POS_LIMIT = 0.1

        episode_frames = []
        done = False

        env_frame = env.render()
        env_frame = to_numpy(env_frame)

        vis_overlay = self._create_visualization_overlay(
            trajectory=trajectory,
            energy_centers=energy_centers,
            energy_scales=energy_scales,
            sigma=sigma,
            current_step=0,
            fig_size=(6, 6)
        )

        combined_frame = self._combine_frames(env_frame, vis_overlay)
        episode_frames.append(combined_frame)

        for step in range(1, len(robot_state_history)):

            target_state = torch.from_numpy(robot_state_history[step]).float()
            current_state = get_robot_state(env).detach().cpu()

            d_pos = (target_state[:3] - current_state[:3]).numpy()
            d_pos_norm = np.clip(d_pos / POS_LIMIT, -1.0, 1.0)

            q_grip = target_state[9].item()
            g_val = (q_grip - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
            if q_grip < 0.025:
                g_val = -0.9

            action = np.concatenate([d_pos_norm, np.zeros(3), [np.clip(g_val, -1.0, 1.0)]])

            obs, _, _, _, info = env.step(action)

            env_frame = env.render()
            env_frame = to_numpy(env_frame)

            vis_overlay = self._create_visualization_overlay(
                trajectory=trajectory,
                energy_centers=energy_centers,
                energy_scales=energy_scales,
                sigma=sigma,
                current_step=step,
                fig_size=(6, 6)
            )

            combined_frame = self._combine_frames(env_frame, vis_overlay)
            episode_frames.append(combined_frame)

            done = bool(info["success"].item())
            if done:
                self.logger.info(f"  任务在步骤 {step} 完成")
                break

        success = done
        execution_time = time.time() - start_time

        video_fps = self.config.get('evaluation', {}).get('video_fps', 20)
        video_name = f"replay_demo_{demo_id:04d}_seed_{seed}.mp4"
        video_path = str(video_dir / video_name)

        if len(episode_frames) > 0:
            imageio.mimsave(video_path, episode_frames, fps=video_fps)
            self.logger.info(f"  视频已保存至 {video_path}")

        env.close()

        result = {
            'demo_id': demo_id,
            'seed': seed,
            'success': success,
            'steps': len(trajectory),
            'execution_time': execution_time,
            'video_path': video_path,
            'energy_centers': energy_centers,
        }

        return result

    def _combine_frames(self, env_frame: np.ndarray, vis_overlay: np.ndarray) -> np.ndarray:

        env_h, env_w = env_frame.shape[:2]
        vis_h, vis_w = vis_overlay.shape[:2]

        scale = env_h / vis_h
        new_vis_h = env_h
        new_vis_w = int(vis_w * scale)

        vis_pil = Image.fromarray(vis_overlay)
        vis_pil_resized = vis_pil.resize((new_vis_w, new_vis_h), Image.Resampling.LANCZOS)
        vis_resized = np.array(vis_pil_resized)

        combined = np.concatenate([env_frame, vis_resized], axis=1)

        return combined

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始 Task 3: 轨迹重放 + 可视化")
        self.logger.info("=" * 60)

        self.logger.info("步骤 1/4: 设置实验...")
        self.setup()

        self.logger.info("步骤 2/4: 加载重放数据...")
        test_data = self.load_data()

        self.logger.info("步骤 3/4: 运行重放...")
        num_episodes = self.config.get('data', {}).get('num_test_episodes', len(test_data))

        results = []
        for i, action_data in enumerate(test_data[:num_episodes]):
            self.logger.info(f"\n重放演示 {i+1}/{num_episodes} (demo_id={action_data['demo_id']}, seed={action_data['seed']})")

            result = self.run_inference(action_data)
            results.append(result)

        self.logger.info("\n步骤 4/4: 保存结果...")
        output_dir = Path(self.config.get('output_dir', 'experiments/outputs/03_task3_replay_visualize'))
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "replay_summary.json"
        summary_data = {
            'num_replays': len(results),
            'results': results,
        }

        with open(summary_path, 'w', encoding='utf-8') as f:

            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_summary = convert_to_serializable(summary_data)
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"结果摘要已保存至: {summary_path}")

        self.logger.info("=" * 60)
        self.logger.info("Task 3 完成！")
        self.logger.info(f"重放了 {len(results)} 个演示")
        self.logger.info("=" * 60)

        return results
