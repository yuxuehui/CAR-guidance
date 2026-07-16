import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.core.base_experiment import BaseExperiment
from experiments.policy.policy_loader import load_policy
from experiments.guidance import NoneGuidance

class Task1StaticOnly(BaseExperiment):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.all_actions = []

    def setup(self):
        self.logger.info("设置 Task 1: 不使用能量场...")

        model_config = self.config['model']
        ckpt_path = model_config['ckpt_path']

        self.logger.info(f"加载模型: {ckpt_path}")

        data_config = self.config.get('data', {})
        demos_path = data_config.get('demos_path', 'experiments_stack_cube/data/success_demos/success_demos.json')
        self.demos_path = Path(demos_path)

        if not self.demos_path.exists():
            raise FileNotFoundError(f"演示数据文件不存在: {self.demos_path}")

        with open(self.demos_path, 'r', encoding='utf-8') as f:
            self.demos_data = json.load(f)

        self.logger.info(f"加载了 {len(self.demos_data['demos'])} 个演示")

        guidance = NoneGuidance(config={})

        self.policy, self.model_config_dict = load_policy(
            ckpt_path,
            guidance=guidance,
            config=self.config
        )

        if self.model_config_dict:
            self.norm_pcd_center = self.model_config_dict.get('model_config', {}).get('norm_pcd_center', None)
            if self.norm_pcd_center:
                self.logger.info(f"模型 norm_pcd_center: {self.norm_pcd_center}")

        self.logger.info("实验设置完成")

    def load_data(self):
        demos = self.demos_data['demos']

        for demo in demos:
            if 'seed' not in demo:
                raise ValueError(f"演示 {demo.get('demo_id', 'unknown')} 缺少 'seed' 字段")
            if 'demo_id' not in demo:
                raise ValueError(f"演示缺少 'demo_id' 字段")

        num_test = self.config.get('data', {}).get('num_test_episodes', len(demos))
        test_demos = demos[:num_test]

        seed_info = ", ".join([f"demo_{d['demo_id']}:seed_{d['seed']}" for d in test_demos[:5]])
        if len(test_demos) > 5:
            seed_info += f", ... (共{len(test_demos)}个)"
        self.logger.info(f"使用 {len(test_demos)} 个演示进行测试，种子信息: {seed_info}")

        return test_demos

    def run_inference(self, demo: Dict[str, Any]) -> Dict[str, Any]:
        import gymnasium as gym
        import mani_skill.envs
        from scripts.eval_maniskill import get_robot_state
        from pfp.utils.pointcloud_utils import (
            get_pointcloud_from_multi_cameras,
            get_ground_ids,
            get_cube_ids,
            get_robot_ids,
        )
        from pfp import set_seeds, DEVICE
        import open3d as o3d
        import imageio

        GRIPPER_LOWER = 0.0
        GRIPPER_UPPER = 0.04
        POS_LIMIT = 0.1

        def to_numpy(data):
            if torch.is_tensor(data):
                data = data.detach().cpu().numpy()
            if isinstance(data, np.ndarray) and data.ndim == 4:
                if data.shape[0] == 1:
                    data = data[0]
            return np.array(data)

        start_time = time.time()

        seed = demo['seed']
        demo_id = demo['demo_id']
        self.logger.info(f"使用演示 {demo_id} 的种子: {seed}")
        set_seeds(seed)

        save_video = self.config.get('evaluation', {}).get('save_videos', False)
        visualize = self.config.get('evaluation', {}).get('visualize', False)
        render_mode = "human" if visualize else "rgb_array"

        env_kwargs = {
            "num_envs": 1,
            "obs_mode": "sensor_data",
            "control_mode": "pd_ee_delta_pose",
            "robot_uids": "panda_wristcam",
            "render_mode": render_mode,
        }

        if save_video:
            video_resolution = self.config.get('evaluation', {}).get('video_resolution', [1280, 720])
            env_kwargs["human_render_camera_configs"] = {
                "width": video_resolution[0],
                "height": video_resolution[1],
            }

        env = gym.make("StackCube-v1", **env_kwargs)

        env.reset(seed=0)
        ground_ids = to_numpy(get_ground_ids(env))
        cube_ids = to_numpy(get_cube_ids(env))
        robot_ids = to_numpy(get_robot_ids(env))

        selected_cameras = ["right_shoulder_camera", "left_shoulder_camera", "hand_camera"]
        voxel_size = 0.003
        n_points = self.model_config_dict['dataset_config']['n_points']
        use_pc_color = self.model_config_dict.get('dataset_config', {}).get('use_pc_color', False)
        ws_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-0.3, -0.3, 0.0), max_bound=(0.3, 0.3, 0.5))

        obs, _ = env.reset(seed=seed)
        base_env = env.unwrapped

        cubeB_pos = base_env.cubeB.pose.p[0].detach().cpu().numpy().astype(np.float32)
        goal_pos = cubeB_pos

        video_dir = None
        episode_frames = []
        if save_video:
            video_output_dir = self.config.get('evaluation', {}).get('video_output_dir', None)
            if video_output_dir:
                video_dir = Path(video_output_dir)
            else:
                output_dir = Path(self.config.get('output_dir', 'experiments_stack_cube/outputs/01_task1_static_only'))
                video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)

        if save_video:
            frame = env.render()
            episode_frames.append(to_numpy(frame))

        max_steps = self.config.get('data', {}).get('max_steps', 300)
        num_actions_per_step = self.config.get('data', {}).get('num_actions_per_step', 32)
        n_pred_steps_model = self.model_config_dict['n_pred_steps']
        n_obs_steps = self.model_config_dict['n_obs_steps']

        done = False
        step_count = 0
        robot_state_history = []
        trajectory = []
        actions_sequence = []

        initial_state = get_robot_state(env).detach().cpu()
        trajectory.append(initial_state[:3].numpy().tolist())

        while not done and step_count < max_steps:

            cur_state = get_robot_state(env).detach().cpu()
            robot_state_history.append(cur_state)

            pcd_data = get_pointcloud_from_multi_cameras(
                obs, ground_ids, voxel_size, n_points, ws_aabb, cube_ids, robot_ids, selected_cameras
            )

            xyz = to_numpy(pcd_data["xyz"])
            if use_pc_color:
                rgb = to_numpy(pcd_data["rgb"])
                pcd = np.concatenate([xyz, rgb], axis=-1)
            else:
                pcd = xyz

            hist = robot_state_history.copy()
            while len(hist) < n_obs_steps:
                hist.insert(0, hist[0])
            hist = hist[-n_obs_steps:]

            pcd_t = torch.from_numpy(pcd).float().unsqueeze(0).repeat(1, n_obs_steps, 1, 1).to(DEVICE)
            state_t = torch.stack(hist, dim=0).unsqueeze(0).to(DEVICE)
            goal_t = torch.from_numpy(goal_pos).float().unsqueeze(0).repeat(1, n_obs_steps, 1).to(DEVICE)

            with torch.no_grad():
                off = torch.tensor(self.norm_pcd_center, device=DEVICE) if self.norm_pcd_center else None
                if off is not None:
                    pcd_t[..., :3] -= off
                    state_t[..., :3] -= off
                    goal_t[..., :3] -= off
                pred_y = self.policy.infer_y(pcd_t, state_t, goal_t)
                if off is not None:
                    pred_y[..., :3] += off

            for i in range(min(num_actions_per_step, n_pred_steps_model)):
                target = pred_y[0, i]
                now_state = get_robot_state(env).detach().cpu()

                d_pos = (target[:3] - now_state[:3].to(DEVICE)).detach().cpu().numpy()
                d_pos_norm = np.clip(d_pos / POS_LIMIT, -1.0, 1.0)

                q_grip = target[9].item()
                g_val = (q_grip - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
                if q_grip < 0.025:
                    g_val = -0.9

                action = np.concatenate([d_pos_norm, np.zeros(3), [np.clip(g_val, -1.0, 1.0)]])

                actions_sequence.append(action.tolist())

                obs, _, _, _, info = env.step(action)

                after_action_state = get_robot_state(env).detach().cpu()
                trajectory.append(after_action_state[:3].numpy().tolist())

                if save_video:
                    frame = env.render()
                    episode_frames.append(to_numpy(frame))
                elif visualize:
                    env.render()

                done = bool(info["success"].item())
                step_count += 1
                if done:
                    break

            if done:
                break

        success = bool(info["success"].item())
        execution_time = time.time() - start_time

        video_path = None
        if save_video and len(episode_frames) > 0:
            video_name = f"demo_{demo_id:04d}_seed_{seed}.mp4"
            video_path = str(video_dir / video_name)
            imageio.mimsave(video_path, episode_frames, fps=20)
            self.logger.info(f"视频已保存至 {video_path} (成功: {success})")

        env.close()

        robot_state_history_np = np.array([s.numpy() if isinstance(s, torch.Tensor) else s for s in robot_state_history])

        result = {
            'demo_id': demo_id,
            'seed': seed,
            'success': success,
            'steps': step_count,
            'execution_time': execution_time,
            'video_path': video_path,
            'trajectory': np.array(trajectory),
            'robot_state_history': robot_state_history_np,
            'actions_sequence': actions_sequence,
        }

        self.logger.info(
            f"演示 {demo_id}: 推理完成 - "
            f"成功: {success}, 步数: {step_count}, 时间: {execution_time:.2f}s"
        )

        return result

    def save_results(self, metrics: Dict[str, float]):

        super().save_results(metrics)

        actions_output_path = self.output_dir / 'actions_summary.json'

        actions_data = {
            'experiment': 'Task1StaticOnly',
            'num_demos': len(self.all_actions),
            'actions': self.all_actions
        }

        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_list(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_list(value) for key, value in obj.items()}
            else:
                return obj

        actions_data = convert_to_list(actions_data)

        with open(actions_output_path, 'w', encoding='utf-8') as f:
            json.dump(actions_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"动作序列汇总已保存至: {actions_output_path}")
        self.logger.info(f"每个demo的单独动作文件保存在: {self.output_dir / 'actions'}")

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("开始 Task 1: 不使用能量场 + 动作保存")
        self.logger.info("=" * 60)

        self.logger.info("步骤 1/5: 设置实验...")
        self.setup()

        self.logger.info("步骤 2/5: 加载测试数据...")
        test_data = self.load_data()

        self.logger.info("步骤 3/5: 运行推理...")
        num_episodes = self.config.get('data', {}).get('num_test_episodes', len(test_data))

        results = []
        for i, demo in enumerate(test_data[:num_episodes]):
            self.logger.info(f"\n处理演示 {i+1}/{num_episodes} (demo_id={demo['demo_id']}, seed={demo['seed']})")

            result = self.run_inference(demo)
            results.append(result)

            def convert_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_list(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_to_list(value) for key, value in obj.items()}
                else:
                    return obj

            demo_actions = convert_to_list({
                'demo_id': result['demo_id'],
                'seed': result['seed'],
                'success': result['success'],
                'steps': result['steps'],
                'execution_time': result['execution_time'],
                'trajectory': result['trajectory'],
                'robot_state_history': result['robot_state_history'],
                'actions_sequence': result['actions_sequence'],
            })

            if 'video_path' in result:
                demo_actions['video_path'] = result['video_path']

            actions_output_dir = Path(self.config.get('output_dir', 'experiments_stack_cube/outputs/01_task1_static_only')) / "actions"
            actions_output_dir.mkdir(parents=True, exist_ok=True)
            action_file = actions_output_dir / f"demo_{demo['demo_id']:04d}_seed_{demo['seed']}.json"

            with open(action_file, 'w', encoding='utf-8') as f:
                json.dump(demo_actions, f, indent=2, ensure_ascii=False)

            self.logger.info(f"演示 {demo['demo_id']} 的动作序列已保存至: {action_file}")

            self.all_actions.append(demo_actions)

            trajectory = result['trajectory']
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            self.trajectories.append(trajectory)
            self.success_flags.append(result['success'])
            self.execution_times.append(result['execution_time'])

        self.logger.info("\n步骤 4/5: 评估结果...")
        metrics = self.evaluate()

        self.logger.info("步骤 5/5: 保存结果...")
        self.save_results(metrics)

        self.logger.info("=" * 60)
        self.logger.info("Task 1 完成！")
        self.logger.info(f"成功率: {metrics.get('success_rate', 0):.2%}")
        self.logger.info(f"保存了 {len(self.all_actions)} 个演示的动作序列")
        self.logger.info("=" * 60)

        return metrics
