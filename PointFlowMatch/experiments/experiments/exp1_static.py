import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import os
import time
import imageio

from ..core.base_experiment import BaseExperiment
from ..policy.policy_loader import load_policy
from ..guidance import StaticGuidance
from ..utils.energy_center_utils import get_energy_centers_for_demo

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

class Exp1Static(BaseExperiment):

    def setup(self):
        self.logger.info("设置实验1：静态能量场引导（基础版本）...")

        model_config = self.config['model']
        ckpt_path = model_config['ckpt_path']

        self.logger.info(f"加载模型: {ckpt_path}")

        data_config = self.config.get('data', {})
        demos_path = data_config.get('demos_path', 'experiments/data/success_demos/success_demos.json')
        self.demos_path = Path(demos_path)

        if not self.demos_path.exists():
            raise FileNotFoundError(f"演示数据文件不存在: {self.demos_path}")

        with open(self.demos_path, 'r', encoding='utf-8') as f:
            self.demos_data = json.load(f)

        self.logger.info(f"加载了 {len(self.demos_data['demos'])} 个演示")

        self.policy, self.model_config_dict = load_policy(
            ckpt_path,
            guidance=None,
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

        fixed_seeds = self.config.get('data', {}).get('fixed_seeds', None)
        if fixed_seeds is not None:
            seed_to_demo = {d['seed']: d for d in demos}
            filtered = []
            for s in fixed_seeds:
                if s in seed_to_demo:
                    filtered.append(seed_to_demo[s])
                else:
                    self.logger.warning(f"指定的种子 {s} 在数据中未找到，跳过")
            test_demos = filtered
            self.logger.info(f"使用指定的 {len(test_demos)} 个种子: {fixed_seeds}")
        else:

            num_test = self.config.get('data', {}).get('num_test_episodes', len(demos))
            test_demos = demos[:num_test]

        seed_info = ", ".join([f"demo_{d['demo_id']}:seed_{d['seed']}" for d in test_demos[:5]])
        if len(test_demos) > 5:
            seed_info += f", ... (共{len(test_demos)}个)"
        self.logger.info(f"使用 {len(test_demos)} 个演示进行测试，种子信息: {seed_info}")

        return test_demos

    def _create_guidance_for_demo(self, demo: Dict[str, Any]) -> StaticGuidance:
        guidance_config = self.config.get('guidance', {}).copy()

        trajectory = np.array(demo['trajectory'])

        demo_id = demo['demo_id']
        num_centers = guidance_config.get('num_energy_centers', 2)
        seed_base = guidance_config.get('seed', 42)

        energy_centers = get_energy_centers_for_demo(
            demo_id=demo_id,
            trajectory=trajectory,
            num_centers=num_centers,
            seed_base=seed_base,
        )

        guidance_config['energy_centers'] = energy_centers
        guidance_config['energy_scales'] = guidance_config.get('energy_scales', [-1.0] * num_centers)

        if hasattr(self, 'norm_pcd_center') and self.norm_pcd_center:
            guidance_config['norm_pcd_center'] = self.norm_pcd_center

        guidance = StaticGuidance(guidance_config)

        self.logger.info(f"演示 {demo_id}: 能量中心数量 = {len(energy_centers)}")

        return guidance

    def run_inference(self, demo: Dict[str, Any]) -> Dict[str, Any]:
        import time
        from scripts.test_pick_cube_simple import PickCubeDiverseEnv
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

        start_time = time.time()

        guidance = self._create_guidance_for_demo(demo)

        self.policy.guidance = guidance

        seed = demo['seed']
        demo_id = demo['demo_id']
        self.logger.info(f"使用演示 {demo_id} 的种子: {seed} (来自 collect_success_demo)")
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
            env_kwargs["human_render_camera_configs"] = {"width": 1280, "height": 720}

        env = gym.make("PickCubeDiverse-v1", **env_kwargs)

        env.reset(seed=0)
        ground_ids = to_numpy(get_ground_ids(env))
        cube_ids = to_numpy(get_cube_ids(env))
        robot_ids = to_numpy(get_robot_ids(env))

        selected_cameras = ["right_shoulder_camera", "left_shoulder_camera", "hand_camera"]
        voxel_size = 0.003
        n_points = self.model_config_dict['dataset_config']['n_points']
        use_pc_color = self.model_config_dict.get('dataset_config', {}).get('use_pc_color', False)

        ws_aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-0.3, -0.3, 0.0),
            max_bound=(0.3, 0.3, 0.5),
        )

        video_dir = None
        if save_video:
            video_dir = self.output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset(seed=seed)
        base_env = env.unwrapped
        goal_pos = base_env.goal_site.pose.p[0].detach().cpu().numpy().astype(np.float32)

        demo_goal_pos = np.array(demo.get('goal_pos', []))
        if len(demo_goal_pos) > 0:
            goal_diff = np.linalg.norm(goal_pos - demo_goal_pos)
            if goal_diff > 1e-3:
                self.logger.warning(
                    f"演示 {demo_id} 的目标位置不一致: "
                    f"demo中={demo_goal_pos}, 环境中={goal_pos}, 差异={goal_diff:.6f}"
                )
            else:
                self.logger.debug(f"演示 {demo_id} 的目标位置验证通过: {goal_pos}")

        robot_state_history = []
        trajectory = []
        episode_frames = []
        actions_sequence = []
        done = False
        step_count = 0
        max_steps = self.config.get('data', {}).get('max_steps', 500)

        enable_dynamic_energy_field = self.config.get('guidance', {}).get('enable_dynamic_energy_field', False)

        cube_grasped = False
        initial_cube_z = None

        if enable_dynamic_energy_field:

            guidance.energy_scales = [0.0, 0.0]
            self.logger.info(f"演示 {demo_id}: 动态能量场模式 - 初始化时禁用能量场（抓取cube前不使用能量场）")
        else:

            original_scales = self.config.get('guidance', {}).get('energy_scales', [-1.0, -1.0])
            guidance.energy_scales = original_scales
            self.logger.info(f"演示 {demo_id}: 静态能量场模式 - 能量场已启用: {original_scales}")

        initial_state = get_robot_state(env).detach().cpu()
        trajectory.append(initial_state[:3].numpy().tolist())

        if save_video:
            frame = env.render()
            episode_frames.append(to_numpy(frame))

        def check_and_update_cube_grasped():
            nonlocal cube_grasped, initial_cube_z
            if not enable_dynamic_energy_field:
                return False

            try:

                cube_pos = base_env.cube.pose.p[0].detach().cpu().numpy()
                cube_z = cube_pos[2]

                if initial_cube_z is None:
                    initial_cube_z = cube_z
                    self.logger.info(f"演示 {demo_id}: 初始cube z坐标: {initial_cube_z:.4f}")

                if not cube_grasped and cube_z > initial_cube_z + 0.05:
                    cube_grasped = True

                    original_scales = self.config.get('guidance', {}).get('energy_scales', [-1.0, -1.0])
                    guidance.energy_scales = original_scales
                    self.logger.info(
                        f"演示 {demo_id} 步骤 {step_count}: Cube已被抓取！"
                        f"(z: {initial_cube_z:.4f} -> {cube_z:.4f})，启用能量场: {original_scales}"
                    )
                    return True
                elif not cube_grasped:

                    if guidance.energy_scales != [0.0, 0.0]:
                        guidance.energy_scales = [0.0, 0.0]
                        self.logger.debug(f"演示 {demo_id} 步骤 {step_count}: Cube未抓取，禁用能量场")
                return False
            except Exception as e:

                if step_count == 0:
                    self.logger.warning(f"演示 {demo_id}: 无法检测cube位置: {e}，将始终使用能量场")
                    cube_grasped = True
                return False

        while not done and step_count < max_steps:

            if enable_dynamic_energy_field:
                check_and_update_cube_grasped()

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

            n_obs_steps = self.model_config_dict['n_obs_steps']
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

            num_actions_per_step = self.config.get('data', {}).get('num_actions_per_step', 32)
            n_pred_steps_model = self.model_config_dict['n_pred_steps']

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

                if enable_dynamic_energy_field and not cube_grasped:
                    was_grasped = check_and_update_cube_grasped()

                    if was_grasped:
                        self.logger.info(
                            f"演示 {demo_id} 步骤 {step_count}: 检测到cube被抓取，"
                            f"中断当前动作执行，重新进行带能量场的推理"
                        )
                        break

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
            video_name = f"demo_{demo['demo_id']:04d}_seed_{seed}.mp4"
            video_path = str(video_dir / video_name)

            imageio.mimsave(video_path, episode_frames, fps=20)
            self.logger.info(f"演示 {demo['demo_id']}: 视频已保存至 {video_path} (成功: {success})")

        env.close()

        energy_centers = guidance.config.get('energy_centers', [])
        energy_scales = guidance.energy_scales if hasattr(guidance, 'energy_scales') else self.config.get('guidance', {}).get('energy_scales', [-1.0] * len(energy_centers))

        robot_state_history_np = np.array([s.numpy() if isinstance(s, torch.Tensor) else s for s in robot_state_history])

        result = {
            'demo_id': demo['demo_id'],
            'seed': seed,
            'success': success,
            'steps': step_count,
            'execution_time': execution_time,
            'video_path': video_path,
            'trajectory': np.array(trajectory),
            'robot_state_history': robot_state_history_np,
            'energy_centers': energy_centers,
            'energy_scales': energy_scales if isinstance(energy_scales, list) else energy_scales.tolist(),
            'actions_sequence': actions_sequence,
        }

        return result

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info(f"开始实验: {self.__class__.__name__}")
        self.logger.info("=" * 60)

        self.logger.info("步骤 1/5: 设置实验...")
        self.setup()

        self.logger.info("步骤 2/5: 加载测试数据...")
        test_demos = self.load_data()

        self.logger.info("步骤 3/5: 运行推理...")
        all_results = []
        for i, demo in enumerate(test_demos):

            self.logger.info(f"处理演示 {i+1}/{len(test_demos)} (demo_id={demo['demo_id']})")

            result = self.run_inference(demo)
            all_results.append(result)

            self.trajectories.append(np.array(result['trajectory']))
            self.success_flags.append(result['success'])
            if hasattr(self, 'execution_times'):
                self.execution_times.append(result['execution_time'])

        self.logger.info("步骤 4/5: 评估结果...")
        metrics = self.evaluate()

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        actions_output_dir = self.output_dir / "actions"
        actions_output_dir.mkdir(parents=True, exist_ok=True)

        for result in all_results:
            demo_id = result['demo_id']
            seed = result['seed']

            serializable_result = convert_to_serializable(result)

            action_file = actions_output_dir / f"demo_{demo_id:04d}_seed_{seed}.json"
            with open(action_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            self.logger.info(f"演示 {demo_id} 的结果已保存至: {action_file}")

        results_file = self.output_dir / "detailed_results.json"
        serializable_results = convert_to_serializable(all_results)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"详细结果汇总已保存到: {results_file}")

        self.logger.info("步骤 5/5: 保存结果...")
        self.save_results(metrics)

        self.logger.info("=" * 60)
        self.logger.info("实验完成！")
        self.logger.info("=" * 60)
