import numpy as np
import torch
from typing import Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .task2_static_energy import Task2StaticEnergy
from experiments.guidance import StaticGuidance, GCovGuidance, NoneGuidance

class Task3GcovFull(Task2StaticEnergy):

    def _create_guidance_for_demo(self, demo: Dict[str, Any]) -> GCovGuidance:
        guidance_config = self.config.get('guidance', {}).copy()

        trajectory = np.array(demo['trajectory'])

        demo_id = demo['demo_id']
        num_centers = guidance_config.get('num_energy_centers', 2)
        seed_base = guidance_config.get('seed', 42)

        from experiments.utils.energy_center_utils import get_energy_centers_for_demo
        energy_centers = get_energy_centers_for_demo(
            demo_id=demo_id,
            trajectory=trajectory,
            num_centers=num_centers,
            seed_base=seed_base,
        )

        guidance_config['energy_centers'] = energy_centers

        energy_scales = guidance_config.get('energy_scales', [-1.0] * num_centers)
        if len(energy_scales) != num_centers:
            self.logger.warning(
                f"演示 {demo_id}: energy_scales 数量 ({len(energy_scales)}) 与 "
                f"能量中心数量 ({num_centers}) 不一致，使用默认值"
            )
            energy_scales = [-1.0] * num_centers

        guidance_config['energy_scales'] = energy_scales

        if hasattr(self, 'norm_pcd_center') and self.norm_pcd_center:
            guidance_config['norm_pcd_center'] = self.norm_pcd_center

        base_guidance = StaticGuidance(guidance_config)

        horizon = self.model_config_dict.get('n_pred_steps', 32)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        gcov_config = guidance_config.copy()
        gcov_config.update({
            'train_online': guidance_config.get('train_online', True),
            'online_train_steps': guidance_config.get('online_train_steps', 1000),
            'online_batch_size': guidance_config.get('online_batch_size', 4),
            'online_lr': guidance_config.get('online_lr', 1e-4),
            'num_ode_steps': guidance_config.get('num_ode_steps', 20),
            'conflict_threshold': guidance_config.get('conflict_threshold', 0.5),
            'conflict_temperature': guidance_config.get('conflict_temperature', 0.1),
            'online_loss_type': guidance_config.get('online_loss_type', 'mse_simple'),
            'residual_unet_channels': guidance_config.get('residual_unet_channels', 64),
            'residual_unet_down_dims': guidance_config.get('residual_unet_down_dims', [256, 512, 1024]),
            'energy_temperature': guidance_config.get('energy_temperature', 1.0),
            'policy_temperature': guidance_config.get('policy_temperature', 1.0),

            'debug_training': self.config.get('debug', {}).get('debug_training', False),
            'debug_learned_correction': self.config.get('debug', {}).get('debug_learned_correction', False),

            'save_visualization_data': guidance_config.get('save_visualization_data', False),

            'record_guidance_details': guidance_config.get('record_guidance_details', False),
        })

        guidance = GCovGuidance(
            base_guidance=base_guidance,
            horizon=horizon,
            device=device,
            config=gcov_config,
        )

        self.logger.info(f"演示 {demo_id}: 创建 GCovGuidance - 能量中心数量 = {len(energy_centers)}")
        self.logger.info(f"  在线训练: {gcov_config.get('train_online', True)}")
        self.logger.info(f"  训练步数: {gcov_config.get('online_train_steps', 1000)}")
        self.logger.info(f"  损失类型: {gcov_config.get('online_loss_type', 'mse_simple')}")

        return guidance

    def run_inference(self, demo: Dict[str, Any]) -> Dict[str, Any]:
        import time

        import sys
        from pathlib import Path
        REPO_ROOT = Path(__file__).parent.parent.parent
        MANISKILL_ROOT = REPO_ROOT / "ManiSkill"
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        if str(MANISKILL_ROOT) not in sys.path:
            sys.path.insert(0, str(MANISKILL_ROOT))

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
        from pathlib import Path

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

        guidance = self._create_guidance_for_demo(demo)

        self.policy.guidance = guidance

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

        goal_pos = base_env.cubeB.pose.p[0].detach().cpu().numpy().astype(np.float32)

        output_dir = Path(self.config.get('output_dir', 'experiments/outputs/exp1_static_gcov'))

        debug_dir = output_dir / "debug_logs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_file = debug_dir / f"demo_{demo_id:04d}_seed_{seed}_run.txt"
        debug_fp = open(debug_file, 'w', encoding='utf-8')

        try:
            initial_cubeA_pos = base_env.cubeA.pose.p[0].detach().cpu().numpy()
            initial_cubeB_pos = base_env.cubeB.pose.p[0].detach().cpu().numpy()
            initial_ee_pos = get_robot_state(env).detach().cpu()[:3].numpy()
            debug_msg = (
                f"[DEBUG] 演示 {demo_id} 初始状态: seed={seed}\n"
                f"[DEBUG]  cubeA_pos=[{initial_cubeA_pos[0]:.4f}, {initial_cubeA_pos[1]:.4f}, {initial_cubeA_pos[2]:.4f}]\n"
                f"[DEBUG]  cubeB_pos=[{initial_cubeB_pos[0]:.4f}, {initial_cubeB_pos[1]:.4f}, {initial_cubeB_pos[2]:.4f}]\n"
                f"[DEBUG]  goal_pos=[{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]\n"
                f"[DEBUG]  ee_pos=[{initial_ee_pos[0]:.4f}, {initial_ee_pos[1]:.4f}, {initial_ee_pos[2]:.4f}]\n"
            )
            self.logger.info(debug_msg.strip())
            debug_fp.write(debug_msg)
            debug_fp.flush()
        except Exception as e:
            error_msg = f"[DEBUG] 无法获取初始状态: {e}\n"
            self.logger.warning(error_msg.strip())
            debug_fp.write(error_msg)
            debug_fp.flush()

        video_dir = None
        episode_frames = []
        if save_video:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)

        done = False
        step_count = 0
        max_steps = self.config.get('data', {}).get('max_steps', 100)
        robot_state_history = []
        trajectory = []
        actions_sequence = []

        cube_grasped = False
        initial_cube_z = None
        first_correction_completed = False

        disable_after_first = (
            self.config.get('adaptive', {}).get('disable_guidance_after_first_round', False)
            or self.config.get('debug', {}).get('disable_guidance_after_first_round', False)
        )

        guidance.base_guidance.energy_scales = [0.0, 0.0]
        if hasattr(guidance, 'energy_scales_tensor') and guidance.energy_scales_tensor is not None:
            guidance.energy_scales_tensor = torch.zeros_like(guidance.energy_scales_tensor)

        self.logger.info(f"演示 {demo_id}: 初始化完成，能量场已禁用 (等待抓取事件)。")
        self.logger.info(f"演示 {demo_id}: 在线修正模式: {'单次 (Single Round)' if disable_after_first else '持续 (Continuous)'}")

        initial_state = get_robot_state(env).detach().cpu()
        trajectory.append(initial_state[:3].numpy().tolist())

        if save_video:
            frame = env.render()
            episode_frames.append(to_numpy(frame))

        def check_and_update_cube_grasped():
            nonlocal cube_grasped, initial_cube_z
            try:
                cube_pos = base_env.cubeA.pose.p[0].detach().cpu().numpy()
                cube_z = cube_pos[2]

                if initial_cube_z is None:
                    initial_cube_z = cube_z

                is_grasped_now = cube_z > (initial_cube_z + 0.05)

                if not cube_grasped and is_grasped_now:
                    cube_grasped = True

                    if not first_correction_completed:
                        original_scales = self.config.get('guidance', {}).get('energy_scales', [-1.0, -1.0])
                        guidance.base_guidance.energy_scales = original_scales
                        if hasattr(guidance, 'energy_scales_tensor') and guidance.energy_scales_tensor is not None:
                            guidance.energy_scales_tensor = torch.tensor(original_scales, device=guidance.energy_scales_tensor.device, dtype=guidance.energy_scales_tensor.dtype)

                        self.logger.info(
                            f"演示 {demo_id} 步骤 {step_count}: CubeA被抓取 (z: {initial_cube_z:.3f}->{cube_z:.3f})！"
                            f"启用能量场 {original_scales}，准备在推理前进行训练。"
                        )
                    return True

                return False
            except Exception as e:
                if step_count == 0:
                    self.logger.warning(f"无法检测Cube位置: {e}")
                return False

        while not done and step_count < max_steps:

            check_and_update_cube_grasped()

            performed_correction_this_iter = False

            inference_round_info = {
                'step': step_count,
                'cube_grasped': cube_grasped,
                'energy_scales': guidance.base_guidance.energy_scales.copy() if hasattr(guidance, 'base_guidance') else None,
                'will_train': False,
            }

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

            train_online_cfg = self.config.get('guidance', {}).get('train_online', True)

            should_train = False
            if cube_grasped and train_online_cfg:
                if not first_correction_completed:
                    should_train = True
                elif not disable_after_first:

                    should_train = True

            if should_train:

                if not first_correction_completed or step_count % 10 == 0:
                    self.logger.info(f"演示 {demo_id} 步骤 {step_count}: === 执行 g_cov 在线训练与修正 ===")

                inference_round_info['will_train'] = True

                pcd_tensor_for_train = torch.from_numpy(pcd).float().unsqueeze(0).repeat(1, n_obs_steps, 1, 1).to(DEVICE)
                state_tensor_for_train = torch.stack(hist, dim=0).unsqueeze(0).to(DEVICE)
                goal_pos_tensor = torch.from_numpy(goal_pos).float().unsqueeze(0).repeat(1, n_obs_steps, 1).to(DEVICE)

                try:

                    guidance.train_model(
                        flow_model=self.policy.base_policy,
                        pcd=pcd_tensor_for_train,
                        robot_state_obs=state_tensor_for_train,
                        goal_pos=goal_pos_tensor,
                        num_samples=self.config.get("guidance", {}).get("online_batch_size", 4),
                    )

                    performed_correction_this_iter = True
                    inference_round_info['performed_correction'] = True
                    if not first_correction_completed:
                        self.logger.info("在线训练完成，本次推理将使用修正后的模型（能量场与方法已激活）。")
                except Exception as e:
                    self.logger.error(f"g_cov 在线训练失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    inference_round_info['performed_correction'] = False
            else:
                inference_round_info['performed_correction'] = False

            should_be_disabled = False

            if first_correction_completed and disable_after_first:

                should_be_disabled = True
            elif cube_grasped and not should_train and not performed_correction_this_iter:

                should_be_disabled = True

            if should_be_disabled:
                 current_scales = guidance.base_guidance.energy_scales
                 if isinstance(current_scales, list) and any(s != 0 for s in current_scales):
                     guidance.base_guidance.energy_scales = [0.0, 0.0]
                     if hasattr(guidance, 'energy_scales_tensor') and guidance.energy_scales_tensor is not None:
                        guidance.energy_scales_tensor = torch.zeros_like(guidance.energy_scales_tensor)

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

                current_scales = guidance.base_guidance.energy_scales
                using_guidance = isinstance(current_scales, list) and any(s != 0 for s in current_scales)
                if using_guidance:
                    if cube_grasped:
                        g_val = -0.9
                    else:
                        g_val = (q_grip - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
                        if q_grip < 0.035:
                            g_val = -0.9
                else:
                    g_val = (q_grip - GRIPPER_LOWER) / (GRIPPER_UPPER - GRIPPER_LOWER) * 2 - 1
                    if q_grip < 0.025:
                        g_val = -0.9

                action = np.concatenate([d_pos_norm, np.zeros(3), [np.clip(g_val, -1.0, 1.0)]])

                cube_grasped_before = cube_grasped

                obs, _, _, _, info = env.step(action)

                actions_sequence.append(action.tolist())

                if not cube_grasped:
                    was_grasped = check_and_update_cube_grasped()
                    if was_grasped:
                        self.logger.info(
                            f"演示 {demo_id} 步骤 {step_count}: 动作执行中检测到抓取！"
                            f"中断当前动作序列，准备进入在线修正流程。"
                        )
                        break

                after_action_state = get_robot_state(env).detach().cpu()
                trajectory.append(after_action_state[:3].numpy().tolist())

                try:
                    cubeA_pos = base_env.cubeA.pose.p[0].detach().cpu().numpy()
                    cubeB_pos = base_env.cubeB.pose.p[0].detach().cpu().numpy()
                    success_flag = bool(info["success"].item())
                    action_str = np.array2string(action, precision=4, suppress_small=True)
                    debug_msg = (
                        f"[DEBUG] 演示 {demo_id} 步骤 {step_count}: "
                        f"动作={action_str}, "
                        f"cubeA_pos=[{cubeA_pos[0]:.4f}, {cubeA_pos[1]:.4f}, {cubeA_pos[2]:.4f}], "
                        f"cubeB_pos=[{cubeB_pos[0]:.4f}, {cubeB_pos[1]:.4f}, {cubeB_pos[2]:.4f}], "
                        f"success={success_flag}, "
                        f"ee_pos=[{after_action_state[0]:.4f}, {after_action_state[1]:.4f}, {after_action_state[2]:.4f}]\n"
                    )
                    self.logger.debug(debug_msg.strip())
                    debug_fp.write(debug_msg)
                    debug_fp.flush()
                except Exception as e:
                    error_msg = f"[DEBUG] 演示 {demo_id} 步骤 {step_count}: 无法获取debug信息: {e}\n"
                    self.logger.warning(error_msg.strip())
                    debug_fp.write(error_msg)
                    debug_fp.flush()

                if save_video:
                    frame = env.render()
                    episode_frames.append(to_numpy(frame))
                elif visualize:
                    env.render()

                done = bool(info["success"].item())
                step_count += 1
                if done: break

            if performed_correction_this_iter:
                if not first_correction_completed:
                    self.logger.info(f"演示 {demo_id}: === 已完成首轮修正动作执行 ===")
                    first_correction_completed = True

                if disable_after_first:
                    self.logger.info(f"演示 {demo_id}: 配置要求单次修正，后续回退到 Base Model。")
                    guidance.base_guidance.energy_scales = [0.0, 0.0]
                    if hasattr(guidance, 'energy_scales_tensor') and guidance.energy_scales_tensor is not None:
                        guidance.energy_scales_tensor = torch.zeros_like(guidance.energy_scales_tensor)

            if done:
                break

        success = bool(info["success"].item())
        execution_time = time.time() - start_time

        try:
            final_cubeA_pos = base_env.cubeA.pose.p[0].detach().cpu().numpy()
            final_cubeB_pos = base_env.cubeB.pose.p[0].detach().cpu().numpy()
            final_ee_pos = get_robot_state(env).detach().cpu()[:3].numpy()
            cubeA_cubeB_distance = np.linalg.norm(final_cubeA_pos - final_cubeB_pos)
            debug_msg = (
                f"[DEBUG] 演示 {demo_id} 最终状态:\n"
                f"[DEBUG]  cubeA_pos=[{final_cubeA_pos[0]:.4f}, {final_cubeA_pos[1]:.4f}, {final_cubeA_pos[2]:.4f}]\n"
                f"[DEBUG]  cubeB_pos=[{final_cubeB_pos[0]:.4f}, {final_cubeB_pos[1]:.4f}, {final_cubeB_pos[2]:.4f}]\n"
                f"[DEBUG]  cubeA到cubeB的距离: {cubeA_cubeB_distance:.4f}米\n"
                f"[DEBUG]  ee_pos=[{final_ee_pos[0]:.4f}, {final_ee_pos[1]:.4f}, {final_ee_pos[2]:.4f}]\n"
                f"[DEBUG]  success={success}, steps={step_count}, actions_sequence长度={len(actions_sequence)}\n"
            )
            self.logger.info(debug_msg.strip())
            debug_fp.write(debug_msg)
            debug_fp.flush()
        except Exception as e:
            error_msg = f"[DEBUG] 无法获取最终状态: {e}\n"
            self.logger.warning(error_msg.strip())
            debug_fp.write(error_msg)
            debug_fp.flush()

        debug_fp.close()
        self.logger.info(f"[DEBUG] 调试信息已保存至: {debug_file}")

        video_path = None
        if save_video and len(episode_frames) > 0:
            video_name = f"demo_{demo_id:04d}_seed_{seed}.mp4"
            video_path = str(video_dir / video_name)
            imageio.mimsave(video_path, episode_frames, fps=20)
            self.logger.info(f"演示 {demo_id}: 视频已保存至 {video_path} (成功: {success})")

        env.close()

        if hasattr(guidance, 'base_guidance'):
            energy_centers = guidance.base_guidance.config.get('energy_centers', [])

            original_energy_scales = guidance.base_guidance.config.get('energy_scales', self.config.get('guidance', {}).get('energy_scales', [-1.0] * len(energy_centers)))

            if len(original_energy_scales) != len(energy_centers):
                energy_scales = original_energy_scales[:len(energy_centers)] if len(original_energy_scales) > len(energy_centers) else original_energy_scales + [-1.0] * (len(energy_centers) - len(original_energy_scales))
            else:
                energy_scales = original_energy_scales
        else:
            energy_centers = []
            energy_scales = []

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
            'energy_centers': energy_centers,
            'energy_scales': energy_scales if isinstance(energy_scales, list) else energy_scales.tolist(),
            'actions_sequence': actions_sequence,
        }

        self.logger.info(
            f"演示 {demo_id}: 推理完成 - "
            f"成功: {success}, 步数: {step_count}, 时间: {execution_time:.2f}s"
        )

        return result
