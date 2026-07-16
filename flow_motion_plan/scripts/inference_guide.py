import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import sys
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffuser.models.flow_guide import *
from diffuser.datasets.normalize import WallLocLimitsNormalizer, TrajectoryLimitsNormalizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

class FlowModelInference:

    def __init__(self, checkpoint_path, config_path=None, energy_center=None, energy_scale=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_config = config['model']
        else:
            model_config = checkpoint.get('config', {}).get('model', {})
            print("model_config", model_config)

            if not model_config:

                model_config = {
                    'position_dim': 2,
                    'horizon': 40,
                    'hidden_dim': 512,
                    'num_layers': 8,
                    'time_embedding_dim': 256,
                    'condition_dim': 4,
                    'max_walls': 6,
                    'wall_feature_dim': 512,
                    'num_attention_heads': 8,
                    'dropout': 0.15
                }

        self.model = TrajFlowModel(**model_config)

        model_keys = set(self.model.state_dict().keys())

        loaded_state_dict = checkpoint['model_state_dict']

        filtered_state_dict = {
            k: v for k, v in loaded_state_dict.items()
            if k in model_keys
        }

        ignored_keys = set(loaded_state_dict.keys()) - model_keys
        if ignored_keys:
            print(f"⚠️ 忽略了检查点中的以下多余参数 ({len(ignored_keys)} 个):")
            for k in sorted(list(ignored_keys)):
                print(f"  - {k}")

        self.model.load_state_dict(filtered_state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.horizon = model_config['horizon']
        self.maze_size = (5, 5)

        print(f"✅ 模型加载成功")
        print(f"   - 设备: {self.device}")
        print(f"   - 轨迹长度: {self.horizon}")
        print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        self.energy_center = energy_center
        self.energy_function = None
        self.energy_scale = energy_scale

    def normalize_inputs(self, start_pos, goal_pos, wall_positions):
        walls_np = np.array(wall_positions, dtype=np.float32).reshape(-1, 2)
        wall_normalizer = WallLocLimitsNormalizer(walls_np, self.maze_size)

        dummy_traj = np.array([start_pos, goal_pos], dtype=np.float32)
        traj_normalizer = TrajectoryLimitsNormalizer(dummy_traj, self.maze_size)

        norm_walls = wall_normalizer.normalize(walls_np)
        norm_start = traj_normalizer.normalize(np.array([start_pos], dtype=np.float32))[0]
        norm_goal  = traj_normalizer.normalize(np.array([goal_pos ], dtype=np.float32))[0]

        return norm_start, norm_goal, norm_walls, traj_normalizer

    def normalize_energy_center(self, energy_center):
        _center = np.array([energy_center], dtype=np.float32)
        energy_center_normalizer = TrajectoryLimitsNormalizer(_center, self.maze_size)
        return energy_center_normalizer.normalize(_center)[0]

    def generate_trajectory(self, start_pos, goal_pos, wall_positions,
                          num_steps=20, dt=0.01, num_samples=1, record_steps=False):

        if len(wall_positions) > 6:
            wall_positions = wall_positions[:6]
        elif len(wall_positions) < 6:

            wall_positions = wall_positions + [[0, 0]] * (6 - len(wall_positions))

        norm_start, norm_goal, norm_walls, traj_normalizer = self.normalize_inputs(
            start_pos, goal_pos, wall_positions
        )
        if len(self.energy_center) == 1:
            norm_energy_center = self.normalize_energy_center(self.energy_center[0])
            J = EnergyFunction(norm_energy_center)
            self.energy_function = [J]
        elif len(self.energy_center) == 2:
            norm_energy_center_1 = self.normalize_energy_center(self.energy_center[0])
            norm_energy_center_2 = self.normalize_energy_center(self.energy_center[1])
            J1 = EnergyFunction(norm_energy_center_1)
            J2 = EnergyFunction(norm_energy_center_2)
            self.energy_function = [J1, J2]
        else:
            self.energy_function = None

        start_tensor = torch.FloatTensor(norm_start).to(self.device)
        goal_tensor = torch.FloatTensor(norm_goal).to(self.device)
        walls_tensor = torch.FloatTensor(norm_walls).to(self.device)

        conditions = {
            0: start_tensor.unsqueeze(0).repeat(num_samples, 1),
            self.horizon - 1: goal_tensor.unsqueeze(0).repeat(num_samples, 1)
        }
        wall_locations = walls_tensor.unsqueeze(0).repeat(num_samples, 1, 1)

        with torch.no_grad():
            trajectories = self.model.sample_trajectory(
                conditions=conditions,
                wall_locations=wall_locations,
                num_steps=num_steps,
                energy_guide=self.energy_function is not None,
                energy_function=self.energy_function,
                energy_scale=self.energy_scale,
                record_steps=record_steps
            )

        trajectories_np = trajectories.cpu().numpy()
        unnorm_trajectories = []

        for i in range(num_samples):
            traj = trajectories_np[i]

            pos_traj = traj[:, :, :2]

            unnorm_traj = traj_normalizer.unnormalize(pos_traj)
            unnorm_traj = unnorm_traj.squeeze(0)
            unnorm_trajectories.append(unnorm_traj)

        return np.array(unnorm_trajectories), traj_normalizer

    def visualize_trajectory(self, start_pos, goal_pos, wall_positions,
                           trajectories, save_path=None, show=False):
        plt.figure(figsize=(6, 6))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        from matplotlib.patches import Rectangle
        for i, wall in enumerate(wall_positions):
            if wall[0] != 0 or wall[1] != 0:
                rect = Rectangle(
                    (wall[0] - 0.5, wall[1] - 0.5),
                    1.0, 1.0,
                    facecolor='blue',
                    alpha=0.7,
                    label='Obstacles' if i == 0 else ""
                )
                plt.gca().add_patch(rect)

        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
        for i, traj in enumerate(trajectories):

            plt.plot(traj[:, 0], traj[:, 1],
                    color=colors[i], alpha=0.8, linewidth=2,
                    label=f'Trajectory {i+1}' if len(trajectories) > 1 else 'Trajectory')

            plt.scatter(traj[:, 0], traj[:, 1],
                      color='red', alpha=0.8, s=50, marker='o',
                      zorder=4)

        plt.scatter(start_pos[0], start_pos[1], c='green', s=150,
                   marker='o', label='Start', zorder=5)
        plt.scatter(goal_pos[0], goal_pos[1], c='blue', s=150,
                   marker='*', label='Goal', zorder=5)

        for i in range(6):
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.legend()
        plt.title('Generated Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图像保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_step_by_step(self, start_pos, goal_pos, wall_positions,
                             step_data, traj_normalizer, save_path=None, show=False):
        num_steps = len(step_data)
        if num_steps == 0:
            print("没有步骤数据可显示")
            return

        max_display_steps = 20
        if num_steps > max_display_steps:

            step_indices = [0]

            if num_steps > 2:
                mid_steps = list(range(1, num_steps-1))
                step_indices.extend(mid_steps[::len(mid_steps)//(max_display_steps-2)])
            step_indices.append(num_steps - 1)
            step_indices = sorted(list(set(step_indices)))[:max_display_steps]
            filtered_step_data = [step_data[i] for i in step_indices]
            print(f"⚠️  步骤数过多({num_steps}步)，仅显示关键步骤: {step_indices}")
        else:
            filtered_step_data = step_data
            step_indices = list(range(num_steps))

        display_steps = len(filtered_step_data)

        max_height_inches = 200
        height_per_step = 3.0
        fig_height = min(height_per_step * display_steps, max_height_inches)

        fig, axes = plt.subplots(display_steps, 4, figsize=(20, fig_height))
        if display_steps == 1:
            axes = axes.reshape(1, -1)

        common_scale = 1
        print(f"🔧 使用固定缩放系数: {common_scale}")

        for display_idx, step_info in enumerate(filtered_step_data):
            original_step_idx = step_indices[display_idx]
            trajectory = step_info['trajectory'][0]
            v_uncond = step_info['v_uncond'][0]
            guidance_grad = step_info['guidance_grad'][0]
            individual_grads = step_info.get('individual_grads', [])
            t_val = step_info['t']

            traj_pos_norm = trajectory[:, :2]
            v_uncond_pos = v_uncond[:, :2]
            guidance_grad_pos = guidance_grad[:, :2]

            traj_pos = traj_normalizer.unnormalize(traj_pos_norm)

            individual_grads_pos = []
            for grad in individual_grads:
                grad_pos = grad[0, :, :]
                individual_grads_pos.append(grad_pos)

            ax_traj = axes[display_idx, 0]
            ax_traj.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=2, alpha=0.8)
            ax_traj.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=30, alpha=0.8)
            ax_traj.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', label='Start')
            ax_traj.scatter(goal_pos[0], goal_pos[1], c='blue', s=100, marker='*', label='Goal')

            from matplotlib.patches import Rectangle
            for wall in wall_positions:
                if wall[0] != 0 or wall[1] != 0:
                    rect = Rectangle((wall[0] - 0.5, wall[1] - 0.5), 1.0, 1.0,
                                   facecolor='gray', alpha=0.7)
                    ax_traj.add_patch(rect)

            ax_traj.set_xlim(0, 5)
            ax_traj.set_ylim(0, 5)

            if isinstance(t_val, np.ndarray):
                t_val_scalar = float(t_val.item()) if t_val.size == 1 else float(t_val[0])
            else:
                t_val_scalar = float(t_val)
            ax_traj.set_title(f'Step {original_step_idx+1}/{num_steps}: Trajectory (t={t_val_scalar:.3f})')
            ax_traj.grid(True, alpha=0.3)
            ax_traj.legend()

            ax_v = axes[display_idx, 1]
            ax_v.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
            ax_v.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

            step_size = max(1, len(traj_pos)//8)
            for i in range(0, len(traj_pos), step_size):
                if i < len(traj_pos):

                    vx, vy = v_uncond_pos[i, 0], v_uncond_pos[i, 1]

                    v_mag = np.sqrt(vx**2 + vy**2)
                    if v_mag > 0:
                        ax_v.arrow(traj_pos[i, 0], traj_pos[i, 1],
                                  vx * common_scale, vy * common_scale,
                                  head_width=0.05, head_length=0.05,
                                  fc='blue', ec='blue', alpha=0.8)

            v_mag_max = np.sqrt((v_uncond_pos**2).sum(axis=1)).max()
            ax_v.text(0.02, 0.98, f'Max |v|: {v_mag_max:.3f}',
                     transform=ax_v.transAxes, fontsize=8,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_v.set_xlim(0, 5)
            ax_v.set_ylim(0, 5)
            ax_v.set_title(f'Step {original_step_idx+1}/{num_steps}: v_uncond')
            ax_v.grid(True, alpha=0.3)

            ax_grad = axes[display_idx, 2]
            ax_grad.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
            ax_grad.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

            step_size = max(1, len(traj_pos)//8)
            for i in range(0, len(traj_pos), step_size):
                if i < len(traj_pos):

                    gx, gy = guidance_grad_pos[i, 0], guidance_grad_pos[i, 1]

                    g_mag = np.sqrt(gx**2 + gy**2)
                    if g_mag > 0:
                        ax_grad.arrow(traj_pos[i, 0], traj_pos[i, 1],
                                     gx * common_scale, gy * common_scale,
                                     head_width=0.05, head_length=0.05,
                                     fc='red', ec='red', alpha=0.8)

            g_mag_max = np.sqrt((guidance_grad_pos**2).sum(axis=1)).max()
            ax_grad.text(0.02, 0.98, f'Max |grad|: {g_mag_max:.3f}',
                        transform=ax_grad.transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_grad.set_xlim(0, 5)
            ax_grad.set_ylim(0, 5)
            ax_grad.set_title(f'Step {original_step_idx+1}/{num_steps}: guidance_grad')
            ax_grad.grid(True, alpha=0.3)

            ax_individual = axes[display_idx, 3]
            ax_individual.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
            ax_individual.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

            colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']

            for grad_idx, grad_pos in enumerate(individual_grads_pos):
                color = colors[grad_idx % len(colors)]

                step_size = max(1, len(traj_pos)//8)
                for i in range(0, len(traj_pos), step_size):
                    if i < len(traj_pos):
                        gx, gy = grad_pos[i, 0], grad_pos[i, 1]
                        g_mag = np.sqrt(gx**2 + gy**2)

                        if g_mag > 0:
                            ax_individual.arrow(traj_pos[i, 0], traj_pos[i, 1],
                                              gx * common_scale, gy * common_scale,
                                              head_width=0.04, head_length=0.04,
                                              fc=color, ec=color, alpha=0.7,
                                              label=f'Energy {grad_idx+1}' if i == 0 else "")

            ax_individual.set_xlim(0, 5)
            ax_individual.set_ylim(0, 5)
            ax_individual.set_title(f'Step {original_step_idx+1}/{num_steps}: Individual Gradients')
            ax_individual.grid(True, alpha=0.3)
            if len(individual_grads_pos) > 0:
                ax_individual.legend(fontsize=6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 步骤可视化图像保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

def main():
    setup_seed(42)

    parser = argparse.ArgumentParser(description='流模型轨迹生成')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径（可选）')
    parser.add_argument('--start', type=float, nargs=2, default=[0.5, 0.5],
                       help='起点坐标 (x y)')
    parser.add_argument('--goal', type=float, nargs=2, default=[4.5, 4.5],
                       help='终点坐标 (x y)')
    parser.add_argument('--walls', type=float, nargs='+',
                       default=[1.5, 1.5, 2.5, 2.5, 3.0, 1.0, 0, 0, 0, 0, 0, 0],
                       help='障碍物坐标 (x1 y1 x2 y2 ...)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='生成轨迹数量')
    parser.add_argument('--steps', type=int, default=20,
                       help='ODE求解步数')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='时间步长')
    parser.add_argument('--save', type=str,
                       help='保存图像路径')
    parser.add_argument('--energy_center', type=float, nargs='+', default=None,
                        help='能量引导中心坐标')
    parser.add_argument('--energy_scale', type=float, nargs='+', default=None,
                        help='能量引导尺度')
    parser.add_argument('--record_steps', type=bool, default=True,
                        help='记录每一步的数据用于可视化')
    parser.add_argument('--step_viz_save', type=str,
                        help='步骤可视化图像保存路径')

    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent.parent / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():

            args.save = str(output_dir / save_path.name)
        else:

            if output_dir not in save_path.parents:
                args.save = str(output_dir / save_path.name)

    if args.step_viz_save:
        step_viz_path = Path(args.step_viz_save)
        if not step_viz_path.is_absolute():
            args.step_viz_save = str(output_dir / step_viz_path.name)
        else:
            if output_dir not in step_viz_path.parents:
                args.step_viz_save = str(output_dir / step_viz_path.name)

    if not args.save:
        args.save = str(output_dir / 'infer_traj_guide.png')
    if args.record_steps and not args.step_viz_save:
        args.step_viz_save = str(output_dir / 'infer_step_guide.png')

    wall_coords = args.walls
    if len(wall_coords) % 2 != 0:
        print("❌ 障碍物坐标数量必须是偶数")
        return

    wall_positions = []
    for i in range(0, len(wall_coords), 2):
        wall_positions.append([wall_coords[i], wall_coords[i+1]])

    print(f"🎯 轨迹生成参数:")
    print(f"   - 起点: {args.start}")
    print(f"   - 终点: {args.goal}")
    print(f"   - 障碍物: {wall_positions}")
    print(f"   - 生成数量: {args.num_samples}")
    print(f"   - 输出目录: {output_dir}")

    if args.energy_center is None:
        inferencer = FlowModelInference(args.checkpoint, args.config, energy_center=[], energy_scale=None)
    elif len(args.energy_center) == 2:
        assert len(args.energy_scale) == 1, "能量场数量和能量尺度数量不匹配"
        inferencer = FlowModelInference(args.checkpoint, args.config, energy_center=[args.energy_center], energy_scale=args.energy_scale)
    elif len(args.energy_center) == 4:
        assert len(args.energy_scale) == 2, "能量场数量和能量尺度数量不匹配"
        inferencer = FlowModelInference(args.checkpoint, args.config, energy_center=[args.energy_center[:2], args.energy_center[2:]], energy_scale=args.energy_scale)
    else:
        print("energy_center数量不对")
        return

    print("🚀 开始生成轨迹...")
    trajectories, traj_normalizer = inferencer.generate_trajectory(
        start_pos=args.start,
        goal_pos=args.goal,
        wall_positions=wall_positions,
        num_steps=args.steps,
        dt=args.dt,
        num_samples=args.num_samples,
        record_steps=args.record_steps
    )

    print(f"✅ 轨迹生成完成: {trajectories.shape}")

    inferencer.visualize_trajectory(
        start_pos=args.start,
        goal_pos=args.goal,
        wall_positions=wall_positions,
        trajectories=trajectories,
        save_path=args.save,
        show=False
    )
    print("record_steps是", args.record_steps)

    if args.record_steps and hasattr(inferencer.model, 'guide_model') and inferencer.model.guide_model is not None:
        step_data = inferencer.model.guide_model.step_data
        if step_data:
            print(f"📊 开始步骤可视化，共{len(step_data)}步...")
            inferencer.visualize_step_by_step(
                start_pos=args.start,
                goal_pos=args.goal,
                wall_positions=wall_positions,
                step_data=step_data,
                traj_normalizer=traj_normalizer,
                save_path=args.step_viz_save,
                show=False
            )
        else:
            print("⚠️ 没有步骤数据可显示")

    traj_save_path = output_dir / 'generated_trajectories_guide.npz'
    np.savez(traj_save_path,
            trajectories=trajectories,
            start_pos=args.start,
            goal_pos=args.goal,
            wall_positions=wall_positions)
    print(f"💾 轨迹数据保存到: {traj_save_path}")

if __name__ == "__main__":
    main()
