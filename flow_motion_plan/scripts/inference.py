import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffuser.models.flow_base_v4 import TrajFlowModel
from diffuser.datasets.normalize import WallLocLimitsNormalizer, TrajectoryLimitsNormalizer

class FlowModelInference:

    def __init__(self, checkpoint_path, config_path=None):
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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.horizon = model_config['horizon']
        self.maze_size = (5, 5)

        print(f"✅ 模型加载成功")
        print(f"   - 设备: {self.device}")
        print(f"   - 轨迹长度: {self.horizon}")
        print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def normalize_inputs(self, start_pos, goal_pos, wall_positions):
        walls_np = np.array(wall_positions, dtype=np.float32).reshape(-1, 2)
        wall_normalizer = WallLocLimitsNormalizer(walls_np, self.maze_size)

        dummy_traj = np.array([start_pos, goal_pos], dtype=np.float32)
        traj_normalizer = TrajectoryLimitsNormalizer(dummy_traj, self.maze_size)

        norm_walls = wall_normalizer.normalize(walls_np)
        norm_start = traj_normalizer.normalize(np.array([start_pos], dtype=np.float32))[0]
        norm_goal  = traj_normalizer.normalize(np.array([goal_pos ], dtype=np.float32))[0]

        return norm_start, norm_goal, norm_walls, traj_normalizer

    def generate_trajectory(self, start_pos, goal_pos, wall_positions,
                          num_steps=20, dt=0.01, num_samples=1):

        if len(wall_positions) > 6:
            wall_positions = wall_positions[:6]
        elif len(wall_positions) < 6:

            wall_positions = wall_positions + [[0, 0]] * (6 - len(wall_positions))

        norm_start, norm_goal, norm_walls, traj_normalizer = self.normalize_inputs(
            start_pos, goal_pos, wall_positions
        )

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

            )
        print("trajectories形状是", trajectories.shape)

        trajectories_np = trajectories.cpu().numpy()
        unnorm_trajectories = []

        for i in range(num_samples):
            traj = trajectories_np[i]

            pos_traj = traj[:, :, :2]
            print("pos_traj形状是", pos_traj.shape)
            unnorm_traj = traj_normalizer.unnormalize(pos_traj)
            unnorm_traj = unnorm_traj.squeeze(0)
            unnorm_trajectories.append(unnorm_traj)
            print("unnorm_traj形状是", unnorm_traj.shape)
        return np.array(unnorm_trajectories)

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

def main():
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
    parser.add_argument('--steps', type=int, default=50,
                       help='ODE求解步数')
    parser.add_argument('--dt', type=float, default=0.05,
                       help='时间步长')
    parser.add_argument('--save', type=str,
                       help='保存图像路径')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图像')

    args = parser.parse_args()

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

    inferencer = FlowModelInference(args.checkpoint, args.config)

    print("🚀 开始生成轨迹...")
    trajectories = inferencer.generate_trajectory(
        start_pos=args.start,
        goal_pos=args.goal,
        wall_positions=wall_positions,
        num_steps=args.steps,
        dt=args.dt,
        num_samples=args.num_samples
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

    if args.save:
        save_dir = Path(args.save).parent
        traj_save_path = save_dir / 'generated_trajectories.npz'
        np.savez(traj_save_path,
                trajectories=trajectories,
                start_pos=args.start,
                goal_pos=args.goal,
                wall_positions=wall_positions)
        print(f"💾 轨迹数据保存到: {traj_save_path}")

if __name__ == "__main__":
    main()
