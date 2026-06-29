#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse
from pathlib import Path

def load_visualization_data(data_path):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def visualize_guidance_interactive(data_path):

    print(f"加载数据: {data_path}")
    data_dict = load_visualization_data(data_path)

    vis_data_list = data_dict['data']
    energy_centers = data_dict['energy_centers']
    energy_scales = data_dict['energy_scales']
    sigma = data_dict['sigma']
    horizon = data_dict['horizon']
    goal_pos = data_dict.get('current_goal_pos', None)
    norm_pcd_center = data_dict.get('norm_pcd_center', None)

    print(f"数据点数量: {len(vis_data_list)}")
    print(f"能量中心: {len(energy_centers)} 个")
    print(f"  {energy_centers}")
    print(f"能量 scales: {energy_scales}")
    print(f"Sigma: {sigma}")
    print(f"Horizon: {horizon}")
    print(f"归一化中心: {norm_pcd_center}")

    if len(vis_data_list) == 0:
        print("没有可视化数据！")
        return

    fig = plt.figure(figsize=(24, 8))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Individual Energy Guidances (分解)', fontsize=12, pad=20)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Base Velocity + Base Guidance (修正前)', fontsize=12, pad=20)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Base Velocity + Total Guidance (修正后)', fontsize=12, pad=20)

    current_idx = [0]

    def update_plot(idx):
        idx = int(idx)
        if idx >= len(vis_data_list):
            idx = len(vis_data_list) - 1

        data = vis_data_list[idx]
        x = data['x']
        t = data['t']
        base_velocity = data['base_velocity']
        base_guidance = data['base_guidance']
        learned_correction = data['learned_correction']
        total_guidance = data['total_guidance']

        positions = x[0, :, :3]
        base_vel_pos = base_velocity[0, :, :3]
        base_guid_pos = base_guidance[0, :, :3]
        learned_corr_pos = learned_correction[0, :, :3]
        total_guid_pos = total_guidance[0, :, :3]

        individual_guidances = []
        for center, scale in zip(energy_centers, energy_scales):
            center_np = np.array(center, dtype=np.float32)

            diff = positions - center_np
            sq_dist = np.sum(diff ** 2, axis=1, keepdims=True)

            energy = np.exp(-sq_dist / (sigma ** 2 + 1e-8))

            dist = np.sqrt(sq_dist + 1e-8)
            dir_to_center = -diff / dist

            guidance_single = scale * energy * dir_to_center
            individual_guidances.append(guidance_single)

            if len(individual_guidances) == 1 and len(positions) > 0:
                first_pos = positions[0]
                first_diff = first_pos - center_np
                first_dist = np.linalg.norm(first_diff)
                first_energy = np.exp(-first_dist**2 / (sigma**2 + 1e-8))
                first_dir = -first_diff / (first_dist + 1e-8)
                first_guid = scale * first_energy * first_dir
                print(f"\n  调试 - 第一个点 (位置={first_pos}, 中心={center_np}):")
                print(f"    距离: {first_dist:.6f}")
                print(f"    能量: {first_energy:.6f}")
                print(f"    方向: {first_dir}")
                print(f"    Guidance: {first_guid}, norm={np.linalg.norm(first_guid):.6f}")

        print(f"\n位置统计:")
        print(f"  轨迹点数量: {len(positions)}")
        print(f"  位置范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
              f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
              f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

        print(f"\n速度场统计:")
        print(f"  Base Velocity Norm:      mean={np.linalg.norm(base_vel_pos, axis=1).mean():.6f}, "
              f"max={np.linalg.norm(base_vel_pos, axis=1).max():.6f}")
        print(f"  Base Guidance Norm:      mean={np.linalg.norm(base_guid_pos, axis=1).mean():.6f}, "
              f"max={np.linalg.norm(base_guid_pos, axis=1).max():.6f}")
        print(f"  Learned Correction Norm: mean={np.linalg.norm(learned_corr_pos, axis=1).mean():.6f}, "
              f"max={np.linalg.norm(learned_corr_pos, axis=1).max():.6f}")
        print(f"  Total Guidance Norm:     mean={np.linalg.norm(total_guid_pos, axis=1).mean():.6f}, "
              f"max={np.linalg.norm(total_guid_pos, axis=1).max():.6f}")

        print(f"\n单个能量场 Guidance:")
        for i, (center, scale, guid) in enumerate(zip(energy_centers, energy_scales, individual_guidances)):
            guid_norm_mean = np.linalg.norm(guid, axis=1).mean()
            guid_norm_max = np.linalg.norm(guid, axis=1).max()
            print(f"  能量场 {i+1} (center={center}, scale={scale:.2f}):")
            print(f"    Guidance Norm: mean={guid_norm_mean:.6f}, max={guid_norm_max:.6f}")

            center_np = np.array(center)
            avg_pos = positions.mean(axis=0)
            vec_to_center = center_np - avg_pos
            avg_guidance = guid.mean(axis=0)
            dot_product = np.dot(vec_to_center, avg_guidance)
            print(f"    方向检查: dot(vec_to_center, guidance)={dot_product:.6f} "
                  f"({'排斥 ✓' if (scale < 0 and dot_product < 0) or (scale > 0 and dot_product > 0) else '异常 ✗'})")

        ax1.cla()
        ax2.cla()
        ax3.cla()

        print(f"\n{'='*60}")
        print(f"时间步 {idx+1}/{len(vis_data_list)} | t={t:.3f}")
        print(f"{'='*60}")

        ax1.set_title(f'Individual Energy Guidances (t={t:.3f})', fontsize=12)

        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', alpha=0.4, linewidth=1.5, label='Trajectory Path')

        ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=30, alpha=0.6, label='Trajectory Points')

        colors_individual = ['red', 'magenta', 'orange', 'brown', 'pink']
        scale_guid = 0.1

        base_guid_norm = np.linalg.norm(base_guid_pos, axis=1).mean()
        max_individual_guid_norm = max([np.linalg.norm(guid, axis=1).mean() for guid in individual_guidances]) if individual_guidances else 1e-6

        if max_individual_guid_norm < 1e-6:
            unified_scale_factor = max(100, int(base_guid_norm / (max_individual_guid_norm + 1e-8) / 10))
        elif max_individual_guid_norm < 1e-4:
            unified_scale_factor = max(10, int(base_guid_norm / (max_individual_guid_norm + 1e-8) / 5))
        elif max_individual_guid_norm < 0.01:
            unified_scale_factor = max(2, int(base_guid_norm / (max_individual_guid_norm + 1e-8)))
        else:
            unified_scale_factor = 1

        individual_scale_factors = [unified_scale_factor] * len(individual_guidances)

        for i, (center, scale, guid) in enumerate(zip(energy_centers, energy_scales, individual_guidances)):
            color = colors_individual[i % len(colors_individual)]
            guid_scaled = guid * unified_scale_factor
            ax1.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                      guid_scaled[:, 0], guid_scaled[:, 1], guid_scaled[:, 2],
                      color=color, alpha=0.8, length=scale_guid, normalize=False,
                      label=f'Energy {i+1} (scale={scale:.1f})',
                      arrow_length_ratio=0.3, linewidths=2.5)

        if unified_scale_factor > 1:
            ax1.text2D(0.02, 0.98, f'All guidances ×{unified_scale_factor} for visualization',
                      transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        for i, (center, scale) in enumerate(zip(energy_centers, energy_scales)):
            color = 'orange' if scale < 0 else 'purple'
            ax1.scatter(center[0], center[1], center[2],
                       c=color, s=200, marker='*',
                       edgecolors='black', linewidths=2)

        if goal_pos is not None:
            ax1.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                       c='cyan', s=300, marker='D',
                       edgecolors='black', linewidths=2,
                       label='Goal Position')

        ax2.set_title(f'Base Velocity + Base Guidance (t={t:.3f})', fontsize=12)

        ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', alpha=0.4, linewidth=1.5, label='Trajectory Path')

        ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=30, alpha=0.6, label='Trajectory Points')

        scale_vel = 0.1
        ax2.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                  base_vel_pos[:, 0], base_vel_pos[:, 1], base_vel_pos[:, 2],
                  color='green', alpha=0.6, length=scale_vel, normalize=False,
                  label='Base Velocity', arrow_length_ratio=0.3)

        ax2.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                  base_guid_pos[:, 0], base_guid_pos[:, 1], base_guid_pos[:, 2],
                  color='red', alpha=0.6, length=scale_guid, normalize=False,
                  label='Base Guidance (sum)', arrow_length_ratio=0.3)

        for i, (center, scale) in enumerate(zip(energy_centers, energy_scales)):
            color = 'orange' if scale < 0 else 'purple'
            ax2.scatter(center[0], center[1], center[2],
                       c=color, s=200, marker='*',
                       edgecolors='black', linewidths=2)

        if goal_pos is not None:
            ax2.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                       c='cyan', s=300, marker='D',
                       edgecolors='black', linewidths=2)

        ax3.set_title(f'Base Velocity + Total Guidance (t={t:.3f})', fontsize=12)

        ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', alpha=0.4, linewidth=1.5, label='Trajectory Path')

        ax3.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=30, alpha=0.6, label='Trajectory Points')

        ax3.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                  base_vel_pos[:, 0], base_vel_pos[:, 1], base_vel_pos[:, 2],
                  color='green', alpha=0.6, length=scale_vel, normalize=False,
                  label='Base Velocity', arrow_length_ratio=0.3)

        ax3.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                  total_guid_pos[:, 0], total_guid_pos[:, 1], total_guid_pos[:, 2],
                  color='purple', alpha=0.6, length=scale_guid, normalize=False,
                  label='Total Guidance', arrow_length_ratio=0.3)

        learned_corr_norm_current = np.linalg.norm(learned_corr_pos, axis=1).mean()

        learned_corr_scale_factor = 1.0
        if learned_corr_norm_current < 0.001:
            learned_corr_scale_factor = 10.0
        elif learned_corr_norm_current < 0.01:
            learned_corr_scale_factor = 5.0

        print(f"\n可视化缩放:")
        print(f"  Learned Correction 放大倍数: {learned_corr_scale_factor:.0f}x")

        ax3.quiver(positions[:, 0], positions[:, 1], positions[:, 2],
                  learned_corr_pos[:, 0] * learned_corr_scale_factor,
                  learned_corr_pos[:, 1] * learned_corr_scale_factor,
                  learned_corr_pos[:, 2] * learned_corr_scale_factor,
                  color='yellow', alpha=0.9, length=scale_guid, normalize=False,
                  label=f'Learned Correction (×{learned_corr_scale_factor:.0f} for viz)',
                  arrow_length_ratio=0.3, linewidths=3)

        for i, (center, scale) in enumerate(zip(energy_centers, energy_scales)):
            color = 'orange' if scale < 0 else 'purple'
            ax3.scatter(center[0], center[1], center[2],
                       c=color, s=200, marker='*',
                       edgecolors='black', linewidths=2)

        if goal_pos is not None:
            ax3.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
                       c='cyan', s=300, marker='D',
                       edgecolors='black', linewidths=2)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc='upper right', fontsize=7)

            all_pos = positions
            x_range = [all_pos[:, 0].min() - 0.1, all_pos[:, 0].max() + 0.1]
            y_range = [all_pos[:, 1].min() - 0.1, all_pos[:, 1].max() + 0.1]
            z_range = [all_pos[:, 2].min() - 0.05, all_pos[:, 2].max() + 0.05]

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)

            ax.view_init(elev=20, azim=45)

        base_vel_norm = np.linalg.norm(base_vel_pos, axis=1).mean()
        base_guid_norm = np.linalg.norm(base_guid_pos, axis=1).mean()
        learned_corr_norm = np.linalg.norm(learned_corr_pos, axis=1).mean()
        total_guid_norm = np.linalg.norm(total_guid_pos, axis=1).mean()

        info_text = (
            f"Step {idx+1}/{len(vis_data_list)} | t={t:.3f}\n"
            f"Base Vel Norm: {base_vel_norm:.6f}\n"
            f"Base Guidance Norm: {base_guid_norm:.6f}\n"
            f"Learned Correction Norm: {learned_corr_norm:.6f}\n"
            f"Total Guidance Norm: {total_guid_norm:.6f}"
        )

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.draw()

    ax_slider = plt.axes([0.15, 0.12, 0.7, 0.03])
    slider = Slider(ax_slider, 'Step', 0, len(vis_data_list) - 1,
                   valinit=0, valstep=1)

    slider.on_changed(update_plot)

    ax_prev = plt.axes([0.15, 0.08, 0.1, 0.03])
    ax_next = plt.axes([0.75, 0.08, 0.1, 0.03])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')

    def prev_step(event):
        new_val = max(0, slider.val - 1)
        slider.set_val(new_val)

    def next_step(event):
        new_val = min(len(vis_data_list) - 1, slider.val + 1)
        slider.set_val(new_val)

    btn_prev.on_clicked(prev_step)
    btn_next.on_clicked(next_step)

    update_plot(0)

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可视化 Guidance 效果")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="可视化数据文件路径 (.pkl)"
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 文件不存在 {data_path}")
        return

    visualize_guidance_interactive(str(data_path))

if __name__ == "__main__":
    main()
