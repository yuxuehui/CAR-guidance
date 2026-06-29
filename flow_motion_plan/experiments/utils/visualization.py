import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

def visualize_trajectory(
    trajectory: np.ndarray,
    start_pos: List[float],
    goal_pos: List[float],
    wall_positions: List[List[float]],
    energy_centers: Optional[List[List[float]]] = None,
    path_functions: Optional[List] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Generated Trajectory (Base Model)",
    goal_tol: float = 0.3,
):
    from matplotlib.patches import Rectangle

    traj_array = np.asarray(trajectory, dtype=np.float64)

    if traj_array.ndim == 0:

        raise ValueError(f"trajectory is a scalar, expected at least 1D array")
    elif traj_array.ndim == 1:

        if traj_array.size % 2 == 0:
            traj_array = traj_array.reshape(-1, 2)
        else:
            raise ValueError(f"trajectory has shape {traj_array.shape}, cannot reshape to [H, 2]")

    if traj_array.ndim == 2:
        traj_array = traj_array[None, ...]
    elif traj_array.ndim > 3:
        raise ValueError(f"trajectory has {traj_array.ndim} dimensions, expected at most 3D [num_traj, H, 2]")

    validated_trajs = []
    for ti, traj in enumerate(traj_array):
        traj = np.asarray(traj, dtype=np.float64)

        traj = np.atleast_1d(traj)

        if traj.ndim == 0:
            raise ValueError(f"trajectory[{ti}] is a scalar after atleast_1d, this should not happen")
        elif traj.ndim == 1:
            if traj.size == 2:

                traj = traj.reshape(1, 2)
            else:

                if traj.size % 2 == 0:
                    traj = traj.reshape(-1, 2)
                else:
                    raise ValueError(f"trajectory[{ti}] has shape {traj.shape}, cannot reshape to [H, 2]")
        elif traj.ndim == 2:
            if traj.shape[1] != 2:
                raise ValueError(f"trajectory[{ti}] has shape {traj.shape}, expected [H, 2]")
        else:
            raise ValueError(f"trajectory[{ti}] has {traj.ndim} dimensions, expected 2D [H, 2]")

        validated_trajs.append(traj)

    traj_array = np.array(validated_trajs)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)

    ax.set_facecolor("#f8f8f8")
    ax.grid(color="#e0e0e0", linestyle="--", linewidth=0.5, alpha=0.8)

    for i, wall in enumerate(wall_positions):
        if wall[0] != 0 or wall[1] != 0:
            rect = Rectangle(
                (wall[0] - 0.5, wall[1] - 0.5),
                1.0,
                1.0,
                facecolor='blue',
                alpha=0.7,
                label="Obstacle" if i == 0 else "",
            )
            ax.add_patch(rect)

    cmap_traj = plt.cm.viridis

    for ti, traj in enumerate(traj_array):

        traj = np.asarray(traj, dtype=np.float64)

        goal_idx = -1
        for j in range(len(traj)):
            dist = np.linalg.norm(traj[j] - np.array(goal_pos))
            if dist <= goal_tol:
                goal_idx = j
                break

        if goal_idx >= 0:

            traj_to_plot = traj[:goal_idx + 1]

            num_connect_points = 5
            connect_points = np.linspace(
                traj[goal_idx],
                np.array(goal_pos),
                num_connect_points + 1
            )

            traj_to_plot = np.vstack([traj_to_plot, connect_points[1:]])
        else:

            traj_to_plot = traj

        t_norm = np.linspace(0, 1, traj_to_plot.shape[0])
        for k in range(traj_to_plot.shape[0] - 1):
            c = cmap_traj(t_norm[k])
            ax.plot(
                traj_to_plot[k : k + 2, 0],
                traj_to_plot[k : k + 2, 1],
                color=c,
                linewidth=2.5,
                alpha=0.9,
            )

        if goal_idx >= 0:

            scatter_traj = traj[:goal_idx + 1]
            scatter_t_norm = np.linspace(0, 1, scatter_traj.shape[0])
        else:
            scatter_traj = traj
            scatter_t_norm = np.linspace(0, 1, scatter_traj.shape[0])

        ax.scatter(
            scatter_traj[:, 0],
            scatter_traj[:, 1],
            c=scatter_t_norm,
            cmap=cmap_traj,
            s=18,
            alpha=0.9,
            edgecolors="none",
        )

    ax.scatter(
        start_pos[0],
        start_pos[1],
        c="#2ca02c",
        s=120,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
        label="Start",
        zorder=6,
    )
    ax.scatter(
        goal_pos[0],
        goal_pos[1],
        c="#d62728",
        s=160,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        label="Goal",
        zorder=6,
    )

    if energy_centers is not None:
        for idx, c in enumerate(energy_centers):
            ax.scatter(
                c[0],
                c[1],
                c="#ff7f0e",
                s=160,
                marker="x",
                linewidths=2.0,
                label="Energy Center" if idx == 0 else "",
                zorder=7,
            )

    if path_functions is not None:
        time_samples = np.linspace(0.0, 1.0, 50)
        path_colors = ['#ff7f0e', '#9467bd']
        for path_idx, path_fn in enumerate(path_functions):
            path_points = np.array([path_fn(t) for t in time_samples])
            path_color = path_colors[path_idx % len(path_colors)]

            ax.plot(path_points[:, 0], path_points[:, 1],
                    color=path_color, linewidth=2.5, alpha=0.6, linestyle='--',
                    label=f'Energy Center {path_idx+1} Path')

            marker_times = [0.0, 0.5, 1.0]
            for mt in marker_times:
                center_pos = np.array(path_fn(mt))
                ax.scatter(center_pos[0], center_pos[1],
                          c=path_color, s=100, marker='x',
                          linewidths=2.5, zorder=7, alpha=0.8)

    for i in range(6):
        ax.axhline(y=i, color="gray", linestyle="-", alpha=0.25, linewidth=0.7)
        ax.axvline(x=i, color="gray", linestyle="-", alpha=0.25, linewidth=0.7)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_xticks(np.linspace(0, 5, 6))
    ax.set_yticks(np.linspace(0, 5, 6))
    ax.tick_params(axis="both", which="both", labelsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"💾 图像保存到: {save_path}")

    if show:
        plt.show()

    return fig

def save_trajectory_plot(
    trajectory: np.ndarray,
    start_pos: List[float],
    goal_pos: List[float],
    wall_positions: List[List[float]],
    output_path: Path,
    **kwargs
):
    visualize_trajectory(
        trajectory=trajectory,
        start_pos=start_pos,
        goal_pos=goal_pos,
        wall_positions=wall_positions,
        save_path=str(output_path),
        show=False,
        **kwargs
    )

def visualize_step_by_step(
    start_pos: List[float],
    goal_pos: List[float],
    wall_positions: List[List[float]],
    step_data: List[Dict],
    traj_normalizer,
    energy_centers: Optional[List[List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Step-by-Step Visualization",
):
    num_steps = len(step_data)
    if num_steps == 0:
        print("⚠️  没有步骤数据可显示")
        return

    fig, axes = plt.subplots(num_steps, 4, figsize=(20, 5 * num_steps))
    if num_steps == 1:
        axes = axes.reshape(1, -1)

    common_scale = 1.0

    for step_idx, step_info in enumerate(step_data):
        trajectory = step_info['trajectory'][0]
        v_uncond = step_info['v_uncond'][0]
        guidance_grad = step_info['guidance_grad'][0]
        individual_grads = step_info.get('individual_grads', [])
        base_guidance = step_info.get('base_guidance', None)
        t_val = step_info['t']

        use_corrected_guidance = base_guidance is not None

        if isinstance(t_val, np.ndarray):
            t_val_scalar = float(t_val.item()) if t_val.size == 1 else float(t_val[0])
        else:
            t_val_scalar = float(t_val)

        traj_pos_norm = trajectory[:, :2]
        v_uncond_pos = v_uncond[:, :2]
        guidance_grad_pos = guidance_grad[:, :2]

        base_guidance_pos = None
        if base_guidance is not None:
            if isinstance(base_guidance, np.ndarray):
                if base_guidance.ndim == 3:
                    base_guidance_pos = base_guidance[0, :, :2]
                else:
                    base_guidance_pos = base_guidance[:, :2]

        traj_pos = traj_normalizer.unnormalize(traj_pos_norm.reshape(1, -1, 2))[0]

        x1_pred = trajectory + (1.0 - t_val_scalar) * v_uncond
        traj_pos_for_guidance = traj_normalizer.unnormalize(x1_pred[:, :2].reshape(1, -1, 2))[0]

        scale_factor = (traj_normalizer.maxs - traj_normalizer.mins) / 2.0
        v_uncond_pos_unnorm = v_uncond_pos * scale_factor
        guidance_grad_pos_unnorm = guidance_grad_pos * scale_factor

        base_guidance_pos_unnorm = None
        if base_guidance_pos is not None:
            base_guidance_pos_unnorm = base_guidance_pos * scale_factor

        individual_grads_pos = []
        for grad in individual_grads:
            if grad.ndim == 3:
                grad_pos = grad[0, :, :2]
            else:
                grad_pos = grad[:, :2]
            individual_grads_pos.append(grad_pos)

        ax_traj = axes[step_idx, 0]
        ax_traj.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=2, alpha=0.8)
        ax_traj.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=30, alpha=0.8)
        ax_traj.scatter(start_pos[0], start_pos[1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax_traj.scatter(goal_pos[0], goal_pos[1], c='blue', s=100, marker='*', label='Goal', zorder=5)

        for wall in wall_positions:
            if wall[0] != 0 or wall[1] != 0:
                rect = Rectangle((wall[0] - 0.5, wall[1] - 0.5), 1.0, 1.0,
                               facecolor='gray', alpha=0.7)
                ax_traj.add_patch(rect)

        if energy_centers is not None:
            for center in energy_centers:
                ax_traj.scatter(center[0], center[1], c='orange', s=150,
                              marker='x', linewidths=2, zorder=5)

        ax_traj.set_xlim(0, 5)
        ax_traj.set_ylim(0, 5)
        ax_traj.set_title(f'Step {step_idx+1}: Trajectory (t={t_val_scalar:.3f})', fontsize=10)
        ax_traj.grid(True, alpha=0.3)
        ax_traj.legend(fontsize=8)

        ax_v = axes[step_idx, 1]
        ax_v.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
        ax_v.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

        step_size = max(1, len(traj_pos) // 8)
        for i in range(0, len(traj_pos), step_size):
            if i < len(traj_pos):
                vx, vy = v_uncond_pos_unnorm[i, 0], v_uncond_pos_unnorm[i, 1]
                v_mag = np.sqrt(vx**2 + vy**2)
                if v_mag > 0:
                    ax_v.arrow(traj_pos[i, 0], traj_pos[i, 1],
                              vx * common_scale, vy * common_scale,
                              head_width=0.05, head_length=0.05,
                              fc='blue', ec='blue', alpha=0.8)

        v_mag_max = np.sqrt((v_uncond_pos_unnorm**2).sum(axis=1)).max()
        ax_v.text(0.02, 0.98, f'Max |v|: {v_mag_max:.3f}',
                 transform=ax_v.transAxes, fontsize=8,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_v.set_xlim(0, 5)
        ax_v.set_ylim(0, 5)
        ax_v.set_title(f'Step {step_idx+1}: v_uncond', fontsize=10)
        ax_v.grid(True, alpha=0.3)
        ax_v.axis('equal')

        ax_grad = axes[step_idx, 2]
        ax_grad.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
        ax_grad.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

        if use_corrected_guidance and base_guidance_pos_unnorm is not None:

            step_size = max(1, len(traj_pos) // 8)
            for i in range(0, len(traj_pos), step_size):
                if i < len(traj_pos):
                    gx, gy = base_guidance_pos_unnorm[i, 0], base_guidance_pos_unnorm[i, 1]
                    g_mag = np.sqrt(gx**2 + gy**2)
                    if g_mag > 0:
                        ax_grad.arrow(traj_pos_for_guidance[i, 0], traj_pos_for_guidance[i, 1],
                                     gx * common_scale, gy * common_scale,
                                     head_width=0.05, head_length=0.05,
                                     fc='orange', ec='orange', alpha=0.8,
                                     label='Base Guidance' if i == 0 else "")

                        mid_x = traj_pos_for_guidance[i, 0] + gx * common_scale / 2
                        mid_y = traj_pos_for_guidance[i, 1] + gy * common_scale / 2
                        ax_grad.text(mid_x, mid_y, f'{g_mag:.2f}',
                                    fontsize=6, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='white',
                                            edgecolor='orange',
                                            alpha=0.7))

            g_mag_max = np.sqrt((base_guidance_pos_unnorm**2).sum(axis=1)).max()
            ax_grad.text(0.02, 0.98, f'Max |grad|: {g_mag_max:.3f}',
                        transform=ax_grad.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_grad.set_title(f'Step {step_idx+1}: Base Guidance (Before Method)', fontsize=10)
        else:

            step_size = max(1, len(traj_pos) // 8)
            for i in range(0, len(traj_pos), step_size):
                if i < len(traj_pos):
                    gx, gy = guidance_grad_pos_unnorm[i, 0], guidance_grad_pos_unnorm[i, 1]
                    g_mag = np.sqrt(gx**2 + gy**2)
                    if g_mag > 0:
                        ax_grad.arrow(traj_pos_for_guidance[i, 0], traj_pos_for_guidance[i, 1],
                                     gx * common_scale, gy * common_scale,
                                     head_width=0.05, head_length=0.05,
                                     fc='red', ec='red', alpha=0.8)

                        mid_x = traj_pos_for_guidance[i, 0] + gx * common_scale / 2
                        mid_y = traj_pos_for_guidance[i, 1] + gy * common_scale / 2
                        ax_grad.text(mid_x, mid_y, f'{g_mag:.2f}',
                                    fontsize=6, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='white',
                                            edgecolor='red',
                                            alpha=0.7))

            g_mag_max = np.sqrt((guidance_grad_pos_unnorm**2).sum(axis=1)).max()
            ax_grad.text(0.02, 0.98, f'Max |grad|: {g_mag_max:.3f}',
                        transform=ax_grad.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_grad.set_title(f'Step {step_idx+1}: Total Guidance Grad', fontsize=10)

        ax_grad.set_xlim(0, 5)
        ax_grad.set_ylim(0, 5)
        ax_grad.grid(True, alpha=0.3)

        ax_ind = axes[step_idx, 3]
        ax_ind.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', linewidth=1, alpha=0.5)
        ax_ind.scatter(traj_pos[:, 0], traj_pos[:, 1], c='red', s=20, alpha=0.8)

        if use_corrected_guidance:

            corrected_guidance_pos = guidance_grad_pos_unnorm

            step_size = max(1, len(traj_pos) // 8)
            for i in range(0, len(traj_pos), step_size):
                if i < len(traj_pos):
                    gx, gy = corrected_guidance_pos[i, 0], corrected_guidance_pos[i, 1]
                    g_mag = np.sqrt(gx**2 + gy**2)
                    if g_mag > 0:

                        ax_ind.arrow(traj_pos_for_guidance[i, 0], traj_pos_for_guidance[i, 1],
                                    gx * common_scale, gy * common_scale,
                                    head_width=0.05, head_length=0.05,
                                    fc='purple', ec='purple', alpha=0.8,
                                    label='Corrected Guidance' if i == 0 else "")

                        mid_x = traj_pos_for_guidance[i, 0] + gx * common_scale / 2
                        mid_y = traj_pos_for_guidance[i, 1] + gy * common_scale / 2
                        ax_ind.text(mid_x, mid_y, f'{g_mag:.2f}',
                                   fontsize=6, ha='center', va='center',
                                   bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='white',
                                           edgecolor='purple',
                                           alpha=0.7))

            g_mag_max = np.sqrt((corrected_guidance_pos**2).sum(axis=1)).max()
            ax_ind.text(0.02, 0.98, f'Max |grad|: {g_mag_max:.3f}',
                       transform=ax_ind.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_ind.set_title(f'Step {step_idx+1}: Corrected Guidance (After Method)', fontsize=10)
        else:

            colors_ind = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']

            if energy_centers is not None:
                for idx, center in enumerate(energy_centers):
                    color = colors_ind[idx % len(colors_ind)]
                    ax_ind.scatter(center[0], center[1], c=color, s=200,
                                  marker='x', linewidths=3, zorder=5,
                                  label=f'Energy Center {idx+1}' if idx == 0 else "")

            guidance_arrow_scale = 1.6
            for idx, grad_pos in enumerate(individual_grads_pos):
                if energy_centers is not None and idx < len(energy_centers):
                    center = np.array(energy_centers[idx])
                else:
                    center = None

                color = colors_ind[idx % len(colors_ind)]
                step_size = max(1, len(traj_pos) // 8)
                for i in range(0, len(traj_pos), step_size):
                    if i < len(traj_pos):
                        gx, gy = grad_pos[i, 0], grad_pos[i, 1]
                        g_mag = np.sqrt(gx**2 + gy**2)
                        if g_mag > 0:

                            ax_ind.arrow(traj_pos_for_guidance[i, 0], traj_pos_for_guidance[i, 1],
                                        gx * guidance_arrow_scale, gy * guidance_arrow_scale,
                                        head_width=0.05, head_length=0.05,
                                        fc=color, ec=color, alpha=0.6,
                                        label=f'Energy {idx+1}' if i == 0 else "")

                            if center is not None:
                                point = traj_pos_for_guidance[i]
                                dir_to_center = center - point
                                dir_to_center_norm = np.linalg.norm(dir_to_center)

                                if dir_to_center_norm > 1e-6:
                                    dir_to_center_unit = dir_to_center / dir_to_center_norm
                                    guidance_dir = np.array([gx, gy])
                                    guidance_dir_unit = guidance_dir / g_mag

                                    dot_product = np.dot(guidance_dir_unit, dir_to_center_unit)

                                    if dot_product > 0:
                                        ax_ind.plot([point[0], center[0]], [point[1], center[1]],
                                                   '--', color='green', alpha=0.3, linewidth=1)
                                        label_text = f'✓ {g_mag:.2f}'
                                    else:
                                        ax_ind.plot([point[0], center[0]], [point[1], center[1]],
                                                   '--', color='red', alpha=0.3, linewidth=1)
                                        label_text = f'✗ {g_mag:.2f}'
                                else:
                                    label_text = f'{g_mag:.2f}'
                            else:
                                label_text = f'{g_mag:.2f}'

                            mid_x = traj_pos_for_guidance[i, 0] + gx * guidance_arrow_scale / 2
                            mid_y = traj_pos_for_guidance[i, 1] + gy * guidance_arrow_scale / 2
                            ax_ind.text(mid_x, mid_y, label_text,
                                       fontsize=6, ha='center', va='center',
                                       bbox=dict(boxstyle='round,pad=0.2',
                                               facecolor='white',
                                               edgecolor=color,
                                               alpha=0.7))

            ax_ind.set_title(f'Step {step_idx+1}: Individual Energy Grads', fontsize=10)

        ax_ind.set_xlim(0, 5)
        ax_ind.set_ylim(0, 5)
        ax_ind.grid(True, alpha=0.3)
        handles, labels = ax_ind.get_legend_handles_labels()
        if len(handles) > 0:
            ax_ind.legend(fontsize=6)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 步骤可视化图像保存到: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
