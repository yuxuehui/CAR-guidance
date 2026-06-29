import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Dict
from pathlib import Path

def plot_trajectory_3d(trajectory: np.ndarray,
                      energy_centers: Optional[List[List[float]]] = None,
                      energy_scales: Optional[List[float]] = None,
                      save_path: Optional[str] = None,
                      title: str = "Trajectory with Energy Field"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    pos = trajectory[:, :3]

    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='red', s=100, marker='s', label='End')

    if energy_centers is not None:
        for i, center in enumerate(energy_centers):
            scale = energy_scales[i] if energy_scales and i < len(energy_scales) else -1.0
            color = 'orange' if scale < 0 else 'purple'
            marker = 'X' if scale < 0 else 'D'
            label = f'Energy Center {i+1} ({"Repulsive" if scale < 0 else "Attractive"})'
            ax.scatter(center[0], center[1], center[2],
                      c=color, s=200, marker=marker, label=label, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_energy_field(energy_centers: List[List[float]],
                     energy_scales: List[float],
                     sigma: float,
                     x_range: tuple = (-0.5, 0.5),
                     y_range: tuple = (-0.5, 0.5),
                     z_value: float = 0.15,
                     save_path: Optional[str] = None):

    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for center, scale in zip(energy_centers, energy_scales):
        cx, cy, cz = center
        sq_dist = (X - cx)**2 + (Y - cy)**2 + (z_value - cz)**2
        energy = np.exp(-sq_dist / (sigma ** 2))
        Z += scale * energy

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(X, Y, Z, levels=20, cmap='coolwarm')
    plt.colorbar(im, ax=ax, label='Energy Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Energy Field (Z={z_value:.2f}, sigma={sigma:.3f})')
    ax.grid(True, alpha=0.3)

    for i, center in enumerate(energy_centers):
        scale = energy_scales[i] if i < len(energy_scales) else -1.0
        color = 'red' if scale < 0 else 'blue'
        ax.scatter(center[0], center[1], c=color, s=200, marker='X',
                  edgecolors='black', linewidths=2, label=f'Center {i+1}')

    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trajectories_comparison(baseline_trajs: List[np.ndarray],
                                static_trajs: List[np.ndarray],
                                energy_centers: Optional[List[List[float]]] = None,
                                save_path: Optional[str] = None):
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    for traj in baseline_trajs[:5]:
        pos = traj[:, :3]
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', alpha=0.5, linewidth=1)
    ax1.set_title('Baseline Trajectories')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)

    ax2 = fig.add_subplot(122, projection='3d')
    for traj in static_trajs[:5]:
        pos = traj[:, :3]
        ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'r-', alpha=0.5, linewidth=1)

    if energy_centers is not None:
        for center in energy_centers:
            ax2.scatter(center[0], center[1], center[2],
                       c='orange', s=200, marker='X', alpha=0.7)

    ax2.set_title('Static Guidance Trajectories')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
