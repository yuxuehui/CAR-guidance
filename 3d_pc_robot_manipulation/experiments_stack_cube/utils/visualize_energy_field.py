import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def compute_energy_field_3d(centers, scales, sigma, x_grid, y_grid, z_grid):
    nx, ny, nz = x_grid.shape
    energy_total = np.zeros((nx, ny, nz))
    guidance_x = np.zeros((nx, ny, nz))
    guidance_y = np.zeros((nx, ny, nz))
    guidance_z = np.zeros((nx, ny, nz))

    for center, scale in zip(centers, scales):
        cx, cy, cz = center

        dx = x_grid - cx
        dy = y_grid - cy
        dz = z_grid - cz
        dist_sq = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist_sq + 1e-8)

        energy = np.exp(-dist_sq / (sigma**2 + 1e-8))

        dir_x = -dx / dist
        dir_y = -dy / dist
        dir_z = -dz / dist

        guidance_x += scale * energy * dir_x
        guidance_y += scale * energy * dir_y
        guidance_z += scale * energy * dir_z

        energy_total += energy

    guidance_mag = np.sqrt(guidance_x**2 + guidance_y**2 + guidance_z**2)

    return energy_total, guidance_x, guidance_y, guidance_z, guidance_mag

def compute_energy_field(centers, scales, sigma, x_grid, y_grid, z_slice=None):
    if z_slice is None:
        z_slice = centers[0][2] if len(centers) > 0 else 0.0

    nx, ny = x_grid.shape
    energy_total = np.zeros((nx, ny))
    guidance_x = np.zeros((nx, ny))
    guidance_y = np.zeros((nx, ny))

    for center, scale in zip(centers, scales):
        cx, cy, cz = center

        dx = x_grid - cx
        dy = y_grid - cy
        dz = z_slice - cz
        dist_sq = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist_sq + 1e-8)

        energy = np.exp(-dist_sq / (sigma**2 + 1e-8))

        dir_x = -dx / dist
        dir_y = -dy / dist

        guidance_x += scale * energy * dir_x
        guidance_y += scale * energy * dir_y

        energy_total += energy

    guidance_mag = np.sqrt(guidance_x**2 + guidance_y**2)

    return energy_total, guidance_x, guidance_y, guidance_mag

def plot_energy_field_2d(centers, scales, sigma, x_range=(-0.2, 0.2), y_range=(-0.2, 0.2),
                         z_slice=None, resolution=100, save_path=None):

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    energy, gx, gy, gmag = compute_energy_field(centers, scales, sigma, x_grid, y_grid, z_slice)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    contour = ax1.contourf(x_grid, y_grid, energy, levels=20, cmap='hot', alpha=0.8)
    ax1.contour(x_grid, y_grid, energy, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='Energy Value')

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        if abs(cz - (z_slice if z_slice is not None else cz)) < 0.01:
            color = 'red' if scale < 0 else 'blue'
            marker = 'X' if scale < 0 else 'o'
            ax1.scatter(cx, cy, c=color, marker=marker, s=200, edgecolors='white', linewidths=2,
                       label=f'Center {i+1} ({"Repulsive" if scale < 0 else "Attractive"})')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Energy Field Contour (sigma={sigma:.3f}, Z={z_slice:.3f})')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)

    skip = max(1, resolution // 20)
    ax2.quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
               gx[::skip, ::skip], gy[::skip, ::skip],
               gmag[::skip, ::skip], cmap='viridis', scale=50, alpha=0.7)

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        if abs(cz - (z_slice if z_slice is not None else cz)) < 0.01:
            color = 'red' if scale < 0 else 'blue'
            marker = 'X' if scale < 0 else 'o'
            ax2.scatter(cx, cy, c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Guidance Force Field Direction (sigma={sigma:.3f})')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    contour3 = ax3.contourf(x_grid, y_grid, gmag, levels=20, cmap='plasma', alpha=0.8)
    plt.colorbar(contour3, ax=ax3, label='Force Magnitude')

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        if abs(cz - (z_slice if z_slice is not None else cz)) < 0.01:
            color = 'red' if scale < 0 else 'blue'
            marker = 'X' if scale < 0 else 'o'
            ax3.scatter(cx, cy, c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Force Magnitude Distribution (sigma={sigma:.3f})')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    distances = np.linspace(0, 0.3, 300)

    if len(centers) > 0:
        energy_curve = np.exp(-distances**2 / (sigma**2 + 1e-8))
        ax4.plot(distances, energy_curve, 'b-', linewidth=2, label=f'Energy (sigma={sigma:.3f})')

        effective_range = distances[energy_curve > 0.1]
        if len(effective_range) > 0:
            ax4.axvspan(0, effective_range[-1], alpha=0.2, color='green',
                       label=f'Effective Range: {effective_range[-1]:.3f}m')

        hwhm_idx = np.argmin(np.abs(energy_curve - 0.5))
        hwhm = distances[hwhm_idx]
        ax4.axvline(hwhm, color='red', linestyle='--', linewidth=1.5,
                   label=f'HWHM: {hwhm:.3f}m')
        ax4.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Distance from Center (m)')
    ax4.set_ylabel('Energy Value')
    ax4.set_title('Energy vs Distance')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")
    else:
        plt.show()

def plot_energy_field_3d(centers, scales, sigma, x_range=(-0.2, 0.2), y_range=(-0.2, 0.2),
                         z_range=(0.05, 0.25), resolution=50, save_path=None):

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')

    energy, gx, gy, gz, gmag = compute_energy_field_3d(centers, scales, sigma, x_grid, y_grid, z_grid)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    z_mid = len(z) // 2
    energy_slice = energy[:, :, z_mid]
    x_slice = x_grid[:, :, z_mid]
    y_slice = y_grid[:, :, z_mid]
    z_slice_val = z_grid[:, :, z_mid]

    surf = ax1.plot_surface(x_slice, y_slice, energy_slice, cmap='hot', alpha=0.8,
                            linewidth=0, antialiased=True)

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        color = 'red' if scale < 0 else 'blue'
        marker = 'X' if scale < 0 else 'o'
        ax1.scatter([cx], [cy], [cz], c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Energy Value')
    ax1.set_title(f'Energy Field 3D (sigma={sigma:.3f})')
    plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    gmag_slice = gmag[:, :, z_mid]
    surf2 = ax2.plot_surface(x_slice, y_slice, gmag_slice, cmap='plasma', alpha=0.8,
                             linewidth=0, antialiased=True)

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        color = 'red' if scale < 0 else 'blue'
        marker = 'X' if scale < 0 else 'o'
        ax2.scatter([cx], [cy], [cz], c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Force Magnitude')
    ax2.set_title(f'Force Magnitude 3D (sigma={sigma:.3f})')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=20)

    ax3 = fig.add_subplot(1, 3, 3)
    skip = max(1, resolution // 15)
    ax3.quiver(x_slice[::skip, ::skip], y_slice[::skip, ::skip],
               gx[::skip, ::skip, z_mid], gy[::skip, ::skip, z_mid],
               gmag[::skip, ::skip, z_mid], cmap='viridis', scale=50, alpha=0.7)

    for i, (center, scale) in enumerate(zip(centers, scales)):
        cx, cy, cz = center
        if abs(cz - z[z_mid]) < (z_range[1] - z_range[0]) / resolution:
            color = 'red' if scale < 0 else 'blue'
            marker = 'X' if scale < 0 else 'o'
            ax3.scatter(cx, cy, c=color, marker=marker, s=200, edgecolors='white', linewidths=2)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Force Field at Z={z[z_mid]:.3f} (sigma={sigma:.3f})')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to: {save_path}")
    else:
        plt.show()

def plot_sigma_comparison(centers, scales, sigma_values, x_range=(-0.2, 0.2),
                         y_range=(-0.2, 0.2), z_slice=None, resolution=100, save_path=None):
    n_sigmas = len(sigma_values)
    fig, axes = plt.subplots(2, n_sigmas, figsize=(5*n_sigmas, 10))

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    x_grid, y_grid = np.meshgrid(x, y)

    for idx, sigma in enumerate(sigma_values):
        energy, gx, gy, gmag = compute_energy_field(centers, scales, sigma, x_grid, y_grid, z_slice)

        ax1 = axes[0, idx]
        contour = ax1.contourf(x_grid, y_grid, energy, levels=20, cmap='hot', alpha=0.8)
        for center, scale in zip(centers, scales):
            cx, cy, cz = center
            if abs(cz - (z_slice if z_slice is not None else cz)) < 0.01:
                color = 'red' if scale < 0 else 'blue'
                marker = 'X' if scale < 0 else 'o'
                ax1.scatter(cx, cy, c=color, marker=marker, s=150, edgecolors='white', linewidths=2)
        ax1.set_title(f'Energy Field (sigma={sigma:.3f})')
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (m)')
        if idx == 0:
            ax1.set_ylabel('Y (m)')

        ax2 = axes[1, idx]
        contour2 = ax2.contourf(x_grid, y_grid, gmag, levels=20, cmap='plasma', alpha=0.8)
        for center, scale in zip(centers, scales):
            cx, cy, cz = center
            if abs(cz - (z_slice if z_slice is not None else cz)) < 0.01:
                color = 'red' if scale < 0 else 'blue'
                marker = 'X' if scale < 0 else 'o'
                ax2.scatter(cx, cy, c=color, marker=marker, s=150, edgecolors='white', linewidths=2)
        ax2.set_title(f'Force Magnitude (sigma={sigma:.3f})')
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (m)')
        if idx == 0:
            ax2.set_ylabel('Y (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison image saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize energy field effects")
    parser.add_argument("--centers", type=str, default="[[0.1, 0.0, 0.15], [-0.1, 0.0, 0.15]]",
                       help="Energy centers list, format: [[x1,y1,z1], [x2,y2,z2], ...]")
    parser.add_argument("--scales", type=str, default="[-2.0, -2.0]",
                       help="Energy scale factors, format: [s1, s2, ...] (negative=repulsive, positive=attractive)")
    parser.add_argument("--sigma", type=float, default=0.1,
                       help="Gaussian standard deviation (controls energy field range)")
    parser.add_argument("--sigma_values", type=str, default=None,
                       help="Sigma values to compare, format: [0.05,0.1,0.15,0.2]")
    parser.add_argument("--mode", type=str, default="3d", choices=["2d", "3d", "both"],
                       help="Visualization mode: 2d (slice), 3d (full 3D), or both")
    parser.add_argument("--z_slice", type=float, default=0.15,
                       help="Z coordinate slice value (for 2D visualization)")
    parser.add_argument("--z_range", type=str, default="0.05,0.25",
                       help="Z coordinate range for 3D visualization, format: min,max")
    parser.add_argument("--x_range", type=str, default="-0.2,0.2",
                       help="X coordinate range, format: min,max")
    parser.add_argument("--y_range", type=str, default="-0.2,0.2",
                       help="Y coordinate range, format: min,max")
    parser.add_argument("--resolution", type=int, default=50,
                       help="Grid resolution (lower for 3D, e.g., 50)")
    parser.add_argument("--save_dir", type=str, default="experiments/outputs/energy_field_viz",
                       help="Save directory")
    parser.add_argument("--save", action="store_true",
                       help="Save images (default: display)")

    args = parser.parse_args()

    import ast
    centers = ast.literal_eval(args.centers)
    scales = ast.literal_eval(args.scales)
    x_range = tuple(map(float, args.x_range.split(',')))
    y_range = tuple(map(float, args.y_range.split(',')))
    z_range = tuple(map(float, args.z_range.split(',')))

    if len(centers) != len(scales):
        raise ValueError(f"Number of energy centers ({len(centers)}) != number of scales ({len(scales)})")

    print("=" * 60)
    print("Energy Field Visualization")
    print("=" * 60)
    print(f"Energy Centers: {centers}")
    print(f"Scales: {scales}")
    print(f"Sigma: {args.sigma}")
    print(f"Mode: {args.mode}")
    if args.mode in ["2d", "both"]:
        print(f"Z Slice: {args.z_slice}")
    if args.mode in ["3d", "both"]:
        print(f"Z Range: {z_range}")
    print("=" * 60)

    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    if args.mode in ["3d", "both"]:
        save_path = save_dir / f"energy_field_3d_sigma_{args.sigma:.3f}.png" if args.save else None
        plot_energy_field_3d(
            centers=centers,
            scales=scales,
            sigma=args.sigma,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            resolution=min(args.resolution, 50),
            save_path=save_path
        )

    if args.mode in ["2d", "both"]:
        save_path = save_dir / f"energy_field_2d_sigma_{args.sigma:.3f}.png" if args.save else None
        plot_energy_field_2d(
            centers=centers,
            scales=scales,
            sigma=args.sigma,
            x_range=x_range,
            y_range=y_range,
            z_slice=args.z_slice,
            resolution=args.resolution,
            save_path=save_path
        )

    if args.sigma_values and args.mode in ["2d", "both"]:
        sigma_list = ast.literal_eval(args.sigma_values)
        save_path = save_dir / "sigma_comparison.png" if args.save else None
        plot_sigma_comparison(
            centers=centers,
            scales=scales,
            sigma_values=sigma_list,
            x_range=x_range,
            y_range=y_range,
            z_slice=args.z_slice,
            resolution=args.resolution,
            save_path=save_path
        )

    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
