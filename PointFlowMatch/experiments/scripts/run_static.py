import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config
from experiments.experiments.exp_static import StaticGuidanceExperiment

def main():

    config_path = Path(__file__).parent.parent / "configs" / "pick_cube_static.yaml"
    config = load_config(str(config_path))

    experiment = StaticGuidanceExperiment(config)

    metrics = experiment.run()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print("\n评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\n结果保存在: {experiment.output_dir}")

    if experiment.config.get('evaluation', {}).get('save_trajectories', False):
        from experiments.utils.visualization import plot_trajectory_3d

        if len(experiment.trajectories) > 0:
            traj = experiment.trajectories[0]
            energy_centers = experiment.config.get('guidance', {}).get('energy_centers', [])
            energy_scales = experiment.config.get('guidance', {}).get('energy_scales', [])

            plot_path = experiment.output_dir / "trajectory_visualization.png"
            plot_trajectory_3d(
                traj,
                energy_centers=energy_centers,
                energy_scales=energy_scales,
                save_path=str(plot_path),
                title="Static Guidance Trajectory"
            )
            print(f"\n轨迹可视化已保存: {plot_path}")

if __name__ == "__main__":
    main()
