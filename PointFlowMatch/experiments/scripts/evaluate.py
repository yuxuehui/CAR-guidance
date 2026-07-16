import sys
from pathlib import Path
import json
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config
from experiments.core.evaluator import Evaluator
from experiments.utils.visualization import plot_trajectories_comparison

def load_experiment_results(experiment_dir: Path) -> dict:
    results = {}

    metrics_file = experiment_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            results['metrics'] = json.load(f)

    traj_file = experiment_dir / "trajectories.npz"
    if traj_file.exists():
        data = np.load(traj_file, allow_pickle=True)
        results['trajectories'] = data['trajectories']
        results['success_flags'] = data['success_flags']

    return results

def main():

    outputs_dir = Path("experiments/outputs")

    baseline_dir = outputs_dir / "pick_cube_baseline"
    static_dir = outputs_dir / "pick_cube_static"

    if not baseline_dir.exists():
        print(f"警告: 基线实验目录不存在: {baseline_dir}")
        print("请先运行: python experiments/scripts/run_baseline.py")
        return

    if not static_dir.exists():
        print(f"警告: 静态能量场实验目录不存在: {static_dir}")
        print("请先运行: python experiments/scripts/run_static.py")
        return

    print("加载实验结果...")
    baseline_results = load_experiment_results(baseline_dir)
    static_results = load_experiment_results(static_dir)

    print("\n" + "=" * 60)
    print("实验对比")
    print("=" * 60)

    baseline_metrics = baseline_results.get('metrics', {})
    static_metrics = static_results.get('metrics', {})

    all_metrics = {
        'Baseline': baseline_metrics,
        'Static Guidance': static_metrics
    }

    print(f"\n{'指标':<25} {'Baseline':<20} {'Static Guidance':<20}")
    print("-" * 65)

    all_keys = set(baseline_metrics.keys()) | set(static_metrics.keys())
    for key in sorted(all_keys):
        baseline_val = baseline_metrics.get(key, 0.0)
        static_val = static_metrics.get(key, 0.0)
        print(f"{key:<25} {baseline_val:<20.4f} {static_val:<20.4f}")

    evaluator = Evaluator()
    report = evaluator.generate_report(
        all_metrics,
        save_path=str(outputs_dir / "comparison_report.txt")
    )
    print("\n" + report)

    if 'trajectories' in baseline_results and 'trajectories' in static_results:
        print("\n生成轨迹对比可视化...")

        config_path = Path(__file__).parent.parent / "configs" / "pick_cube_static.yaml"
        config = load_config(str(config_path))
        energy_centers = config.get('guidance', {}).get('energy_centers', [])

        plot_path = outputs_dir / "trajectories_comparison.png"
        plot_trajectories_comparison(
            baseline_trajs=baseline_results['trajectories'][:10],
            static_trajs=static_results['trajectories'][:10],
            energy_centers=energy_centers,
            save_path=str(plot_path)
        )
        print(f"轨迹对比图已保存: {plot_path}")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
