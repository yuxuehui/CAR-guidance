import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config
from experiments.experiments.baseline import BaselineExperiment

def main():

    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    experiment = BaselineExperiment(config)

    metrics = experiment.run()

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print("\n评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\n结果保存在: {experiment.output_dir}")

if __name__ == "__main__":
    main()
