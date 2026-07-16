#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.config_loader import load_config

def test_config(config_path: str, task_name: str):
    print(f"\n{'='*60}")
    print(f"测试 {task_name} 配置")
    print(f"{'='*60}")

    try:
        config = load_config(Path(config_path))
        print(f"✓ 配置文件加载成功: {config_path}")
        print(f"  实验名称: {config.get('experiment_name', 'N/A')}")
        print(f"  模型路径: {config.get('model', {}).get('ckpt_path', 'N/A')}")
        print(f"  测试演示数量: {config.get('data', {}).get('num_test_episodes', 'N/A')}")

        if 'guidance' in config:
            guidance = config['guidance']
            print(f"  Guidance 类型: {guidance.get('type', 'N/A')}")
            print(f"  能量中心数量: {guidance.get('num_energy_centers', 'N/A')}")

        if 'save_actions' in config:
            print(f"  保存动作: {config.get('save_actions', False)}")
            print(f"  动作输出路径: {config.get('actions_output_path', 'N/A')}")

        if 'adaptive' in config:
            adaptive = config['adaptive']
            print(f"  自适应配置:")
            print(f"    - 第一轮后禁用 guidance: {adaptive.get('disable_guidance_after_first_round', False)}")
            print(f"    - 最大轮数: {adaptive.get('max_rounds', 'N/A')}")

        if 'visualization' in config:
            viz = config['visualization']
            print(f"  可视化配置:")
            print(f"    - 显示能量中心: {viz.get('show_energy_centers', False)}")
            print(f"    - 显示轨迹: {viz.get('show_trajectory', False)}")
            print(f"    - 显示能量场: {viz.get('show_energy_field', False)}")

        return True

    except Exception as e:
        print(f"✗ 配置文件加载失败: {config_path}")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("开始测试所有任务的配置文件")
    print("="*60)

    configs = [
        ("experiments/configs/01_task1_static_only.yaml", "Task 1"),
        ("experiments/configs/02_task2_gcov_adaptive.yaml", "Task 2"),
        ("experiments/configs/03_task3_replay_visualize.yaml", "Task 3"),
    ]

    results = []
    for config_path, task_name in configs:
        result = test_config(config_path, task_name)
        results.append((task_name, result))

    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)

    for task_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{task_name}: {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n✓ 所有配置文件测试通过！")
        return 0
    else:
        print("\n✗ 部分配置文件测试失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
