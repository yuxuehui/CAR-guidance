#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    print("=" * 70)
    print("测试导入...")
    print("=" * 70)

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False

    try:
        import yaml
        print(f"✅ PyYAML")
    except ImportError as e:
        print(f"❌ PyYAML导入失败: {e}")
        return False

    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False

    return True

def test_project_structure():
    print("\n" + "=" * 70)
    print("测试项目结构...")
    print("=" * 70)

    required_dirs = [
        'exp_mppi/core',
        'exp_mppi/experiments',
        'exp_mppi/configs',
        'exp_mppi/scripts',
        'exp_mppi/utils',
    ]

    required_files = [
        'exp_mppi/core/mppi_flow_controller.py',
        'exp_mppi/core/energy_cost.py',
        'exp_mppi/experiments/exp1_mppi_static.py',
        'exp_mppi/experiments/exp2_mppi_goal.py',
        'exp_mppi/experiments/exp3_mppi_dynamic.py',
        'exp_mppi/experiments/exp4_mppi_mixed.py',
        'exp_mppi/configs/mppi_exp1_static.yaml',
        'exp_mppi/configs/mppi_exp2_goal.yaml',
        'exp_mppi/configs/mppi_exp3_dynamic.yaml',
        'exp_mppi/configs/mppi_exp4_mixed.yaml',
        'exp_mppi/scripts/run_mppi_exp1.py',
        'exp_mppi/scripts/run_mppi_exp2.py',
        'exp_mppi/scripts/run_mppi_exp3.py',
        'exp_mppi/scripts/run_mppi_exp4.py',
        'exp_mppi/README.md',
    ]

    all_ok = True

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} 不存在")
            all_ok = False

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_ok = False

    return all_ok

def test_experiments_module():
    print("\n" + "=" * 70)
    print("测试experiments模块...")
    print("=" * 70)

    try:
        from experiments.core.evaluator import Evaluator
        print("✅ experiments.core.evaluator.Evaluator")
    except ImportError as e:
        print(f"❌ 导入Evaluator失败: {e}")
        return False

    try:
        from experiments.utils.inference import FlowModelInference
        print("✅ experiments.utils.inference.FlowModelInference")
    except ImportError as e:
        print(f"❌ 导入FlowModelInference失败: {e}")
        return False

    try:
        from experiments.experiments.exp1_static import select_energy_centers_from_base_traj
        print("✅ experiments.experiments.exp1_static.select_energy_centers_from_base_traj")
    except ImportError as e:
        print(f"❌ 导入select_energy_centers_from_base_traj失败: {e}")
        return False

    return True

def test_mppi_module():
    print("\n" + "=" * 70)
    print("测试MPPI模块...")
    print("=" * 70)

    try:
        from exp_mppi.core.mppi_flow_controller import MPPIFlowController
        print("✅ exp_mppi.core.mppi_flow_controller.MPPIFlowController")
    except ImportError as e:
        print(f"❌ 导入MPPIFlowController失败: {e}")
        return False

    try:
        from exp_mppi.core.energy_cost import EnergyCostFunction
        print("✅ exp_mppi.core.energy_cost.EnergyCostFunction")
    except ImportError as e:
        print(f"❌ 导入EnergyCostFunction失败: {e}")
        return False

    try:
        from exp_mppi.experiments.exp1_mppi_static import Exp1MPPIStatic
        print("✅ exp_mppi.experiments.exp1_mppi_static.Exp1MPPIStatic")
    except ImportError as e:
        print(f"❌ 导入Exp1MPPIStatic失败: {e}")
        return False

    try:
        from exp_mppi.experiments.exp2_mppi_goal import Exp2MPPIGoal
        print("✅ exp_mppi.experiments.exp2_mppi_goal.Exp2MPPIGoal")
    except ImportError as e:
        print(f"❌ 导入Exp2MPPIGoal失败: {e}")
        return False

    try:
        from exp_mppi.experiments.exp3_mppi_dynamic import Exp3MPPIDynamic
        print("✅ exp_mppi.experiments.exp3_mppi_dynamic.Exp3MPPIDynamic")
    except ImportError as e:
        print(f"❌ 导入Exp3MPPIDynamic失败: {e}")
        return False

    try:
        from exp_mppi.experiments.exp4_mppi_mixed import Exp4MPPIMixed
        print("✅ exp_mppi.experiments.exp4_mppi_mixed.Exp4MPPIMixed")
    except ImportError as e:
        print(f"❌ 导入Exp4MPPIMixed失败: {e}")
        return False

    return True

def test_configs():
    print("\n" + "=" * 70)
    print("测试配置文件...")
    print("=" * 70)

    import yaml

    config_files = [
        'exp_mppi/configs/mppi_exp1_static.yaml',
        'exp_mppi/configs/mppi_exp2_goal.yaml',
        'exp_mppi/configs/mppi_exp3_dynamic.yaml',
        'exp_mppi/configs/mppi_exp4_mixed.yaml',
    ]

    all_ok = True

    for config_file in config_files:
        full_path = project_root / config_file
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'mppi' not in config:
                print(f"❌ {config_file}: 缺少'mppi'字段")
                all_ok = False
            elif 'horizon' not in config['mppi']:
                print(f"❌ {config_file}: 缺少'mppi.horizon'字段")
                all_ok = False
            else:
                print(f"✅ {config_file}")
        except Exception as e:
            print(f"❌ {config_file}: {e}")
            all_ok = False

    return all_ok

def main():
    print("\n" + "=" * 70)
    print("MPPI实验环境测试")
    print("=" * 70 + "\n")

    results = []

    results.append(("导入测试", test_imports()))

    results.append(("项目结构", test_project_structure()))

    results.append(("experiments模块", test_experiments_module()))

    results.append(("MPPI模块", test_mppi_module()))

    results.append(("配置文件", test_configs()))

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 所有测试通过！可以开始运行实验。")
        print("\n运行命令：")
        print("  bash exp_mppi/scripts/run_all_mppi.sh")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
