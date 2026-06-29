#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

def run_command(cmd, description):
    print(f">>> {description}")
    print(f"    命令: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=project_root)
    if result.returncode != 0:
        print(f"❌ 失败: {description}")
        return False
    print(f"✅ 完成: {description}\n")
    return True

def main():
    print_header("MPPI Baseline 实验 - 一键运行")

    print_header("步骤 1/5: 测试环境")
    if not run_command(
        "python exp_mppi/scripts/test_environment.py",
        "环境测试"
    ):
        print("⚠️  环境测试失败，但继续运行...")

    print_header("步骤 2/5: 运行实验1 - 静态能量场")
    if not run_command(
        "bash exp_mppi/scripts/run_mppi_exp1.sh",
        "实验1"
    ):
        print("❌ 实验1失败")
        return 1

    print_header("步骤 3/5: 运行实验2 - 目标吸引场干扰")
    if not run_command(
        "bash exp_mppi/scripts/run_mppi_exp2.sh",
        "实验2"
    ):
        print("❌ 实验2失败")
        return 1

    print_header("步骤 4/5: 运行实验3 - 动态障碍物")
    if not run_command(
        "bash exp_mppi/scripts/run_mppi_exp3.sh",
        "实验3"
    ):
        print("❌ 实验3失败")
        return 1

    print_header("步骤 5/5: 运行实验4 - 混合场景")
    if not run_command(
        "bash exp_mppi/scripts/run_mppi_exp4.sh",
        "实验4"
    ):
        print("❌ 实验4失败")
        return 1

    print_header("🎉 所有实验完成！")
    print("结果保存在:")
    print("  - exp_mppi/outputs/exp1_static/")
    print("  - exp_mppi/outputs/exp2_goal/")
    print("  - exp_mppi/outputs/exp3_dynamic/")
    print("  - exp_mppi/outputs/exp4_mixed/")
    print("\n查看详细结果:")
    print("  - 轨迹可视化: case_*.png")
    print("  - 评估指标: results.json")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
