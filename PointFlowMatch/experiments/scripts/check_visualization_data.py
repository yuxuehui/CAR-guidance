#!/usr/bin/env python3

import pickle
import numpy as np
import argparse
from pathlib import Path

def check_visualization_data(data_path):
    print(f"\n{'='*60}")
    print(f"检查文件: {data_path}")
    print(f"{'='*60}\n")

    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    vis_data_list = data_dict['data']
    energy_centers = data_dict['energy_centers']
    energy_scales = data_dict['energy_scales']
    goal_pos = data_dict.get('current_goal_pos', None)

    print(f"📊 基本信息:")
    print(f"  - 时间步数量: {len(vis_data_list)}")
    print(f"  - 能量中心: {len(energy_centers)} 个")
    print(f"  - 能量 scales: {energy_scales}")
    print(f"  - 目标位置: {goal_pos}\n")

    if len(vis_data_list) == 0:
        print("❌ 没有可视化数据！")
        return

    base_vel_norms = []
    base_guid_norms = []
    learned_corr_norms = []
    total_guid_norms = []

    for data in vis_data_list:
        base_velocity = data['base_velocity'][0, :, :3]
        base_guidance = data['base_guidance'][0, :, :3]
        learned_correction = data['learned_correction'][0, :, :3]
        total_guidance = data['total_guidance'][0, :, :3]

        base_vel_norms.append(np.linalg.norm(base_velocity, axis=1).mean())
        base_guid_norms.append(np.linalg.norm(base_guidance, axis=1).mean())
        learned_corr_norms.append(np.linalg.norm(learned_correction, axis=1).mean())
        total_guid_norms.append(np.linalg.norm(total_guidance, axis=1).mean())

    base_vel_norms = np.array(base_vel_norms)
    base_guid_norms = np.array(base_guid_norms)
    learned_corr_norms = np.array(learned_corr_norms)
    total_guid_norms = np.array(total_guid_norms)

    print(f"📈 统计数据 (所有时间步):")
    print(f"\n  Base Velocity Norm:")
    print(f"    Min:  {base_vel_norms.min():.6f}")
    print(f"    Max:  {base_vel_norms.max():.6f}")
    print(f"    Mean: {base_vel_norms.mean():.6f}")

    print(f"\n  Base Guidance Norm:")
    print(f"    Min:  {base_guid_norms.min():.6f}")
    print(f"    Max:  {base_guid_norms.max():.6f}")
    print(f"    Mean: {base_guid_norms.mean():.6f}")

    print(f"\n  Learned Correction Norm:")
    print(f"    Min:  {learned_corr_norms.min():.6f}")
    print(f"    Max:  {learned_corr_norms.max():.6f}")
    print(f"    Mean: {learned_corr_norms.mean():.6f}")

    print(f"\n  Total Guidance Norm:")
    print(f"    Min:  {total_guid_norms.min():.6f}")
    print(f"    Max:  {total_guid_norms.max():.6f}")
    print(f"    Mean: {total_guid_norms.mean():.6f}")

    print(f"\n{'='*60}")
    print(f"🔍 诊断结果:")
    print(f"{'='*60}\n")

    mean_learned_corr = learned_corr_norms.mean()
    mean_base_vel = base_vel_norms.mean()
    mean_base_guid = base_guid_norms.mean()

    ratio_to_vel = mean_learned_corr / (mean_base_vel + 1e-8)
    ratio_to_guid = mean_learned_corr / (mean_base_guid + 1e-8)

    print(f"  Learned Correction / Base Velocity:  {ratio_to_vel:.4f} ({ratio_to_vel*100:.2f}%)")
    print(f"  Learned Correction / Base Guidance:  {ratio_to_guid:.4f} ({ratio_to_guid*100:.2f}%)")

    print(f"\n  评估:")

    if mean_learned_corr < 1e-5:
        print(f"  ❌ 极小 - Learned correction 几乎为 0！")
        print(f"     建议: 检查训练是否成功，goal_reward_weight 是否 > 0")
        print(f"     建议: 增大 learned_correction_scale 到 10000+")
    elif mean_learned_corr < 1e-3:
        print(f"  ⚠️  很小 - Learned correction 比 base velocity 小很多")
        print(f"     建议: 增大 learned_correction_scale (当前可能 < 1000)")
        print(f"     建议: 检查 goal_reward_weight 是否设置合理")
    elif mean_learned_corr < 0.01:
        print(f"  ⚠️  较小 - Learned correction 可能影响有限")
        print(f"     建议: 可以尝试增大 learned_correction_scale")
    elif mean_learned_corr < 0.1:
        print(f"  ✅ 合理 - Learned correction 与 base guidance 量级相当")
        print(f"     效果: 应该能看到明显的修正效果")
    else:
        print(f"  ⚠️  很大 - Learned correction 可能过强")
        print(f"     建议: 减小 learned_correction_scale 或增加训练稳定性")

    print(f"\n  可视化建议:")
    if mean_learned_corr < 1e-3:
        scale_factor = max(10, int(0.01 / (mean_learned_corr + 1e-8)))
        print(f"    黄色箭头太小，可视化脚本会自动放大 {scale_factor}x")
        print(f"    如果还是看不到，请增大 learned_correction_scale 配置")
    else:
        print(f"    黄色箭头应该可见")

    nonzero_count = np.sum(learned_corr_norms > 1e-8)
    print(f"\n  非零 correction 的时间步: {nonzero_count}/{len(vis_data_list)} ({nonzero_count/len(vis_data_list)*100:.1f}%)")

    if nonzero_count == 0:
        print(f"  ❌ 所有时间步的 learned correction 都是 0！")
        print(f"     原因: 模型未训练或训练失败")
    elif nonzero_count < len(vis_data_list) * 0.5:
        print(f"  ⚠️  只有部分时间步有 correction")
    else:
        print(f"  ✅ 大部分时间步都有 correction")

    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="检查可视化数据")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="可视化数据文件路径 (.pkl)"
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 错误: 文件不存在 {data_path}")
        return

    check_visualization_data(str(data_path))

if __name__ == "__main__":
    main()
