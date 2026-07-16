import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from collections import defaultdict

def setup_chinese_font():

    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'Noto Sans CJK SC',
        'STHeiti',
        'Arial Unicode MS',
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    font_found = None
    for font in chinese_fonts:
        if font in available_fonts:
            font_found = font
            break

    if font_found:
        plt.rcParams['font.sans-serif'] = [font_found]
        print(f"使用字体: {font_found}")
    else:

        print("警告: 未找到中文字体，中文可能无法正常显示")
        print(f"可用字体示例: {available_fonts[:5]}")

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

def load_all_episodes(data_dir: str):
    episodes = []
    episode_dirs = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("episode")])

    print(f"找到 {len(episode_dirs)} 个episode目录")

    for episode_dir in episode_dirs:
        episode_path = os.path.join(data_dir, episode_dir)
        episode_idx = int(episode_dir.replace("episode", ""))

        try:

            actions = np.load(os.path.join(episode_path, "actions.npz"))["data"]
            robot_states = np.load(os.path.join(episode_path, "robot_states.npz"))["data"]
            pointcloud = np.load(os.path.join(episode_path, "pointcloud.npz"))
            info = np.load(os.path.join(episode_path, "info.npz"))

            episodes.append({
                "idx": episode_idx,
                "actions": actions,
                "robot_states": robot_states,
                "pointcloud_xyz": pointcloud["xyz"],
                "pointcloud_rgb": pointcloud["rgb"],
                "success": info["success"].item(),
                "episode_length": info["episode_length"].item(),
                "cube_pose": info["cube_pose"],
                "goal_pose": info["goal_pose"],
            })
        except Exception as e:
            print(f"警告: 无法加载 {episode_dir}: {e}")
            continue

    return episodes

def analyze_actions(episodes):
    print("\n" + "="*70)
    print("动作分布分析")
    print("="*70)

    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)

    delta_pos = all_actions[:, :3]
    delta_rot = all_actions[:, 3:6]
    gripper = all_actions[:, 6]

    print(f"\n总动作数: {len(all_actions)}")
    print(f"动作形状: {all_actions.shape}")

    print("\n>>> 位置增量 (delta_pos) 统计:")
    print(f"  形状: {delta_pos.shape}")
    print(f"  范围: [{delta_pos.min():.4f}, {delta_pos.max():.4f}]")
    print(f"  均值: [{delta_pos.mean(axis=0)[0]:.4f}, {delta_pos.mean(axis=0)[1]:.4f}, {delta_pos.mean(axis=0)[2]:.4f}]")
    print(f"  标准差: [{delta_pos.std(axis=0)[0]:.4f}, {delta_pos.std(axis=0)[1]:.4f}, {delta_pos.std(axis=0)[2]:.4f}]")
    print(f"  是否在[-1, 1]范围内: {np.all(delta_pos >= -1) and np.all(delta_pos <= 1)}")

    print("\n>>> 旋转增量 (delta_rot) 统计:")
    print(f"  形状: {delta_rot.shape}")
    print(f"  范围: [{delta_rot.min():.6f}, {delta_rot.max():.6f}]")
    print(f"  均值: [{delta_rot.mean(axis=0)[0]:.6f}, {delta_rot.mean(axis=0)[1]:.6f}, {delta_rot.mean(axis=0)[2]:.6f}]")
    print(f"  标准差: [{delta_rot.std(axis=0)[0]:.6f}, {delta_rot.std(axis=0)[1]:.6f}, {delta_rot.std(axis=0)[2]:.6f}]")
    print(f"  是否全为0: {np.allclose(delta_rot, 0)}")

    print("\n>>> 夹爪动作统计:")
    print(f"  形状: {gripper.shape}")
    print(f"  唯一值: {np.unique(gripper)}")
    print(f"  打开(1)次数: {np.sum(gripper == 1)} ({100*np.sum(gripper == 1)/len(gripper):.1f}%)")
    print(f"  闭合(-1)次数: {np.sum(gripper == -1)} ({100*np.sum(gripper == -1)/len(gripper):.1f}%)")

    return {
        "delta_pos": delta_pos,
        "delta_rot": delta_rot,
        "gripper": gripper,
    }

def analyze_robot_states(episodes):
    print("\n" + "="*70)
    print("机器人状态分布分析")
    print("="*70)

    all_states = np.concatenate([ep["robot_states"] for ep in episodes], axis=0)

    ee_pos = all_states[:, :3]
    ee_rot_6d = all_states[:, 3:9]
    gripper_state = all_states[:, 9]

    print(f"\n总状态数: {len(all_states)}")
    print(f"状态形状: {all_states.shape}")

    print("\n>>> 末端执行器位置 (ee_pos) 统计:")
    print(f"  形状: {ee_pos.shape}")
    print(f"  X范围: [{ee_pos[:, 0].min():.4f}, {ee_pos[:, 0].max():.4f}]")
    print(f"  Y范围: [{ee_pos[:, 1].min():.4f}, {ee_pos[:, 1].max():.4f}]")
    print(f"  Z范围: [{ee_pos[:, 2].min():.4f}, {ee_pos[:, 2].max():.4f}]")
    print(f"  均值: [{ee_pos.mean(axis=0)[0]:.4f}, {ee_pos.mean(axis=0)[1]:.4f}, {ee_pos.mean(axis=0)[2]:.4f}]")
    print(f"  标准差: [{ee_pos.std(axis=0)[0]:.4f}, {ee_pos.std(axis=0)[1]:.4f}, {ee_pos.std(axis=0)[2]:.4f}]")

    print("\n>>> 末端执行器旋转 (ee_rot_6d) 统计:")
    print(f"  形状: {ee_rot_6d.shape}")
    print(f"  范围: [{ee_rot_6d.min():.4f}, {ee_rot_6d.max():.4f}]")
    print(f"  均值: {ee_rot_6d.mean(axis=0)}")
    print(f"  标准差: {ee_rot_6d.std(axis=0)}")

    print("\n>>> 夹爪状态统计:")
    print(f"  形状: {gripper_state.shape}")
    print(f"  范围: [{gripper_state.min():.4f}, {gripper_state.max():.4f}]")
    print(f"  均值: {gripper_state.mean():.4f}")
    print(f"  唯一值数量: {len(np.unique(gripper_state))}")

    return {
        "ee_pos": ee_pos,
        "ee_rot_6d": ee_rot_6d,
        "gripper_state": gripper_state,
    }

def analyze_pointclouds(episodes):
    print("\n" + "="*70)
    print("点云分布分析")
    print("="*70)

    all_pcd_xyz = np.concatenate([ep["pointcloud_xyz"] for ep in episodes], axis=0)

    num_points_per_step = [ep["pointcloud_xyz"].shape[1] for ep in episodes]
    num_points_per_step = np.array(num_points_per_step)

    print(f"\n总时间步数: {len(all_pcd_xyz)}")
    print(f"点云形状: {all_pcd_xyz.shape}")

    print("\n>>> 点数统计:")
    print(f"  每步点数: {num_points_per_step[0]} (固定)")
    print(f"  是否所有episode点数相同: {np.all(num_points_per_step == num_points_per_step[0])}")

    print("\n>>> 点云位置范围:")
    print(f"  X范围: [{all_pcd_xyz[:, :, 0].min():.4f}, {all_pcd_xyz[:, :, 0].max():.4f}]")
    print(f"  Y范围: [{all_pcd_xyz[:, :, 1].min():.4f}, {all_pcd_xyz[:, :, 1].max():.4f}]")
    print(f"  Z范围: [{all_pcd_xyz[:, :, 2].min():.4f}, {all_pcd_xyz[:, :, 2].max():.4f}]")

    print("\n>>> 数据质量检查:")
    print(f"  NaN数量: {np.isnan(all_pcd_xyz).sum()}")
    print(f"  Inf数量: {np.isinf(all_pcd_xyz).sum()}")

    return {
        "pcd_xyz": all_pcd_xyz,
        "num_points": num_points_per_step[0],
    }

def analyze_episodes(episodes):
    print("\n" + "="*70)
    print("Episode级别统计")
    print("="*70)

    episode_lengths = [ep["episode_length"] for ep in episodes]
    successes = [ep["success"] for ep in episodes]
    cube_positions = np.array([ep["cube_pose"][:3] for ep in episodes])
    goal_positions = np.array([ep["goal_pose"][:3] for ep in episodes])

    print(f"\n总episode数: {len(episodes)}")

    print("\n>>> Episode长度统计:")
    print(f"  范围: [{min(episode_lengths)}, {max(episode_lengths)}]")
    print(f"  均值: {np.mean(episode_lengths):.2f}")
    print(f"  中位数: {np.median(episode_lengths):.2f}")
    print(f"  标准差: {np.std(episode_lengths):.2f}")

    print("\n>>> 成功率统计:")
    success_count = sum(successes)
    print(f"  成功: {success_count}/{len(episodes)} ({100*success_count/len(episodes):.1f}%)")
    print(f"  失败: {len(episodes)-success_count}/{len(episodes)} ({100*(len(episodes)-success_count)/len(episodes):.1f}%)")

    print("\n>>> Cube初始位置分布:")
    print(f"  X范围: [{cube_positions[:, 0].min():.4f}, {cube_positions[:, 0].max():.4f}]")
    print(f"  Y范围: [{cube_positions[:, 1].min():.4f}, {cube_positions[:, 1].max():.4f}]")
    print(f"  Z范围: [{cube_positions[:, 2].min():.4f}, {cube_positions[:, 2].max():.4f}]")
    print(f"  均值: [{cube_positions.mean(axis=0)[0]:.4f}, {cube_positions.mean(axis=0)[1]:.4f}, {cube_positions.mean(axis=0)[2]:.4f}]")

    print("\n>>> Goal位置分布:")
    print(f"  X范围: [{goal_positions[:, 0].min():.4f}, {goal_positions[:, 0].max():.4f}]")
    print(f"  Y范围: [{goal_positions[:, 1].min():.4f}, {goal_positions[:, 1].max():.4f}]")
    print(f"  Z范围: [{goal_positions[:, 2].min():.4f}, {goal_positions[:, 2].max():.4f}]")
    print(f"  均值: [{goal_positions.mean(axis=0)[0]:.4f}, {goal_positions.mean(axis=0)[1]:.4f}, {goal_positions.mean(axis=0)[2]:.4f}]")

    distances = np.linalg.norm(cube_positions - goal_positions, axis=1)
    print("\n>>> Cube-Goal距离分布:")
    print(f"  范围: [{distances.min():.4f}, {distances.max():.4f}] 米")
    print(f"  均值: {distances.mean():.4f} 米")
    print(f"  中位数: {np.median(distances):.4f} 米")
    print(f"  标准差: {distances.std():.4f} 米")

    return {
        "episode_lengths": episode_lengths,
        "successes": successes,
        "cube_positions": cube_positions,
        "goal_positions": goal_positions,
        "distances": distances,
    }

def save_plot_with_description(fig, filename, description, save_dir: str):

    fig_new = plt.figure(figsize=(12, 10))
    gs = fig_new.add_gridspec(2, 1, height_ratios=[8, 2], hspace=0.4)

    ax_new = fig_new.add_subplot(gs[0])

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ax_new.imshow(img)
    ax_new.axis('off')

    ax_text = fig_new.add_subplot(gs[1])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, description,
                 ha='center', va='center',
                 fontsize=10, wrap=True,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 family='monospace')

    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig_new)
    print(f"  保存: {os.path.join(save_dir, filename)}")

def plot_distributions(episodes, actions_data, states_data, pcd_data, episode_data, save_dir: str):
    print("\n" + "="*70)
    print("生成分布图...")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    delta_pos = actions_data["delta_pos"]
    for i, axis_name in enumerate(["X", "Y", "Z"]):
        ax.hist(delta_pos[:, i], bins=50, alpha=0.6, label=f"{axis_name}")
    ax.set_xlabel("Position Delta Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Position Delta Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of position delta values (delta_pos) in X, Y, Z directions. " \
                  "These values represent the normalized position increments in the action space, " \
                  "typically in the range [-1, 1] which are scaled to [-0.1, 0.1] meters per step by the controller."
    save_plot_with_description(fig, "01_position_delta_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    gripper = actions_data["gripper"]
    unique, counts = np.unique(gripper, return_counts=True)
    ax.bar(unique, counts, alpha=0.6, color=['green', 'red'])
    ax.set_xlabel("Gripper Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Gripper Action Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of gripper actions. Value 1 represents open gripper, " \
                  "and -1 represents closed gripper. This shows the proportion of " \
                  "gripper opening and closing actions in the dataset."
    save_plot_with_description(fig, "02_gripper_action_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    episode_lengths = episode_data["episode_lengths"]
    ax.hist(episode_lengths, bins=30, alpha=0.6, edgecolor='black', color='skyblue')
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Frequency")
    ax.set_title("Episode Length Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of episode lengths in number of steps. " \
                  "This helps identify if episodes are too short or too long, " \
                  "and whether the data collection is consistent."
    save_plot_with_description(fig, "03_episode_length_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    distances = episode_data["distances"]
    ax.hist(distances, bins=30, alpha=0.6, edgecolor='black', color='orange')
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Cube-Goal Distance Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of distances between the initial cube position and goal position " \
                  "for each episode. Longer distances typically indicate more challenging tasks. " \
                  "This helps assess the difficulty distribution of the collected demonstrations."
    save_plot_with_description(fig, "04_cube_goal_distance_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ee_pos = states_data["ee_pos"]
    for i, axis_name in enumerate(["X", "Y", "Z"]):
        ax.hist(ee_pos[:, i], bins=50, alpha=0.6, label=f"{axis_name}")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("End-Effector Position Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of end-effector positions in X, Y, Z directions across all time steps. " \
                  "This shows the workspace coverage of the robot during demonstrations. " \
                  "A wider distribution indicates better coverage of the robot's operational space."
    save_plot_with_description(fig, "05_end_effector_position_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(ee_pos[:, 0], ee_pos[:, 1], alpha=0.1, s=1, c='blue')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("End-Effector Position (XY Projection)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    description = "2D projection of end-effector positions onto the XY plane. " \
                  "Each point represents the robot's end-effector position at a specific time step. " \
                  "This visualization helps understand the horizontal movement patterns and workspace coverage."
    save_plot_with_description(fig, "06_end_effector_xy_projection.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cube_pos = episode_data["cube_positions"]
    goal_pos = episode_data["goal_positions"]
    ax.scatter(cube_pos[:, 0], cube_pos[:, 1], alpha=0.6, label="Cube", s=20, c='red')
    ax.scatter(goal_pos[:, 0], goal_pos[:, 1], alpha=0.6, label="Goal", s=20, c='green')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Cube and Goal Position Distribution (XY Projection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    description = "Distribution of cube initial positions (red) and goal positions (green) in the XY plane. " \
                  "This shows the spatial diversity of task scenarios. A wider distribution indicates " \
                  "more diverse training scenarios, which is beneficial for generalization."
    save_plot_with_description(fig, "07_cube_goal_position_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    successes = episode_data["successes"]
    success_count = sum(successes)
    fail_count = len(successes) - success_count
    ax.pie([success_count, fail_count], labels=["Success", "Failure"],
           autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    ax.set_title("Success Rate Distribution")
    plt.tight_layout()
    description = f"Overall success rate of the collected demonstrations. " \
                  f"Success: {success_count} episodes ({100*success_count/len(successes):.1f}%), " \
                  f"Failure: {fail_count} episodes ({100*fail_count/len(successes):.1f}%). " \
                  f"A high success rate indicates good quality expert demonstrations."
    save_plot_with_description(fig, "08_success_rate_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    pcd_xyz = pcd_data["pcd_xyz"]

    sample_indices = np.random.choice(len(pcd_xyz), min(1000, len(pcd_xyz)), replace=False)
    sampled_z = pcd_xyz[sample_indices, :, 2].flatten()
    ax.hist(sampled_z, bins=50, alpha=0.6, edgecolor='black', color='purple')
    ax.set_xlabel("Z Coordinate (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Point Cloud Height Distribution (Z Coordinate)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    description = "Distribution of point cloud heights (Z coordinates). " \
                  "This helps verify that ground points have been properly filtered out. " \
                  "If ground filtering worked correctly, there should be few or no points " \
                  "at the ground level (typically Z ≈ 0 or very low values)."
    save_plot_with_description(fig, "09_pointcloud_height_distribution.png", description, save_dir)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    episode_lengths = episode_data["episode_lengths"]
    successes = episode_data["successes"]
    ax.scatter(episode_lengths, successes, alpha=0.5, s=50)
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Success (1) / Failure (0)")
    ax.set_title("Episode Length vs Success Rate")
    ax.grid(True, alpha=0.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Failure", "Success"])
    plt.tight_layout()
    description = "Relationship between episode length and success rate. " \
                  "Each point represents one episode. Points at y=1 are successful episodes, " \
                  "and y=0 are failed episodes. This helps identify if longer episodes " \
                  "are more likely to fail, which might indicate issues with the expert policy."
    save_plot_with_description(fig, "10_episode_length_vs_success.png", description, save_dir)
    plt.close(fig)

    print("\n所有分布图已生成完成！")

def main():
    parser = argparse.ArgumentParser(description="分析采集数据的分布")
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="./data/demo_data_diverse_small",
        help="数据目录 (默认: ./demo_data)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="输出目录 (默认: {data_dir}/analysis)"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="不生成分布图"
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return

    output_dir = args.output_dir or os.path.join(args.data_dir, "analysis")

    print("正在加载数据...")
    episodes = load_all_episodes(args.data_dir)

    if len(episodes) == 0:
        print("错误: 没有找到任何episode数据")
        return

    actions_data = analyze_actions(episodes)
    states_data = analyze_robot_states(episodes)
    pcd_data = analyze_pointclouds(episodes)
    episode_data = analyze_episodes(episodes)

    if not args.no_plot:
        plot_distributions(episodes, actions_data, states_data, pcd_data, episode_data, output_dir)

    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == "__main__":
    main()
