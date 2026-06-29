#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.utils.inference import FlowModelInference
from experiments.utils.visualization import visualize_trajectory

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def point_in_box(point, center, box_size=1.0, margin=0.0) -> bool:
    px, py = point
    cx, cy = center
    half = box_size / 2.0 + margin
    return (abs(px - cx) <= half) and (abs(py - cy) <= half)

def trajectory_collides(traj: np.ndarray,
                        walls: np.ndarray,
                        box_size: float = 1.0,
                        margin: float = 0.0) -> bool:
    valid_walls = [w for w in walls if not (abs(w[0]) < 1e-6 and abs(w[1]) < 1e-6)]
    if len(valid_walls) == 0:
        return False

    for p in traj:
        for w in valid_walls:
            if point_in_box(p, w, box_size=box_size, margin=margin):
                return True
    return False

def reached_goal(traj: np.ndarray,
                 goal: np.ndarray,
                 tol: float = 0.3) -> bool:
    last = traj[-1]
    dist = np.linalg.norm(last - goal)
    return dist <= tol

def get_region_id(x: float, y: float, maze_size=(5.0, 5.0)) -> int:
    w, h = maze_size
    col = int(x / (w / 3))
    row = int(y / (h / 3))
    col = min(max(col, 0), 2)
    row = min(max(row, 0), 2)
    return row * 3 + col

def get_region_bounds(region_id: int, maze_size=(5.0, 5.0)) -> tuple:
    w, h = maze_size
    row = region_id // 3
    col = region_id % 3

    x_min = col * (w / 3)
    x_max = (col + 1) * (w / 3) if col < 2 else w
    y_min = row * (h / 3)
    y_max = (row + 1) * (h / 3) if row < 2 else h

    return (x_min, x_max, y_min, y_max)

def is_corner_region(region_id: int) -> bool:
    return region_id in [0, 2, 6, 8]

def get_allowed_goal_regions(start_region_id: int) -> list:
    all_regions = list(range(9))
    allowed = [r for r in all_regions if r != start_region_id]

    if is_corner_region(start_region_id):
        corner_regions = [0, 2, 6, 8]
        allowed = [r for r in allowed if r not in corner_regions]

    return allowed

def sample_point_in_region(region_id, walls, maze_size=(5.0, 5.0), margin=0.1, max_tries=20):
    x_min, x_max, y_min, y_max = get_region_bounds(region_id, maze_size)
    x_min = max(x_min + margin, 0.0)
    x_max = min(x_max - margin, maze_size[0])
    y_min = max(y_min + margin, 0.0)
    y_max = min(y_max - margin, maze_size[1])

    if x_max <= x_min or y_max <= y_min:
        return None

    def is_valid_point(p):
        for w_center in walls:
            if abs(w_center[0]) < 1e-6 and abs(w_center[1]) < 1e-6:
                continue
            if point_in_box(p, w_center, box_size=1.0, margin=0.1):
                return False
        return True

    for _ in range(max_tries):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        point = np.array([x, y], dtype=np.float32)
        if is_valid_point(point):
            return point

    return None

def sample_start_goal(walls: np.ndarray,
                      maze_size=(5.0, 5.0),
                      min_dist: float = 1.0,
                      max_tries: int = 1000) -> tuple:
    w = float(maze_size[0])
    h = float(maze_size[1])
    effective_min_dist = max(min_dist, 1.5)
    max_dist = 3.5

    for attempt in range(max_tries):

        start_region_candidates = list(range(9))
        random.shuffle(start_region_candidates)

        start = None
        start_region_id = None

        for region_id in start_region_candidates:
            candidate_start = sample_point_in_region(region_id, walls, maze_size, margin=0.1)
            if candidate_start is not None:
                start = candidate_start
                start_region_id = region_id
                break

        if start is None:
            continue

        allowed_goal_regions = get_allowed_goal_regions(start_region_id)
        random.shuffle(allowed_goal_regions)

        goal = None
        for goal_region_id in allowed_goal_regions:
            candidate_goal = sample_point_in_region(goal_region_id, walls, maze_size, margin=0.1)
            if candidate_goal is not None:
                dist = np.linalg.norm(start - candidate_goal)
                if effective_min_dist <= dist <= max_dist:
                    goal = candidate_goal
                    break

        if goal is not None:
            return start, goal

    raise RuntimeError("在给定约束下无法采样到合法的起点和终点")

def collect_success_trajectories(
    checkpoint: str,
    config: str,
    sample_json: str,
    num_success_per_maze: int = 5,
    max_attempts_per_maze: int = 200,
    steps: int = 20,
    dt: float = 0.05,
    goal_tol: float = 0.3,
    collision_margin: float = 0.0,
    use_deterministic_trajectory: bool = True,
    save_images: bool = True,
):

    with open(sample_json, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    sampled_mazes = sample_data.get("sampled_mazes", [])
    num_mazes = len(sampled_mazes)
    print(f"📄 从 {sample_json} 中读取到 {num_mazes} 个迷宫样本")

    inferencer = FlowModelInference(checkpoint, config)

    output_dir = Path("experiments/data/base_model_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "success_trajectories.json"

    all_results = {
        "checkpoint": checkpoint,
        "config": config,
        "sample_json": sample_json,
        "num_mazes": num_mazes,
        "num_success_per_maze": num_success_per_maze,
        "goal_tolerance": goal_tol,
        "collision_margin": collision_margin,
        "results": [],
    }

    total_success = 0
    target_total = num_mazes * num_success_per_maze
    demo_counter = 0

    for maze_entry in sampled_mazes:
        maze_idx = maze_entry.get("maze_idx")
        walls = np.array(maze_entry.get("walls", []), dtype=np.float32)
        if walls.ndim != 2 or walls.shape[1] != 2:
            print(f"⚠️ 迷宫 {maze_idx} 的 walls 形状异常: {walls.shape}，跳过")
            continue

        print(f"\n================ 评估迷宫 {maze_idx} ================")
        print(f"障碍物数量(含填充): {walls.shape[0]}")

        maze_result = {
            "maze_idx": int(maze_idx),
            "walls": walls.tolist(),
            "success_cases": [],
        }

        success_count = 0
        attempts = 0

        while success_count < num_success_per_maze and attempts < max_attempts_per_maze:
            attempts += 1

            try:
                start, goal = sample_start_goal(walls, maze_size=(5.0, 5.0))
            except RuntimeError as e:
                print(f"  ❌ 迷宫 {maze_idx} 采样起点终点失败: {e}")
                break

            if use_deterministic_trajectory:
                seed_value = hash((tuple(start), tuple(goal))) % (2**31)
                torch.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)
                np.random.seed(seed_value)

            trajectories, _ = inferencer.generate_trajectory(
                start_pos=start.tolist(),
                goal_pos=goal.tolist(),
                wall_positions=walls.tolist(),
                num_steps=steps,
                dt=dt,
                num_samples=1,
                record_steps=False,
            )
            traj = trajectories[0]

            has_collision = trajectory_collides(
                traj, walls, box_size=1.0, margin=collision_margin
            )
            reached = reached_goal(traj, goal, tol=goal_tol)

            if (not has_collision) and reached:
                success_count += 1
                total_success += 1
                print(
                    f"  ✅ 成功 {success_count}/{num_success_per_maze} "
                    f"(全局: {total_success}/{target_total}) | "
                    f"start={start}, goal={goal}"
                )

                image_path = None
                if save_images:
                    demo_counter += 1
                    image_filename = f"demo_{demo_counter:04d}.png"
                    image_path = output_dir / image_filename
                    visualize_trajectory(
                        trajectory=traj,
                        start_pos=start.tolist(),
                        goal_pos=goal.tolist(),
                        wall_positions=walls.tolist(),
                        save_path=str(image_path),
                        show=False,
                        title=f"Demo {demo_counter} - Maze {maze_idx}",
                        goal_tol=goal_tol,
                    )

                maze_result["success_cases"].append({
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "reached": True,
                    "collision_free": True,
                    "image_path": str(image_path) if image_path is not None else None,
                })
            else:
                status = []
                if has_collision:
                    status.append("碰撞")
                if not reached:
                    status.append("未到达终点")
                if attempts % 20 == 0:
                    print(
                        f"  ❌ 失败 (尝试 {attempts}/{max_attempts_per_maze}) "
                        f"[{', '.join(status)}]"
                    )

        print(
            f"迷宫 {maze_idx} 总尝试 {attempts} 次，成功 {success_count} 条"
        )
        all_results["results"].append(maze_result)

    print(
        f"\n✅ 完成收集: 共成功 {total_success}/{target_total} 条 "
        f"(期望 {target_total} 条)"
    )

    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
    print(f"💾 成功轨迹已保存到: {output_json}")

def main():
    setup_seed(42)

    parser = argparse.ArgumentParser(
        description="收集成功轨迹（无碰撞且到达终点）"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型检查点路径"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="配置文件路径（可选）"
    )
    parser.add_argument(
        "--sample-json",
        type=str,
        default="data/sample_maze.json",
        help="采样迷宫 JSON 文件路径",
    )
    parser.add_argument(
        "--num-success-per-maze",
        type=int,
        default=5,
        help="每个迷宫需要的成功轨迹数量",
    )
    parser.add_argument(
        "--max-attempts-per-maze",
        type=int,
        default=200,
        help="每个迷宫最多尝试次数",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="ODE 求解步数"
    )
    parser.add_argument(
        "--dt", type=float, default=0.05, help="时间步长"
    )
    parser.add_argument(
        "--goal-tol",
        type=float,
        default=0.1,
        help="判定到达终点的距离阈值",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.0,
        help="碰撞检测的额外安全边距",
    )
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="使用非确定性轨迹生成",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="不保存轨迹图片",
    )

    args = parser.parse_args()

    collect_success_trajectories(
        checkpoint=args.checkpoint,
        config=args.config,
        sample_json=args.sample_json,
        num_success_per_maze=args.num_success_per_maze,
        max_attempts_per_maze=args.max_attempts_per_maze,
        steps=args.steps,
        dt=args.dt,
        goal_tol=args.goal_tol,
        collision_margin=args.collision_margin,
        use_deterministic_trajectory=not args.non_deterministic,
        save_images=not args.no_images,
    )

if __name__ == "__main__":
    main()
