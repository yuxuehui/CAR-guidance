import functools
from typing import Dict
import os
import torch
import numpy as np
import open3d as o3d

def get_ground_ids(env) -> list:
    ground_ids = []
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if "ground" in name.lower():
            ground_ids.append(seg_id)
    return ground_ids

def get_cube_ids(env) -> list:
    cube_ids = []
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if "cube" in name.lower() or "target" in name.lower():
            cube_ids.append(seg_id)
    return cube_ids

def get_robot_ids(env) -> list:
    robot_ids = []
    base_env = env.unwrapped

    for seg_id, obj in base_env.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if ("panda" in name.lower() or
            "link" in name.lower() or
            "arm" in name.lower() or
            "gripper" in name.lower() or
            "hand" in name.lower() or
            "finger" in name.lower() or
            "wrist" in name.lower() or
            "camera_link" in name.lower()):
            if "ground" not in name.lower() and "cube" not in name.lower() and "target" not in name.lower():
                robot_ids.append(seg_id)

    try:
        if hasattr(base_env, 'agent') and hasattr(base_env.agent, 'robot'):
            robot = base_env.agent.robot
            for link in robot.get_links():
                if hasattr(link, 'per_scene_id'):
                    link_seg_id = link.per_scene_id
                    if isinstance(link_seg_id, torch.Tensor):
                        link_seg_id = link_seg_id.item()
                    if link_seg_id not in robot_ids:
                        robot_ids.append(link_seg_id)
    except Exception as e:
        print(f"  警告: 无法从机器人link获取分割ID: {e}")

    return robot_ids

def make_pcd_from_tensor(
    xyzw: torch.Tensor,
    rgb: torch.Tensor,
) -> o3d.geometry.PointCloud:
    if xyzw.shape[1] == 4:
        xyz = xyzw[:, :3].cpu().numpy()
    else:
        xyz = xyzw.cpu().numpy()
    rgb_np = rgb.cpu().numpy()

    if rgb_np.max() > 1.0:
        rgb_np = rgb_np / 255.0

    points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    colors = o3d.utility.Vector3dVector(rgb_np.reshape(-1, 3).astype(np.float64))
    pcd = o3d.geometry.PointCloud(points)
    pcd.colors = colors
    return pcd

def farthest_point_sampling(xyz: np.ndarray, n_points: int) -> np.ndarray:
    N = len(xyz)
    if N == 0:
        return np.array([], dtype=np.int64)
    if N <= n_points:
        return np.arange(N, dtype=np.int64)

    if N > n_points * 10:
        random_indices = np.random.choice(N, n_points * 10, replace=False)
        xyz = xyz[random_indices]
        N = len(xyz)

    indices = np.zeros(n_points, dtype=np.int64)
    distances = np.ones(N) * np.inf
    farthest = np.random.randint(0, N)

    for i in range(n_points):
        indices[i] = farthest
        centroid = xyz[farthest:farthest+1]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return indices

def sync_segmentation_after_voxel_downsample(original_pcd, downsampled_pcd, original_seg):
    if original_seg is None or len(original_pcd.points) == 0:
        return None
    try:
        from scipy.spatial import cKDTree
        original_xyz = np.asarray(original_pcd.points, dtype=np.float32)
        downsampled_xyz = np.asarray(downsampled_pcd.points, dtype=np.float32)
        tree = cKDTree(original_xyz)
        _, indices = tree.query(downsampled_xyz, k=1)
        return original_seg[indices]
    except ImportError:

        original_xyz = np.asarray(original_pcd.points, dtype=np.float32)
        downsampled_xyz = np.asarray(downsampled_pcd.points, dtype=np.float32)
        indices = []
        for pt in downsampled_xyz:
            dists = np.sum((original_xyz - pt) ** 2, axis=1)
            indices.append(np.argmin(dists))
        return original_seg[np.array(indices)]

def merge_pcds_rlbench_style(
    voxel_size: float,
    n_points: int,
    pcds: list[o3d.geometry.PointCloud],
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
    preserve_cube_points: bool = True,
    cube_seg_ids: list = None,
    preserve_robot_points: bool = True,
    robot_seg_ids: list = None,
    pcd_segmentations: list = None,
) -> o3d.geometry.PointCloud:

    merged_pcd = functools.reduce(lambda a, b: a + b, pcds, o3d.geometry.PointCloud())

    merged_seg = None
    if pcd_segmentations is not None and len(pcd_segmentations) > 0:
        seg_list_valid = [seg for seg in pcd_segmentations if seg is not None]
        if len(seg_list_valid) > 0:
            merged_seg = np.concatenate(seg_list_valid, axis=0)

    if len(merged_pcd.points) == 0:
        zeros = np.zeros((n_points, 3), dtype=np.float32)
        zeros_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
        zeros_pcd.colors = o3d.utility.Vector3dVector(np.zeros((n_points, 3), dtype=np.float64))
        return zeros_pcd

    if ws_aabb is not None:
        mask = ws_aabb.get_point_indices_within_bounding_box(merged_pcd.points)
        if len(mask) > 0:
            merged_pcd = merged_pcd.select_by_index(mask)
            if merged_seg is not None:
                merged_seg = merged_seg[mask]
        else:
            zeros = np.zeros((n_points, 3), dtype=np.float32)
            zeros_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
            zeros_pcd.colors = o3d.utility.Vector3dVector(np.zeros((n_points, 3), dtype=np.float64))
            return zeros_pcd

    fine_voxel_size = voxel_size * 0.5
    downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size=fine_voxel_size)
    if merged_seg is not None and len(merged_pcd.points) > 0:
        merged_seg = sync_segmentation_after_voxel_downsample(merged_pcd, downsampled_pcd, merged_seg)

    if len(downsampled_pcd.points) > n_points * 3:
        original_pcd = downsampled_pcd
        downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size=voxel_size)
        if merged_seg is not None and len(original_pcd.points) > 0:
            merged_seg = sync_segmentation_after_voxel_downsample(original_pcd, downsampled_pcd, merged_seg)

    if len(downsampled_pcd.points) > n_points:
        xyz = np.asarray(downsampled_pcd.points, dtype=np.float32)
        rgb = np.asarray(downsampled_pcd.colors, dtype=np.float32)

        if len(downsampled_pcd.points) > n_points * 20:
            coarse_voxel_size = voxel_size * 2.0
            original_pcd = downsampled_pcd
            downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size=coarse_voxel_size)
            if merged_seg is not None and len(original_pcd.points) > 0:
                merged_seg = sync_segmentation_after_voxel_downsample(original_pcd, downsampled_pcd, merged_seg)
            xyz = np.asarray(downsampled_pcd.points, dtype=np.float32)
            rgb = np.asarray(downsampled_pcd.colors, dtype=np.float32)

        if (preserve_cube_points or preserve_robot_points) and merged_seg is not None and len(xyz) > n_points:
            cube_indices_all = np.array([], dtype=np.int64)
            robot_indices_all = np.array([], dtype=np.int64)

            if preserve_cube_points and cube_seg_ids is not None and len(cube_seg_ids) > 0:
                cube_mask = np.zeros(len(merged_seg), dtype=bool)
                for cube_id in cube_seg_ids:
                    cube_mask = cube_mask | (merged_seg == cube_id)
                cube_indices_all = np.where(cube_mask)[0]

            if preserve_robot_points and robot_seg_ids is not None and len(robot_seg_ids) > 0:
                robot_mask = np.zeros(len(merged_seg), dtype=bool)
                for robot_id in robot_seg_ids:
                    robot_mask = robot_mask | (merged_seg == robot_id)
                robot_indices_all = np.where(robot_mask)[0]

            total_important_points = len(cube_indices_all) + len(robot_indices_all)

            if total_important_points > 0:

                cube_ratio = 0.0
                robot_ratio = 0.0

                if len(cube_indices_all) > 0:
                    cube_ratio = min(0.25, max(0.15, len(cube_indices_all) / len(xyz)))
                if len(robot_indices_all) > 0:
                    robot_ratio = min(0.45, max(0.30, len(robot_indices_all) / len(xyz)))

                if cube_ratio + robot_ratio > 0.65:
                    scale = 0.65 / (cube_ratio + robot_ratio)
                    cube_ratio *= scale
                    robot_ratio *= scale

                n_cube_points = int(n_points * cube_ratio) if len(cube_indices_all) > 0 else 0
                n_robot_points = int(n_points * robot_ratio) if len(robot_indices_all) > 0 else 0
                n_other_points = n_points - n_cube_points - n_robot_points

                cube_indices = np.array([], dtype=np.int64)
                if n_cube_points > 0 and len(cube_indices_all) > 0:
                    if len(cube_indices_all) > n_cube_points:
                        cube_indices = np.random.choice(cube_indices_all, n_cube_points, replace=False)
                    else:
                        cube_indices = cube_indices_all

                robot_indices = np.array([], dtype=np.int64)
                if n_robot_points > 0 and len(robot_indices_all) > 0:
                    robot_candidates = np.setdiff1d(robot_indices_all, cube_indices)
                    if len(robot_candidates) > n_robot_points:
                        robot_xyz = xyz[robot_candidates]
                        robot_fps_indices = farthest_point_sampling(robot_xyz, n_robot_points)
                        robot_indices = robot_candidates[robot_fps_indices]
                    else:
                        robot_indices = robot_candidates

                important_indices = np.concatenate([cube_indices, robot_indices])
                other_indices = np.setdiff1d(np.arange(len(xyz)), important_indices)

                if len(other_indices) > n_other_points:
                    other_xyz = xyz[other_indices]
                    other_fps_indices = farthest_point_sampling(other_xyz, n_other_points)
                    other_selected = other_indices[other_fps_indices]
                else:
                    other_selected = other_indices

                final_indices = np.concatenate([cube_indices, robot_indices, other_selected])
                xyz_sampled = xyz[final_indices]
                rgb_sampled = rgb[final_indices]

            else:

                print(f"  警告: 未找到cube或机器臂点，使用普通FPS采样")
                fps_indices = farthest_point_sampling(xyz, n_points)
                xyz_sampled = xyz[fps_indices]
                rgb_sampled = rgb[fps_indices]
        else:

            fps_indices = farthest_point_sampling(xyz, n_points)
            xyz_sampled = xyz[fps_indices]
            rgb_sampled = rgb[fps_indices]

        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(xyz_sampled)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(rgb_sampled.astype(np.float64))

    if len(downsampled_pcd.points) < n_points:
        num_missing_points = n_points - len(downsampled_pcd.points)
        zeros = np.zeros((num_missing_points, 3), dtype=np.float32)
        zeros_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
        zeros_pcd.colors = o3d.utility.Vector3dVector(np.zeros((num_missing_points, 3), dtype=np.float64))
        downsampled_pcd += zeros_pcd

    return downsampled_pcd

def get_pointcloud_from_multi_cameras(
    obs: dict,
    ground_ids: list,
    voxel_size: float = 0.003,
    n_points: int = 4096,
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
    cube_ids: list = None,
    robot_ids: list = None,
    selected_cameras: list = None,
) -> Dict[str, np.ndarray]:

    if "pointcloud" in obs:
        from mani_skill.utils.observations.observations import sensor_data_to_pointcloud

        pass

    sensor_data = obs.get("sensor_data", {})
    camera_params = obs.get("sensor_param", {})
    pcd_list = []
    seg_list = []

    if len(sensor_data) == 0:
        print(f"  警告: sensor_data为空，无法获取点云")
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    processed_cameras = []
    for cam_uid, images in sensor_data.items():

        if selected_cameras is not None and cam_uid not in selected_cameras:
            continue

        processed_cameras.append(cam_uid)

        if cam_uid not in camera_params:

            continue

        position = None
        rgb = None
        segmentation = None

        if "PositionSegmentation" in images:
            pos_seg = images["PositionSegmentation"].clone()
            position = pos_seg[..., :3]
            segmentation = pos_seg[..., 3:4]
        elif "position" in images:
            position = images["position"].clone()
            if position.shape[-1] == 4:
                position = position[..., :3]
            segmentation = images.get("segmentation", None)

        if position is None:
            print(f"  警告: 相机 {cam_uid} 没有 position/PositionSegmentation 数据")
            continue

        if "Color" in images:
            rgb = images["Color"][..., :3].clone()
        elif "rgb" in images:
            rgb = images["rgb"][..., :3].clone()

        if rgb is None:
            print(f"  警告: 相机 {cam_uid} 没有 rgb/Color 数据")
            continue

        position_valid = (position[..., :3] != 0).any()
        if not position_valid:
            print(f"  警告: 相机 {cam_uid} 的 position 数据全为零")
            continue

        if position.dim() == 3:
            position = position.unsqueeze(0)
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if segmentation is not None and segmentation.dim() == 3:
            segmentation = segmentation.unsqueeze(0)

        position = position.float()
        position = position / 1000.0

        cam2world = camera_params[cam_uid]["cam2world_gl"].to(position.device)
        position_flat = position.reshape(position.shape[0], -1, 3)
        position_homogeneous = torch.cat([
            position_flat,
            torch.ones(position_flat.shape[0], position_flat.shape[1], 1, device=position.device)
        ], dim=-1)
        xyzw = position_homogeneous @ cam2world.transpose(1, 2)

        rgb_flat = rgb.reshape(rgb.shape[0], -1, 3)

        if segmentation is not None:
            seg_flat = segmentation.reshape(segmentation.shape[0], -1, 1).squeeze(-1)
            mask = torch.ones(seg_flat.shape[1], dtype=torch.bool, device=seg_flat.device)
            mask = mask & (position_flat[0, :, 2] != 0)
            for ground_id in ground_ids:
                mask = mask & (seg_flat[0] != ground_id)

            xyzw = xyzw[0][mask]
            rgb_reshaped = rgb_flat[0][mask]
            seg_filtered = seg_flat[0][mask].cpu().numpy()

            if len(xyzw) == 0:
                print(f"  警告: 相机 {cam_uid} 过滤后点云为空")
                continue
        else:
            mask = (position_flat[0, :, 2] != 0)
            xyzw = xyzw[0][mask]
            rgb_reshaped = rgb_flat[0][mask]
            seg_filtered = None

            if len(xyzw) == 0:
                print(f"  警告: 相机 {cam_uid} 过滤无效点后点云为空")
                continue

        pcd = make_pcd_from_tensor(xyzw, rgb_reshaped)

        pcd_list.append(pcd)
        seg_list.append(seg_filtered)

    if len(pcd_list) == 0:
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    merged_pcd = merge_pcds_rlbench_style(
        voxel_size,
        n_points,
        pcd_list,
        ws_aabb,
        preserve_cube_points=True,
        cube_seg_ids=cube_ids,
        preserve_robot_points=True,
        robot_seg_ids=robot_ids,
        pcd_segmentations=seg_list,
    )

    xyz = np.asarray(merged_pcd.points, dtype=np.float32)
    rgb = np.asarray(merged_pcd.colors, dtype=np.float32)

    return {"xyz": xyz, "rgb": rgb}
