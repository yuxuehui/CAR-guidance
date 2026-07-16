import functools
from typing import Dict, List, Optional, Tuple
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

def get_peg_ids(env) -> list:
    peg_ids = []
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if "peg" in name.lower():
            peg_ids.append(seg_id)
    return peg_ids

def get_cube_ids(env) -> list:
    cube_ids = []

    seg_map = getattr(env.unwrapped, "segmentation_id_map", {})
    for seg_id, obj in seg_map.items():
        name = getattr(obj, "name", "")
        if "cube" in name.lower() or "target" in name.lower():
            cube_ids.append(seg_id)
    return cube_ids

def get_robot_ids(env) -> list:
    robot_ids = []
    base_env = env.unwrapped

    seg_map = getattr(base_env, "segmentation_id_map", {})
    for seg_id, obj in seg_map.items():
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
        pass

    return robot_ids

def get_box_ids(env) -> list:
    box_ids = []
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = getattr(obj, "name", "")
        if "box" in name.lower() or "hole" in name.lower():
            box_ids.append(seg_id)
    return box_ids

def voxel_grid_sample_fast(
    xyz: np.ndarray,
    rgb: np.ndarray,
    seg: Optional[np.ndarray],
    voxel_size: float
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(xyz) == 0:
        return xyz, rgb, seg

    quantized_coords = np.floor(xyz / voxel_size).astype(np.int32)

    _, unique_indices = np.unique(quantized_coords, axis=0, return_index=True)

    xyz_down = xyz[unique_indices]
    rgb_down = rgb[unique_indices]
    seg_down = seg[unique_indices] if seg is not None else None

    return xyz_down, rgb_down, seg_down

def make_pcd_from_numpy(
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if rgb.dtype == np.uint8 or np.max(rgb) > 1.1:
         rgb = rgb.astype(np.float64) / 255.0
    else:
         rgb = rgb.astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def merge_pcds_rlbench_style(
    voxel_size: float,
    n_points: int,
    merged_xyz: np.ndarray,
    merged_rgb: np.ndarray,
    merged_seg: np.ndarray = None,
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
    preserve_cube_points: bool = True,
    cube_seg_ids: list = None,
    preserve_robot_points: bool = True,
    robot_seg_ids: list = None,
) -> o3d.geometry.PointCloud:
    if len(merged_xyz) == 0:
        return _create_zeros_pcd(n_points)

    if ws_aabb is not None:
        min_bound = ws_aabb.get_min_bound()
        max_bound = ws_aabb.get_max_bound()

        mask = np.all((merged_xyz >= min_bound) & (merged_xyz <= max_bound), axis=1)

        if not np.any(mask):
            return _create_zeros_pcd(n_points)

        merged_xyz = merged_xyz[mask]
        merged_rgb = merged_rgb[mask]
        if merged_seg is not None:
            merged_seg = merged_seg[mask]

    merged_xyz, merged_rgb, merged_seg = voxel_grid_sample_fast(
        merged_xyz, merged_rgb, merged_seg, voxel_size
    )

    if len(merged_xyz) <= n_points:
        pcd = make_pcd_from_numpy(merged_xyz, merged_rgb)
        return _pad_pcd_with_zeros(pcd, n_points)

    final_pcd = o3d.geometry.PointCloud()

    use_stratified = (preserve_cube_points or preserve_robot_points) and (merged_seg is not None)

    if use_stratified:

        cube_mask = np.isin(merged_seg, cube_seg_ids) if (preserve_cube_points and cube_seg_ids) else np.zeros(len(merged_seg), dtype=bool)
        robot_mask = np.isin(merged_seg, robot_seg_ids) if (preserve_robot_points and robot_seg_ids) else np.zeros(len(merged_seg), dtype=bool)

        robot_mask = robot_mask & (~cube_mask)
        other_mask = ~(cube_mask | robot_mask)

        n_cube = np.sum(cube_mask)
        n_robot = np.sum(robot_mask)
        n_total = len(merged_xyz)

        cube_quota = 0
        robot_quota = 0

        if n_cube > 0:
            cube_ratio = min(0.25, max(0.15, n_cube / n_total))
            cube_quota = int(n_points * cube_ratio)

        if n_robot > 0:
            robot_ratio = min(0.45, max(0.30, n_robot / n_total))
            robot_quota = int(n_points * robot_ratio)

        if cube_quota + robot_quota > int(n_points * 0.65):
             scale = int(n_points * 0.65) / (cube_quota + robot_quota + 1e-6)
             cube_quota = int(cube_quota * scale)
             robot_quota = int(robot_quota * scale)

        n_other_quota = n_points - cube_quota - robot_quota

        def sample_subcloud(mask, quota, is_background=False):
            if np.sum(mask) == 0 or quota <= 0:
                return o3d.geometry.PointCloud()

            sub_xyz = merged_xyz[mask]
            sub_rgb = merged_rgb[mask]

            if is_background and len(sub_xyz) > quota * 5:

                indices = np.random.choice(len(sub_xyz), quota * 5, replace=False)
                sub_xyz = sub_xyz[indices]
                sub_rgb = sub_rgb[indices]

            sub_pcd = make_pcd_from_numpy(sub_xyz, sub_rgb)

            if len(sub_xyz) > quota:
                return sub_pcd.farthest_point_down_sample(quota)
            else:
                return sub_pcd

        pcd_cube = sample_subcloud(cube_mask, cube_quota)
        pcd_robot = sample_subcloud(robot_mask, robot_quota)

        current_sampled = len(pcd_cube.points) + len(pcd_robot.points)
        n_other_real_quota = max(0, n_points - current_sampled)

        pcd_other = sample_subcloud(other_mask, n_other_real_quota, is_background=True)

        final_pcd += pcd_cube
        final_pcd += pcd_robot
        final_pcd += pcd_other

    else:

        if len(merged_xyz) > n_points * 8:
             indices = np.random.choice(len(merged_xyz), n_points * 8, replace=False)
             merged_xyz = merged_xyz[indices]
             merged_rgb = merged_rgb[indices]

        merged_pcd = make_pcd_from_numpy(merged_xyz, merged_rgb)
        final_pcd = merged_pcd.farthest_point_down_sample(n_points)

    if len(final_pcd.points) < n_points:
        final_pcd = _pad_pcd_with_zeros(final_pcd, n_points)

    return final_pcd

def merge_pcds_peg_insert_style(
    voxel_size: float,
    n_points: int,
    merged_xyz: np.ndarray,
    merged_rgb: np.ndarray,
    merged_seg: np.ndarray = None,
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
    preserve_robot_points: bool = True,
    robot_seg_ids: list = None,
    preserve_peg_points: bool = True,
    peg_seg_ids: list = None,
    preserve_box_points: bool = True,
    box_seg_ids: list = None,
) -> o3d.geometry.PointCloud:
    if len(merged_xyz) == 0:
        return _create_zeros_pcd(n_points)

    if ws_aabb is not None:
        min_bound = ws_aabb.get_min_bound()
        max_bound = ws_aabb.get_max_bound()

        mask = np.all((merged_xyz >= min_bound) & (merged_xyz <= max_bound), axis=1)

        if not np.any(mask):
            return _create_zeros_pcd(n_points)

        merged_xyz = merged_xyz[mask]
        merged_rgb = merged_rgb[mask]
        if merged_seg is not None:
            merged_seg = merged_seg[mask]

    merged_xyz, merged_rgb, merged_seg = voxel_grid_sample_fast(
        merged_xyz, merged_rgb, merged_seg, voxel_size
    )

    if len(merged_xyz) <= n_points:
        pcd = make_pcd_from_numpy(merged_xyz, merged_rgb)
        return _pad_pcd_with_zeros(pcd, n_points)

    final_pcd = o3d.geometry.PointCloud()

    use_stratified = (preserve_robot_points or preserve_peg_points or preserve_box_points) and (merged_seg is not None)

    if use_stratified:

        robot_mask = np.isin(merged_seg, robot_seg_ids) if (preserve_robot_points and robot_seg_ids) else np.zeros(len(merged_seg), dtype=bool)
        peg_mask = np.isin(merged_seg, peg_seg_ids) if (preserve_peg_points and peg_seg_ids) else np.zeros(len(merged_seg), dtype=bool)
        box_mask = np.isin(merged_seg, box_seg_ids) if (preserve_box_points and box_seg_ids) else np.zeros(len(merged_seg), dtype=bool)

        peg_mask = peg_mask & (~robot_mask)
        box_mask = box_mask & (~robot_mask) & (~peg_mask)
        other_mask = ~(robot_mask | peg_mask | box_mask)

        n_robot = np.sum(robot_mask)
        n_peg = np.sum(peg_mask)
        n_box = np.sum(box_mask)
        n_total = len(merged_xyz)

        robot_quota = 0
        peg_quota = 0
        box_quota = 0

        if n_robot > 0:
            robot_ratio = min(0.45, max(0.30, n_robot / n_total))
            robot_quota = int(n_points * robot_ratio)

        if n_peg > 0:
            peg_ratio = min(0.25, max(0.15, n_peg / n_total))
            peg_quota = int(n_points * peg_ratio)

        if n_box > 0:
            box_ratio = min(0.20, max(0.10, n_box / n_total))
            box_quota = int(n_points * box_ratio)

        total_priority_quota = robot_quota + peg_quota + box_quota
        if total_priority_quota > int(n_points * 0.75):
            scale = int(n_points * 0.75) / (total_priority_quota + 1e-6)
            robot_quota = int(robot_quota * scale)
            peg_quota = int(peg_quota * scale)
            box_quota = int(box_quota * scale)

        n_other_quota = n_points - robot_quota - peg_quota - box_quota

        def sample_subcloud(mask, quota, is_background=False):
            if np.sum(mask) == 0 or quota <= 0:
                return o3d.geometry.PointCloud()

            sub_xyz = merged_xyz[mask]
            sub_rgb = merged_rgb[mask]

            if is_background and len(sub_xyz) > quota * 5:
                indices = np.random.choice(len(sub_xyz), quota * 5, replace=False)
                sub_xyz = sub_xyz[indices]
                sub_rgb = sub_rgb[indices]

            sub_pcd = make_pcd_from_numpy(sub_xyz, sub_rgb)

            if len(sub_xyz) > quota:
                return sub_pcd.farthest_point_down_sample(quota)
            else:
                return sub_pcd

        pcd_robot = sample_subcloud(robot_mask, robot_quota)
        pcd_peg = sample_subcloud(peg_mask, peg_quota)
        pcd_box = sample_subcloud(box_mask, box_quota)

        current_sampled = len(pcd_robot.points) + len(pcd_peg.points) + len(pcd_box.points)
        n_other_real_quota = max(0, n_points - current_sampled)

        pcd_other = sample_subcloud(other_mask, n_other_real_quota, is_background=True)

        final_pcd += pcd_robot
        final_pcd += pcd_peg
        final_pcd += pcd_box
        final_pcd += pcd_other

    else:

        if len(merged_xyz) > n_points * 8:
             indices = np.random.choice(len(merged_xyz), n_points * 8, replace=False)
             merged_xyz = merged_xyz[indices]
             merged_rgb = merged_rgb[indices]

        merged_pcd = make_pcd_from_numpy(merged_xyz, merged_rgb)
        final_pcd = merged_pcd.farthest_point_down_sample(n_points)

    if len(final_pcd.points) < n_points:
        final_pcd = _pad_pcd_with_zeros(final_pcd, n_points)

    return final_pcd

def _create_zeros_pcd(n_points):
    zeros = np.zeros((n_points, 3), dtype=np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(zeros))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((n_points, 3), dtype=np.float64))
    return pcd

def _pad_pcd_with_zeros(pcd, target_n):
    curr_n = len(pcd.points)
    if curr_n < target_n:
        zeros_pcd = _create_zeros_pcd(target_n - curr_n)
        pcd += zeros_pcd
    return pcd

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

        pass

    sensor_data = obs.get("sensor_data", {})
    camera_params = obs.get("sensor_param", {})

    if len(sensor_data) == 0:
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    all_xyz_list = []
    all_rgb_list = []
    all_seg_list = []

    device = None

    for cam_uid, images in sensor_data.items():
        if selected_cameras is not None and cam_uid not in selected_cameras:
            continue
        if cam_uid not in camera_params:
            continue

        pos_src = None
        seg_src = None
        rgb_src = None

        if "PositionSegmentation" in images:
            pos_seg = images["PositionSegmentation"]
            pos_src = pos_seg[..., :3]
            seg_src = pos_seg[..., 3:4]
        elif "position" in images:
            pos_src = images["position"]
            if pos_src.shape[-1] == 4:
                pos_src = pos_src[..., :3]
            seg_src = images.get("segmentation", None)

        if "Color" in images:
            rgb_src = images["Color"][..., :3]
        elif "rgb" in images:
            rgb_src = images["rgb"][..., :3]

        if pos_src is None or rgb_src is None:
            continue

        if device is None:
            device = pos_src.device

        pos_flat = pos_src.reshape(-1, 3).float() / 1000.0

        rgb_flat = rgb_src.reshape(-1, 3).float()
        if rgb_flat.max() > 1.1:
            rgb_flat = rgb_flat / 255.0

        seg_flat = None
        if seg_src is not None:
            seg_flat = seg_src.reshape(-1)

        cam2world = camera_params[cam_uid]["cam2world_gl"].to(device)
        if cam2world.ndim == 3: cam2world = cam2world[0]

        R_T = cam2world[:3, :3]
        T = cam2world[:3, 3]

        xyz_world = torch.matmul(pos_flat, R_T.T) + T

        valid_mask = (pos_flat[:, 2] != 0)

        if seg_flat is not None:

            for gid in ground_ids:
                valid_mask = valid_mask & (seg_flat != gid)

        if valid_mask.any():
            xyz_valid = xyz_world[valid_mask]
            rgb_valid = rgb_flat[valid_mask]

            all_xyz_list.append(xyz_valid)
            all_rgb_list.append(rgb_valid)

            if seg_flat is not None:
                all_seg_list.append(seg_flat[valid_mask])

    if not all_xyz_list:
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    merged_xyz_np = torch.cat(all_xyz_list, dim=0).detach().cpu().numpy()
    merged_rgb_np = torch.cat(all_rgb_list, dim=0).detach().cpu().numpy()

    merged_seg_np = None
    if all_seg_list:
        merged_seg_np = torch.cat(all_seg_list, dim=0).detach().cpu().numpy()

    final_pcd = merge_pcds_rlbench_style(
        voxel_size=voxel_size,
        n_points=n_points,
        merged_xyz=merged_xyz_np,
        merged_rgb=merged_rgb_np,
        merged_seg=merged_seg_np,
        ws_aabb=ws_aabb,
        preserve_cube_points=True,
        cube_seg_ids=cube_ids,
        preserve_robot_points=True,
        robot_seg_ids=robot_ids,
    )

    return {
        "xyz": np.asarray(final_pcd.points, dtype=np.float32),
        "rgb": np.asarray(final_pcd.colors, dtype=np.float32)
    }

def get_pointcloud_from_multi_cameras_peg_insert(
    obs: dict,
    ground_ids: list,
    voxel_size: float = 0.003,
    n_points: int = 6144,
    ws_aabb: o3d.geometry.AxisAlignedBoundingBox = None,
    robot_ids: list = None,
    peg_ids: list = None,
    box_ids: list = None,
    selected_cameras: list = None,
) -> Dict[str, np.ndarray]:
    if "pointcloud" in obs:

        pass

    sensor_data = obs.get("sensor_data", {})
    camera_params = obs.get("sensor_param", {})

    if len(sensor_data) == 0:
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    all_xyz_list = []
    all_rgb_list = []
    all_seg_list = []

    device = None

    for cam_uid, images in sensor_data.items():
        if selected_cameras is not None and cam_uid not in selected_cameras:
            continue
        if cam_uid not in camera_params:
            continue

        pos_src = None
        seg_src = None
        rgb_src = None

        if "PositionSegmentation" in images:
            pos_seg = images["PositionSegmentation"]
            pos_src = pos_seg[..., :3]
            seg_src = pos_seg[..., 3:4]
        elif "position" in images:
            pos_src = images["position"]
            if pos_src.shape[-1] == 4:
                pos_src = pos_src[..., :3]
            seg_src = images.get("segmentation", None)

        if "Color" in images:
            rgb_src = images["Color"][..., :3]
        elif "rgb" in images:
            rgb_src = images["rgb"][..., :3]

        if pos_src is None or rgb_src is None:
            continue

        if device is None:
            device = pos_src.device

        pos_flat = pos_src.reshape(-1, 3).float() / 1000.0

        rgb_flat = rgb_src.reshape(-1, 3).float()
        if rgb_flat.max() > 1.1:
            rgb_flat = rgb_flat / 255.0

        seg_flat = None
        if seg_src is not None:
            seg_flat = seg_src.reshape(-1)

        cam2world = camera_params[cam_uid]["cam2world_gl"].to(device)
        if cam2world.ndim == 3: cam2world = cam2world[0]

        R_T = cam2world[:3, :3]
        T = cam2world[:3, 3]

        xyz_world = torch.matmul(pos_flat, R_T.T) + T

        valid_mask = (pos_flat[:, 2] != 0)

        if seg_flat is not None:

            for gid in ground_ids:
                valid_mask = valid_mask & (seg_flat != gid)

        if valid_mask.any():
            xyz_valid = xyz_world[valid_mask]
            rgb_valid = rgb_flat[valid_mask]

            all_xyz_list.append(xyz_valid)
            all_rgb_list.append(rgb_valid)

            if seg_flat is not None:
                all_seg_list.append(seg_flat[valid_mask])

    if not all_xyz_list:
        return {
            "xyz": np.zeros((n_points, 3), dtype=np.float32),
            "rgb": np.zeros((n_points, 3), dtype=np.float32),
        }

    merged_xyz_np = torch.cat(all_xyz_list, dim=0).detach().cpu().numpy()
    merged_rgb_np = torch.cat(all_rgb_list, dim=0).detach().cpu().numpy()

    merged_seg_np = None
    if all_seg_list:
        merged_seg_np = torch.cat(all_seg_list, dim=0).detach().cpu().numpy()

    final_pcd = merge_pcds_peg_insert_style(
        voxel_size=voxel_size,
        n_points=n_points,
        merged_xyz=merged_xyz_np,
        merged_rgb=merged_rgb_np,
        merged_seg=merged_seg_np,
        ws_aabb=ws_aabb,
        preserve_robot_points=True,
        robot_seg_ids=robot_ids,
        preserve_peg_points=True,
        peg_seg_ids=peg_ids,
        preserve_box_points=True,
        box_seg_ids=box_ids,
    )

    return {
        "xyz": np.asarray(final_pcd.points, dtype=np.float32),
        "rgb": np.asarray(final_pcd.colors, dtype=np.float32)
    }
