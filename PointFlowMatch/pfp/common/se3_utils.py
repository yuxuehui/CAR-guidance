import torch
import numpy as np
from spatialmath.base import r2q
from spatialmath.base.transforms3d import isrot

try:
    from pytorch3d.ops import corresponding_points_alignment
except ImportError:
    print("pytorch3d not installed")
from pfp import DEVICE

def transform_th(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    new_points = points @ transform[..., :3, :3].mT + transform[..., :3, 3]
    return new_points

def vec_projection_np(v: np.ndarray, e: np.ndarray) -> np.ndarray:
    proj = np.sum(v * e, axis=-1, keepdims=True) * e
    return proj

def vec_projection_th(v: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    proj = torch.sum(v * e, dim=-1, keepdim=True) * e
    return proj

def grahm_schmidt_np(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    u1 = v1
    e1 = u1 / np.linalg.norm(u1, axis=-1, keepdims=True)
    u2 = v2 - vec_projection_np(v2, e1)
    e2 = u2 / np.linalg.norm(u2, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2, axis=-1)
    rot_matrix = np.concatenate([e1[..., None], e2[..., None], e3[..., None]], axis=-1)
    return rot_matrix

def grahm_schmidt_th(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    u1 = v1
    e1 = u1 / torch.norm(u1, dim=-1, keepdim=True)
    u2 = v2 - vec_projection_th(v2, e1)
    e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)
    rot_matrix = torch.cat(
        [e1.unsqueeze(dim=-1), e2.unsqueeze(dim=-1), e3.unsqueeze(dim=-1)], dim=-1
    )
    return rot_matrix

def pfp_to_pose_np(robot_states: np.ndarray) -> np.ndarray:
    T = robot_states.shape[0]
    poses = np.eye(4)[np.newaxis, ...]
    poses = np.tile(poses, (T, 1, 1))
    poses[:, :3, 3] = robot_states[:, :3]
    poses[:, :3, :3] = grahm_schmidt_np(robot_states[:, 3:6], robot_states[:, 6:9])
    return poses

def pfp_to_pose_th(robot_states: torch.Tensor) -> torch.Tensor:
    B = robot_states.shape[0]
    T = robot_states.shape[1]
    poses = (
        torch.eye(4, device=robot_states.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, T, 4, 4)
        .contiguous()
    )
    poses[..., :3, 3] = robot_states[..., :3]
    poses[..., :3, :3] = grahm_schmidt_th(robot_states[..., 3:6], robot_states[..., 6:9])
    gripper = robot_states[..., -1:]
    return poses, gripper

def rot6d_to_quat_np(rot6d: np.ndarray, order: str = "xyzs") -> np.ndarray:
    rot = grahm_schmidt_np(rot6d[:3], rot6d[3:])
    quat = r2q(rot, order=order)
    return quat

def rot6d_to_rot_np(rot6d: np.ndarray) -> np.ndarray:
    rot = grahm_schmidt_np(rot6d[:3], rot6d[3:])
    return rot

def check_valid_rot(rot: np.ndarray) -> bool:
    valid = isrot(rot, check=True, tol=1e10)
    return valid

def get_canonical_5p_th() -> torch.Tensor:
    gripper_width = 0.08
    left_y = 0.5 * gripper_width
    right_y = -0.5 * gripper_width
    mid_z = -0.041
    top_z = -0.1034
    a = [0, 0, top_z]
    b = [0, left_y, mid_z]
    c = [0, right_y, mid_z]
    d = [0, left_y, 0]
    e = [0, right_y, 0]
    pose_5p = torch.tensor([a, b, c, d, e])
    return pose_5p

def pfp_to_state5p_th(robot_states: torch.Tensor) -> torch.Tensor:
    device = robot_states.device
    poses, gripper = pfp_to_pose_th(robot_states)
    canonical_5p = get_canonical_5p_th().to(device)
    canonical_5p_homog = torch.cat([canonical_5p, torch.ones(5, 1, device=device)], dim=-1)
    poses_5p_homog = (poses @ canonical_5p_homog.mT).mT
    poses_5p = poses_5p_homog[..., :3].contiguous().flatten(start_dim=-2)
    state5p = torch.cat([poses_5p, gripper], dim=-1)
    return state5p

def state5p_to_pfp_th(state5p: torch.Tensor) -> torch.Tensor:
    device = state5p.device
    leading_dims = state5p.shape[0:2]

    state5p = state5p.reshape(-1, *state5p.shape[2:])
    poses_5p, gripper = state5p[..., :-1], state5p[..., -1:]
    poses_5p = poses_5p.reshape(-1, 5, 3)
    canonical_5p = get_canonical_5p_th().expand(poses_5p.shape[0], 5, 3).to(device)
    with torch.cuda.amp.autocast(enabled=False):
        result = corresponding_points_alignment(canonical_5p, poses_5p)
    rotations = result.R.mT
    translations = result.T
    pfp_state = torch.cat([translations, rotations[..., 0], rotations[..., 1], gripper], dim=-1)

    pfp_state = pfp_state.reshape(*leading_dims, -1)
    return pfp_state

def init_random_traj_th(B: int, T: int, noise_scale: float) -> torch.Tensor:

    random_xyz = torch.randn((B, 1, 3), device=DEVICE) * noise_scale
    direction = torch.randn((B, 1, 3), device=DEVICE)
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)
    t = torch.linspace(0, 1, T, device=DEVICE).unsqueeze(0).unsqueeze(-1)
    random_xyz = random_xyz + t * direction

    random_r1 = torch.randn((B, 1, 3), device=DEVICE)
    random_r1 = random_r1 / torch.norm(random_r1, dim=-1, keepdim=True)
    random_r2 = torch.randn((B, 1, 3), device=DEVICE)
    random_r2 = random_r2 - vec_projection_th(random_r2, random_r1)
    random_r2 = random_r2 / torch.norm(random_r2, dim=-1, keepdim=True)
    random_r6d = torch.cat([random_r1, random_r2], dim=-1)
    random_r6d = random_r6d.expand(B, T, 6)

    gripper = torch.ones((B, T, 1), device=DEVICE)

    random_traj = torch.cat([random_xyz, random_r6d, gripper], dim=-1)
    return random_traj
