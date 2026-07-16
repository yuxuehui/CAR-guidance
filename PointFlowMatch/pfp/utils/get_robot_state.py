import torch
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_rotation_6d

def get_robot_state(env) -> torch.Tensor:
    base_env = env.unwrapped

    tcp_pose = base_env.agent.tcp_pose
    ee_pos = tcp_pose.p[0]
    ee_quat = tcp_pose.q[0]

    ee_rot_matrix = quaternion_to_matrix(ee_quat)
    ee_rot_6d = matrix_to_rotation_6d(ee_rot_matrix)

    qpos = base_env.agent.robot.get_qpos()
    gripper = qpos[0, -1:]

    robot_state = torch.cat([ee_pos, ee_rot_6d, gripper], dim=-1)
    return robot_state

def get_robot_state_no_transform(env) -> torch.Tensor:
    base_env = env.unwrapped

    tcp_pose = base_env.agent.tcp_pose
    ee_pos = tcp_pose.p[0]
    ee_quat = tcp_pose.q[0]

    qpos = base_env.agent.robot.get_qpos()
    gripper = qpos[0, -1:]

    robot_state = torch.cat([ee_pos, ee_quat, gripper], dim=-1)
    return robot_state
