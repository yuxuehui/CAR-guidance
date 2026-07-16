import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch

def load_test_data(data_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"测试数据路径不存在: {data_path}")

    if data_path.is_dir():
        data_files = sorted(list(data_path.glob("*.npz")))
    else:
        data_files = [data_path]

    test_data = []
    n_obs_steps = config.get('n_obs_steps', 2)

    for data_file in data_files:
        try:
            data = np.load(data_file, allow_pickle=True)

            pcd = data.get('pcd', None)
            robot_state = data.get('robot_state', None)
            goal = data.get('goal', None)
            success = data.get('success', True)

            if pcd is None or robot_state is None:
                continue

            if pcd.ndim == 2:

                pcd = pcd[np.newaxis, np.newaxis, :, :]
            elif pcd.ndim == 3:
                pcd = pcd[np.newaxis, :, :, :]

            if robot_state.ndim == 1:
                robot_state = robot_state[np.newaxis, np.newaxis, :]
            elif robot_state.ndim == 2:
                robot_state = robot_state[np.newaxis, :, :]

            obs_dict = {
                'pcd': torch.from_numpy(pcd).float(),
                'robot_state': torch.from_numpy(robot_state).float(),
            }

            if goal is not None:
                goal_tensor = torch.from_numpy(goal).float()
                if goal_tensor.dim() == 1:
                    goal_tensor = goal_tensor.unsqueeze(0)

                obs_dict['goal'] = goal_tensor

            test_data.append({
                'obs': obs_dict,
                'success': bool(success) if success is not None else True
            })
        except Exception as e:
            print(f"加载数据文件 {data_file} 时出错: {e}")
            continue

    return test_data

def preprocess_observation(obs: Dict[str, np.ndarray], norm_pcd_center: List[float]) -> Dict[str, torch.Tensor]:
    pcd = torch.from_numpy(obs['pcd']).float()
    robot_state = torch.from_numpy(obs['robot_state']).float()

    if norm_pcd_center is not None:
        pcd[..., :3] -= torch.tensor(norm_pcd_center, device=pcd.device)

    if norm_pcd_center is not None:
        robot_state[..., :3] -= torch.tensor(norm_pcd_center, device=robot_state.device)
        robot_state[..., 9] -= 0.5

    processed_obs = {
        'pcd': pcd,
        'robot_state': robot_state,
    }

    if 'goal' in obs:
        processed_obs['goal'] = torch.from_numpy(obs['goal']).float()

    return processed_obs
