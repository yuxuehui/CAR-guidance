import json
from pathlib import Path
from typing import List, Dict, Any

def load_success_trajectories(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = []
    for maze_result in data.get('results', []):
        walls = maze_result.get('walls', [])
        for case in maze_result.get('success_cases', []):

            start = [float(x) for x in case['start']]
            goal = [float(x) for x in case['goal']]
            walls_float = [[float(x) for x in w] for w in walls]

            test_cases.append({
                'start': start,
                'goal': goal,
                'walls': walls_float,
                'maze_idx': maze_result.get('maze_idx'),
            })

    return test_cases

def load_test_cases(data_path: str) -> List[Dict[str, Any]]:
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'results' in data and isinstance(data['results'], list):
        return load_success_trajectories(str(data_path))

    raise ValueError(f"不支持的数据格式: {data_path}")
