from tqdm import tqdm
from pfp import DATA_DIRS
from pfp.data.replay_buffer import RobotReplayBuffer

task_name = "calibration"
mode = "train"

data_path = DATA_DIRS.PFP_REAL / task_name / mode
new_data_path = DATA_DIRS.PFP_REAL / (task_name + "_new") / mode
if new_data_path.exists():
    raise FileExistsError(f"{new_data_path} already exists")

replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
new_replay_buffer = RobotReplayBuffer.create_from_path(new_data_path, mode="w")

for episode_idx in tqdm(range(replay_buffer.n_episodes)):
    episode = replay_buffer.get_episode(episode_idx)
    episode_len = len(episode["robot_state"])
    data_list = []
    for step_idx in range(episode_len):
        data_dict = {}
        for key in episode.keys():
            if key.startswith("rgb"):
                data_dict[key] = episode[key][step_idx][::4, ::4, :]
            else:
                data_dict[key] = episode[key][step_idx]
        data_list.append(data_dict)
    new_replay_buffer.add_episode_from_list(data_list)
    print("debug")

print("done")
