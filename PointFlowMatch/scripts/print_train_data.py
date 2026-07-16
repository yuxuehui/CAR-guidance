import numpy as np

file_name = "/home/cfc/桌面/PointFlowMatch/data/demo_data_pcd_from_three_cameras_small/episode2/robot_states.npz"

def print_npz_file(npz_path):
    data = np.load(npz_path)
    print(f"文件: {npz_path}")
    print("包含以下字段:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    for key in data.files:
        arr = data[key]
        print(f"--- {key} ---")
        print(arr)
        print()

if __name__ == "__main__":
    print_npz_file(file_name)
