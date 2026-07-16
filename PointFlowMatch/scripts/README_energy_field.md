# 能量场引导功能使用说明

## 概述

参考 `flow_motion_plan` 中实验一的能量场实现，在 `PointFlowMatch` 的训练好的 pick-cube 模型上添加了能量场引导功能。

## 实现内容

1. **能量场模块** (`pfp/policy/energy_guide.py`)
   - `EnergyFunction`: 定义能量函数（使用线性衰减）
   - `EnergyGuideVectorField`: 能量引导向量场（用于计算梯度）

2. **模型修改** (`pfp/policy/fm_policy_maniskill.py`)
   - 在 `FMPolicy.infer_y` 方法中添加了能量场引导支持
   - 支持在推理时传入能量场参数

3. **测试脚本** (`scripts/test_pick_cube_with_energy.py`)
   - 在 pick-cube 模型上测试能量场引导

## 使用方法

### 基本用法

```bash
# 使用默认参数测试（无能量场）
python scripts/test_pick_cube_with_energy.py --seed 1

# 添加一个排斥能量场（中心在 [0.1, 0.1, 0.1]）
python scripts/test_pick_cube_with_energy.py \
    --seed 1 \
    --energy_centers 0.1 0.1 0.1 \
    --energy_scales -1.0 \
    --energy_radius 0.3

# 添加两个排斥能量场
python scripts/test_pick_cube_with_energy.py \
    --seed 1 \
    --energy_centers 0.1 0.1 0.1 0.2 0.2 0.2 \
    --energy_scales -1.0 -1.0 \
    --energy_radius 0.3
```

### 参数说明

- `--ckpt_name`: checkpoint目录名称（默认: `maniskill_train_pcd_from_three_cameras_more_gripper`）
- `--ckpt_episode`: checkpoint的episode标识（默认: `ep1500-ba160500`）
- `--seed`: 环境seed（默认: 1）
- `--max_steps`: 最大步数（默认: 300）
- `--energy_centers`: 能量场中心坐标列表，格式: `x1 y1 z1 x2 y2 z2 ...`（每个能量场3个坐标）
- `--energy_scales`: 能量场缩放系数列表（负数表示排斥，正数表示吸引，默认: `[-1.0]`）
- `--energy_radius`: 能量场作用半径（默认: 0.3）
- `--save_video`: 是否保存视频（默认: True）
- `--video_output_dir`: 视频保存目录（默认: `outputs/videos/test_energy`）

### 能量场参数说明

- **能量中心** (`energy_centers`): 能量场中心在3D空间中的位置（x, y, z）
  - 注意：坐标应该在归一化空间中（相对于 `norm_pcd_center`）
  - 在测试脚本中，会自动进行归一化处理

- **能量缩放系数** (`energy_scales`): 控制能量场的强度
  - 负数（如 -1.0）: 排斥能量场，轨迹会远离能量中心
  - 正数（如 1.0）: 吸引能量场，轨迹会靠近能量中心

- **能量作用半径** (`energy_radius`): 能量场的作用范围
  - 使用线性衰减：距离中心越近，作用越强
  - 超出半径范围的点不受能量场影响

## 实现细节

### 能量场计算

能量场使用线性衰减模式：

```
weight = clamp((radius - dist) / radius, min=0.0)
grad = scale * weight * direction_to_center
```

其中：
- `dist`: 点到能量中心的距离
- `radius`: 能量场作用半径
- `scale`: 能量场缩放系数
- `direction_to_center`: 从点指向中心的单位向量

### 集成方式

在 `FMPolicy.infer_y` 方法中：

1. 计算无条件速度场 `v_uncond`
2. 预测去噪后的轨迹位置 `x_1_pred = z + (1 - t) * v_uncond`
3. 计算能量梯度（只对位置部分，前3维）
4. 将能量梯度添加到速度场：`v_guided = v_uncond + energy_grad`

## 参考

- `flow_motion_plan/experiments/exp1_batch_from_demos_g_cov_a_gm_online.py`: 实验一的能量场实现
- `flow_motion_plan/experiments/energy_guide.py`: 能量场模块实现



