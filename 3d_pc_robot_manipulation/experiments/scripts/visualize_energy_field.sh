#!/bin/bash
# 能量场可视化脚本

# 默认参数（与 exp1_static.yaml 中的配置一致）
CENTERS="[[0.1, 0.0, 0.15], [-0.1, 0.0, 0.15]]"
SCALES="[-2.0, -2.0]"
SIGMA=0.1
MODE="3d"  # 默认3D可视化
Z_SLICE=0.15
Z_RANGE="0.05,0.25"
X_RANGE="-0.2,0.2"
Y_RANGE="-0.2,0.2"
RESOLUTION=50  # 3D使用较低分辨率

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --centers)
            CENTERS="$2"
            shift 2
            ;;
        --scales)
            SCALES="$2"
            shift 2
            ;;
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --z_slice)
            Z_SLICE="$2"
            shift 2
            ;;
        --z_range)
            Z_RANGE="$2"
            shift 2
            ;;
        --x_range)
            X_RANGE="$2"
            shift 2
            ;;
        --y_range)
            Y_RANGE="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --save)
            SAVE_FLAG="--save"
            shift
            ;;
        --sigma_values)
            SIGMA_VALUES="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 运行可视化脚本
cd "$(dirname "$0")/../.." || exit 1

python experiments/utils/visualize_energy_field.py \
    --centers "$CENTERS" \
    --scales "$SCALES" \
    --sigma "$SIGMA" \
    --mode "$MODE" \
    --z_slice "$Z_SLICE" \
    --z_range "$Z_RANGE" \
    --x_range "$X_RANGE" \
    --y_range "$Y_RANGE" \
    --resolution "$RESOLUTION" \
    ${SAVE_FLAG:+"$SAVE_FLAG"} \
    ${SIGMA_VALUES:+--sigma_values "$SIGMA_VALUES"}

