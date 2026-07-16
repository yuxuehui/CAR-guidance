#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def parse_loss_file(file_path):
    epochs = []
    losses = []
    metadata = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('# run_name:'):
            metadata['run_name'] = line.split(':', 1)[1].strip()
        elif line.startswith('# epochs:'):
            metadata['epochs'] = line.split(':', 1)[1].strip()
        elif line.startswith('epoch='):

            match = re.match(r'epoch=(\d+),\s*loss=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    return epochs, losses, metadata

def plot_loss(epochs, losses, metadata=None, output_path=None, show_plot=True):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, linewidth=1.5, alpha=0.8, color='#2E86AB')

    title = 'Training Loss'
    if metadata and 'run_name' in metadata:
        title += f" - {metadata['run_name']}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='绘制训练损失折线图')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='ckpt/maniskill_train/logs/epoch_loss.txt',
        help='输入损失日志文件路径（默认: ckpt/maniskill_train/logs/epoch_loss.txt）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（可选，默认不保存）'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='不显示图表（仅保存）'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return

    print(f"正在读取文件: {input_path}")
    epochs, losses, metadata = parse_loss_file(input_path)

    if not epochs:
        print("错误: 未能从文件中解析出任何数据")
        return

    print(f"成功解析 {len(epochs)} 个数据点")
    print(f"Epoch范围: {min(epochs)} - {max(epochs)}")
    print(f"Loss范围: {min(losses):.6f} - {max(losses):.6f}")

    filtered_epochs = []
    filtered_losses = []
    for epoch, loss in zip(epochs, losses):
        if loss <= 3.0:
            filtered_epochs.append(epoch)
            filtered_losses.append(loss)

    filtered_count = len(epochs) - len(filtered_epochs)
    if filtered_count > 0:
        print(f"已过滤 {filtered_count} 个loss > 3.0的数据点")
        print(f"剩余 {len(filtered_epochs)} 个数据点用于绘图")
        if filtered_epochs:
            print(f"过滤后Loss范围: {min(filtered_losses):.6f} - {max(filtered_losses):.6f}")

    if not filtered_epochs:
        print("错误: 过滤后没有剩余数据点")
        return

    output_path = args.output
    if output_path is None and args.no_show:

        output_path = input_path.parent / 'loss_plot.png'

    plot_loss(filtered_epochs, filtered_losses, metadata, output_path, show_plot=not args.no_show)

if __name__ == '__main__':
    main()
