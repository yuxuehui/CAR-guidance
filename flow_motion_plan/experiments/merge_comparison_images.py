#!/usr/bin/env python3

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def merge_images_side_by_side(img1_path, img2_path, output_path,
                               label1="Base", label2="GCov",
                               add_labels=True, spacing=20):

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    width1, height1 = img1.size
    width2, height2 = img2.size

    max_height = max(height1, height2)
    total_width = width1 + width2 + spacing

    label_height = 40 if add_labels else 0

    merged_img = Image.new('RGB', (total_width, max_height + label_height), color='white')

    y_offset1 = (max_height - height1) // 2 + label_height
    merged_img.paste(img1, (0, y_offset1))

    y_offset2 = (max_height - height2) // 2 + label_height
    merged_img.paste(img2, (width1 + spacing, y_offset2))

    if add_labels:
        draw = ImageDraw.Draw(merged_img)
        try:

            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:

            font = ImageFont.load_default()

        text1_bbox = draw.textbbox((0, 0), label1, font=font)
        text1_width = text1_bbox[2] - text1_bbox[0]
        text1_x = (width1 - text1_width) // 2
        draw.text((text1_x, 10), label1, fill='black', font=font)

        text2_bbox = draw.textbbox((0, 0), label2, font=font)
        text2_width = text2_bbox[2] - text2_bbox[0]
        text2_x = width1 + spacing + (width2 - text2_width) // 2
        draw.text((text2_x, 10), label2, fill='black', font=font)

    merged_img.save(output_path, quality=95)

def main():

    base_dir = Path(__file__).resolve().parent / "outputs"

    dir1 = base_dir / "exp2_goal_base" / "images"
    dir2 = base_dir / "exp2_goal_gcov" / "images"
    output_dir = base_dir / "comparison_base_vs_gcov" / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("图片对比拼接脚本")
    print("=" * 60)
    print(f"目录1: {dir1.name}")
    print(f"目录2: {dir2.name}")
    print(f"输出: {output_dir}")
    print("=" * 60)

    images1 = sorted(dir1.glob("demo_*.png"))
    images2 = sorted(dir2.glob("demo_*.png"))

    print(f"\n找到 {len(images1)} 张图片（目录1）")
    print(f"找到 {len(images2)} 张图片（目录2）")

    names1 = {img.name for img in images1}
    names2 = {img.name for img in images2}
    common_names = sorted(names1 & names2)

    print(f"共同图片: {len(common_names)} 张")

    if len(common_names) == 0:
        print("\n⚠️  没有找到共同的图片名称！")
        return

    print(f"\n开始拼接图片...")

    for name in tqdm(common_names, desc="拼接进度"):
        img1_path = dir1 / name
        img2_path = dir2 / name
        output_path = output_dir / name

        try:
            merge_images_side_by_side(
                img1_path,
                img2_path,
                output_path,
                label1="Base (center_2)",
                label2="GCov (learned*200)",
                add_labels=True,
                spacing=20
            )
        except Exception as e:
            print(f"\n⚠️  处理 {name} 时出错: {e}")

    print(f"\n✅ 完成！拼接后的图片已保存到:")
    print(f"   {output_dir}")
    print(f"\n共处理 {len(common_names)} 张图片")

    readme_path = output_dir.parent / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("图片对比说明\n")
        f.write("=" * 60 + "\n\n")
        f.write("左侧: exp1_static_base_center_2_scale_1 (Base方法)\n")
        f.write("右侧: exp1_static_gcov_step20_learned*200_scale_1 (GCov方法)\n\n")
        f.write(f"共 {len(common_names)} 张对比图片\n")
        f.write(f"生成时间: {Path(__file__).stat().st_mtime}\n")

    print(f"\n说明文件已保存: {readme_path}")

if __name__ == '__main__':
    main()
