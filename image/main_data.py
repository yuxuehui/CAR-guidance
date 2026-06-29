# coding=utf-8

import os
from datetime import datetime
import numpy as np
import imageio
import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

# Custom library imports
from utils import run_lib_gcar

FLAGS = flags.FLAGS

# Configuration
config_flags.DEFINE_config_file("config", 'RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py', "Rectified Flow Model configuration.", lock_config=True)

# Method selection
flags.DEFINE_string('method', 'gcar_gcovG_multiprompt', '[gcar_gcovG_multiprompt]')

# Optimization Hyperparameters
flags.DEFINE_integer("batch_size", 1, "Batch size")
flags.DEFINE_integer("index", 0, "Position of samples")

# Conflict & GCAR Parameters
flags.DEFINE_float('conflict_weight', 0.3, 'Weight for conflict score minimization')
flags.DEFINE_float('conflict_lr', 2.5, 'Learning rate for conflict methods')
flags.DEFINE_bool('use_L_best', True, 'Return controls from best metric (L_best); else return last step')

# Global Constants
ALPHA = 0.7
LR_DEFAULT = 5.0 # 2.5
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(_SCRIPT_DIR, 'demo')
_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
# All images under the demo/ directory.
IMAGE_PATHS = sorted([
    os.path.join(DEMO_DIR, f) for f in os.listdir(DEMO_DIR)
    if os.path.isfile(os.path.join(DEMO_DIR, f)) and f.lower().endswith(_IMAGE_EXTENSIONS)
])
# Pretrained Rectified Flow CelebA-HQ checkpoint. By default it is looked up at
# ``./checkpoint_10.pth`` next to this script (see README). Override with the
# env var CAR_IMAGE_CKPT if your checkpoint lives elsewhere.
MODEL_PATH = os.environ.get('CAR_IMAGE_CKPT', os.path.join(_SCRIPT_DIR, 'checkpoint_10.pth'))

def get_img(path=None):
    """Helper to load and preprocess single image."""
    img = imageio.imread(path)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('Read image from:', path, 'Range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)
    return img

def main(argv):
    # --- Prompt Definitions ---
    # Index: 0=old, 1=sad, 2=smiling, 3=angry, 4=curly hair
    text_prompts = [
        'A photo of an old face.',
        'A photo of a sad face.',
        'A photo of a smiling face.',
        'A photo of an angry face.',
        'A photo of a face with curly hair.'
    ]

    print(f"=== Starting Optimization: {FLAGS.method} ===")

    if FLAGS.method != 'gcar_gcovG_multiprompt':
        raise ValueError(f"Unknown method: {FLAGS.method}")

    # Single output folder for the whole run: gcar_{timestamp}
    run_folder = f'gcar_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # --- Execution Logic: gcar (Trained Residual Guidance) ---
    # Task list: each entry is (list of prompts, task name prefix).
    tasks = [
        # Task 1: Sad + Angry
        (
            [text_prompts[1], text_prompts[3]],  # Sad, Angry
            'sad_angry'
        ),
        # Task 2: Sad + Smile
        (
            [text_prompts[1], text_prompts[2]],  # Sad, Smile
            'sad_smile'
        ),
        # Task 3: Sad + Curly Hair
        (
            [text_prompts[1], text_prompts[4]],  # Sad, Curly
            'sad_curly'
        )
    ]

    # Outer loop: iterate over tasks (prompt combinations).
    for current_prompts, task_prefix in tasks:

        # Inner loop: iterate over all images.
        for img_path in IMAGE_PATHS:
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            full_suffix = f"{task_prefix}_{img_name}"
            output_dir = f"{run_folder}/{full_suffix}"

            print(f"\n--- Processing Image: {img_name} | Task: {task_prefix} ---")
            print(f"Prompts: {current_prompts}")
            print(f"Output Directory: {output_dir}")

            current_batch_paths = [img_path]

            print("Using GCAR (Trained Residual Guidance)")
            run_lib_gcar.gcar_edit_batch_multiprompt(
                FLAGS.config, MODEL_PATH, current_batch_paths, current_prompts, output_dir,
                method=FLAGS.method, alpha=ALPHA, lr_gcov=LR_DEFAULT, lr_res=FLAGS.conflict_lr,
                conflict_weight=FLAGS.conflict_weight,
                use_L_best=FLAGS.use_L_best
            )

    print("\nAll Tasks & Images Processed!")

if __name__ == "__main__":
    app.run(main)


# Run from inside the image/ directory, e.g.:
# CUDA_VISIBLE_DEVICES=0 nohup python -u ./main_data.py > run.log 2>&1 &