# coding=utf-8
"""
Evaluation script: score generated results by type / prompt combination / image id.

Usage examples:
  # Evaluate gcar (trained) results, automatically over all sad_angry images.
  python utils/run_eval.py --dir examples/gcar_20260624_020209  --prefix sad_angry --type gcar

  # Evaluate single-prompt results (singleprompt1 / singleprompt2).
  python utils/run_eval.py --dir examples/gcar_20260624_020209 --prefix sad_angry --type gcov-G_multi

  # Evaluate linear-sum (multiprompt) results for specific image ids only.
  python run_eval.py --dir <dir> --prefix sad_smile --type gcov-G_multi --ids 000442 celeba

Directory layout convention:  {dir}/{prefix}_{id}/{id}{suffix}.jpg
  --type gcar          -> suffix _gcar
  --type gcov-G_multi   -> suffix _gcov-G_multiprompt
  --type gcov-G_single -> suffix _gcov-G_singleprompt1 / _gcov-G_singleprompt2 ...
                          (each image is evaluated against its own single prompt)
"""
import os
import sys
import argparse

import torch
import imageio
import numpy as np
import lpips

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.dirname(CURRENT_DIR)
if IMAGE_DIR not in sys.path:
    sys.path.insert(0, IMAGE_DIR)

from eval_metrics import (
    _compute_clipiqa_score,
    _compute_dsd_score,
    _compute_blip_itm_scores,
    _compute_vqa_scores,
)
from gcar_utils import clip_semantic_loss

# ===========================================================================
# Global configuration
# ===========================================================================
PROMPT_MAP = {
    "angry": "A photo of a angry face.",
    "smile": "A photo of a smiling face.",
    "sad":   "A photo of a sad face.",
    "curly": "A photo of a curly hair face.",
}

# Each eval type -> filename suffix + whether to score per single prompt.
EVAL_TYPES = {
    "gcar":          {"suffix": "_gcar",                "single": False},
    "gcov-G_multi":   {"suffix": "_gcov-G_multiprompt",  "single": False},
    "gcov-G_single": {"suffix": "_gcov-G_singleprompt", "single": True},
}

DEFAULT_ORIG_DIR = os.path.join(IMAGE_DIR, "demo")
IMAGE_SIZE = 256
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def build_prompts_from_prefix(prefix):
    """prefix='sad_angry' -> [sad_prompt, angry_prompt] (same order as generation)."""
    prompts = []
    for p in prefix.split("_"):
        if p not in PROMPT_MAP:
            raise ValueError(f"Unknown prompt keyword: '{p}' (prefix={prefix})")
        prompts.append(PROMPT_MAP[p])
    return prompts


def find_image(dir_path, stem):
    """Find an image named `stem` (without extension) in dir_path; return its full path or None."""
    for ext in VALID_EXTS:
        p = os.path.join(dir_path, stem + ext)
        if os.path.exists(p):
            return p
    return None


def discover_ids(orig_dir):
    """Use the image filenames in the original (demo) dir as the canonical id list.

    The demo ids are unique and match the image set used at generation time; if
    an id is missing from the generation dir, the main loop skips it.
    """
    if not os.path.isdir(orig_dir):
        return []
    ids = []
    for f in os.listdir(orig_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in VALID_EXTS:
            ids.append(stem)
    return sorted(ids)


def load_and_preprocess_image(image_path, device, size=256):
    img = imageio.imread(image_path)
    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    img_tensor = torch.tensor(img).float().to(device)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=size)
    img_tensor = img_tensor.clamp(0.0, 1.0)
    return img_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-metric evaluation of generated results")
    parser.add_argument("--dir", required=True,
                        help="results root dir, e.g. .../examples/gcar_20260624_013503")
    parser.add_argument("--prefix", required=True,
                        help="prompt combination, e.g. sad_angry / sad_smile / sad_curly")
    parser.add_argument("--type", required=True, choices=list(EVAL_TYPES.keys()),
                        help="eval target: gcar / gcov-G_single / gcov-G_multi")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="specific image ids (one or more); if omitted, evaluate all images in the dir")
    parser.add_argument("--orig_dir", default=DEFAULT_ORIG_DIR,
                        help="original image dir (reference for LPIPS / DSD / CLIP)")
    return parser.parse_args()


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = build_prompts_from_prefix(args.prefix)
    num_prompts = len(prompts)
    type_info = EVAL_TYPES[args.type]

    # Determine which image ids to evaluate.
    id_list = args.ids if args.ids else discover_ids(args.orig_dir)
    if not id_list:
        print(f"[Error] No images under original dir {args.orig_dir}; cannot determine ids. "
              f"Check --orig_dir or pass --ids explicitly.")
        return

    print(f"Initializing metrics on {device}...")
    lpips_f = lpips.LPIPS(net='alex').to(device)
    identity_scaler = lambda x: x

    # Stats: clip/blip/vqa bucketed per prompt; lpips/clipiqa/dsd are global.
    global_stats = {
        "clip_loss": [[] for _ in range(num_prompts)],
        "lpips": [],
        "clipiqa": [],
        "dsd": [],
        "blip_itm": [[] for _ in range(num_prompts)],
        "vqa": [[] for _ in range(num_prompts)],
    }

    print("=" * 60)
    print(f"Type   : {args.type}   (suffix='{type_info['suffix']}')")
    print(f"Prefix : {args.prefix}  ->  prompts={prompts}")
    print(f"IDs    : {id_list}")
    print("=" * 60)

    for img_id in id_list:
        print(f"\n>>> Processing ID: {img_id}")

        orig_img_path = os.path.join(args.orig_dir, f"{img_id}.jpg")
        gen_dir = os.path.join(args.dir, f"{args.prefix}_{img_id}")

        if not os.path.isdir(gen_dir):
            print(f"  [Warning] Generation dir not found: {gen_dir}, skipping.")
            continue

        # Load the original image (reference for LPIPS / DSD / CLIP).
        img_orig = None
        img_orig_norm = None
        if os.path.exists(orig_img_path):
            img_orig = load_and_preprocess_image(orig_img_path, device, IMAGE_SIZE)
            img_orig_norm = img_orig * 2.0 - 1.0
        else:
            print(f"  [Warning] Original image not found: {orig_img_path}; skipping DSD / LPIPS / CLIP.")

        # Pre-build a CLIP loss object for each prompt.
        clip_loss_evals = []
        if img_orig is not None:
            for prompt in prompts:
                clip_loss_evals.append(clip_semantic_loss(
                    prompt, img_orig, device, alpha=1.0, inverse_scaler=identity_scaler
                ))

        # Build the list of files to evaluate: [(file_path, [prompt indices])].
        eval_files = []
        if type_info["single"]:
            # Single prompt: singleprompt{k} is scored only against the k-th prompt.
            for k in range(1, num_prompts + 1):
                stem = f"{img_id}{type_info['suffix']}{k}"
                fp = find_image(gen_dir, stem)
                if fp:
                    eval_files.append((fp, [k - 1]))
                else:
                    print(f"  [Warning] Missing file: {stem}.* in {gen_dir}")
        else:
            # gcar / multiprompt: one image scored against all prompts.
            stem = f"{img_id}{type_info['suffix']}"
            fp = find_image(gen_dir, stem)
            if fp:
                eval_files.append((fp, list(range(num_prompts))))
            else:
                print(f"  [Warning] Missing file: {stem}.* in {gen_dir}")

        for img_path, p_indices in eval_files:
            img_name = os.path.basename(img_path)
            img_edit = load_and_preprocess_image(img_path, device, IMAGE_SIZE)
            img_edit_norm = img_edit * 2.0 - 1.0
            prompts_subset = [prompts[i] for i in p_indices]

            print(f"\n  Evaluating File: {img_name}  (prompts={[i + 1 for i in p_indices]})")

            # 1. CLIP loss (only for the prompts this file is responsible for).
            clip_losses = {}
            if img_orig is not None:
                with torch.no_grad():
                    for i in p_indices:
                        clip_losses[i] = clip_loss_evals[i].L_N(img_edit).item()

            # 2. LPIPS
            lpips_score = None
            if img_orig_norm is not None:
                with torch.no_grad():
                    lpips_score = lpips_f(img_edit_norm, img_orig_norm).item()

            # 3. Other metrics.
            clipiqa_score = _compute_clipiqa_score(img_edit, device)
            dsd_score = _compute_dsd_score(img_orig, img_edit, device) if img_orig is not None else None
            blip = _compute_blip_itm_scores(img_edit, prompts_subset, device)
            vqa = _compute_vqa_scores(img_edit, prompts_subset, device)

            # --- Collect global (prompt-independent) metrics ---
            if lpips_score is not None:
                global_stats["lpips"].append(lpips_score)
            if clipiqa_score is not None:
                global_stats["clipiqa"].append(clipiqa_score)
            if dsd_score is not None:
                global_stats["dsd"].append(dsd_score)

            # --- Collect per-prompt metrics ---
            for j, i in enumerate(p_indices):
                if i in clip_losses and clip_losses[i] is not None:
                    global_stats["clip_loss"][i].append(clip_losses[i])
                if blip and j < len(blip) and blip[j] is not None:
                    global_stats["blip_itm"][i].append(blip[j])
                if vqa and j < len(vqa) and vqa[j] is not None:
                    global_stats["vqa"][i].append(vqa[j])

            # --- Print per-image results ---
            for j, i in enumerate(p_indices):
                if i in clip_losses:
                    print(f"    [Prompt {i+1}] CLIP loss : {clip_losses[i]:.4f}")
            print(f"    LPIPS score   : {lpips_score if lpips_score is not None else 'N/A'}")
            print(f"    CLIPIQA score : {clipiqa_score if clipiqa_score is not None else 'N/A'}")
            print(f"    DSD score     : {dsd_score if dsd_score is not None else 'N/A'}")
            for j, i in enumerate(p_indices):
                s = blip[j] if (blip and j < len(blip)) else None
                print(f"    [Prompt {i+1}] BLIP-ITM : {s:.4f}" if s is not None else f"    [Prompt {i+1}] BLIP-ITM : N/A")
            for j, i in enumerate(p_indices):
                s = vqa[j] if (vqa and j < len(vqa)) else None
                print(f"    [Prompt {i+1}] VQAScore : {s:.4f}" if s is not None else f"    [Prompt {i+1}] VQAScore : N/A")

    # ===========================================================================
    # FINAL STATISTICS
    # ===========================================================================
    print("\n" + "=" * 60)
    print("FINAL GLOBAL STATISTICS")
    print("=" * 60)

    single_metrics = {
        "lpips": "LPIPS Score (Lower is better)",
        "clipiqa": "CLIPIQA Score (Higher is better)",
        "dsd": "DSD Score (Lower is better)",
    }
    multi_prompt_metrics = {
        "clip_loss": "CLIP Loss (Lower is better)",
        "blip_itm": "BLIP-ITM Score (Higher is better)",
        "vqa": "VQAScore (Higher is better)",
    }

    for key, name in single_metrics.items():
        vals = global_stats[key]
        if vals:
            print(f"  {name:<46} : {np.mean(vals):.4f}  (N={len(vals)})")
        else:
            print(f"  {name:<46} : N/A")

    print("  " + "-" * 56)

    for key, name in multi_prompt_metrics.items():
        for i in range(num_prompts):
            vals = global_stats[key][i]
            display_name = f"{name} [Prompt {i+1}]"
            if vals:
                print(f"  {display_name:<46} : {np.mean(vals):.4f}  (N={len(vals)})")
            else:
                print(f"  {display_name:<46} : N/A")

    print("=" * 60)


if __name__ == "__main__":
    main()
