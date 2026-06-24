# coding=utf-8
"""
Evaluation metrics for multi-prompt image editing experiments.

Provides lazy-loaded, device-cached implementations of:
  - CLIP loss          (via clip_semantic_loss, passed externally)
  - LPIPS              (passed externally)
  - CLIPIQA            (pyiqa)
  - DINO Structure Distance (timm ViT-B/8)
  - BLIP-ITM           (Salesforce/blip-itm-base-coco)
  - VQAScore           (Salesforce/blip-vqa-base, yes/no framing)
  - FER                (deepface)
"""

import torch
import torch.nn.functional as F

try:
    from .gcar_utils import clip_semantic_loss
except ImportError:
    from gcar_utils import clip_semantic_loss

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    pyiqa = None
    PYIQA_AVAILABLE = False
    print("Warning: pyiqa not available, CLIPIQA metric will be disabled")

# ===========================================================================
# Global lazy-init state
# ===========================================================================

_CLIPIQA_METRIC = None
_CLIPIQA_DEVICE = None
_CLIPIQA_INIT_FAILED = False

_DINO_MODEL = None
_DINO_DEVICE = None
_DINO_INIT_FAILED = False

_BLIP_ITM_MODEL = None
_BLIP_ITM_PROCESSOR = None
_BLIP_ITM_DEVICE = None
_BLIP_ITM_INIT_FAILED = False

_BLIP_VQA_MODEL = None
_BLIP_VQA_PROCESSOR = None
_BLIP_VQA_DEVICE = None
_BLIP_VQA_INIT_FAILED = False


# ===========================================================================
# CLIPIQA
# ===========================================================================

def _get_clipiqa_metric(device):
    global _CLIPIQA_METRIC, _CLIPIQA_DEVICE, _CLIPIQA_INIT_FAILED
    if _CLIPIQA_INIT_FAILED or not PYIQA_AVAILABLE:
        return None
    device_str = str(torch.device(device))
    if _CLIPIQA_METRIC is not None and _CLIPIQA_DEVICE == device_str:
        return _CLIPIQA_METRIC
    try:
        print(f"Loading CLIPIQA metric on {device_str}...")
        _CLIPIQA_METRIC = pyiqa.create_metric("clipiqa", device=device_str)
        _CLIPIQA_DEVICE = device_str
        return _CLIPIQA_METRIC
    except Exception as e:
        _CLIPIQA_INIT_FAILED = True
        print(f"Warning: failed to initialize CLIPIQA metric: {e}")
        return None


def _compute_clipiqa_score(img_edit, device):
    metric = _get_clipiqa_metric(device)
    if metric is None:
        return None
    try:
        with torch.no_grad():
            score = metric(img_edit.to(torch.device(device)))
        if isinstance(score, torch.Tensor):
            return score.mean().item()
        return float(score)
    except Exception as e:
        print(f"Warning: failed to compute CLIPIQA score: {e}")
        return None


# ===========================================================================
# DINO Structure Distance (DSD)
# ===========================================================================

def _get_dino_model(device):
    """
    Lazy-load DINO ViT-B/8 via timm (avoids torch.hub sys.path conflict with
    the project's local `utils/` package that shadows DINO's utils.py).
    Model: vit_base_patch8_224.dino  (pretrained DINO weights from timm hub).
    """
    global _DINO_MODEL, _DINO_DEVICE, _DINO_INIT_FAILED
    if _DINO_INIT_FAILED:
        return None
    device_str = str(torch.device(device))
    if _DINO_MODEL is not None and _DINO_DEVICE == device_str:
        return _DINO_MODEL
    try:
        import timm
        print(f"Loading DINO (vit_base_patch8_224.dino via timm) on {device_str}...")
        model = timm.create_model('vit_base_patch8_224.dino', pretrained=True)
        model.eval()
        model.to(torch.device(device_str))
        _DINO_MODEL = model
        _DINO_DEVICE = device_str
        return _DINO_MODEL
    except Exception as e:
        _DINO_INIT_FAILED = True
        print(f"Warning: failed to initialize DINO model: {e}")
        return None


def _preprocess_for_dino(img_tensor, device):
    """
    Resize a [B, C, H, W] tensor in [0, 1] to 224×224 and apply
    DINO's ImageNet normalisation.  Returns a tensor on `device`.
    (timm's vit_base_patch8_224 expects 224×224 input.)
    """
    img = torch.nn.functional.interpolate(
        img_tensor.to(device), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img - mean) / std


def _dino_patch_tokens(model, x):
    """
    Extract L2-normalised patch tokens from a timm ViT model.
    timm's forward_features returns [B, N+1, dim]; index 0 is the CLS token.
    """
    features = model.forward_features(x)   # [B, N+1, dim]
    patch_tokens = features[:, 1:, :]      # drop CLS token
    return torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)


def _compute_dsd_score(img_orig, img_edit, device):
    """
    DINO Structure Distance (DSD): MSE between self-similarity matrices of
    DINO patch tokens.  Lower = more structural similarity to the original.
    """
    model = _get_dino_model(device)
    if model is None:
        return None
    try:
        with torch.no_grad():
            t_orig = _preprocess_for_dino(img_orig, device)
            t_edit = _preprocess_for_dino(img_edit, device)
            feat_orig = _dino_patch_tokens(model, t_orig)
            feat_edit = _dino_patch_tokens(model, t_edit)
            ssm_orig = torch.bmm(feat_orig, feat_orig.transpose(1, 2))
            ssm_edit = torch.bmm(feat_edit, feat_edit.transpose(1, 2))
            dsd = torch.nn.functional.mse_loss(ssm_orig, ssm_edit)
        return dsd.item()
    except Exception as e:
        print(f"Warning: failed to compute DSD score: {e}")
        return None


# ===========================================================================
# BLIP-ITM / VQAScore / FER helpers
# ===========================================================================

def _tensor_to_pil(img_tensor):
    """Convert a [1, C, H, W] or [C, H, W] tensor in [0,1] to a PIL Image."""
    from torchvision.transforms.functional import to_pil_image
    t = img_tensor.squeeze(0).clamp(0, 1).cpu()
    return to_pil_image(t)


# ─── BLIP-ITM ────────────────────────────────────────────────────────────────

def _get_blip_itm_model(device):
    """
    Lazy-load BLIP ITM (Salesforce/blip-itm-base-coco).
    Outputs a 2-class score; softmax[:,1] = P(image matches text).
    """
    global _BLIP_ITM_MODEL, _BLIP_ITM_PROCESSOR, _BLIP_ITM_DEVICE, _BLIP_ITM_INIT_FAILED
    if _BLIP_ITM_INIT_FAILED:
        return None, None
    device_str = str(torch.device(device))
    if _BLIP_ITM_MODEL is not None and _BLIP_ITM_DEVICE == device_str:
        return _BLIP_ITM_MODEL, _BLIP_ITM_PROCESSOR
    try:
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        print(f"Loading BLIP-ITM (blip-itm-base-coco) on {device_str}...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        model.eval()
        model.to(torch.device(device_str))
        _BLIP_ITM_MODEL = model
        _BLIP_ITM_PROCESSOR = processor
        _BLIP_ITM_DEVICE = device_str
        return model, processor
    except Exception as e:
        _BLIP_ITM_INIT_FAILED = True
        print(f"Warning: failed to initialize BLIP-ITM: {e}")
        return None, None


def _compute_blip_itm_scores(img_tensor, prompts, device):
    """
    BLIP Image-Text Matching score for each prompt.
    Returns list of P(match) ∈ [0, 1] per prompt; None entries on failure.
    Higher = better text-image alignment.
    """
    model, processor = _get_blip_itm_model(device)
    if model is None:
        return [None] * len(prompts)
    try:
        pil_img = _tensor_to_pil(img_tensor)
        scores = []
        dev = torch.device(device)
        with torch.no_grad():
            for prompt in prompts:
                inputs = processor(images=pil_img, text=prompt,
                                   return_tensors="pt").to(dev)
                outputs = model(**inputs, use_itm_head=True)
                p_match = torch.nn.functional.softmax(
                    outputs.itm_score, dim=1)[0][1].item()
                scores.append(p_match)
        return scores
    except Exception as e:
        print(f"Warning: failed to compute BLIP-ITM scores: {e}")
        return [None] * len(prompts)


# ─── VQAScore (BLIP-VQA yes/no) ──────────────────────────────────────────────

def _get_blip_vqa_model(device):
    """
    Lazy-load BLIP VQA (Salesforce/blip-vqa-base).
    Used for VQAScore: P(yes) for prompt-specific yes/no questions.
    """
    global _BLIP_VQA_MODEL, _BLIP_VQA_PROCESSOR, _BLIP_VQA_DEVICE, _BLIP_VQA_INIT_FAILED
    if _BLIP_VQA_INIT_FAILED:
        return None, None
    device_str = str(torch.device(device))
    if _BLIP_VQA_MODEL is not None and _BLIP_VQA_DEVICE == device_str:
        return _BLIP_VQA_MODEL, _BLIP_VQA_PROCESSOR
    try:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print(f"Loading BLIP-VQA (blip-vqa-base) on {device_str}...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        model.eval()
        model.to(torch.device(device_str))
        _BLIP_VQA_MODEL = model
        _BLIP_VQA_PROCESSOR = processor
        _BLIP_VQA_DEVICE = device_str
        return model, processor
    except Exception as e:
        _BLIP_VQA_INIT_FAILED = True
        print(f"Warning: failed to initialize BLIP-VQA: {e}")
        return None, None


def _prompt_to_desc(prompt: str) -> str:
    """
    Extract a natural-language description from a photo-caption prompt.
    e.g. "A photo of a sad face."  →  "a sad face"
         "A photo of a face with curly hair."  →  "a face with curly hair"
    Falls back to the original prompt (lowercased, trailing period stripped)
    if no "a photo of" prefix is found.
    """
    p = prompt.strip().rstrip(".")
    lower = p.lower()
    for prefix in ("a photo of a ", "a photo of an ", "a photo of "):
        if lower.startswith(prefix):
            return p[len(prefix):]
    return p


def _compute_vqa_scores(img_tensor, prompts, device):
    """
    VQAScore: for each prompt, ask "Is this [desc]?" and return P(yes)
    at the first generated token.
    The prompt is normalised via _prompt_to_desc to produce a grammatical
    question (e.g. "A photo of a sad face." → "Is this a sad face?").
    Returns list of P(yes) ∈ [0, 1] per prompt.
    """
    model, processor = _get_blip_vqa_model(device)
    if model is None:
        return [None] * len(prompts)
    try:
        pil_img = _tensor_to_pil(img_tensor)
        dev = torch.device(device)
        # Pre-compute yes/no token IDs once
        yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
        scores = []
        with torch.no_grad():
            for prompt in prompts:
                desc = _prompt_to_desc(prompt)
                question = f"Is this {desc}?"
                inputs = processor(images=pil_img, text=question,
                                   return_tensors="pt").to(dev)
                out = model.generate(
                    **inputs, max_new_tokens=1,
                    output_scores=True, return_dict_in_generate=True)
                first_logits = out.scores[0][0]          # [vocab_size]
                yn_probs = torch.softmax(
                    first_logits[[yes_id, no_id]], dim=0)
                scores.append(yn_probs[0].item())        # P(yes)
        return scores
    except Exception as e:
        print(f"Warning: failed to compute VQAScore: {e}")
        return [None] * len(prompts)


# ─── FER (Facial Expression Recognition via DeepFace) ────────────────────────

def _compute_fer_scores(img_tensor):
    """
    Facial Expression Recognition via DeepFace (RetinaFace detector).

    Uses enforce_detection=True so that images with no detectable face return
    None cleanly instead of producing unreliable whole-image estimates.
    For portrait editing tasks there is typically one face; if multiple are
    detected the largest bounding-box region is selected.

    Returns a dict mapping emotion → probability (sum=1), e.g.
    {'angry': 0.72, 'sad': 0.15, 'happy': 0.03, ...}.
    Returns None if deepface is unavailable or no face is detected.

    Suggested usage:
      • angry_sad task  → score = fer['angry'] + fer['sad']
      • smile task      → score = fer['happy']
    """
    try:
        import numpy as np
        import cv2
        from deepface import DeepFace
        # Convert tensor [1, C, H, W] in [0,1] → numpy uint8 RGB
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0)
                  .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result = DeepFace.analyze(
            img_bgr,
            actions=['emotion'],
            enforce_detection=True,       # return None (via exception) when no face found
            detector_backend='retinaface',  # more accurate than default mtcnn
            silent=True,
        )
        faces = result if isinstance(result, list) else [result]
        if not faces:
            return None
        # Pick the face with the largest bounding-box area
        def _area(f):
            r = f.get('region', {})
            return r.get('w', 0) * r.get('h', 0)
        best = max(faces, key=_area)
        emotions = best['emotion']
        total = sum(emotions.values()) or 1.0
        return {k: v / total for k, v in emotions.items()}
    except ImportError:
        return None
    except Exception:
        # Covers "Face could not be detected" from enforce_detection=True
        # as well as any other runtime error – silently return None.
        return None


# ===========================================================================
# High-level helpers used by run_lib_gcar.py
# ===========================================================================

def compute_evaluation_metrics(
    traj,
    traj_oc,
    original_img,
    prompts,
    config,
    inverse_scaler,
    lpips_f,
    clip_loss_list=None,
):
    """
    Compute all evaluation metrics for a single edited output.

    Parameters
    ----------
    traj           : baseline (reconstruction) trajectory
    traj_oc        : optimised trajectory
    original_img   : original image tensor (used to build clip loss if needed)
    prompts        : list of text prompts
    config         : experiment config (provides config.device)
    inverse_scaler : maps normalised tensor → [0, 1]
    lpips_f        : LPIPS network (already on device)
    clip_loss_list : pre-built ClipLoss objects (optional); if None,
                     clip_semantic_loss is called per prompt

    Returns
    -------
    clip_losses   : list[float]   – per-prompt CLIP loss (lower = better)
    lpips_score   : float         – perceptual distortion vs. reconstruction
    clipiqa_score : float|None    – no-reference image quality (higher = better)
    dsd_score     : float|None    – DINO structure distance (lower = better)
    extra_metrics : dict          – {blip_itm_scores, vqa_scores, fer_scores}
    """
    # with torch.no_grad():
    #     clip_losses = []
    #     if clip_loss_list is not None:
    #         for clip_loss in clip_loss_list:
    #             clip_losses.append(
    #                 clip_loss.L_N(traj_oc[-1].to(config.device)).item())
    #     else:
    #         for prompt in prompts:
    #             clip_loss_eval = clip_semantic_loss(
    #                 prompt, original_img, config.device,
    #                 alpha=1.0, inverse_scaler=inverse_scaler)
    #             clip_losses.append(
    #                 clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item())

    with torch.no_grad():
        clip_losses = []
        for prompt in prompts:
            # Always rebuild with alpha=1.0 for evaluation.
            # The clip_loss_list objects use the training alpha (e.g. 0.7),
            # which mixes in an L1 identity penalty:
            #   L = 0.7 * (−CLIP) + 0.3 * ||edit − orig||_1
            # That L1 term grows with iterations and makes the reported
            # "CLIP score" decrease even when semantic alignment improves.
            # alpha=1.0 gives pure text-image cosine similarity only.
            clip_loss_eval = clip_semantic_loss(
                prompt, original_img, config.device,
                alpha=1.0, inverse_scaler=inverse_scaler)
            clip_losses.append(
                clip_loss_eval.L_N(traj_oc[-1].to(config.device)).item())

    img_recon = inverse_scaler(traj[-1].to(config.device)).clamp(0, 1)
    img_edit  = inverse_scaler(traj_oc[-1].to(config.device)).clamp(0, 1)
    img_recon_norm = img_recon * 2 - 1
    img_edit_norm  = img_edit  * 2 - 1

    lpips_score   = lpips_f(img_edit_norm, img_recon_norm).item()
    clipiqa_score = _compute_clipiqa_score(img_edit, config.device)
    dsd_score     = _compute_dsd_score(img_recon, img_edit, config.device)

    extra_metrics = {
        "blip_itm_scores": _compute_blip_itm_scores(img_edit, prompts, config.device),
        "vqa_scores":      _compute_vqa_scores(img_edit, prompts, config.device),
        "fer_scores":      _compute_fer_scores(img_edit),
    }
    return clip_losses, lpips_score, clipiqa_score, dsd_score, extra_metrics


def build_evaluation_entry(method, clip_losses, lpips_score, clipiqa_score,
                            dsd_score=None, prompts=None,
                            extra_fields=None, extra_metrics=None):
    """Build a consistent metrics dict for single/multi-prompt methods."""
    if len(clip_losses) == 1:
        entry = {
            "clip_loss": clip_losses[0],
            "lpips_score": lpips_score,
            "clipiqa_score": clipiqa_score,
            "dsd_score": dsd_score,
            "method": method,
        }
    else:
        entry = {
            "clip_losses": clip_losses,
            "clip_loss_avg": sum(clip_losses) / len(clip_losses),
            "lpips_score": lpips_score,
            "clipiqa_score": clipiqa_score,
            "dsd_score": dsd_score,
            "prompts": prompts,
            "method": method,
        }
    if extra_metrics:
        entry.update(extra_metrics)
        # Expand per-prompt list metrics with avg / min aggregates,
        # consistent with how clip_loss_avg is stored for multi-prompt.
        for key in ("blip_itm_scores", "vqa_scores"):
            vals = extra_metrics.get(key)
            if vals and all(v is not None for v in vals):
                base = key.replace("_scores", "")
                entry[f"{base}_avg"] = sum(vals) / len(vals)
                entry[f"{base}_min"] = min(vals)
    if extra_fields:
        entry.update(extra_fields)
    return entry


def print_evaluation_summary(prompts, clip_losses, lpips_score, clipiqa_score,
                              elapsed_s, dsd_score=None, extra_metrics=None,
                              header="=== Results ==="):
    print(f"\n{header}")
    if len(clip_losses) == 1:
        label = prompts[0] if prompts else "prompt"
        print(f'Prompt "{label}": CLIP loss = {clip_losses[0]:.4f}')
    else:
        for idx, (prompt, loss) in enumerate(zip(prompts, clip_losses)):
            print(f'Prompt {idx+1} "{prompt}": CLIP loss = {loss:.4f}')
    print(f"LPIPS score: {lpips_score:.4f}")
    if clipiqa_score is None:
        print("CLIPIQA score: N/A (pyiqa unavailable or CLIPIQA init failed)")
    else:
        print(f"CLIPIQA score: {clipiqa_score:.4f}")
    if dsd_score is None:
        print("DSD score: N/A (DINO unavailable or init failed)")
    else:
        print(f"DSD score: {dsd_score:.6f}")
    # ── extra metrics ─────────────────────────────────────────────────────────
    em = extra_metrics or {}
    blip_itm = em.get("blip_itm_scores")
    vqa      = em.get("vqa_scores")
    fer      = em.get("fer_scores")
    if blip_itm is None or all(s is None for s in blip_itm):
        print("BLIP-ITM: N/A (model unavailable)")
    else:
        for idx, (prompt, s) in enumerate(zip(prompts, blip_itm)):
            tag = f'Prompt {idx+1} "{prompt}"' if len(prompts) > 1 else f'"{prompt}"'
            val = f"{s:.4f}" if s is not None else "N/A"
            print(f"BLIP-ITM {tag}: {val}")
    if vqa is None or all(s is None for s in vqa):
        print("VQAScore: N/A (model unavailable)")
    else:
        for idx, (prompt, s) in enumerate(zip(prompts, vqa)):
            tag = f'Prompt {idx+1} "{prompt}"' if len(prompts) > 1 else f'"{prompt}"'
            val = f"{s:.4f}" if s is not None else "N/A"
            print(f"VQAScore {tag}: {val}")
    if fer is None:
        print("FER: N/A (deepface unavailable)")
    else:
        top = sorted(fer.items(), key=lambda x: -x[1])[:4]
        print("FER: " + ", ".join(f"{k}={v:.3f}" for k, v in top))
    print(f"Total time: {elapsed_s:.4f} s")
