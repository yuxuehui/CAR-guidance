import gc
import io
import os
import time

import numpy as np
import logging
import json
from tqdm import tqdm
# Keep the import below for registering all model definitions
from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from RectifiedFlow.utils import save_checkpoint, restore_checkpoint
import RectifiedFlow.datasets as datasets

from RectifiedFlow.models.utils import get_model_fn
from RectifiedFlow.models import utils as mutils

from .gcar_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj
# from id_loss.loss_fn import IDLoss

import torch.nn.functional as F
import torch.backends.cuda

import warnings
warnings.filterwarnings("ignore")
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'



FLAGS = flags.FLAGS

def gcar_edit_batch_multiprompt(config, model_path, image_paths, text_prompts, output_dir, 
                                           method='gcar', alpha=0.7, 
                                           lr_gcov=1.0, lr_res=2.5, conflict_weight=None,
                                           use_L_best=True,
                                           save_single_prompt=True,
                                           save_combined=True):
    # Initialize model, EMA and data scalers, then restore the checkpoint.
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)
    state = restore_checkpoint(model_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    model_fn = mutils.get_model_fn(score_model, train=False)
    
    N = 100
    num_prompts = len(text_prompts)

    print(f"\n{'='*60}")
    print(f"GCAR Multi-prompt (Trained Residual Guidance):")
    for idx, prompt in enumerate(text_prompts):
        print(f"  Prompt {idx+1}: {prompt}")
    print(f"{'='*60}\n")

    for img_path in tqdm(image_paths):
        # Output dir: examples/{output_dir} (output_dir like gcar_{timestamp}/{task}_{id}).
        target_dir = f'examples/{output_dir}'
        os.makedirs(target_dir, exist_ok=True)
        # Image id, e.g. 000442.
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # Trained gcar result image: {id}_gcar.jpg
        opt_img_path = os.path.join(target_dir, f'{img_id}_gcar.jpg')

        image = get_img(img_path)  
        original_img = image.to(config.device)
        
        clip_loss_list = []
        for prompt in text_prompts:
            clip_loss = clip_semantic_loss(prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)
            clip_loss_list.append(clip_loss)
        
        import math
        t_s = time.time()
        # Invert the original image to its deterministic latent y(0), shape [1, C, H, W].
        y_0 = embed_to_latent(model_fn, scaler(original_img)) 
        # Also run a baseline trajectory for reference.
        traj = generate_traj(model_fn, y_0, N=N)

        print(f'\nGCAR optimization starts: {img_path} -> {opt_img_path}')
        u_ind = [_ for _ in range(N)]
        L_N_list = [clip_loss.L_N for clip_loss in clip_loss_list]
        
        # Build a robust batch latent for training.
        train_batch_size = getattr(config, "guidance_batch_size", 4)
        # The function-signature alpha is for the CLIP loss; the alpha in the
        # init formula below is named init_alpha to avoid confusion.
        init_alpha = getattr(config, "init_alpha", 0.9) 

        # Expand the single y_0 into a batch, shape [B, C, H, W].
        y_0_batch = y_0.repeat(train_batch_size, 1, 1, 1)
        
        # Sample standard Gaussian noise z ~ p_0(x_0).
        z = torch.randn_like(y_0_batch)

        # Initialization: x_0 = sqrt(alpha)*y(0) + sqrt(1-alpha)*z
        latent_batch = math.sqrt(init_alpha) * y_0_batch + math.sqrt(1.0 - init_alpha) * z


        # Build the guidance vector field for the prompt combination. The CLIP
        # losses (L_N_list) play the role of the classifiers in the 2D version,
        # and ImageGCovGGMOnlineGuidance wraps a learnable residual net.
        is_learnable = True  # enable training of the residual net
        
        from utils.composed_guidance import ImageGCovGGMOnlineGuidance
        lr_gcov = 4400
        guided_field = ImageGCovGGMOnlineGuidance(
            base_model=model_fn,                  # base velocity field
            loss_fns=L_N_list,                    # CLIP-loss gradient providers
            scales=[lr_gcov] * num_prompts,       # per-prompt guidance scales
            config=config,
            learnable=is_learnable,
            conflict_weight=conflict_weight,
        )

        # Online-train the residual net to resolve prompt conflicts, feeding the
        # perturbed batch for better generalization and convergence.
        print("\n--- Training Residual Net to resolve prompt conflicts ---")
        guided_field.train_model(latent_batch, num_steps=N, steps=15)

        # Final sampling: guided_field is itself a callable dynamic that computes
        # v_total(x, t) = v_base(x, t) + v_gcov(x, t) + v_res_net(x, t),
        # so we pass it directly to the solver without an explicit control u.
        print("\n--- Generating final trajectory with GCAR (Trained Residual Guidance) ---")
        # Run inference from the clean y_0 to maximally preserve the original
        # image's structure prior (the trained residual already handles conflict).
        traj_oc = generate_traj(guided_field, z0=y_0, u=None, N=N)

        # Save the result image.
        if opt_img_path is not None:
            save_img(inverse_scaler(traj_oc[-1]), path=opt_img_path)

        # Debug 1: save each prompt's standalone guidance result.
        if save_single_prompt and opt_img_path is not None:
            print("\n--- Generating Single Prompt Trajectories for Debugging ---")
            lr_gcov = 4400
            for p_idx in range(num_prompts):
                # A single prompt has no conflict, so disable learnable and
                # conflict_weight: plain g_cov-G guidance, which is very fast.
                single_guided_field = ImageGCovGGMOnlineGuidance(
                    base_model=model_fn,
                    loss_fns=[L_N_list[p_idx]],       # only this one CLIP loss
                    scales=[lr_gcov],                 # match g_cov-G strength
                    config=config,
                    learnable=False,
                    conflict_weight=0.0,
                )
                
                # Run a clean trajectory from y_0 with this single-prompt field.
                traj_single = generate_traj(single_guided_field, z0=y_0, u=None, N=N)
                
                # Single-prompt result: {id}_gcov-G_singleprompt{k}.jpg
                single_path = os.path.join(target_dir, f'{img_id}_gcov-G_singleprompt{p_idx+1}.jpg')
                save_img(inverse_scaler(traj_single[-1]), path=single_path)
                print(f"  -> Saved single effect for Prompt {p_idx+1}: {single_path}")

        # Debug 2: naive linear sum of the prompts (no residual guidance).
        if save_combined and opt_img_path is not None:
            print("\n--- Generating Combined Trajectory ---")
            lr_gcov = 4400
            # Combined field with both network learning and the conflict term off.
            from utils.composed_guidance import ImageGCovGGMOnlineGuidance
            combined_guided_field = ImageGCovGGMOnlineGuidance(
                base_model=model_fn,
                loss_fns=L_N_list,                          
                scales=[lr_gcov] * num_prompts,    
                config=config,
                learnable=False,                  # residual net fully disabled
                conflict_weight=0.0,              # no conflict computation
            )

            # Inference: generate the plain linearly-combined trajectory.
            traj_combined = generate_traj(combined_guided_field, z0=y_0, u=None, N=N)

            # Linear-sum result (no residual guidance): {id}_gcov-G_multiprompt.jpg
            combined_path = os.path.join(target_dir, f'{img_id}_gcov-G_multiprompt.jpg')
            save_img(inverse_scaler(traj_combined[-1]), path=combined_path)
            print(f"  -> Saved linear combined image to: {combined_path}")