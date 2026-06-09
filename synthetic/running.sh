#!/bin/bash
# =============================================================================
# CAR-guidance — Synthetic experiment runners
#
# Reproduces the synthetic 2D experiment from
# "Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards".
#
# USAGE:
#   cd synthetic
#   conda activate car_guidance
#   bash running.sh
#
# NOTES:
#   * All commands are launched from inside the `synthetic/` directory.
#   * --result_output_dir is resolved relative to the script directory, so
#     results are written under  synthetic/<RESULT_DIR>/...
#   * --guidance_ckpt_dir is resolved relative to the current working directory
#     (i.e. synthetic/), and stores the trained online guidance model g_psi.
#   * The pretrained velocity field and the two classifiers ship with the repo
#     (pretrain_vf_model/ and guidance/pretrained_guidance/), so no extra
#     pretraining step is required.
#   * CUDA_VISIBLE_DEVICES below assigns one GPU per run for parallel launches;
#     edit them to match your hardware (use the same id to run sequentially).
# =============================================================================

CKPT_DIR="guidance/pretrained_guidance_online_mar"
RESULT_DIR="result_online_mar"

# #############################################################################
# # CAR guidance (ours) — optimal hyperparameters
# #   g^car = g^approx + w_t * g_psi, with a conflict gate (conflict_threshold).
# #   conflict_threshold 0.495 (≈perpendicular on the [0,1] conflict scale), conflict_temperature 0.01, gradient online loss.
# #   active_ratio_threshold controls early-stopping of the online g_psi training.
# #############################################################################

# art = 0.00 (no early stop on active-ratio)
CUDA_VISIBLE_DEVICES=0 python -u ./main.py \
  --guidance_fn car_guidance \
  --conflict_score_method direct \
  --conflict_threshold 0.495 \
  --conflict_temperature 0.01 \
  --x1_conflict_threshold 0.0 \
  --active_ratio_threshold 0.00 \
  --online_loss_type gradient \
  --guidance_ckpt_dir "${CKPT_DIR}/guidance_ckpt_cth0495_gradient_art000" \
  --result_output_dir "${RESULT_DIR}/result_cth0495_gradient_art000"

# art = 0.05
CUDA_VISIBLE_DEVICES=0 python -u ./main.py \
  --guidance_fn car_guidance \
  --conflict_score_method direct \
  --conflict_threshold 0.495 \
  --conflict_temperature 0.01 \
  --x1_conflict_threshold 0.0 \
  --active_ratio_threshold 0.05 \
  --online_loss_type gradient \
  --guidance_ckpt_dir "${CKPT_DIR}/guidance_ckpt_cth0495_gradient_art005" \
  --result_output_dir "${RESULT_DIR}/result_cth0495_gradient_art005"


# #############################################################################
# # Baselines
# #############################################################################

# # g_cov_g (non-learnable approximate guidance g^approx only, no training)
# CUDA_VISIBLE_DEVICES=0 python -u ./main.py \
#   --guidance_fn g_cov_g \
#   --guidance_ckpt_dir "${CKPT_DIR}/guidance_ckpt_g_cov_g" \
#   --result_output_dir "${RESULT_DIR}/result_g_cov_g"

# # guidance_matching (learnable baseline)
# CUDA_VISIBLE_DEVICES=0 python -u ./main.py \
#   --guidance_fn guidance_matching \
#   --guidance_ckpt_dir "${CKPT_DIR}/guidance_ckpt_guidance_matching" \
#   --result_output_dir "${RESULT_DIR}/result_guidance_matching"


# #############################################################################
# # GLASS Flow FK corrector baseline (5 seeds, one GPU per seed)
# #############################################################################

# for SEED in 42 43 44 45 46; do
#   CUDA_VISIBLE_DEVICES=0 python -u ./run_glass_fk_only.py \
#     --seed ${SEED} --fk_corr_rho 0.5 --fk_n_backbone 6 --fk_n_inner_steps 20 \
#     --result_output_dir "${RESULT_DIR}/result_glass_fk_seed${SEED}"
# done
