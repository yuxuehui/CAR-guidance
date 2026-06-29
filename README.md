# Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards (coming soon)

<p align="center">
  <a href="https://arxiv.org/abs/2605.20758"><img src="https://img.shields.io/badge/arXiv-2605.20758-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Venue-ICML%202026-blue.svg" alt="ICML 2026">
</p>

Official implementation of **Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards** (ICML 2026). The paper is available at [arXiv:2605.20758](https://arxiv.org/abs/2605.20758)

**Authors:** [Xuehui Yu](https://github.com/yuxuehui), [Fucheng Cai](https://github.com/HITCai), [Meiyi Wang](https://github.com/mywang44), Xiaopeng Fan, [Harold Soh](https://github.com/haroldsoh)

## Overview

📌 **Challenge.** Inference-time guidance can easily push your sampling process off the data manifold. 

👉 How do we harness large, complex pretrained generative priors to satisfy multiple constraints at inference time, without drifting off-manifold (i.e., avoiding hallucinated generation)? In this work, we introduce CAR guidance, a plug-and-play module that corrects off-manifold drift on the fly.

🔑 **Key insight.** In compositional reward settings, the approximation error grows sharply with gradient misalignment $(1 - \cos\varphi)$ and the number of reward functions $G$, where $\varphi$ is the average angular divergence between guidance channels.

https://github.com/user-attachments/assets/4fefa401-ea46-4bf4-8a55-3fdce9e1ded3

## Requirements

Each task has its own directory with a self-contained environment. Set up the one for the experiment you want to run. For example, for the image editing task:

```bash
cd image
conda env create -f environment.yml
conda activate car_guidance
pip install -r requirements.txt
```

The other tasks follow the same steps — replace `image` with `synthetic`, `planning` or `manipulation`.

## Datasets

We evaluate CAR guidance on three tasks. Download the corresponding dataset before running each experiment.

- **Text-Guided Image Manipulation — CelebA-HQ-1024.** A high-quality version of CelebA containing 30,000 images at 1024×1024 resolution. Download it from the [Kaggle CelebA-HQ dataset](https://www.kaggle.com/datasets/lamsimon/celebahq).
- **Robot Planning — Maze2D.** Available on Hugging Face: [car-guidance/GAR_guidance](https://huggingface.co/datasets/car-guidance/GAR_guidance).
- **Robot Manipulation — ManiSkill2.** Available on Hugging Face: [car-guidance/GAR_guidance](https://huggingface.co/datasets/car-guidance/GAR_guidance).

## Checkpoints

- **Text-Guided Image Manipulation — CelebA-HQ-1024.**
- **Robot Planning — Maze2D.**
- **Robot Manipulation — ManiSkill2.**

## 📎 Citation

If you find CAR guidance useful in your research, please consider citing our paper:

```bibtex
@inproceedings{
anonymous2026conflictaware,
title={Conflict-Aware Additive Guidance for  Flow Models under Compositional Rewards},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=DJtX0Fo7ii}
}
```