# Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards

<p align="center">
  <a href="https://arxiv.org/abs/2605.20758"><img src="https://img.shields.io/badge/arXiv-2605.20758-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Venue-ICML%202026-blue.svg" alt="ICML 2026">
</p>

Official implementation of **Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards** (ICML 2026).

📄 **Paper:** [arXiv:2605.20758](https://arxiv.org/abs/2605.20758)

**Authors:** [Xuehui Yu](https://github.com/yuxuehui), [Fucheng Cai](https://github.com/HITCai), [Meiyi Wang](https://github.com/mywang44), Xiaopeng Fan, [Harold Soh](https://github.com/haroldsoh)

## Overview

> **Challenge.** Inference-time guidance can easily push your sampling process off the data manifold.

How do we harness large, complex pretrained generative priors to satisfy multiple constraints at inference time, without drifting off-manifold (i.e., avoiding hallucinated generation)?

👉 In this work, we introduce **CAR guidance**, a plug-and-play module that corrects off-manifold drift on the fly.

🔑 **Key insight.** In compositional reward settings, the approximation error grows sharply with:

- **Gradient misalignment** $(1 - \cos\varphi)$, where $\varphi$ is the average angular divergence between guidance channels.
- **The number of reward functions** $G$.

## Demo

[▶ **Watch the demo video**](docs/car_guidance.mp4)

<!--
To embed an inline video player here (instead of just a link):
  1. Open a new GitHub issue, or edit this README on github.com.
  2. Drag docs/car_guidance.mp4 into the comment box and wait for the upload to finish.
  3. Copy the generated https://github.com/user-attachments/assets/... URL.
  4. Paste that URL on its own line below, replacing this comment.
GitHub renders an inline player only for user-attachments URLs — not for repo raw URLs.
-->

