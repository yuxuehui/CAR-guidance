import functools

import torch
import torchvision
import numpy as np
import abc

from RectifiedFlow.models.utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate


import imageio

import lpips
import clip

try:
  from .DiffAugment_pytorch import DiffAugment
except ImportError:
  from DiffAugment_pytorch import DiffAugment

import os
import time

@torch.no_grad()
def embed_to_latent(model_fn, img):
  device = img.device
  def ode_func(t, x):
    x = from_flattened_numpy(x, img.shape).to(device).type(torch.float32)
    vec_t = torch.ones(img.shape[0], device=x.device) * t
    drift = model_fn(x, vec_t*999)
    return to_flattened_numpy(drift)

  rtol=atol=1e-5
  method='RK45'
  eps=1e-3
  
  # Initial sample
  x = img.detach().clone()

  solution = integrate.solve_ivp(ode_func, (1., eps), to_flattened_numpy(x),
                                 rtol=rtol, atol=atol, method=method)
  nfe = solution.nfev
  x = torch.tensor(solution.y[:, -1]).reshape(img.shape).to(device).type(torch.float32)

  return x

@torch.no_grad()
def generate_traj(dynamic, z0, u=None, N=100, straightness_threshold=None):
  traj = []

  # Initial sample (keep on same device as z0 for VJP / backward pass)
  z = z0.detach().clone()
  traj.append(z.detach().clone())
  batchsize = z0.shape[0]

  dt = 1./N
  eps = 1e-3
  pred_list = []
  for i in range(N):
    if (u is not None):
        try:
            z = z + u[i]
        except:
            pass

    t = torch.ones(z0.shape[0], device=z0.device) * i / N * (1.-eps) + eps
    pred = dynamic(z, t*999)
    z = z.detach().clone() + pred * dt
      
    traj.append(z.detach().clone())

    pred_list.append(pred.detach().clone().cpu())

  if straightness_threshold is not None:
      ### compute straightness and construct G
      non_uniform_set = {}
      non_uniform_set['indices'] = []
      non_uniform_set['length'] = {}
      accumulate_length = 0
      accumulate_straightness = 0
      cur_index = 0
      for i in range(N):
          try:
            d1 = (pred_list[i-1] - pred_list[i]).pow(2).sum() / pred_list[i].pow(2).sum()
          except:
            d1 = 0
          
          try:
            d2 = (pred_list[i+1] - pred_list[i]).pow(2).sum() / pred_list[i].pow(2).sum()
          except:
            d2 = 0
          
          d = max(d1, d2)
          accumulate_straightness += d
          accumulate_length += 1
          if (accumulate_straightness > straightness_threshold) or (i==(N-1)):
            non_uniform_set['length'][cur_index] = accumulate_length
            non_uniform_set['indices'].append(cur_index)

            accumulate_straightness = 0
            accumulate_length = 0
            cur_index = i+1

      return traj, non_uniform_set
  else:
      return traj

def get_img(path=None):
    img = imageio.imread(path) ### 4-no expression
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('read image from:', path, 'img range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)

    return img

def save_img(img, path=None):
    torchvision.utils.save_image(img.clamp_(0.0, 1.0), os.path.join(path), nrow=16, normalize=False)

class clip_semantic_loss():
    def __init__(self, text, img, device, alpha=0.5, replicate=20, inverse_scaler=None):
        # self.loss_fn_alex = lpips.LPIPS(net='alex', spatial=False).to(device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        clip_mode="ViT-B/32"
        self.interp_mode='bilinear'
        self.clip_model, _ = clip.load(clip_mode, device=device)
        self.clip_c = self.clip_model.logit_scale.exp()
        self.text_tok = clip.tokenize([text]).to(device)
        self.policy = 'color,translation,resize,cutout'
        self.replicate = 20 # second before is 20
        self.img = img
        self.alpha = alpha
        self.inverse_scaler = inverse_scaler
        self.verbose = False  # whether to print regu/reward

    def L_N(self, x):
        # Use x's own batch size: the input may be B trajectories while self.img
        # holds a single reference image.
        batch_size = x.shape[0] 
        
        # Per-image mean L1 error, keeping the batch dim -> shape [B].
        sim = (self.inverse_scaler(x) - self.img).abs().view(batch_size, -1).mean(dim=1)

        img_aug = DiffAugment(x.repeat(self.replicate, 1, 1, 1), policy=self.policy)
        img_aug = self.inverse_scaler(img_aug)
        img_aug = torch.nn.functional.interpolate(img_aug, size=224, mode=self.interp_mode)
        img_aug.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])

        logits_per_image, logits_per_text = self.clip_model(img_aug, self.text_tok)
        logits_per_image = logits_per_image.view(batch_size, self.replicate, -1).mean(dim=1)
        logits_per_image = logits_per_image / self.clip_c
        
        # Squeeze [B, 1] -> [B] to align with sim.
        concept_loss = ((-1.) * logits_per_image).squeeze(-1) 

        if self.verbose:
            print(f'regu: {sim.sum().item():.4f}, reward: {concept_loss.sum().item():.4f}')

        # Return the per-sample sum as a 1-D tensor of shape [B].
        return self.alpha * concept_loss + (1.-self.alpha) * sim

