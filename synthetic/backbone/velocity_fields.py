"""Velocity-field architectures for the unconditional flow v(x, t).

The velocity field predicts, at any time ``t`` and position ``x``, the
instantaneous direction in which a sample should move to flow from the prior
``x_0`` to the data distribution ``x_1``. Input and output share the same shape.

Three interchangeable backbones are provided (selected via ``cfg.vf_model``):
``MLP``, ``FiLMResNet`` and ``MiniUnetVelocityField`` (the default).
"""

import math
import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    """Basic MLP velocity field for 2D data.

    Input:  x ∈ ℝ^{B×2},  t ∈ ℝ^{B×1}  (concatenated to B×3)
    └─ Cat(x, t) → (B×3)
    └─ Linear(3 → 128) + Swish                → (B×128)
    └─ Linear(128 → 128) + Swish  (×3)        → (B×128)
    └─ Linear(128 → 2)                        → (B×2)
    """
    def __init__(self, input_dim=2, time_dim=1, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).expand(x.shape[0], 1).float()
        h = torch.cat([x, t], dim=1)
        return self.main(h)

class SinusoidalEmbed(torch.nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int = 64):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(1.), math.log(1000.), dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

class FiLMResBlock(torch.nn.Module):
    """FiLM-based residual block."""
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.ln1, self.ln2 = torch.nn.LayerNorm(ch), torch.nn.LayerNorm(ch)
        self.fc1 = weight_norm(torch.nn.Linear(ch, ch))
        self.fc2 = weight_norm(torch.nn.Linear(ch, ch))
        self.film = torch.nn.Linear(t_dim, ch * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s, b = self.film(t_emb).chunk(2, dim=-1)
        h = self.ln1(x) * (1 + s) + b
        h = F.silu(self.fc1(h))
        h = self.ln2(h)
        h = self.fc2(F.silu(h))
        return x + h

class FiLMResNet(torch.nn.Module):
    """FiLM residual network velocity field (feature trunk + time modulation).

    Input:  x ∈ ℝ^{B×2},  t ∈ ℝ^{B×1}
    ├─ x  ── Linear(2 → 128)                  → (B×128)   # feature trunk
    └─ t  ── SinusoidalEmbed(→ 64-d)          → (B×64)    # time embedding e_t

    Backbone (6 FiLM residual blocks): each block derives FiLM parameters
    (γ_k, β_k) from e_t and applies a channel-wise affine modulation
    h = γ_k ⊙ h + β_k with a residual connection.

    Head:
    └─ Linear(128 → 2)                        → (B×2)
    """
    def __init__(self):
        super().__init__()
        self.t_embed = SinusoidalEmbed()
        self.input = torch.nn.Linear(2, 128)
        self.blocks = torch.nn.ModuleList([FiLMResBlock(128, 64) for _ in range(6)])
        self.out = torch.nn.Linear(128, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.input(x))
        t_emb = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(F.silu(h))


class DownLayer1D(nn.Module):
    """MiniUnet downsampling ResNet block for 1D vector data."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16):
        super(DownLayer1D, self).__init__()

        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.act = nn.ReLU()

        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x
        x = x + self.fc(temb)
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)

        x = x + res
        return x


class UpLayer1D(nn.Module):
    """MiniUnet upsampling block for 1D vector data."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim=16):
        super(UpLayer1D, self).__init__()

        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.act = nn.ReLU()

        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x

        x = x + self.fc(temb)
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.act(x)

        if self.shortcut is not None:
            res = self.shortcut(res)
        x = x + res

        return x


class MiddleLayer1D(nn.Module):
    """MiniUnet bottleneck block for 1D vector data."""

    def __init__(self, in_channels, out_channels, time_emb_dim=16):
        super(MiddleLayer1D, self).__init__()

        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.act = nn.ReLU()

        self.fc = nn.Linear(time_emb_dim, in_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        res = x

        x = x + self.fc(temb)
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.act(x)

        if self.shortcut is not None:
            x = self.shortcut(x)
        x = x + res

        return x


class MiniUnetVelocityField(nn.Module):
    """MiniUnet-based velocity field for 2D vector data.
    Input:  x ∈ ℝ^{B×input_dim},  t ∈ ℝ^{B×1}
    Output: v ∈ ℝ^{B×output_dim}
    """

    def __init__(self, input_dim=2, output_dim=2, base_channels=128, time_emb_dim=None):
        super(MiniUnetVelocityField, self).__init__()

        if time_emb_dim is None:
            self.time_emb_dim = base_channels // 2 # Use half for sinusoidal embedding
        else:
            self.time_emb_dim = time_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_channels = base_channels

        self.linear_in = nn.Linear(input_dim, base_channels)

        self.down1 = nn.ModuleList([
            DownLayer1D(base_channels,
                        base_channels * 2,
                        time_emb_dim=self.time_emb_dim),
            DownLayer1D(base_channels * 2,
                        base_channels * 2,
                        time_emb_dim=self.time_emb_dim)
        ])

        self.down2 = nn.ModuleList([
            DownLayer1D(base_channels * 2,
                        base_channels * 4,
                        time_emb_dim=self.time_emb_dim),
            DownLayer1D(base_channels * 4,
                        base_channels * 4,
                        time_emb_dim=self.time_emb_dim)
        ])

        self.middle = MiddleLayer1D(base_channels * 4,
                                  base_channels * 4,
                                  time_emb_dim=self.time_emb_dim)

        self.up1 = nn.ModuleList([
            UpLayer1D(
                base_channels * 8,  # concat
                base_channels * 2,
                time_emb_dim=self.time_emb_dim),
            UpLayer1D(base_channels * 2,
                    base_channels * 2,
                    time_emb_dim=self.time_emb_dim)
        ])

        self.up2 = nn.ModuleList([
            UpLayer1D(base_channels * 4,
                    base_channels,
                    time_emb_dim=self.time_emb_dim),
            UpLayer1D(base_channels,
                    base_channels,
                    time_emb_dim=self.time_emb_dim)
        ])

        self.linear_out = nn.Linear(base_channels, output_dim)

    def time_emb(self, t, dim):
        """Sinusoidal time embedding (absolute-position style).

        Maps the scalar time ``t`` to a ``dim``-dimensional vector via sine/cosine
        features so the network can condition on ``t``, analogous to the
        positional encodings used in transformers.
        """
        t = t * 1000
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim) # Flatten to B x input_dim

        temb = self.time_emb(t.reshape(-1), self.time_emb_dim)

        x = self.linear_in(x)

        # Downsampling path
        h1 = x
        for layer in self.down1:
            h1 = layer(h1, temb)

        h2 = h1
        for layer in self.down2:
            h2 = layer(h2, temb)

        # Middle path
        h_middle = self.middle(h2, temb)

        # Upsampling path
        h = torch.cat([h_middle, h2], dim=1) # Skip connection
        for layer in self.up1:
            h = layer(h, temb)

        h = torch.cat([h, h1], dim=1) # Skip connection
        for layer in self.up2:
            h = layer(h, temb)

        out = self.linear_out(h)
        return out.reshape(sz) # Reshape back to original spatial dimensions if any
