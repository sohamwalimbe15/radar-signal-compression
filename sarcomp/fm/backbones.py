"""Velocity-field networks v(x_t, t, cond) for flow matching on the latent1 grid.

Both backbones are REAL-valued and see the latent as plain channels (for the complex autoencoder the
latent is the stacked [Re | Im] tensor), so experiments 3 and 4 share the exact same FM network.
Zero-init of residual/output layers is OFF by default so gradients reach every layer at step 0
(handy for smoke tests); flip `zero_init: true` for the usual DiT-style init.

Add a backbone: @BACKBONES.register("name") on a class with
    __init__(self, latent_ch, grid, cond_dim, **kwargs)  and  forward(x, t, cond) -> same shape as x.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Registry

BACKBONES = Registry("fm_backbone")


def timestep_embedding(t, dim, max_period=10000.0):
    """t: (B,) in [0,1] -> (B,dim) sinusoidal."""
    t = t * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2)) if dim % 2 else emb


class _Embed(nn.Module):
    """time + condition vector -> one embedding vector."""

    def __init__(self, cond_dim, emb):
        super().__init__()
        self.emb = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb))

    def forward(self, t, cond):
        return self.t_mlp(timestep_embedding(t, self.emb)) + self.c_mlp(cond)


def _gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)


class _ResBlock(nn.Module):
    def __init__(self, c, emb, zero_init):
        super().__init__()
        self.n1, self.c1 = _gn(c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = _gn(c), nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(emb, 2 * c)
        if zero_init:
            nn.init.zeros_(self.c2.weight), nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.film(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + s) + b                      # FiLM from (time + condition)
        return x + self.c2(F.silu(h))


@BACKBONES.register("conv_film")
class ConvFiLM(nn.Module):
    """Residual conv net at constant (latent) resolution; FiLM conditioning. Good default for a tiny grid."""

    def __init__(self, latent_ch, grid, cond_dim, hidden=512, n_blocks=6, emb=512, zero_init=False):
        super().__init__()
        self.embed = _Embed(cond_dim, emb)
        self.inp = nn.Conv2d(latent_ch, hidden, 1)
        self.blocks = nn.ModuleList([_ResBlock(hidden, emb, zero_init) for _ in range(n_blocks)])
        self.out_norm, self.out = _gn(hidden), nn.Conv2d(hidden, latent_ch, 1)
        if zero_init:
            nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        e = self.embed(t, cond)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, e)
        return self.out(F.silu(self.out_norm(h)))


class _DiTBlock(nn.Module):
    def __init__(self, d, heads, emb, zero_init):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.mod = nn.Linear(emb, 6 * d)                  # adaLN: shift/scale/gate for attn and mlp
        if zero_init:
            nn.init.zeros_(self.mod.weight), nn.init.zeros_(self.mod.bias)

    def forward(self, x, e):
        s1, c1, g1, s2, c2, g2 = self.mod(e)[:, None].chunk(6, dim=-1)
        h = self.n1(x) * (1 + c1) + s1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + c2) + s2
        return x + g2 * self.mlp(h)


@BACKBONES.register("dit")
class DiT(nn.Module):
    """Transformer over the grid*grid latent positions (tokens); adaLN conditioning on (time + condition)."""

    def __init__(self, latent_ch, grid, cond_dim, hidden=512, depth=6, heads=8, emb=512, zero_init=False):
        super().__init__()
        self.grid = grid
        self.embed = _Embed(cond_dim, emb)
        self.inp = nn.Linear(latent_ch, hidden)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, hidden) * 0.02)
        self.blocks = nn.ModuleList([_DiTBlock(hidden, heads, emb, zero_init) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, latent_ch)
        if zero_init:
            nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        B, C, H, W = x.shape
        e = self.embed(t, cond)
        h = self.inp(x.flatten(2).transpose(1, 2)) + self.pos
        for blk in self.blocks:
            h = blk(h, e)
        return self.out(self.out_norm(h)).transpose(1, 2).reshape(B, C, H, W)
