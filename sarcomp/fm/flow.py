"""Conditional (optimal-transport / rectified) flow matching on latent1, conditioned on latent2.

  x_t = (1 - t) * noise + t * x1          x1 = normalised E1(x),  noise ~ N(0, I)
  target velocity  u = x1 - noise,   loss = MSE(v_theta(x_t, t, c), u)
  sampling: integrate dx/dt = v_theta from t=0 (noise) to t=1 (latent1)

Latents / conditions are standardised per channel with statistics fitted once on training data
(`set_stats`), stored as buffers so they travel with the checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import BACKBONES


def make_generator(seed, device):
    g = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
    return g.manual_seed(seed)


class FlowMatcher(nn.Module):
    def __init__(self, backbone, latent_shape, cond_dim, t_sampling="uniform"):
        super().__init__()
        self.backbone, self.latent_shape, self.cond_dim, self.t_sampling = backbone, tuple(latent_shape), cond_dim, t_sampling
        C = latent_shape[0]
        self.register_buffer("lat_mean", torch.zeros(1, C, 1, 1))
        self.register_buffer("lat_std", torch.ones(1, C, 1, 1))
        self.register_buffer("cond_mean", torch.zeros(1, cond_dim))
        self.register_buffer("cond_std", torch.ones(1, cond_dim))

    # ---- normalisation
    @torch.no_grad()
    def set_stats(self, lat_mean, lat_std, cond_mean, cond_std):
        self.lat_mean.copy_(lat_mean.view(1, -1, 1, 1)), self.lat_std.copy_(lat_std.view(1, -1, 1, 1).clamp_min(1e-6))
        self.cond_mean.copy_(cond_mean.view(1, -1)), self.cond_std.copy_(cond_std.view(1, -1).clamp_min(1e-6))

    def norm_lat(self, l1):
        return (l1 - self.lat_mean) / self.lat_std

    def denorm_lat(self, x):
        return x * self.lat_std + self.lat_mean

    def norm_cond(self, l2):
        return (l2.flatten(1) - self.cond_mean) / self.cond_std

    # ---- training
    def _sample_t(self, B, device):
        if self.t_sampling == "logit_normal":
            return torch.sigmoid(torch.randn(B, device=device))
        return torch.rand(B, device=device)

    def loss(self, l1, l2):
        x1, c = self.norm_lat(l1), self.norm_cond(l2)
        x0 = torch.randn_like(x1)
        t = self._sample_t(x1.shape[0], x1.device)
        xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        return F.mse_loss(self.backbone(xt, t, c).float(), (x1 - x0).float())

    # ---- sampling
    @torch.no_grad()
    def sample(self, l2, steps=20, solver="euler", generator=None):
        """l2: (B,C2,1,1) latent2 code -> l1_hat (B,C1,h,w) in the ORIGINAL latent1 scale."""
        c = self.norm_cond(l2)
        B = c.shape[0]
        x = torch.randn((B,) + self.latent_shape, device=c.device, dtype=c.dtype, generator=generator)
        dt = 1.0 / steps

        def v(x, t):
            return self.backbone(x, torch.full((B,), t, device=x.device, dtype=x.dtype), c)

        for i in range(steps):
            t = i * dt
            if solver == "euler":
                x = x + dt * v(x, t)
            elif solver == "midpoint":
                x = x + dt * v(x + 0.5 * dt * v(x, t), t + 0.5 * dt)
            elif solver == "heun":
                k1 = v(x, t)
                x = x + 0.5 * dt * (k1 + v(x + dt * k1, t + dt))
            else:
                raise ValueError("solver must be euler|midpoint|heun")
        return self.denorm_lat(x)


def build_fm(fm_cfg, info):
    """fm_cfg = config["fm"]; info = autoencoder.latent_info(model_cfg)."""
    kw = dict(fm_cfg.get("backbone_args", {}))
    bb = BACKBONES.build(fm_cfg.get("backbone", "conv_film"), info["l1_shape"][0], info["grid"], info["cond_dim"], **kw)
    return FlowMatcher(bb, info["l1_shape"], info["cond_dim"], fm_cfg.get("t_sampling", "uniform"))
