"""Hierarchical autoencoder  E1 -> E2 -> D1 -> D2  (see idea.md).

  E1: image (B,2,S,S)          -> latent1 (B,C1,S/2^n1,S/2^n1)   [stride-2 stages, n1 = len(e1_widths)]
  E2: latent1                  -> latent2 (B,C2,1,1)             [stride-2 stages, n2 = len(e2_widths)]
  D1: latent2                  -> latent1_hat                    [x2 upsampling stages]
  D2: latent1                  -> image (B,2,S,S)

Latents are linear (no activation after the last conv of E1/E2/D1). Each of the four parts is an
independent nn.Module, so they can be swapped / frozen / re-initialised separately.
"""
import torch.nn as nn

from .arch import ARCHS


def _stage(ops, first, cin, cout, last):
    layers = [first(cin, cout), ops.norm(cout), ops.act(), ops.conv(cout, cout, 3, 1)]
    if not last:
        layers += [ops.norm(cout), ops.act()]
    return nn.Sequential(*layers)


def _down(ops, cin, cout, last):
    return _stage(ops, lambda i, o: ops.conv(i, o, 3, 2), cin, cout, last)


def _up(ops, cin, cout, last):
    return _stage(ops, ops.convT, cin, cout, last)


def latent_info(m):
    """Shapes implied by a model config (no model needed)."""
    S, n1, n2 = m["image_size"], len(m["e1_widths"]), len(m["e2_widths"])
    grid = S // 2 ** n1
    assert grid * 2 ** n1 == S, "image_size must be divisible by 2^len(e1_widths)"
    assert grid == 2 ** n2, ("latent1 grid %d must collapse to 1x1 with len(e2_widths)=%d stages (need %d)"
                             % (grid, n2, int(grid).bit_length() - 1))
    c1, c2 = m["e1_widths"][-1], m["e2_widths"][-1]
    return {"l1_shape": (c1, grid, grid), "l2_shape": (c2, 1, 1), "cond_dim": c2, "grid": grid}


def build_e1(ops, w):
    layers = [ops.stem(w[0]), ops.act()]
    prev = w[0]
    for i, c in enumerate(w):
        layers.append(_down(ops, prev, c, last=(i == len(w) - 1)))
        prev = c
    return nn.Sequential(*layers)


def build_e2(ops, c1, w):
    layers, prev = [], c1
    for i, c in enumerate(w):
        layers.append(_down(ops, prev, c, last=(i == len(w) - 1)))
        prev = c
    return nn.Sequential(*layers)


def build_d1(ops, e2_w, c1):
    widths = list(reversed(e2_w[:-1])) + [c1]
    layers, prev = [], e2_w[-1]
    for i, c in enumerate(widths):
        layers.append(_up(ops, prev, c, last=(i == len(widths) - 1)))
        prev = c
    return nn.Sequential(*layers)


def build_d2(ops, e1_w):
    widths = list(reversed(e1_w[:-1])) + [e1_w[0]]
    layers, prev = [], e1_w[-1]
    for c in widths:
        layers.append(_up(ops, prev, c, last=False))
        prev = c
    layers.append(ops.head(prev))
    return nn.Sequential(*layers)


class HierarchicalAE(nn.Module):
    def __init__(self, e1, e2, d1, d2):
        super().__init__()
        self.e1, self.e2, self.d1, self.d2 = e1, e2, d1, d2

    def encode(self, x):
        l1 = self.e1(x)
        return l1, self.e2(l1)

    def forward(self, x):
        l1, l2 = self.encode(x)
        l1_hat = self.d1(l2)
        return {"recon": self.d2(l1_hat), "l1": l1, "l2": l2, "l1_hat": l1_hat}


def make_ops(m):
    return ARCHS.build(m["arch"], act=m.get("act", "relu"), norm=m.get("norm", "none"))


def build_autoencoder(m):
    """m = config["model"]:  arch, image_size, e1_widths, e2_widths, act, norm."""
    info = latent_info(m)
    ops = make_ops(m)
    c1 = info["l1_shape"][0]
    return HierarchicalAE(build_e1(ops, m["e1_widths"]), build_e2(ops, c1, m["e2_widths"]),
                          build_d1(ops, m["e2_widths"], c1), build_d2(ops, m["e1_widths"]))
