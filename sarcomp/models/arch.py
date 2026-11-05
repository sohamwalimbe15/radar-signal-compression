"""Architecture 'ops' = the only place that differs between the baseline and the complex model.

Everything is specified in REAL channel counts (so latent sizes mean the same thing for both):
the complex ops use real_channels // 2 complex channels internally.

To add an architecture: subclass Ops, decorate with @ARCHS.register("name").
"""
import torch.nn as nn

from ..utils import Registry
from .ops import ComplexConv2d, ComplexConvTranspose2d, ComplexGroupNorm, group_count, he_normal_

ARCHS = Registry("arch")

ACTS = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1), "silu": nn.SiLU, "gelu": nn.GELU}


class Ops:
    def __init__(self, act="relu", norm="none"):
        self.act_name, self.norm_name = act, norm

    def act(self):
        return ACTS[self.act_name]()

    # subclasses implement:
    def conv(self, cin, cout, k=3, s=1): raise NotImplementedError
    def convT(self, cin, cout): raise NotImplementedError          # exact x2 upsampling
    def norm(self, c): raise NotImplementedError
    def stem(self, cout): raise NotImplementedError                # (B,2,H,W) -> (B,cout,H,W)
    def head(self, cin): raise NotImplementedError                 # (B,cin,H,W) -> (B,2,H,W) [re, im]


def _he(layer, fan_in):
    he_normal_(layer.weight, fan_in)
    nn.init.zeros_(layer.bias)
    return layer


@ARCHS.register("baseline")
class BaselineOps(Ops):
    """Real-valued network. Stem = separate conv on Re and on Im, then concat (baseline_model.md)."""

    def conv(self, cin, cout, k=3, s=1):
        return _he(nn.Conv2d(cin, cout, k, s, (k - 1) // 2), cin * k * k)

    def convT(self, cin, cout):
        return _he(nn.ConvTranspose2d(cin, cout, 4, 2, 1), cin * 16 / 4.0)   # 4x4 kernel, stride 2

    def norm(self, c):
        return nn.Identity() if self.norm_name == "none" else nn.GroupNorm(group_count(c), c)

    def stem(self, cout):
        assert cout % 2 == 0
        return _BaselineStem(cout // 2)

    def head(self, cin):
        return _he(nn.Conv2d(cin, 2, 3, 1, 1), cin * 9)


class _BaselineStem(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_a = _he(nn.Conv2d(1, c, 3, 1, 1), 9)   # sees the real part only
        self.conv_b = _he(nn.Conv2d(1, c, 3, 1, 1), 9)   # sees the imaginary part only

    def forward(self, x):
        import torch
        return torch.cat([self.conv_a(x[:, 0:1]), self.conv_b(x[:, 1:2])], dim=1)


@ARCHS.register("complex")
class ComplexOps(Ops):
    """Complex conv / conv-transpose built from real kernels (complex_model.md) + split activation."""

    @staticmethod
    def _half(c):
        assert c % 2 == 0, "complex arch needs even real channel counts, got %d" % c
        return c // 2

    def conv(self, cin, cout, k=3, s=1):
        return ComplexConv2d(self._half(cin), self._half(cout), k, s, (k - 1) // 2)

    def convT(self, cin, cout):
        return ComplexConvTranspose2d(self._half(cin), self._half(cout), 4, 2, 1)

    def norm(self, c):
        return nn.Identity() if self.norm_name == "none" else ComplexGroupNorm(self._half(c))

    def stem(self, cout):
        return ComplexConv2d(1, self._half(cout), 3, 1, 1)   # 1 complex input channel

    def head(self, cin):
        return ComplexConv2d(self._half(cin), 1, 3, 1, 1)    # -> 1 complex channel = [re, im]
