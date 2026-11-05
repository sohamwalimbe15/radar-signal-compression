"""Complex-valued layers on a *stacked real* layout.

A complex feature map with C complex channels is stored as a real tensor with 2C channels,
laid out [Re_0..Re_{C-1} | Im_0..Im_{C-1}]. Every module therefore has the plain
(B, channels, H, W) -> (B, channels', H', W') interface, so complex and real networks are
interchangeable in the rest of the code (latents, flow matching, ...).

Complex product (a + ib)(x + iy) = (ax - by) + i(bx + ay) is implemented as ONE real conv with the
block weight [[x, -y], [y, x]]. Split activations ReLU(a) + i ReLU(b) are just ordinary elementwise
activations on the stacked layout.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def he_normal_(w, fan_in):
    """Kaiming/He init for ReLU nets: std = sqrt(2 / fan_in). (PyTorch's default conv init is ~sqrt(1/3) of
    this per layer, which makes a deep no-norm net's gradients vanish.)"""
    nn.init.normal_(w, 0.0, math.sqrt(2.0 / fan_in))


class ComplexConv2d(nn.Module):
    """cin / cout are COMPLEX channel counts (real tensor has 2*cin -> 2*cout channels)."""

    def __init__(self, cin, cout, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.x = nn.Parameter(torch.empty(cout, cin, k, k))   # real part of the kernel
        self.y = nn.Parameter(torch.empty(cout, cin, k, k))   # imaginary part of the kernel
        fan_in = 2 * cin * k * k                               # each output sums over re+im inputs
        he_normal_(self.x, fan_in)
        he_normal_(self.y, fan_in)
        # free complex bias (re and im independent); NOT bias_x - bias_y
        self.bias = nn.Parameter(torch.zeros(2 * cout)) if bias else None

    def weight(self):
        x, y = self.x, self.y
        return torch.cat([torch.cat([x, -y], 1), torch.cat([y, x], 1)], 0)  # (2cout, 2cin, k, k)

    def forward(self, z):
        return F.conv2d(z, self.weight(), self.bias, self.stride, self.padding)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.stride, self.padding, self.output_padding = stride, padding, output_padding
        self.x = nn.Parameter(torch.empty(cin, cout, k, k))   # ConvTranspose weight layout (in, out, k, k)
        self.y = nn.Parameter(torch.empty(cin, cout, k, k))
        fan_in = 2 * cin * k * k / float(stride * stride)      # inputs that contribute to one output pixel
        he_normal_(self.x, fan_in)
        he_normal_(self.y, fan_in)
        self.bias = nn.Parameter(torch.zeros(2 * cout)) if bias else None

    def weight(self):
        x, y = self.x, self.y
        return torch.cat([torch.cat([x, y], 1), torch.cat([-y, x], 1)], 0)  # (2cin, 2cout, k, k)

    def forward(self, z):
        return F.conv_transpose2d(z, self.weight(), self.bias, self.stride, self.padding, self.output_padding)


def group_count(c, cap=32, min_size=4):
    """Largest divisor g <= cap of c with >= min_size channels per group (1 if none).
    min_size matters at 1x1 spatial size: a 1-channel group there normalises a single value -> exactly 0."""
    for g in range(min(cap, c // min_size if c >= min_size else 1), 0, -1):
        if c % g == 0:
            return g
    return 1


class ComplexGroupNorm(nn.Module):
    """GroupNorm on the stacked layout whose groups never mix Re and Im channels."""

    def __init__(self, c_complex):
        super().__init__()
        g = group_count(c_complex)
        self.gn = nn.GroupNorm(2 * g, 2 * c_complex)

    def forward(self, z):
        return self.gn(z)
