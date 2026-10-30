"""
scratch_layers.py

Hand-rolled neural network building blocks, written directly from torch
tensor operations -- no nn.Conv2d, nn.Linear, nn.BatchNorm2d, or
nn.functional conv/pool helpers.

What "from scratch" means here, precisely: you control every multiply and
every parameter. torch.autograd still computes gradients for you --
writing your own backward pass / autodiff engine is a separate, much
larger project (that's the option you didn't pick). Every class below
still subclasses nn.Module, but only so that:
  - nn.Parameter tensors auto-register with the optimizer
  - .to(device) moves everything correctly
  - .train()/.eval() mode propagates to layers that need it (BatchNorm, Dropout)

Nothing here calls nn.Conv2d, F.conv2d, nn.Linear, nn.BatchNorm2d, or
nn.functional.relu/dropout. Padding uses F.pad only for zero-padding
input -- it has no learnable weights, so it isn't "a layer."

Rule: nothing outside car_damage.models.research should depend on this
module -- in particular, car_damage.models.production and
car_damage.training must not import from here.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dScratch(nn.Module):
    """
    2D convolution via im2col (unfold) + matmul.
    Mathematically identical to nn.Conv2d -- we extract each k x k patch
    the kernel would slide over, flatten it, and turn the whole convolution
    into one matrix multiply instead of calling a built-in operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # He (Kaiming) init -- critical with no pretrained weights. Wrong
        # init here means activations vanish or explode within a few layers.
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        N, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding

        if p > 0:
            x = F.pad(x, (p, p, p, p))  # zero-padding only, no learnable weights

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # im2col: slide a k x k window across the image and stack every
        # patch as a row, so convolution becomes one big matmul.
        patches = x.unfold(2, k, s).unfold(3, k, s)          # (N, C, H_out, W_out, k, k)
        patches = patches.contiguous().view(N, C, H_out * W_out, k * k)
        patches = patches.permute(0, 2, 1, 3).reshape(N, H_out * W_out, C * k * k)

        weight_flat = self.weight.view(self.weight.size(0), -1)  # (out_channels, C*k*k)

        out = patches @ weight_flat.t() + self.bias  # (N, H_out*W_out, out_channels)
        out = out.permute(0, 2, 1).view(N, -1, H_out, W_out)
        return out


class BatchNorm2dScratch(nn.Module):
    """
    Batch normalization, hand-implemented: normalize each channel to
    zero mean / unit variance across the batch, then apply a learnable
    scale (gamma) and shift (beta). Tracks running stats for eval mode
    the same way nn.BatchNorm2d does.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)


class ReLUScratch(nn.Module):
    """max(0, x) -- written directly, no nn.functional.relu."""
    def forward(self, x):
        return torch.clamp(x, min=0.0)


class MaxPool2dScratch(nn.Module):
    """Downsamples by taking the max value in each k x k window."""
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.k = kernel_size
        self.s = stride or kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.s
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        patches = x.unfold(2, k, s).unfold(3, k, s)  # (N, C, H_out, W_out, k, k)
        out, _ = patches.contiguous().view(N, C, H_out, W_out, -1).max(dim=-1)
        return out


class GlobalAvgPoolScratch(nn.Module):
    """Averages each channel's whole spatial map down to one number."""
    def forward(self, x):
        return x.mean(dim=(2, 3))


class LinearScratch(nn.Module):
    """Fully connected layer: y = xW^T + b, written directly."""
    def __init__(self, in_features, out_features):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class DropoutScratch(nn.Module):
    """
    Randomly zeroes activations during training to reduce overfitting,
    scaling survivors by 1/(1-p) so the expected sum stays the same
    (inverted dropout). No-op during eval.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float() / (1 - self.p)
        return x * mask
