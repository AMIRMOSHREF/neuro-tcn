"""Dilated causal convolutions (Temporal Convolutional Network, Bai et al. 2018 style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D convolution whose output at time t only depends on inputs <= t."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, T) tensor, applied independently at every time step.

    ``nn.GroupNorm(1, C)`` would normalise over (C, T) jointly, which makes every bin depend on every other
    bin (including the future) and silently breaks the causal guarantee of the TCN. This variant keeps the
    representation at bin t a function of bins <= t only.
    """

    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.weight, self.bias).transpose(1, 2)


class TemporalBlock(nn.Module):
    """Two causal dilated convs + GELU + channel dropout with a residual connection (pre-norm, per-step norm)."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm1 = ChannelLayerNorm(c_in)
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.norm2 = ChannelLayerNorm(c_out)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        # Channel-wise dropout (one mask per channel, shared over time): ~7x cheaper than element-wise
        # dropout on CPU, where the Bernoulli RNG dominated the training step.
        self.drop = nn.Dropout1d(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.gelu(self.conv1(self.norm1(x))))
        h = self.drop(F.gelu(self.conv2(self.norm2(h))))
        return h + self.skip(x)


class DilatedCausalTCN(nn.Module):
    """Stack of temporal blocks with exponentially growing dilation.

    Each block has two causal convolutions, so the receptive field is
    1 + 2 * (k - 1) * (2**n_blocks - 1) bins. With k=3 and 5 blocks that is 125 bins, i.e. the whole
    1.2 s delay at 10 ms resolution.
    """

    def __init__(self, c_in: int, d_model: int, n_blocks: int, kernel_size: int, dropout: float):
        super().__init__()
        blocks = []
        c = c_in
        for i in range(n_blocks):
            blocks.append(TemporalBlock(c, d_model, kernel_size, 2 ** i, dropout))
            c = d_model
        self.blocks = nn.Sequential(*blocks)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** n_blocks - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, D, T)
        return self.blocks(x)
