"""Attention blocks: learned neuron gates, causal temporal self-attention, attention pooling and
cross-region fusion. Every block returns its attention weights so they can be visualised."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronGate(nn.Module):
    """Per-neuron multiplicative gate g = sigmoid(theta) in (0, 1).

    Gates are learned with an L1 penalty so that the model is pushed to *switch off* neurons that
    do not help the objectives. The learned gate values provide a model-based ranking of neuron
    importance that can be compared with the statistical selection criteria.
    """

    def __init__(self, n_neurons: int, init: float = 2.0):
        super().__init__()
        self.theta = nn.Parameter(torch.full((n_neurons,), init))

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, T)
        return x * self.gates()[None, :, None]

    def l1(self) -> torch.Tensor:
        """Summed gate mass (so the pressure per neuron does not vanish as K grows)."""
        return self.gates().sum()

    def hoyer(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Scale-invariant sparsity penalty 1 - Hoyer(g) in [0, 1] (0 = one-hot, 1 = uniform).

        Hoyer(g) = (sqrt(K) - ||g||_1 / ||g||_2) / (sqrt(K) - 1). Because it is invariant to a global
        rescaling of the gates, the network cannot lower it by shrinking every gate and growing the
        downstream weights - the degeneracy that makes a plain L1 penalty uninformative after normalisation.
        """
        g = self.gates()
        if mask is not None:
            g = g[mask]
        k = g.numel()
        if k < 2:
            return torch.zeros((), device=g.device)
        ratio = g.sum() / (g.norm() + 1e-8)
        return 1.0 - (k ** 0.5 - ratio) / (k ** 0.5 - 1.0)


class CausalTemporalAttention(nn.Module):
    """Multi-head self-attention with a causal mask over the time axis (pre-norm, residual)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # attention-probability dropout off (an extra Bernoulli draw per layer); the block's output dropout stays
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self._causal_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _causal(self, T: int, device) -> torch.Tensor:
        key = (T, str(device))
        if key not in self._causal_cache:
            self._causal_cache[key] = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return self._causal_cache[key]

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """h: (B, T, D). pad_mask: (B, T) True where the time step should be ignored (context truncation)."""
        B, T, _ = h.shape
        causal = self._causal(T, h.device)
        if pad_mask is None:
            mask = causal
        else:
            # Combine causal + padding masks per sample; keep the diagonal open so that fully padded
            # rows (early bins of a truncated context) never see an all -inf softmax.
            mask = causal[None] | pad_mask[:, None, :]
            eye = torch.eye(T, dtype=torch.bool, device=h.device)[None]
            mask = mask & ~eye
            mask = mask.repeat_interleave(self.attn.num_heads, dim=0)  # (B*heads, T, T)
        x = self.norm(h)
        a, w = self.attn(x, x, x, attn_mask=mask, need_weights=True, average_attn_weights=True)
        h = h + self.drop(a)
        h = h + self.ff(h)
        return h, w  # w: (B, T, T)


class AttentionPool(nn.Module):
    """Learned-query attention pooling over time: which past bins are used for the read-out."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.key = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor | None = None):
        # h: (B, T, D) -> pooled (B, D), weights (B, T)
        scores = self.key(h) @ self.query / math.sqrt(h.shape[-1])
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        return torch.einsum("bt,btd->bd", w, h), w


class CrossRegionAttention(nn.Module):
    """Self-attention across the four region tokens, then a learned-query pooling to one vector."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, n_regions: int = 4):
        super().__init__()
        self.region_embed = nn.Parameter(torch.randn(n_regions, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pool = AttentionPool(d_model)

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, R, D)
        x = tokens + self.region_embed[None]
        a, w = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=True)
        x = x + a
        fused, rw = self.pool(x)
        return fused, x, w, rw  # fused (B, D), region tokens (B, R, D), (B, R, R), (B, R)
