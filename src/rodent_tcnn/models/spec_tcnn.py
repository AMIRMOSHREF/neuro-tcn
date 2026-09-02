"""SPEC-TCNN: Selective Predictive Epoch Context Temporal CNN.

Four region streams → dilated causal convolution (no future leak) →
neuron attention (which units) → temporal attention (which delay bins) →
cross-attention with wavelet/STFT features → dual heads:
  (1) predict lick-period rasters
  (2) classify Ignore / Left / Right
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class GatedDCCBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.filter_conv = DilatedCausalConv1d(channels, kernel_size, dilation)
        self.gate_conv = DilatedCausalConv1d(channels, kernel_size, dilation)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.drop(self.proj(h))
        return self.norm(x + h)


class RegionEncoder(nn.Module):
    def __init__(self, n_units: int, d_model: int, n_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Conv1d(n_units, d_model, 1)
        self.blocks = nn.ModuleList(
            [GatedDCCBlock(d_model, kernel_size, dilation=2**i, dropout=dropout) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, N, T  →  B, D, T
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return h


class SpectralEncoder(nn.Module):
    def __init__(self, n_freq: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * n_freq, d_model)

    def forward(self, tf: torch.Tensor) -> torch.Tensor:
        # tf: B, 4, N, F, T  → mean over units → B, 4, D, T
        b, r, n, f, t = tf.shape
        x = tf.mean(dim=2).reshape(b * r, 1, f, t)
        h = self.net(x)  # B*R, 32, F, T
        h = h.permute(0, 3, 1, 2).reshape(b * r, t, -1)
        h = self.proj(h)  # B*R, T, D
        return h.reshape(b, r, t, -1).permute(0, 1, 3, 2)  # B, 4, D, T


class NeuronAttention(nn.Module):
    """Soft selection over units using delay-pooled features."""

    def __init__(self, n_units: int, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.unit_proj = nn.Conv1d(n_units, d_model, 1)

    def forward(self, raster: torch.Tensor, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # raster: B, N, T ; encoded: B, D, T
        pooled = encoded.mean(dim=-1)
        raw = self.score(pooled)
        # also score from per-unit mean rate projected
        unit_feat = self.unit_proj(raster).mean(dim=-1)  # B, D
        raw = raw + 0.35 * self.score(unit_feat)
        weights = torch.softmax(raw, dim=0) if raw.dim() == 1 else torch.softmax(raw.squeeze(-1), dim=-1)
        # weights broadcast: we want B, N — use a learned map from D to N via raster energy
        energy = raster.mean(dim=-1)  # B, N
        mix = torch.softmax(energy + 0.25 * raw.expand(-1, energy.size(-1))[:, :1] * 0.0 + energy, dim=-1)
        # cleaner: score each unit from its time-averaged activity + encoded context
        return mix, encoded


class PerUnitNeuronAttention(nn.Module):
    def __init__(self, n_units: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, raster: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        # raster B,N,T ; encoded B,D,T
        rate = raster.mean(dim=-1, keepdim=True)
        peak = raster.max(dim=-1, keepdim=True).values
        feat = torch.cat([rate, peak], dim=-1)
        logits = self.net(feat).squeeze(-1)  # B, N
        return torch.softmax(logits, dim=-1)


class TemporalAttention(nn.Module):
    """Causal attention over delay bins: later bins query the past."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: B, D, T
        x = h.transpose(1, 2)  # B, T, D
        t = x.size(1)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        ctx, weights = self.attn(x, x, x, attn_mask=causal, need_weights=True, average_attn_weights=True)
        ctx = self.norm(x + ctx)
        return ctx, weights  # ctx: B,T,D ; weights: B,T,T


class SPECTCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.n_units = m.units_per_region
        self.d_model = m.d_model
        self.n_freq = m.n_freq
        self.encoders = nn.ModuleList(
            [RegionEncoder(m.units_per_region, m.d_model, m.n_dcc_layers, m.kernel_size, m.dropout) for _ in range(4)]
        )
        self.spec = SpectralEncoder(m.n_freq, m.d_model)
        self.neuron_attn = nn.ModuleList([PerUnitNeuronAttention(m.units_per_region, m.d_model) for _ in range(4)])
        self.temp_attn = TemporalAttention(m.d_model, m.n_heads, m.dropout)
        self.fuse = nn.Sequential(
            nn.Linear(m.d_model * 2, m.d_model),
            nn.GELU(),
            nn.Linear(m.d_model, m.d_model),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(m.d_model, m.d_model),
            nn.GELU(),
            nn.Linear(m.d_model, m.units_per_region),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(m.d_model * 4, m.d_model),
            nn.GELU(),
            nn.Dropout(m.dropout),
            nn.Linear(m.d_model, 3),
        )
        self.lick_len = cfg.epochs.lick_bins

    def _tf_or_zeros(self, delay: torch.Tensor, tf: torch.Tensor | None) -> torch.Tensor:
        if tf is not None:
            return tf
        b, r, n, t = delay.shape
        return delay.new_zeros(b, r, n, self.n_freq, t)

    def forward(self, delay: torch.Tensor, tf: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # delay: B, 4, N, T
        tf = self._tf_or_zeros(delay, tf)
        spec_h = self.spec(tf)  # B, 4, D, T
        region_ctx = []
        neuron_w = []
        temp_w = []
        for r in range(4):
            x = delay[:, r]  # B, N, T
            enc = self.encoders[r](x)  # B, D, T
            nw = self.neuron_attn[r](x, enc)  # B, N
            gated = torch.einsum("bn,bnt->bt", nw, x).unsqueeze(1)  # B,1,T
            # mix encoded stream with unit-gated trace
            mixed = enc + 0.25 * F.gelu(self.encoders[r].in_proj(x)) * nw.unsqueeze(-1).mean(dim=1, keepdim=True)
            # fuse spectral
            fused = self.fuse(torch.cat([mixed.transpose(1, 2), spec_h[:, r].transpose(1, 2)], dim=-1))
            fused = fused.transpose(1, 2)  # B, D, T
            ctx, tw = self.temp_attn(fused)
            region_ctx.append(ctx)
            neuron_w.append(nw)
            temp_w.append(tw)
            _ = gated
        # ctx: list of B,T,D
        pooled = torch.cat([c[:, -1] for c in region_ctx], dim=-1)  # last delay bin, causal
        # also attention-pool last 25% of delay
        tail = []
        for c in region_ctx:
            k = max(1, c.size(1) // 4)
            tail.append(c[:, -k:].mean(dim=1))
        pooled = 0.5 * pooled + 0.5 * torch.cat(tail, dim=-1)
        logits = self.cls_head(pooled)

        pred_regions = []
        for c in region_ctx:
            # interpolate last-context sequence to lick length
            seq = c.transpose(1, 2)  # B, D, T
            seq = F.interpolate(seq, size=self.lick_len, mode="linear", align_corners=False)
            pred = self.pred_head(seq.transpose(1, 2))  # B, T_lick, N
            pred_regions.append(F.softplus(pred.transpose(1, 2)))  # B, N, T_lick
        y_lick = torch.stack(pred_regions, dim=1)  # B, 4, N, T_lick

        return {
            "y_lick": y_lick,
            "logits": logits,
            "neuron_attn": torch.stack(neuron_w, dim=1),  # B, 4, N
            "temp_attn": torch.stack(temp_w, dim=1),  # B, 4, T, T
            "context": torch.stack([c[:, -1] for c in region_ctx], dim=1),
        }
