"""DelayCAST-Net.

                 per region r (shared weights unless stated)                             heads
 selected      ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────┐
 neurons  ───► │NeuronGate │─►│ read-in    │─►│ dilated  │─►│ causal    │─►│attn  │─┐
 (K x T)       │(session)  │  │ 1x1 conv   │  │ causal   │  │ temporal  │  │pool  │ │   ┌──────────────┐   ┌─► 3-class logits
               └───────────┘  │(session)   │  │ TCN      │  │ attention │  └──────┘ ├──►│ cross-region │───┤
 STFT band ───────────────────►│            │  └──────────┘  └───────────┘           │   │ attention    │   └─► response-epoch
 power (B x T)                └────────────┘                                        ┘   └──────────────┘       rates (K x T_tgt)
                                                                                                                 (session read-out)

Session-specific parameters (read-in, gates, read-out) adapt to the unit identities of each
session/dataset while the temporal backbone and the attention blocks are shared across all
sessions of both datasets, which is what allows joint training on ``Data`` and ``Data2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import CLASSES, REGIONS
from .attention import AttentionPool, CausalTemporalAttention, CrossRegionAttention, NeuronGate
from .tcn import DilatedCausalTCN


@dataclass
class ModelOutput:
    logits: torch.Tensor                       # (B, 3)
    forecast_log_rate: dict[str, torch.Tensor]  # region -> (B, K, T_tgt)
    temporal_attn: dict[str, torch.Tensor]      # region -> (B, T) pooling weights over delay bins
    self_attn: dict[str, torch.Tensor]          # region -> (B, T, T) causal attention maps
    region_attn: torch.Tensor                   # (B, R) cross-region pooling weights
    gates: dict[str, torch.Tensor]              # region -> (K,) neuron gates of the batch's session
    gate_l1: torch.Tensor                       # mean over regions of the summed gate mass


class NormalizedReadIn(nn.Module):
    """1x1 convolution whose weight column for every input channel has unit L2 norm.

    Without this the read-in could absorb any rescaling of the neuron gates, which would make the
    gates meaningless as importance scores; with it the gate is the only per-neuron scale factor.
    """

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, c_in) / math.sqrt(c_in))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        w = self.weight / (self.weight.norm(dim=0, keepdim=True) + 1e-6)
        return F.conv1d(x, (self.scale * w)[:, :, None], self.bias)


class SessionAdapter(nn.Module):
    """Session-specific read-in / gates / read-out for the four regions."""

    def __init__(self, k: int, n_spec: int, d_model: int, t_tgt: int):
        super().__init__()
        self.gates = nn.ModuleDict({r: NeuronGate(k, init=1.0) for r in REGIONS})
        self.read_in = nn.ModuleDict({r: NormalizedReadIn(k + n_spec, d_model) for r in REGIONS})
        self.read_out = nn.ModuleDict({r: nn.Linear(d_model, k) for r in REGIONS})
        self.log_base = nn.ParameterDict({r: nn.Parameter(torch.zeros(k)) for r in REGIONS})
        # Persistence path: weight of each neuron's own late-delay log-rate in its response forecast.
        self.persist = nn.ParameterDict({r: nn.Parameter(torch.zeros(k)) for r in REGIONS})


class DelayCASTNet(nn.Module):
    def __init__(self, sessions: list[str], k: int, t_ctx: int, t_tgt: int, n_spec: int, cfg):
        super().__init__()
        m = cfg.model
        d = int(m.d_model)
        self.k, self.t_ctx, self.t_tgt, self.n_spec = k, t_ctx, t_tgt, n_spec
        self.late_bins = int(m.get_path("persistence_late_bins", 20))
        self.count_scale = float(cfg.data.target_bin_ms) / float(cfg.data.bin_ms)  # ctx-bin counts -> target-bin counts
        self.use_spec = bool(m.use_spectral_branch) and n_spec > 0
        n_spec_in = n_spec if self.use_spec else 0
        self.adapters = nn.ModuleDict({_key(s): SessionAdapter(k, n_spec_in, d, t_tgt) for s in sessions})
        self.tcn = DilatedCausalTCN(d, d, int(m.tcn_blocks), int(m.kernel_size), float(m.dropout))
        self.temporal_attn = CausalTemporalAttention(d, int(m.n_heads), float(m.dropout))
        self.pool = AttentionPool(d)
        self.cross = CrossRegionAttention(d, int(m.n_heads), float(m.dropout), len(REGIONS))
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(float(m.dropout)), nn.Linear(d, len(CLASSES)))
        # Forecast decoder: fused + region token -> D x T_tgt latent trajectory (shared), then a
        # session-specific linear read-out to the K selected neurons of that region.
        self.time_embed = nn.Parameter(torch.randn(t_tgt, d) * 0.02)
        self.dec_in = nn.Linear(2 * d, d)
        self.decoder = DilatedCausalTCN(d, d, 3, int(m.kernel_size), float(m.dropout))
        self.gate_l1 = float(m.neuron_gate_l1)

    def add_session(self, session: str) -> None:
        if _key(session) not in self.adapters:
            d = self.dec_in.out_features
            self.adapters[_key(session)] = SessionAdapter(self.k, self.n_spec if self.use_spec else 0, d, self.t_tgt)

    def adapter_parameters(self, session: str):
        return self.adapters[_key(session)].parameters()

    def backbone_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("adapters."):
                yield p

    def forward(self, x: dict[str, torch.Tensor], spec: dict[str, torch.Tensor] | None, session: str,
                pad_mask: torch.Tensor | None = None) -> ModelOutput:
        """x[r]: (B, K, T) spike counts of selected neurons; spec[r]: (B, n_spec, T); pad_mask: (B, T)."""
        ad: SessionAdapter = self.adapters[_key(session)]
        tokens, tattn, sattn, gates, late_log = [], {}, {}, {}, {}
        l1 = torch.zeros((), device=next(self.parameters()).device)
        for r in REGIONS:
            # Late-delay mean count per neuron (masked mean when the context is truncated).
            valid = torch.ones_like(x[r][:, :1, :]) if pad_mask is None else (~pad_mask)[:, None, :].to(x[r].dtype)
            lb = self.late_bins
            late = (x[r][:, :, -lb:] * valid[:, :, -lb:]).sum(-1) / valid[:, :, -lb:].sum(-1).clamp(min=1)
            late_log[r] = torch.log(late * self.count_scale + 0.05)  # (B, K)
            xr = ad.gates[r](torch.sqrt(x[r] + 0.0))  # variance-stabilising sqrt on counts
            if self.use_spec and spec is not None:
                xr = torch.cat([xr, torch.log1p(spec[r])], dim=1)
            if pad_mask is not None:
                xr = xr.masked_fill(pad_mask[:, None, :], 0.0)
            h = self.tcn(ad.read_in[r](xr))                 # (B, D, T)
            h = h.transpose(1, 2)                            # (B, T, D)
            h, w_self = self.temporal_attn(h, pad_mask)
            pooled, w_pool = self.pool(h, pad_mask)
            tokens.append(pooled)
            tattn[r], sattn[r], gates[r] = w_pool, w_self, ad.gates[r].gates()
            l1 = l1 + ad.gates[r].l1()
        tokens = torch.stack(tokens, dim=1)                  # (B, R, D)
        fused, region_tokens, _, w_region = self.cross(tokens)
        logits = self.classifier(fused)
        forecast = {}
        for i, r in enumerate(REGIONS):
            z = torch.cat([fused, region_tokens[:, i]], dim=-1)           # (B, 2D)
            z = self.dec_in(z)[:, None, :] + self.time_embed[None]        # (B, T_tgt, D)
            z = self.decoder(z.transpose(1, 2)).transpose(1, 2)          # (B, T_tgt, D)
            persist = (ad.persist[r][None] * late_log[r])[:, None, :]              # (B, 1, K)
            forecast[r] = (ad.read_out[r](z) + ad.log_base[r] + persist).transpose(1, 2)  # (B, K, T_tgt) log-rate
        return ModelOutput(logits, forecast, tattn, sattn, w_region, gates, l1 / len(REGIONS))


def _key(session: str) -> str:
    return session.replace("/", "__").replace(".", "_")


def poisson_nll(log_rate: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Poisson negative log-likelihood per bin, averaged over valid (non-padded) neurons."""
    nll = torch.exp(log_rate) - counts * log_rate
    if mask is not None:  # mask: (B, K) True for real neurons
        nll = nll * mask[:, :, None]
        return nll.sum() / (mask.sum() * log_rate.shape[-1]).clamp(min=1)
    return nll.mean()


def poisson_deviance(mu: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Summed Poisson deviance 2*sum(y log(y/mu) - (y - mu)) over valid neurons."""
    eps = 1e-8
    dev = 2 * (counts * torch.log((counts + eps) / (mu + eps)) - (counts - mu))
    if mask is not None:
        dev = dev * mask[:, :, None].to(dev.dtype)
    return dev.sum()


def poisson_deviance_explained(log_rate: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor | None = None,
                               null_mu: torch.Tensor | None = None) -> torch.Tensor:
    """1 - D(model)/D(null). ``null_mu`` (K, T_tgt) is the per-neuron mean PSTH from *training* trials;
    falls back to the in-batch per-neuron mean when not given."""
    if null_mu is None:
        null_mu = counts.mean(dim=(0, 2), keepdim=True).expand_as(counts)
    else:
        null_mu = null_mu[None].expand_as(counts)
    dev = poisson_deviance(torch.exp(log_rate), counts, mask)
    dev0 = poisson_deviance(null_mu, counts, mask)
    return 1 - dev / dev0.clamp(min=1e-8)
