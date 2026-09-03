"""DelayCAST-Net.

                 per region r (shared weights unless stated)                                  heads
 selected      ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────┐
 neurons  ───► │NeuronGate │─►│ read-in    │─►│ dilated  │─►│ N x causal  │─►│attn  │─┐
 (K x T)       │(session)  │  │ 1x1 conv   │  │ causal   │  │ Transformer │  │pool  │ │   ┌──────────────┐   ┌─► 3-class logits
               └─────┬─────┘  │(session)   │  │ TCN      │  │ blocks (+   │  └──────┘ ├──►│ cross-region │───┤
   gated population  │        │            │  └──────────┘  │ time-to-go  │           │   │ attention    │   └─► response-epoch
   rate ─► causal    └───────►│            │                │ encoding)   │           ┘   └──────────────┘       rates (K x T_tgt)
   Gabor filterbank ─────────►└────────────┘                └─────────────┘                                        (session read-out)
   (band power, past bins only)

Every operation is causal within the delay: dilated convolutions with left padding only, per-time-step
channel normalisation, causally masked attention, and a spectral branch that is a fixed *causal* Gabor
filterbank applied inside the model to the gated population rate (so it is recomputed under any context
mask / occlusion and cannot bypass the neuron gates).  The representation at bin t is a function of bins
<= t only, which is what makes the context sweep and the temporal occlusion maps meaningful.

Session-specific parameters (read-in, gates, read-out) adapt to the unit identities of each
session/dataset while the temporal backbone and the attention blocks are shared across all sessions of
both datasets, which is what allows joint training on ``Data`` and ``Data2``.
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
    self_attn: dict[str, torch.Tensor]          # region -> (B, T, T) causal attention maps (last block)
    region_attn: torch.Tensor                   # (B, R) cross-region pooling weights
    gates: dict[str, torch.Tensor]              # region -> (K,) neuron gates of the batch's session
    gate_penalty: torch.Tensor                  # sparsity penalty (mean over regions)
    spec: dict[str, torch.Tensor]               # region -> (B, n_spec, T) causal band power actually used


class NormalizedReadIn(nn.Module):
    """1x1 convolution whose weight column for every input channel has unit L2 norm.

    Without this the read-in could absorb any rescaling of the neuron gates, which would make the
    gates meaningless as importance scores; with it the gate is the only per-neuron scale factor.
    (There is deliberately no free global scale either: the following per-step LayerNorm is
    scale-invariant, so a global scale would give the sparsity penalty a free direction.)
    """

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, c_in) / math.sqrt(c_in))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        w = self.weight / (self.weight.norm(dim=0, keepdim=True) + 1e-6)
        return F.conv1d(x, w[:, :, None], self.bias)


class CausalBandPower(nn.Module):
    """Fixed causal Gabor filterbank: band power of a population rate trace from *past bins only*.

    For every band a Hann-windowed cosine/sine pair at the band's geometric centre frequency is convolved
    with left padding only (output at bin t sees bins t-nper+1 .. t). A causal running mean over the same
    window is subtracted first (masked so that truncated contexts do not leak zeros into the mean).
    ``mode='popmean'`` returns just the causal running mean (one channel) - the matched control that has
    the same window and the same gating but no spectral information.
    """

    def __init__(self, bin_ms: float, bands: dict[str, list[float]], win_ms: float = 300.0, mode: str = "bands"):
        super().__init__()
        self.mode = mode
        dt = bin_ms / 1000.0
        self.nper = max(4, int(round(win_ms / bin_ms)))
        t = torch.arange(self.nper, dtype=torch.float32) * dt
        hann = torch.hann_window(self.nper, periodic=False)
        kernels = []
        self.band_names = list(bands)
        for lo, hi in bands.values():
            fc = math.sqrt(max(lo, 0.5) * hi)
            kernels.append(hann * torch.cos(2 * math.pi * fc * t))
            kernels.append(hann * torch.sin(2 * math.pi * fc * t))
        self.register_buffer("kernel", torch.stack(kernels)[:, None, :])          # (2*n_bands, 1, nper)
        self.register_buffer("mean_kernel", torch.ones(1, 1, self.nper) / self.nper)

    @property
    def n_out(self) -> int:
        return 1 if self.mode == "popmean" else len(self.band_names)

    def forward(self, pop: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """pop: (B, T) population rate; valid: (B, T) 1.0 for bins that are part of the context."""
        x = (pop * valid)[:, None]                                                  # (B, 1, T)
        v = valid[:, None]
        pad = (self.nper - 1, 0)
        mean = F.conv1d(F.pad(x, pad), self.mean_kernel) / F.conv1d(F.pad(v, pad), self.mean_kernel).clamp(min=1e-3)
        if self.mode == "popmean":
            return torch.log1p(mean * v)
        xc = (x - mean) * v
        z = F.conv1d(F.pad(xc, pad), self.kernel)                                   # (B, 2*n_bands, T)
        power = z[:, 0::2] ** 2 + z[:, 1::2] ** 2
        return torch.log1p(power) * v


def time_to_go_encoding(T: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal encoding of the number of bins remaining before the go cue (T-1-t).

    Encoding time-to-go rather than time-since-onset keeps the code meaningful when the context length
    changes (``pre_delay_ms`` / ``include_sample``) and is the quantity ALM preparatory activity is
    aligned to. Fixed (not learned) so that attention maps are not shaped by a learned position prior.
    """
    pos = torch.arange(T - 1, -1, -1, dtype=torch.float32)[:, None]               # T-1 ... 0
    i = torch.arange(0, d, 2, dtype=torch.float32)
    div = torch.exp(-math.log(max(T, 10) * 2.0) * i / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2 + (d % 2 == 1)][:, : pe[:, 1::2].shape[1]]
    return pe * 0.3


class SessionAdapter(nn.Module):
    """Session-specific read-in / gates / read-out for the four regions."""

    def __init__(self, k: int, n_spec: int, d_model: int, t_tgt: int):
        super().__init__()
        self.gates = nn.ModuleDict({r: NeuronGate(k, init=1.0) for r in REGIONS})
        self.read_in = nn.ModuleDict({r: NormalizedReadIn(k + n_spec, d_model) for r in REGIONS})
        self.read_out = nn.ModuleDict({r: nn.Linear(d_model, k) for r in REGIONS})
        self.log_base = nn.ParameterDict({r: nn.Parameter(torch.zeros(k)) for r in REGIONS})
        # Persistence path: weight of each neuron's own late-delay log-rate in its response forecast.  This
        # is an *ungated* per-neuron self-term by design (it models "no change"), so importance claims are
        # based on permutation occlusion of the raster, never on the gates alone.
        self.persist = nn.ParameterDict({r: nn.Parameter(torch.zeros(k)) for r in REGIONS})


class DelayCASTNet(nn.Module):
    def __init__(self, sessions: list[str], k: int, t_ctx: int, t_tgt: int, cfg, n_spec: int | None = None):
        super().__init__()
        m = cfg.model
        d = int(m.d_model)
        self.k, self.t_ctx, self.t_tgt = k, t_ctx, t_tgt
        self.late_bins = int(m.get_path("persistence_late_bins", 20))
        self.count_scale = float(cfg.data.target_bin_ms) / float(cfg.data.bin_ms)  # ctx-bin counts -> target-bin counts
        # ---- spectral branch (in-model, causal)
        branch = str(m.get_path("spectral_branch", "bands" if bool(m.get_path("use_spectral_branch", True)) else "none"))
        bands = {kk: list(v) for kk, v in cfg.selection.bands_hz.items()}
        self.spectral_branch = branch
        self.bandpower = CausalBandPower(float(cfg.data.bin_ms), bands, float(m.get_path("spectral_win_ms", 300.0)),
                                         mode="popmean" if branch == "popmean" else "bands") if branch != "none" else None
        self.n_spec = self.bandpower.n_out if self.bandpower is not None else 0
        if n_spec is not None and n_spec != self.n_spec:
            raise ValueError(f"checkpoint has n_spec={n_spec} but config gives {self.n_spec} (spectral_branch={branch})")
        self.adapters = nn.ModuleDict({_key(s): SessionAdapter(k, self.n_spec, d, t_tgt) for s in sessions})
        self.tcn = DilatedCausalTCN(d, d, int(m.tcn_blocks), int(m.kernel_size), float(m.dropout))
        # ---- position: fixed time-to-go encoding (or learned / none)
        self.pos_mode = str(m.get_path("positional_encoding", "sinusoidal"))
        if self.pos_mode == "learned":
            self.pos_embed = nn.Parameter(torch.randn(t_ctx, d) * 0.02)
        else:
            self.register_buffer("pos_embed", time_to_go_encoding(t_ctx, d) if self.pos_mode == "sinusoidal" else torch.zeros(t_ctx, d))
        n_layers = int(m.get_path("n_transformer_layers", 2))
        self.temporal_attn = nn.ModuleList([CausalTemporalAttention(d, int(m.n_heads), float(m.dropout)) for _ in range(max(n_layers, 1))])
        self.pool = AttentionPool(d)
        self.cross = CrossRegionAttention(d, int(m.n_heads), float(m.dropout), len(REGIONS))
        self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(float(m.dropout)), nn.Linear(d, len(CLASSES)))
        # Forecast decoder: fused + region token -> D x T_tgt latent trajectory (shared), then a
        # session-specific linear read-out to the K selected neurons of that region.
        self.time_embed = nn.Parameter(torch.randn(t_tgt, d) * 0.02)
        self.dec_in = nn.Linear(2 * d, d)
        self.decoder = DilatedCausalTCN(d, d, 3, int(m.kernel_size), float(m.dropout))
        self.gate_penalty_type = str(m.get_path("neuron_gate_penalty", "hoyer"))
        self.gate_weight = float(m.get_path("neuron_gate_weight", m.get_path("neuron_gate_l1", 0.01)))

    # ------------------------------------------------------------------ session handling
    def add_session(self, session: str) -> None:
        if _key(session) not in self.adapters:
            d = self.dec_in.out_features
            self.adapters[_key(session)] = SessionAdapter(self.k, self.n_spec, d, self.t_tgt)

    def adapter_parameters(self, session: str, include_gates: bool = True):
        for name, p in self.adapters[_key(session)].named_parameters():
            if include_gates or not name.startswith("gates."):
                yield p

    def backbone_parameters(self):
        for name, p in self.named_parameters():
            if not name.startswith("adapters."):
                yield p

    # ------------------------------------------------------------------ forward
    def forward(self, x: dict[str, torch.Tensor], session: str, pad_mask: torch.Tensor | None = None,
                neuron_mask: dict[str, torch.Tensor] | None = None, drop_region: str | None = None,
                late_log_override: dict[str, torch.Tensor] | None = None) -> ModelOutput:
        """x[r]: (B, K, T) spike counts of the selected neurons.

        pad_mask: (B, T) True for context bins that must be ignored (truncated context / window occlusion).
        neuron_mask[r]: (B, K) True for real (non-padded) neurons.  drop_region: region whose input is
        removed (region dropout during training; in-distribution region ablation at test time).
        late_log_override[r]: (B, K) replaces the persistence input (temporal-occlusion analysis of the
        backbone alone).
        """
        ad: SessionAdapter = self.adapters[_key(session)]
        ref = x[REGIONS[0]]
        B, _, T = ref.shape
        valid = torch.ones(B, T, device=ref.device, dtype=ref.dtype) if pad_mask is None else (~pad_mask).to(ref.dtype)
        tokens, tattn, sattn, gates, late_log, spec_used = [], {}, {}, {}, {}, {}
        penalty = torch.zeros((), device=ref.device)
        lb = self.late_bins
        for r in REGIONS:
            xr_raw = x[r]
            nm = None if neuron_mask is None else neuron_mask[r].to(ref.dtype)                      # (B, K)
            if drop_region == r:
                xr_raw = torch.zeros_like(xr_raw)
            # Persistence input: masked mean count of the last ``lb`` valid bins.
            v_late = valid[:, None, -lb:]
            late = (xr_raw[:, :, -lb:] * v_late).sum(-1) / v_late.sum(-1).clamp(min=1)
            ll = torch.log(late * self.count_scale + 0.05)
            if late_log_override is not None and r in late_log_override:
                ll = late_log_override[r]
            late_log[r] = ll
            xr = ad.gates[r](torch.sqrt(xr_raw)) * valid[:, None, :]                                # gated, variance-stabilised
            if nm is not None:
                xr = xr * nm[:, :, None]
            if self.bandpower is not None:
                denom = (nm.sum(1) if nm is not None else torch.full((B,), float(self.k), device=ref.device)).clamp(min=1)
                pop = xr.sum(1) / denom[:, None]                                                     # (B, T) gated population
                sp = self.bandpower(pop, valid)                                                      # (B, n_spec, T), causal
                spec_used[r] = sp
                xr = torch.cat([xr, sp], dim=1)
            h = self.tcn(ad.read_in[r](xr))                 # (B, D, T)
            h = h.transpose(1, 2) + self.pos_embed[None, :T]  # (B, T, D)
            w_self = None
            for blk in self.temporal_attn:                  # causal Transformer stack; last block's map is exported
                h, w_self = blk(h, pad_mask)
            pooled, w_pool = self.pool(h, pad_mask)
            tokens.append(pooled)
            tattn[r], sattn[r], gates[r] = w_pool, w_self, ad.gates[r].gates()
            gm = None if nm is None else nm[0].bool()
            penalty = penalty + (ad.gates[r].hoyer(gm) if self.gate_penalty_type == "hoyer" else ad.gates[r].l1())
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
        return ModelOutput(logits, forecast, tattn, sattn, w_region, gates, penalty / len(REGIONS), spec_used)


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
