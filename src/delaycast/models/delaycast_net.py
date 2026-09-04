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

import numpy as np
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
    logits_backbone: torch.Tensor | None = None  # (B, 3) contribution of the TCN/Transformer path (None when off)
    logits_skip: torch.Tensor | None = None      # (B, 3) contribution of the linear count read-out (None when off)


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
    """Causal filterbank: band power of a population rate trace from *past bins only*.

    For every band a Hann-windowed cosine/sine pair at the band's geometric centre frequency is convolved
    with left padding only (output at bin t sees bins t-nper+1 .. t). A causal running mean over the same
    window is subtracted first (masked so that truncated contexts do not leak zeros into the mean).

    ``mode='bands'``: the Gabor kernels are fixed (an STFT with a causal Hann window, one centre frequency per
    band).  ``mode='learned'``: the same kernels are the *initialisation* of a learned causal filterbank - every
    quadrature pair is a free parameter, so the branch can move its centre frequencies, bandwidths and shapes to
    whatever spectro-temporal feature of the population rate carries information about the upcoming action;
    it can only match or improve on the fixed bank in training, and stays causal by construction (left padding).
    ``mode='popmean'`` returns just the causal running mean (one channel) - the matched control that has
    the same window and the same gating but no spectral information.
    """

    def __init__(self, bin_ms: float, bands: dict[str, list[float]], win_ms: float = 300.0, mode: str = "bands"):
        super().__init__()
        if mode not in ("bands", "learned", "popmean"):
            raise ValueError(f"spectral branch mode must be bands | learned | popmean, got {mode!r}")
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
        kern = torch.stack(kernels)[:, None, :]                                     # (2*n_bands, 1, nper)
        if mode == "learned":
            self.kernel = nn.Parameter(kern)
        else:
            self.register_buffer("kernel", kern)
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


class CountReadout(nn.Module):
    """The *wide* path of the classifier: a linear read-out of ``n_feat`` mean sqrt-count features per selected
    neuron - over the visible context, over its last ``late_bins`` (the two features of the linear baseline) and
    over ``n_feat - 2`` equal windows of the context (a time-resolved linear decoder: which part of the delay
    carries the side is then a learned weight, not an assumption) - standardised with statistics of the training
    trials (buffers set by ``DelayCASTNet.fit_count_stats``) and multiplied by the neuron gates.  A feature whose
    window holds no visible bin (context sweep, window occlusion) is zeroed *after* standardisation, so an
    invisible window contributes nothing rather than "the mean".  Its logits are *added* to the backbone
    classifier's, so the network contains the tuned linear decoder as a special case and the TCN/Transformer path
    only has to learn what a linear read-out of windowed rates cannot express.  Zero-initialised: at the start the
    classifier is the backbone alone.
    """

    def __init__(self, k: int, n_classes: int, n_feat: int = 2):
        super().__init__()
        self.k, self.n_feat = k, n_feat
        self.lin = nn.Linear(n_feat * k, n_classes)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.register_buffer("mu", torch.zeros(n_feat * k))
        self.register_buffer("sd", torch.ones(n_feat * k))

    def forward(self, feats: torch.Tensor, fvalid: torch.Tensor, gates: torch.Tensor, nm: torch.Tensor | None) -> torch.Tensor:
        """feats, fvalid: (B, n_feat*K); gates: (K,); nm: (B, K) or None."""
        z = (feats - self.mu) / self.sd * fvalid
        z = z * gates.repeat(self.n_feat)[None, :]
        if nm is not None:
            z = z * nm.repeat(1, self.n_feat)
        return self.lin(z)


class UnitStats(nn.Module):
    """Per-unit training statistics of one region of one session: mean / sd of the sqrt-count per bin (input
    standardisation of the backbone) and the mean response-epoch count per target bin (forecast-loss scale)."""

    def __init__(self, k: int):
        super().__init__()
        self.register_buffer("mu", torch.zeros(k))
        self.register_buffer("sd", torch.ones(k))
        self.register_buffer("rate", torch.ones(k))


class SessionAdapter(nn.Module):
    """Session-specific read-in / gates / read-out for the four regions."""

    def __init__(self, k: int, n_spec: int, d_model: int, t_tgt: int, n_feat: int = 2):
        super().__init__()
        self.gates = nn.ModuleDict({r: NeuronGate(k, init=1.0) for r in REGIONS})
        self.read_in = nn.ModuleDict({r: NormalizedReadIn(k + n_spec, d_model) for r in REGIONS})
        self.skip = nn.ModuleDict({r: CountReadout(k, len(CLASSES), n_feat) for r in REGIONS})
        self.stats = nn.ModuleDict({r: UnitStats(k) for r in REGIONS})
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
        self.n_windows = max(0, int(m.get_path("skip_windows", 4)))                 # time windows of the wide path
        self.n_feat = 2 + self.n_windows
        self.standardize_input = bool(m.get_path("standardize_input", True))
        self.forecast_norm = str(cfg.train.get_path("forecast_norm", "mean_count"))
        self.forecast_norm_floor = float(cfg.train.get_path("forecast_norm_floor", 0.1))
        self.max_log_rate = float(m.get_path("max_log_rate", 5.5))           # log(255): the uint8 cache never holds more
        self.skip_init = str(m.get_path("skip_init", "logreg"))                # logreg (warm start) | zeros
        # ---- spectral branch (in-model, causal)
        branch = str(m.get_path("spectral_branch", "learned" if bool(m.get_path("use_spectral_branch", True)) else "none"))
        bands = {kk: list(v) for kk, v in cfg.selection.bands_hz.items()}
        self.spectral_branch = branch
        self.bandpower = CausalBandPower(float(cfg.data.bin_ms), bands, float(m.get_path("spectral_win_ms", 300.0)),
                                         mode=branch) if branch != "none" else None
        self.n_spec = self.bandpower.n_out if self.bandpower is not None else 0
        if n_spec is not None and n_spec != self.n_spec:
            raise ValueError(f"checkpoint has n_spec={n_spec} but config gives {self.n_spec} (spectral_branch={branch})")
        self.adapters = nn.ModuleDict({_key(s): SessionAdapter(k, self.n_spec, d, t_tgt, self.n_feat) for s in sessions})
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
        if self.skip_init == "logreg":
            # With the wide path warm-started at the tuned logistic regression (train.warm_start_skip), the deep head
            # starts at zero: at epoch 0 the classifier *is* the linear decoder and the deep path learns the residual.
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
        # Forecast decoder: fused + region token -> D x T_tgt latent trajectory (shared), then a
        # session-specific linear read-out to the K selected neurons of that region.
        self.time_embed = nn.Parameter(torch.randn(t_tgt, d) * 0.02)
        self.dec_in = nn.Linear(2 * d, d)
        self.decoder = DilatedCausalTCN(d, d, 3, int(m.kernel_size), float(m.dropout))
        self.gate_penalty_type = str(m.get_path("neuron_gate_penalty", "hoyer"))
        self.gate_weight = float(m.get_path("neuron_gate_weight", m.get_path("neuron_gate_l1", 0.01)))
        # ---- classifier paths: backbone (TCN -> Transformer -> pooling -> cross-region) and/or the linear count read-out
        self.linear_skip = bool(m.get_path("linear_skip", True))
        self.classifier_from_backbone = bool(m.get_path("classifier_from_backbone", True))
        if not (self.linear_skip or self.classifier_from_backbone):
            raise ValueError("model.linear_skip and model.classifier_from_backbone cannot both be false")

    # ------------------------------------------------------------------ count read-out statistics
    @staticmethod
    def count_features(x: torch.Tensor, valid: torch.Tensor, late_bins: int, n_windows: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """(features, validity), both (B, (2 + n_windows) K): masked mean sqrt-count of every unit over the visible
        context, over its last ``late_bins``, and over ``n_windows`` equal windows of the context; validity is 1
        where the window holds at least one visible bin."""
        sq = torch.sqrt(x)
        B, K, T = sq.shape
        feats, fval = [], []

        def window(sl: slice) -> None:
            v = valid[:, sl]
            n = v.sum(-1, keepdim=True)
            feats.append((sq[:, :, sl] * v[:, None, :]).sum(-1) / n.clamp(min=1))
            fval.append((n > 0).to(sq.dtype).expand(B, K))

        window(slice(0, T))
        window(slice(T - late_bins, T))
        edges = torch.linspace(0, T, n_windows + 1).round().long().tolist() if n_windows else []
        for a, b in zip(edges[:-1], edges[1:]):
            window(slice(int(a), max(int(b), int(a) + 1)))
        return torch.cat(feats, dim=1), torch.cat(fval, dim=1)

    def forecast_scale(self, session: str, r: str) -> torch.Tensor | None:
        """(K,) per-unit divisor of the Poisson forecast loss (training mean count per target bin, floored), or
        None when ``train.forecast_norm`` is ``none``.  With it the gradient of the forecast term is O(1) per unit
        whatever the count scale - single units or population channels - instead of growing with the rate."""
        if self.forecast_norm == "none":
            return None
        return self.adapters[_key(session)].stats[r].rate.clamp(min=self.forecast_norm_floor)

    @torch.no_grad()
    def fit_count_stats(self, session: str, x: dict[str, np.ndarray | torch.Tensor], y: dict | None = None) -> None:
        """Per-unit statistics from the given trials (training / adaptation trials only, never test): standardisation
        of the count read-out features and of the backbone input, and the mean response count for the forecast
        loss scale (``y``: response-epoch counts of the same trials)."""
        ad: SessionAdapter = self.adapters[_key(session)]
        for r in REGIONS:
            xr = torch.as_tensor(np.asarray(x[r]), dtype=torch.float32)
            if xr.ndim != 3 or xr.shape[0] == 0:
                continue
            valid = torch.ones(xr.shape[0], xr.shape[2])
            f, _ = self.count_features(xr, valid, self.late_bins, self.n_windows)
            mu, sd = f.mean(0), f.std(0, unbiased=False)
            sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
            ad.skip[r].mu.copy_(mu.to(ad.skip[r].mu.device))
            ad.skip[r].sd.copy_(sd.to(ad.skip[r].sd.device))
            sq = torch.sqrt(xr)
            imu, isd = sq.mean((0, 2)), sq.std((0, 2), unbiased=False)
            isd = torch.where(isd > 1e-6, isd, torch.ones_like(isd))
            st = ad.stats[r]
            st.mu.copy_(imu.to(st.mu.device))
            st.sd.copy_(isd.to(st.sd.device))
            if y is not None and r in y:
                yr = torch.as_tensor(np.asarray(y[r]), dtype=torch.float32)
                if yr.ndim == 3 and yr.shape[0]:
                    rate = yr.mean((0, 2))
                    st.rate.copy_(rate.to(st.rate.device))
                    # The forecast starts at each unit's training mean log-rate (the level of the PSTH null) and the
                    # decoder learns deviations; from log-rate 0 a channel with 30 counts per bin needed thousands of
                    # steps just to reach its scale, and the forecast stayed far below the null within early stopping.
                    ad.log_base[r].data.copy_(torch.log(rate + 0.05).to(ad.log_base[r].device))

    # ------------------------------------------------------------------ session handling
    def add_session(self, session: str) -> None:
        if _key(session) not in self.adapters:
            d = self.dec_in.out_features
            self.adapters[_key(session)] = SessionAdapter(self.k, self.n_spec, d, self.t_tgt, self.n_feat)

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
        skip_logits = torch.zeros(B, len(CLASSES), device=ref.device, dtype=ref.dtype)
        lb = self.late_bins
        for r in REGIONS:
            xr_raw = x[r]
            nm = None if neuron_mask is None else neuron_mask[r].to(ref.dtype)                      # (B, K)
            if drop_region == r:
                xr_raw = torch.zeros_like(xr_raw)
            if self.linear_skip:   # wide path: gated, standardised windowed mean counts of the visible context (drop = zero input)
                feats, fvalid = self.count_features(xr_raw, valid, lb, self.n_windows)
                skip_logits = skip_logits + ad.skip[r](feats, fvalid, ad.gates[r].gates(), nm)
            # Persistence input: masked mean count of the last ``lb`` valid bins.
            v_late = valid[:, None, -lb:]
            late = (xr_raw[:, :, -lb:] * v_late).sum(-1) / v_late.sum(-1).clamp(min=1)
            ll = torch.log(late * self.count_scale + 0.05)
            if late_log_override is not None and r in late_log_override:
                ll = late_log_override[r]
            late_log[r] = ll
            sq = torch.sqrt(xr_raw)                                                                  # variance-stabilised counts
            xr_pop = ad.gates[r](sq) * valid[:, None, :]                                             # gated, non-negative (population trace)
            if nm is not None:
                xr_pop = xr_pop * nm[:, :, None]
            if self.standardize_input:   # per-unit z-score with training statistics: every unit enters on the same scale
                st = ad.stats[r]
                sq = (sq - st.mu[None, :, None]) / st.sd[None, :, None]
            xr = ad.gates[r](sq) * valid[:, None, :]
            if nm is not None:
                xr = xr * nm[:, :, None]
            if self.bandpower is not None:
                denom = (nm.sum(1) if nm is not None else torch.full((B,), float(self.k), device=ref.device)).clamp(min=1)
                pop = xr_pop.sum(1) / denom[:, None]                                                 # (B, T) gated population rate
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
        logits_bb = self.classifier(fused) if self.classifier_from_backbone else None
        logits = (logits_bb if logits_bb is not None else 0) + (skip_logits if self.linear_skip else 0)
        forecast = {}
        for i, r in enumerate(REGIONS):
            z = torch.cat([fused, region_tokens[:, i]], dim=-1)           # (B, 2D)
            z = self.dec_in(z)[:, None, :] + self.time_embed[None]        # (B, T_tgt, D)
            z = self.decoder(z.transpose(1, 2)).transpose(1, 2)          # (B, T_tgt, D)
            persist = (ad.persist[r][None] * late_log[r])[:, None, :]              # (B, 1, K)
            # (B, K, T_tgt) log-rate, capped at log(255): no cached bin can hold more, and one runaway prediction
            # otherwise dominates the deviance of a whole region
            forecast[r] = (ad.read_out[r](z) + ad.log_base[r] + persist).transpose(1, 2).clamp(max=self.max_log_rate)
        return ModelOutput(logits, forecast, tattn, sattn, w_region, gates, penalty / len(REGIONS), spec_used,
                           logits_bb, skip_logits if self.linear_skip else None)


def _key(session: str) -> str:
    return session.replace("/", "__").replace(".", "_")


def poisson_nll(log_rate: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor | None = None,
                scale: torch.Tensor | None = None, max_log_rate: float = 7.0) -> torch.Tensor:
    """Poisson negative log-likelihood per bin, averaged over valid (non-padded) neurons.

    ``scale`` (K,) divides every neuron's NLL by its training mean count (see ``DelayCASTNet.forecast_scale``):
    the gradient (rate - count) / scale is then O(1) per neuron regardless of the count magnitude, so the forecast
    term neither swamps the classification loss for population channels (counts of tens per bin) nor vanishes for
    sparse single units.  The log-rate is clamped at ``max_log_rate`` (e^7 ~ 1100 counts per bin, far above any
    real value) so that one runaway prediction cannot blow the loss up."""
    lr = log_rate.clamp(max=max_log_rate)
    nll = torch.exp(lr) - counts * lr
    if scale is not None:
        nll = nll / scale.to(nll.dtype)[None, :, None]
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
