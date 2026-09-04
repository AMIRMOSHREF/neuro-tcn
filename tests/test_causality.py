"""Causality and gate-identifiability guarantees of DelayCAST-Net.

Every evaluation analysis that makes a *temporal* claim (context sweep, tau95, temporal occlusion,
attention centre of mass) is only meaningful if the representation at delay bin ``t`` is a function of
bins ``<= t``.  These tests perturb inputs at or after a bin and check that nothing before it moves - for
each component in isolation (dilated TCN with per-step channel normalisation, causal Transformer block,
causal Gabor filterbank) and for the assembled model under the prefix ``pad_mask`` used by the context
sweep.  A second group checks that the neuron gates are *identifiable*: the read-in has no free scale that
could absorb a gate rescaling and the Hoyer penalty is invariant to a global rescaling of the gates, so a
learned gate value is a per-neuron quantity and not an artefact of an arbitrary overall scale.

The model is small (d_model 32, K 8) and built from the default configuration; parameters are jittered
away from their initial values so that the checks are not trivially satisfied by zero-initialised paths
(e.g. the persistence weights start at 0, which would hide a leak through the late-delay input).
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from delaycast import REGIONS
from delaycast.config import load_config
from delaycast.models.attention import CausalTemporalAttention, NeuronGate
from delaycast.models.delaycast_net import CausalBandPower, DelayCASTNet, NormalizedReadIn
from delaycast.models.tcn import ChannelLayerNorm, DilatedCausalTCN

K, T_CTX, T_TGT, B = 8, 120, 30, 3
SESSION = "A/S1"
ATOL = 1e-5


# ----------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def cfg():
    return load_config(None, ["model.d_model=32"])


@pytest.fixture(scope="module")
def model(cfg) -> DelayCASTNet:
    torch.manual_seed(0)
    m = DelayCASTNet([SESSION], K, T_CTX, T_TGT, cfg)
    with torch.no_grad():
        for p in m.parameters():                      # move every parameter off its init (gates, persistence, ...)
            p.add_(0.3 * torch.randn_like(p))
    return m.eval()


@pytest.fixture(scope="module")
def x() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {r: torch.poisson(torch.full((B, K, T_CTX), 0.15)) for r in REGIONS}


def _clone(x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {r: v.clone() for r, v in x.items()}


def _assert_same_output(a, b) -> None:
    assert torch.allclose(a.logits, b.logits, atol=ATOL)
    for r in REGIONS:
        assert torch.allclose(a.forecast_log_rate[r], b.forecast_log_rate[r], atol=ATOL)
        assert torch.allclose(a.temporal_attn[r], b.temporal_attn[r], atol=ATOL)
        assert torch.allclose(a.spec[r], b.spec[r], atol=ATOL)
    assert torch.allclose(a.region_attn, b.region_attn, atol=ATOL)


# ----------------------------------------------------------------------------- (a) TCN + norm, Transformer
def test_channel_layer_norm_is_per_time_step():
    """The per-step norm touches only the perturbed bin; ``GroupNorm(1, C)`` would spread it over every bin."""
    torch.manual_seed(2)
    C, T = 16, 40
    x = torch.randn(2, C, T)
    norm = ChannelLayerNorm(C).eval()
    ref = norm(x)
    x2 = x.clone()
    x2[:, :, 25] += 5.0 * torch.randn(2, C)      # channel-varying (a constant offset would be removed by the norm itself)
    out = norm(x2)
    unchanged = torch.ones(T, dtype=torch.bool)
    unchanged[25] = False
    assert torch.allclose(out[:, :, unchanged], ref[:, :, unchanged], atol=ATOL)
    assert not torch.allclose(out[:, :, 25], ref[:, :, 25])
    # the anti-pattern the docstring warns about: GroupNorm over (C, T) leaks the perturbation into the past
    gn = nn.GroupNorm(1, C).eval()
    assert not torch.allclose(gn(x2)[:, :, :25], gn(x)[:, :, :25], atol=ATOL)


@pytest.mark.parametrize("t", [1, 17, 64, T_CTX - 1])
def test_dilated_tcn_is_causal(t):
    """Perturbing every bin >= t leaves the TCN output at bins < t untouched (and does move bins >= t)."""
    torch.manual_seed(3)
    tcn = DilatedCausalTCN(K + 3, 32, 5, 3, 0.15).eval()
    x = torch.randn(B, K + 3, T_CTX)
    ref = tcn(x)
    x2 = x.clone()
    x2[:, :, t:] += 3.0 * torch.randn_like(x2[:, :, t:])
    out = tcn(x2)
    assert torch.allclose(out[:, :, :t], ref[:, :, :t], atol=ATOL)
    assert not torch.allclose(out[:, :, t:], ref[:, :, t:], atol=ATOL)
    assert tcn.receptive_field == 1 + 2 * (3 - 1) * (2 ** 5 - 1) == 125


@pytest.mark.parametrize("t", [1, 40, 119])
def test_transformer_block_is_causal(model, t):
    """Causal self-attention: positions < t do not see a perturbation at positions >= t (each block and the stack)."""
    torch.manual_seed(4)
    d = model.pos_embed.shape[1]
    h = torch.randn(B, T_CTX, d)
    h2 = h.clone()
    h2[:, t:] += 3.0 * torch.randn_like(h2[:, t:])
    for blk in model.temporal_attn:
        assert isinstance(blk, CausalTemporalAttention)
        ref, _ = blk(h)
        out, _ = blk(h2)
        assert torch.allclose(out[:, :t], ref[:, :t], atol=ATOL)
        assert not torch.allclose(out[:, t:], ref[:, t:], atol=ATOL)
    # the whole stack (what the model actually runs) - the last block's map is what gets exported
    ref, out = h, h2
    for blk in model.temporal_attn:
        ref, _ = blk(ref)
        out, _ = blk(out)
    assert torch.allclose(out[:, :t], ref[:, :t], atol=ATOL)


# ----------------------------------------------------------------------------- (b) prefix pad_mask
@pytest.mark.parametrize("t0", [50, 110])
def test_prefix_pad_mask_hides_masked_bins(model, x, t0):
    """Context sweep guarantee: with the first ``t0`` bins masked, their content (spike counts *and* the band
    power computed from them inside the model) cannot influence logits or forecasts.  ``t0 = 110`` reaches
    into the 20-bin persistence window, which checks the masked mean of the late-delay input as well."""
    pm = torch.zeros(B, T_CTX, dtype=torch.bool)
    pm[:, :t0] = True
    with torch.no_grad():
        ref = model(x, SESSION, pad_mask=pm)
        x2 = _clone(x)
        for r in REGIONS:
            x2[r][:, :, :t0] = torch.poisson(torch.full_like(x2[r][:, :, :t0], 2.0))   # very different masked content
        out = model(x2, SESSION, pad_mask=pm)
        full = model(x2, SESSION)                                                    # without the mask it *does* matter
    _assert_same_output(ref, out)
    assert not torch.allclose(full.logits, ref.logits, atol=ATOL)
    # masked bins carry no attention-pool weight and no band power
    for r in REGIONS:
        assert float(ref.temporal_attn[r][:, :t0].abs().max()) == 0.0
        assert float(ref.spec[r][:, :, :t0].abs().max()) == 0.0


# ----------------------------------------------------------------------------- (c) causal band power
@pytest.mark.parametrize("mode", ["bands", "learned", "popmean"])
@pytest.mark.parametrize("t", [5, 60, 119])
def test_causal_band_power(cfg, mode, t):
    """Perturbing bin t leaves the filterbank output at bins < t unchanged, in both the spectral and the
    matched population-mean mode.  (The Hann taper is zero at its last tap, so in ``bands`` mode the first
    bin that can respond is t + 1 - the test therefore only requires *some* later bin to move.)"""
    bands = {k: list(v) for k, v in cfg.selection.bands_hz.items()}
    bp = CausalBandPower(float(cfg.data.bin_ms), bands, float(cfg.model.spectral_win_ms), mode=mode).eval()
    assert bp.n_out == (1 if mode == "popmean" else len(bands))
    torch.manual_seed(5)
    pop = torch.rand(B, T_CTX) * 2
    valid = torch.ones(B, T_CTX)
    ref = bp(pop, valid)
    pop2 = pop.clone()
    pop2[:, t] += 4.0
    out = bp(pop2, valid)
    assert out.shape == (B, bp.n_out, T_CTX)
    assert torch.allclose(out[:, :, :t], ref[:, :, :t], atol=ATOL)
    if t < T_CTX - 1:
        assert not torch.allclose(out[:, :, t:], ref[:, :, t:], atol=ATOL)


def test_causal_band_power_masked_prefix_is_ignored(cfg):
    """With a prefix of invalid bins, their values never leak into the running mean or the band power of the
    valid bins (the masked mean divides by the number of *valid* bins in the window, not by the window length)."""
    bands = {k: list(v) for k, v in cfg.selection.bands_hz.items()}
    torch.manual_seed(6)
    pop = torch.rand(B, T_CTX) * 2
    valid = torch.ones(B, T_CTX)
    valid[:, :40] = 0.0
    for mode in ("bands", "learned", "popmean"):
        bp = CausalBandPower(float(cfg.data.bin_ms), bands, float(cfg.model.spectral_win_ms), mode=mode).eval()
        ref = bp(pop, valid)
        pop2 = pop.clone()
        pop2[:, :40] = 50.0
        out = bp(pop2, valid)
        assert torch.allclose(out, ref, atol=ATOL)
        assert float(ref[:, :, :40].abs().max()) == 0.0


# ----------------------------------------------------------------------------- (d) attention maps
def test_self_attention_map_is_lower_triangular(model, x):
    """The exported (B, T, T) map has no weight on future keys and every row is a distribution."""
    with torch.no_grad():
        out = model(x, SESSION)
        pm = torch.zeros(B, T_CTX, dtype=torch.bool)
        pm[:, :30] = True
        out_pm = model(x, SESSION, pad_mask=pm)
    for r in REGIONS:
        w = out.self_attn[r]
        assert w.shape == (B, T_CTX, T_CTX)
        assert float(torch.triu(w, diagonal=1).abs().max()) == 0.0
        assert torch.allclose(w.sum(-1), torch.ones(B, T_CTX), atol=1e-5)
        w = out_pm.self_attn[r]
        assert float(torch.triu(w, diagonal=1).abs().max()) == 0.0
        assert float(w[:, 30:, :30].abs().max()) == 0.0      # valid queries never attend to masked keys


# ----------------------------------------------------------------------------- (e) region drop
def test_region_drop_equals_zero_input(model, x):
    """``drop_region`` (region dropout / in-distribution ablation) is exactly the model applied to a zeroed region."""
    with torch.no_grad():
        dropped = model(x, SESSION, drop_region="ALM_L")
        x0 = _clone(x)
        x0["ALM_L"].zero_()
        zeroed = model(x0, SESSION)
        intact = model(x, SESSION)
    _assert_same_output(dropped, zeroed)
    assert not torch.allclose(dropped.logits, intact.logits, atol=ATOL)


# ----------------------------------------------------------------------------- gate identifiability
def test_normalized_read_in_has_no_free_scale(model, x):
    """Every input column of the read-in has unit norm after normalisation, and there is no other scale
    parameter: rescaling the raw weight changes nothing, so the gate is the only per-neuron scale factor."""
    ri = NormalizedReadIn(K + 3, 32)
    assert sorted(n for n, _ in ri.named_parameters()) == ["bias", "weight"]
    torch.manual_seed(7)
    inp = torch.randn(B, K + 3, T_CTX)
    ref = ri(inp)
    with torch.no_grad():
        ri.weight.mul_(7.0)
    assert torch.allclose(ri(inp), ref, atol=ATOL)
    w_eff = ri.weight / (ri.weight.norm(dim=0, keepdim=True) + 1e-6)
    assert torch.allclose(w_eff.norm(dim=0), torch.ones(K + 3), atol=1e-4)
    # inside the assembled model: rescaling a session's read-in weights leaves logits and forecasts unchanged
    # (up to the 1e-6 epsilon of the normaliser, hence a relative tolerance rather than the exact ATOL)
    ad = model.adapters[SESSION.replace("/", "__")]
    assert all(isinstance(ad.read_in[r], NormalizedReadIn) for r in REGIONS)
    saved = {r: ad.read_in[r].weight.detach().clone() for r in REGIONS}
    try:
        with torch.no_grad():
            before = model(x, SESSION)
            for r in REGIONS:
                ad.read_in[r].weight.mul_(0.2)
            after = model(x, SESSION)
    finally:
        with torch.no_grad():
            for r in REGIONS:
                ad.read_in[r].weight.copy_(saved[r])
    assert torch.allclose(after.logits, before.logits, rtol=1e-4, atol=1e-4)
    for r in REGIONS:
        assert torch.allclose(after.forecast_log_rate[r], before.forecast_log_rate[r], rtol=1e-4, atol=1e-4)
        assert torch.allclose(after.temporal_attn[r], before.temporal_attn[r], rtol=1e-4, atol=1e-4)
    assert torch.allclose(after.region_attn, before.region_attn, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("scale", [0.1, 0.5, 3.0])
def test_hoyer_penalty_is_scale_invariant(scale):
    """1 - Hoyer(g) does not change when every gate is multiplied by the same constant (a plain L1 mass does),
    so the network cannot lower the penalty by shrinking all gates and growing downstream weights."""
    torch.manual_seed(8)
    gate = NeuronGate(K)
    with torch.no_grad():
        gate.theta.copy_(torch.randn(K) * 1.5)
    with torch.no_grad():
        g = gate.gates().clone()
        ref, ref_l1 = float(gate.hoyer()), float(gate.l1())
        mask = torch.tensor([True, True, False, True, True, True, False, True])
        ref_masked = float(gate.hoyer(mask))
        with mock.patch.object(gate, "gates", return_value=scale * g):
            assert float(gate.hoyer()) == pytest.approx(ref, abs=1e-6)
            assert float(gate.hoyer(mask)) == pytest.approx(ref_masked, abs=1e-6)
            assert float(gate.l1()) == pytest.approx(scale * ref_l1, rel=1e-6)
    assert 0.0 <= ref <= 1.0


def test_hoyer_penalty_extremes():
    """0 for a one-hot gate vector, 1 for a uniform one; fewer than two gated neurons give 0 (nothing to sparsify)."""
    gate = NeuronGate(K)
    with mock.patch.object(gate, "gates", return_value=torch.tensor([1.0] + [0.0] * (K - 1))):
        assert float(gate.hoyer()) == pytest.approx(0.0, abs=1e-6)
    with mock.patch.object(gate, "gates", return_value=torch.full((K,), 0.37)):
        assert float(gate.hoyer()) == pytest.approx(1.0, abs=1e-6)
    one = torch.zeros(K, dtype=torch.bool)
    one[0] = True
    assert float(gate.hoyer(one)) == 0.0


def test_model_output_shapes(model, x):
    """Contract of ``ModelOutput`` used by the evaluation code."""
    with torch.no_grad():
        out = model(x, SESSION)
    assert out.logits.shape == (B, 3)
    assert out.region_attn.shape == (B, len(REGIONS))
    assert out.gate_penalty.ndim == 0 and 0.0 <= float(out.gate_penalty) <= 1.0
    for r in REGIONS:
        assert out.forecast_log_rate[r].shape == (B, K, T_TGT)
        assert out.temporal_attn[r].shape == (B, T_CTX)
        assert out.gates[r].shape == (K,)
        assert out.spec[r].shape == (B, model.n_spec, T_CTX)
    assert np.isfinite(out.logits.numpy()).all()


# ----------------------------------------------------------------------------- linear count read-out (wide path)
def test_linear_count_readout_paths(cfg, x):
    """The classifier is backbone + linear count read-out; each path can be switched off, the read-out is
    standardised on the given trials, ignores masked bins and padded units, and a dropped region equals a zeroed
    input on both paths."""
    from delaycast.config import load_config
    torch.manual_seed(0)
    m = DelayCASTNet([SESSION], K, T_CTX, T_TGT, cfg).eval()
    assert m.linear_skip and m.classifier_from_backbone
    m.fit_count_stats(SESSION, {r: x[r].numpy() for r in REGIONS})
    ad = m.adapters[SESSION.replace("/", "__")]
    f, fv = m.count_features(x["ALM_L"], torch.ones(B, T_CTX), m.late_bins, m.n_windows)
    nf = m.n_feat * K
    assert f.shape == (B, nf) and fv.shape == (B, nf) and float(fv.min()) == 1.0 and m.n_feat == 2 + m.n_windows
    z = (f - ad.skip["ALM_L"].mu) / ad.skip["ALM_L"].sd
    assert torch.allclose(z.mean(0), torch.zeros(nf), atol=1e-4) and torch.allclose(z.std(0, unbiased=False), torch.ones(nf), atol=1e-3)
    # the window features tile the context: their bin-count-weighted mean equals the whole-context feature
    if m.n_windows:
        edges = torch.linspace(0, T_CTX, m.n_windows + 1).round().long().tolist()
        w = torch.tensor([float(b - a) for a, b in zip(edges[:-1], edges[1:])])
        win = f[:, 2 * K:].reshape(B, m.n_windows, K)
        assert torch.allclose((win * w[None, :, None]).sum(1) / w.sum(), f[:, :K], atol=1e-5)
        # an invisible window is flagged invalid (and therefore contributes nothing after standardisation)
        valid = torch.ones(B, T_CTX)
        valid[:, : edges[1]] = 0.0
        _, fv2 = m.count_features(x["ALM_L"], valid, m.late_bins, m.n_windows)
        assert float(fv2[:, 2 * K: 3 * K].max()) == 0.0 and float(fv2[:, 3 * K:].min()) == 1.0
    with torch.no_grad():
        # zero-initialised read-out: at init the logits are the backbone's
        out = m(x, SESSION)
        assert out.logits_backbone is not None and out.logits_skip is not None
        assert torch.allclose(out.logits, out.logits_backbone, atol=ATOL) and float(out.logits_skip.abs().max()) == 0.0
        for p in ad.skip.parameters():
            p.add_(0.5 * torch.randn_like(p))
        out = m(x, SESSION)
        assert torch.allclose(out.logits, out.logits_backbone + out.logits_skip, atol=ATOL)
        assert float(out.logits_skip.abs().max()) > 0.0
        # masked bins do not reach the read-out; a padded unit contributes nothing
        pm = torch.zeros(B, T_CTX, dtype=torch.bool)
        pm[:, :60] = True
        x2 = {r: v.clone() for r, v in x.items()}
        x2["ALM_L"][:, :, :60] += 5.0
        assert torch.allclose(m(x, SESSION, pad_mask=pm).logits_skip, m(x2, SESSION, pad_mask=pm).logits_skip, atol=ATOL)
        nm = {r: torch.ones(B, K, dtype=torch.bool) for r in REGIONS}
        nm["STR_R"][:, 0] = False
        x3 = {r: v.clone() for r, v in x.items()}
        x3["STR_R"][:, 0] += 7.0
        assert torch.allclose(m(x, SESSION, neuron_mask=nm).logits_skip, m(x3, SESSION, neuron_mask=nm).logits_skip, atol=ATOL)
        # region drop == zeroed input on the whole classifier
        x0 = {r: v.clone() for r, v in x.items()}
        x0["ALM_R"].zero_()
        assert torch.allclose(m(x, SESSION, drop_region="ALM_R").logits, m(x0, SESSION).logits, atol=ATOL)
    # variants
    c1 = load_config(None); c1.set_path("model.linear_skip", False)
    m1 = DelayCASTNet([SESSION], K, T_CTX, T_TGT, c1).eval()
    c2 = load_config(None); c2.set_path("model.classifier_from_backbone", False)
    m2 = DelayCASTNet([SESSION], K, T_CTX, T_TGT, c2).eval()
    with torch.no_grad():
        o1, o2 = m1(x, SESSION), m2(x, SESSION)
    assert o1.logits_skip is None and torch.allclose(o1.logits, o1.logits_backbone, atol=ATOL)
    assert o2.logits_backbone is None and torch.allclose(o2.logits, o2.logits_skip, atol=ATOL)
    c3 = load_config(None); c3.set_path("model.linear_skip", False); c3.set_path("model.classifier_from_backbone", False)
    with pytest.raises(ValueError):
        DelayCASTNet([SESSION], K, T_CTX, T_TGT, c3)



def test_learned_filterbank_starts_at_the_gabor_bank_and_is_trainable(cfg):
    """``learned`` mode initialises at the fixed Gabor kernels (same output at init) and exposes them as parameters."""
    bands = {k: list(v) for k, v in cfg.selection.bands_hz.items()}
    fixed = CausalBandPower(float(cfg.data.bin_ms), bands, float(cfg.model.spectral_win_ms), mode="bands").eval()
    learned = CausalBandPower(float(cfg.data.bin_ms), bands, float(cfg.model.spectral_win_ms), mode="learned").eval()
    torch.manual_seed(3)
    pop, valid = torch.rand(B, T_CTX) * 2, torch.ones(B, T_CTX)
    assert torch.allclose(fixed(pop, valid), learned(pop, valid), atol=ATOL)
    assert sum(p.numel() for p in fixed.parameters()) == 0
    assert sum(p.numel() for p in learned.parameters()) == 2 * len(bands) * learned.nper
    learned(pop, valid).sum().backward()
    assert learned.kernel.grad is not None and float(learned.kernel.grad.abs().max()) > 0


def test_forecast_loss_scale_is_count_invariant(cfg):
    """With the per-unit mean-count normalisation the gradient of the Poisson term w.r.t. the log-rate is O(1)
    whatever the count scale (single units vs population channels); without it the gradient grows with the rate."""
    from delaycast.models.delaycast_net import poisson_nll
    torch.manual_seed(0)
    n_el = B * K * T_TGT                       # the loss is a mean over elements: per-element gradient x n_el
    g_norms = []
    for mean in (0.2, 5.0, 60.0):
        counts = torch.poisson(torch.full((B, K, T_TGT), mean))
        lr = torch.full((B, K, T_TGT), float(np.log(mean)) + 0.3, requires_grad=True)
        scale = torch.full((K,), mean)
        poisson_nll(lr, counts, None, scale=scale).backward()
        g_norm = float(lr.grad.abs().mean()) * n_el
        lr2 = lr.detach().clone().requires_grad_(True)
        poisson_nll(lr2, counts, None).backward()
        g_raw = float(lr2.grad.abs().mean()) * n_el
        assert 0.05 < g_norm < 3.0, (mean, g_norm)
        assert abs(g_raw / g_norm - mean) < 1e-4 * mean + 1e-6
        g_norms.append((g_norm, g_raw))
    # normalised: within one order of magnitude across a 300-fold range of count scales (a Poisson of mean 0.2 has a
    # large relative error by nature); raw: the gradient grows with the count scale
    assert max(g for g, _ in g_norms) / min(g for g, _ in g_norms) < 8.0, g_norms
    assert max(g for _, g in g_norms) / min(g for _, g in g_norms) > 20.0, g_norms
    # the log-rate clamp keeps a runaway prediction finite
    big = torch.full((B, K, T_TGT), 50.0)
    assert torch.isfinite(poisson_nll(big, torch.zeros(B, K, T_TGT)))


def test_standardised_input_uses_training_statistics(cfg, x):
    """The backbone input is the per-unit z-score of the sqrt-count with the statistics fitted on the given trials;
    the wide path and the spectral population trace are unaffected by that choice (the population trace stays
    non-negative, which the log1p of the branch requires)."""
    torch.manual_seed(0)
    m = DelayCASTNet([SESSION], K, T_CTX, T_TGT, cfg).eval()
    assert m.standardize_input
    m.fit_count_stats(SESSION, {r: x[r].numpy() for r in REGIONS}, {r: torch.poisson(torch.full((B, K, T_TGT), 2.0)).numpy() for r in REGIONS})
    st = m.adapters[SESSION.replace("/", "__")].stats["ALM_L"]
    sq = torch.sqrt(x["ALM_L"])
    assert torch.allclose(st.mu, sq.mean((0, 2)), atol=1e-5) and torch.allclose(st.sd, sq.std((0, 2), unbiased=False), atol=1e-5)
    assert torch.allclose(st.rate, torch.full((K,), 2.0), atol=0.5)
    with torch.no_grad():
        out = m(x, SESSION)
        assert torch.isfinite(out.logits).all() and all(torch.isfinite(out.spec[r]).all() for r in REGIONS)
    m2 = DelayCASTNet([SESSION], K, T_CTX, T_TGT, load_config(None, ["model.d_model=32", "model.standardize_input=false"])).eval()
    assert not m2.standardize_input


def test_forecast_starts_at_the_mean_log_rate_and_is_capped(cfg, x):
    """fit_count_stats puts each unit's forecast base at its training mean log-rate, and the forecast log-rate never
    exceeds log(255)."""
    torch.manual_seed(0)
    m = DelayCASTNet([SESSION], K, T_CTX, T_TGT, cfg).eval()
    y = {r: torch.poisson(torch.full((B, K, T_TGT), 30.0)).numpy() for r in REGIONS}
    m.fit_count_stats(SESSION, {r: x[r].numpy() for r in REGIONS}, y)
    ad = m.adapters[SESSION.replace("/", "__")]
    for r in REGIONS:
        assert torch.allclose(ad.log_base[r], torch.log(torch.as_tensor(y[r]).mean((0, 2)) + 0.05), atol=1e-5)
    with torch.no_grad():
        ad.log_base["ALM_L"].fill_(40.0)
        out = m(x, SESSION)
        assert float(out.forecast_log_rate["ALM_L"].max()) <= m.max_log_rate + 1e-6
        assert abs(m.max_log_rate - float(np.log(255))) < 0.1


def test_wide_path_warm_start_equals_the_tuned_logistic_regression(cfg, x):
    """With skip_init=logreg the deep head is zero and the wide path carries the sklearn fit: the network's logits at
    epoch 0 are the logistic regression's decision function on the standardised windowed count features."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from delaycast.train import warm_start_skip
    torch.manual_seed(0)
    n = 40
    xx = {r: torch.poisson(torch.full((n, K, T_CTX), 0.2)) for r in REGIONS}
    labels = np.array([0, 1, 2] * 13 + [1])
    xx["ALM_L"][labels == 1, :4] += 1.0                          # a decodable signal
    xx["STR_R"][labels == 2, 4:] += 1.0
    m = DelayCASTNet([SESSION], K, T_CTX, T_TGT, cfg).eval()
    assert m.skip_init == "logreg" and float(m.classifier[-1].weight.abs().max()) == 0.0
    m.fit_count_stats(SESSION, {r: xx[r].numpy() for r in REGIONS}, {r: xx[r][:, :, :T_TGT].numpy() for r in REGIONS})
    info = warm_start_skip(m, SESSION, {r: xx[r].numpy() for r in REGIONS}, labels, cfg)
    assert info["fitted"] and info["n_features"] == 4 * m.n_feat * K
    # reference fit on the same standardised features
    ad = m.adapters[SESSION.replace("/", "__")]
    Z = []
    for r in REGIONS:
        f, _ = m.count_features(xx[r], torch.ones(n, T_CTX), m.late_bins, m.n_windows)
        Z.append(((f - ad.skip[r].mu) / ad.skip[r].sd).numpy())
    Z = np.concatenate(Z, axis=1)
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 6), cv=StratifiedKFold(5, shuffle=True, random_state=int(cfg.train.seed)),
                               class_weight="balanced", max_iter=1000, n_jobs=1).fit(Z, labels)
    with torch.no_grad():
        out = m(xx, SESSION)
    ref = clf.decision_function(Z)
    assert np.allclose(out.logits.numpy(), ref, atol=1e-3)
    assert (out.logits.argmax(1).numpy() == clf.predict(Z)).all()
    assert float(out.logits_backbone.abs().max()) == 0.0
