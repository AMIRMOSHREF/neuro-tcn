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
@pytest.mark.parametrize("mode", ["bands", "popmean"])
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
    for mode in ("bands", "popmean"):
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
