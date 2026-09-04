"""Verification of the neuron-selection guarantees on in-memory synthetic sessions.

The selection is the part of the pipeline a reviewer will attack first, so these tests pin down:

* the eligibility rule (one criterion is never enough; the descriptive T / I statistics never enter the score);
* that label-free selection (used for held-out sessions) produces byte-identical tables when the labels are
  permuted - i.e. no label information can leak into the units chosen for a session whose labels the model
  must not see;
* that planted structure is recovered (Left-selective, ramping units are flagged, stable and selected) while a
  constant unit is not, and that permuting the labels removes the selectivity criterion;
* that the vectorised rank statistics reproduce scipy on heavily tied Poisson counts;
* that the class-conditioned coupling test is blind to correlations created by class means alone.

Sessions are built directly as ``SessionCache`` objects from Poisson counts (3 classes, 60 trials, 12 units
per region, 120 context bins of 10 ms, 30 target bins of 50 ms, uint8) so nothing touches the disk except
the feature cache directory, which is redirected to ``tmp_path``.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from delaycast import CLASSES, REGIONS
from delaycast.config import load_config
from delaycast.data.cache import SessionCache
from delaycast.features.selection import CRITERION_KEYS, _flag_and_score, compute_features, select_neurons
from delaycast.features.stats import (kruskal_vectorised, mannwhitney_vectorised, spearman_vectorised, wilcoxon_vectorised,
                                      within_class_rank_corr)

IGNORE, LEFT, RIGHT = (CLASSES.index(c) for c in ("Ignore", "Left", "Right"))
N_TRIALS, N_UNITS, T, T_TGT = 60, 12, 120, 30
PLANTED = (0, 1, 2)        # Left-selective + ramping (+ coupled) units in every region
CONSTANT = 3               # one spike in every bin of every trial: passes the floor, carries no information


# ----------------------------------------------------------------------------- synthetic session
def make_cache(seed: int, plant: bool = True) -> SessionCache:
    """Poisson session with optional planted structure.

    Planted units fire twice as much on Left trials (criterion S), ramp from 0.1x to 1.9x of their base rate
    across the delay (criterion R, present in every class) and share a per-trial log-normal gain between the
    last 400 ms of the delay and the response epoch (criterion C).  Background units are homogeneous Poisson.
    The effects are deliberately strong: a criterion must still reach q < 0.05 inside a 30-trial half-subsample
    (12 Left / 12 Right / 6 Ignore) for the unit to count as stable, and the Wilcoxon p of a 12-trial class
    cannot go below 0.002 x 3 (Bonferroni over classes), so a weak ramp would fail BH there for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    labels = np.repeat([IGNORE, LEFT, RIGHT], [12, 24, 24])
    rng.shuffle(labels)
    context, target, unit_ids = {}, {}, {}
    ramp = np.linspace(0.1, 1.9, T)
    for r in REGIONS:
        base = rng.uniform(0.05, 0.12, size=N_UNITS)                                 # spikes per 10 ms bin (5-12 Hz)
        lam = np.tile(base[None, :, None], (N_TRIALS, 1, T))
        lam_t = np.tile(5.0 * base[None, :, None], (N_TRIALS, 1, T_TGT))             # 50 ms bins
        if plant:
            gain = np.exp(rng.normal(0.0, 0.8, size=N_TRIALS))
            for u in PLANTED:
                lam[:, u, :] = base[u] * ramp[None]
                lam[labels == LEFT, u, :] *= 2.0
                lam[:, u, -40:] *= gain[:, None]
                lam_t[:, u, :] *= gain[:, None]
        X = rng.poisson(lam).astype(np.uint8)
        Y = rng.poisson(lam_t).astype(np.uint8)
        if plant:
            X[:, CONSTANT, :] = 1
            Y[:, CONSTANT, :] = 1
        context[r], target[r] = X, Y
        unit_ids[r] = np.arange(N_UNITS) + 100 * REGIONS.index(r)
    meta = pd.DataFrame({"trial": np.arange(1, N_TRIALS + 1), "label": [CLASSES[i] for i in labels]})
    return SessionCache(session="A/Synthetic", dataset="A", subject="Synthetic", context=context, target=target, unit_ids=unit_ids,
                        labels=labels, trials=np.arange(1, N_TRIALS + 1), meta=meta, bin_ms=10.0, target_bin_ms=50.0)


def permuted(cache: SessionCache, seed: int) -> SessionCache:
    """Copy of the session with labels permuted (what the negative control does before everything else)."""
    cc = copy.copy(cache)
    cc.labels = np.random.default_rng(seed).permutation(cache.labels)
    return cc


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(None)
    cfg.set_path("data.cache_dir", str(tmp_path / "cache"))
    cfg.set_path("selection.n_subsamples", 5)
    cfg.set_path("selection.locus_n_perm", 20)
    cfg.set_path("selection.coupling_n_perm", 50)
    cfg.set_path("selection.n_null_permutations", 0)
    cfg.set_path("selection.top_k_per_region", 4)
    return cfg


def _features(cache: SessionCache, cfg):
    """Always recomputed: the on-disk feature cache is keyed by shapes, not labels, and must not short-circuit
    the label-permutation tests."""
    return compute_features(cache, cfg, use_cache=False)


# ----------------------------------------------------------------------------- (a) eligibility rule
def test_eligibility_needs_two_scored_criteria():
    cfg = load_config(None)
    assert int(cfg.selection.min_criteria) == 2
    ones = {f"q_{k}": 1.0 for k in CRITERION_KEYS + ("ignore", "locus")}
    rows = [
        {**ones, "q_selectivity": 1e-9},                                  # S only
        {**ones, "q_selectivity": 1e-9, "q_ramp": 1e-3},                  # S + R
        {**ones, "q_selectivity": 1e-9, "q_coupling": 1e-9, "q_spectral": 1e-9, "q_ramp": 1e-9, "pass_floor": False},
        {**ones, "q_locus": 1e-9, "q_ignore": 1e-9},                      # descriptive T + I only
    ]
    df = pd.DataFrame(rows)
    df["pass_floor"] = df.get("pass_floor", True).fillna(True).astype(bool)
    df["label_free"] = False
    out = _flag_and_score(df, cfg)
    assert out.c_selectivity.tolist() == [True, True, True, False]
    assert out.n_criteria.tolist() == [1, 2, 4, 0]
    assert out.eligible.tolist() == [False, True, False, False]
    assert out.score.iloc[2] == -np.inf                                    # below the floor: never ranked
    assert out.c_locus.iloc[3] and out.c_ignore.iloc[3]
    assert out.score.iloc[3] == 0.0                                        # T and I are never in the score
    assert out.score.iloc[1] > out.score.iloc[0] > 0.0


def test_label_free_eligibility_threshold():
    """Held-out sessions use ``min_criteria_label_free`` (1 of {C, R}) because S / W cannot be tested without labels."""
    cfg = load_config(None)
    ones = {f"q_{k}": 1.0 for k in CRITERION_KEYS + ("ignore", "locus")}
    df = pd.DataFrame([{**ones, "q_ramp": 1e-4}, {**ones}])
    df["pass_floor"], df["label_free"] = True, True
    out = _flag_and_score(df, cfg)
    assert out.eligible.tolist() == [True, False]


# ----------------------------------------------------------------------------- (b) label-free selection
def test_label_free_selection_ignores_labels(cfg):
    cache = make_cache(11)
    res_a = select_neurons(cache, cfg, features=_features(cache, cfg), seed=0, label_free=True)
    cache_p = permuted(cache, 5)
    assert not np.array_equal(cache_p.labels, cache.labels)
    res_b = select_neurons(cache_p, cfg, features=_features(cache_p, cfg), seed=0, label_free=True)
    pd.testing.assert_frame_equal(res_a.table, res_b.table)
    pd.testing.assert_frame_equal(res_a.funnel, res_b.funnel)
    for r in REGIONS:
        assert np.array_equal(res_a.selected[r], res_b.selected[r])
    t = res_a.table
    assert t.label_free.all()
    # S / W / I / T are reported as not tested; only floor, C and the net ramp can select
    assert t.p_selectivity.isna().all() and t.p_spectral.isna().all() and t.p_ignore.isna().all()
    assert "p_locus" not in t.columns
    assert not t.c_selectivity.any() and not t.c_spectral.any()
    assert (t.n_fit_left == 0).all() and (t.n_fit_right == 0).all() and (t.n_fit_ignore == N_TRIALS).all()
    assert t.loc[t.eligible].n_criteria.min() >= int(cfg.selection.min_criteria_label_free)


# ----------------------------------------------------------------------------- (c) planted structure, (d) permutation
def test_planted_units_are_selected_and_constant_unit_is_not(cfg):
    cache = make_cache(11)
    res = select_neurons(cache, cfg, features=_features(cache, cfg), seed=0)
    t = res.table
    assert len(t) == len(REGIONS) * N_UNITS and (t.n_fit_trials == N_TRIALS).all()
    for r in REGIONS:
        reg = t[t.region == r].set_index("unit_index")
        planted = reg.loc[list(PLANTED)]
        assert planted.pass_floor.all()
        assert planted.c_selectivity.all(), reg[["c_selectivity", "q_selectivity"]]
        assert (planted.preferred_class == "Left").all() and (planted.auroc_left_right > 0.5).all()
        assert planted.c_ramp.all() and (planted.ramp_slope_hz_s > 0).all()
        assert (planted.n_criteria >= 2).all() and planted.eligible.all()
        assert (planted.stability >= 0.6).all() and planted.stable.all()
        assert planted.selected.all() and set(planted["rank"].astype(int)) <= {1, 2, 3, 4}
        assert set(PLANTED) <= set(res.selected[r].tolist()) and len(res.selected[r]) <= 4
        const = reg.loc[CONSTANT]
        assert const.pass_floor and const.n_criteria == 0 and not const.eligible and not const.selected
        assert np.isnan(const.p_selectivity) and np.isnan(const.p_coupling) and np.isnan(const.p_ramp)
        assert const.reason_short.startswith(("lALM", "rALM", "lSTR", "rSTR")) and "0 crit" in const.reason_short
        assert "not eligible: 0 of >= 2 criteria" in const.reasons
        first = reg.loc[reg.selected].sort_values("rank").iloc[0]
        assert first.reason_short.count("|") == 5 and len(first.reason_short) <= 110
        assert "selected in" in first.reasons and "S Left-preferring" in first.reasons
    f = res.funnel
    assert (f.n_fit_trials == N_TRIALS).all() and (f.K == 4).all() and (f.selected <= 4).all() and (f.selected >= 3).all()
    assert (f.eligible >= f.stable).all() and (f.stable >= f.selected).all() and (f.recorded == N_UNITS).all()
    assert (f.n_subsamples if "n_subsamples" in f else t.n_subsamples).max() == 5
    # (d) permuted labels remove choice selectivity (ramping / coupling are label-agnostic and may survive)
    cache_p = permuted(cache, 3)
    res_p = select_neurons(cache_p, cfg, features=_features(cache_p, cfg), seed=0)
    assert int(res_p.table.c_selectivity.sum()) <= 1
    assert int(res_p.table.selected.sum()) < int(t.selected.sum()) or not res_p.table.loc[res_p.table.selected].c_selectivity.any()


FIT = np.sort(np.random.default_rng(0).choice(N_TRIALS, size=45, replace=False))
HELD = np.setdiff1d(np.arange(N_TRIALS), FIT)


def _with_corrupted_held_out_trials(cache: SessionCache) -> SessionCache:
    """Same session with every non-fit trial overwritten (3 spikes in every bin): if the selection on ``FIT``
    is really restricted to the fit trials, nothing in its table may change."""
    cache2 = copy.copy(cache)
    cache2.context = {r: v.copy() for r, v in cache.context.items()}
    for r in REGIONS:
        cache2.context[r][HELD] = 3
    return cache2


def test_selection_is_deterministic_and_reports_fit_trials(cfg):
    """Same seed -> same table; the fit-trial bookkeeping (n_fit_*) reflects ``trial_idx`` only."""
    cache = make_cache(4)
    feats = _features(cache, cfg)
    a = select_neurons(cache, cfg, trial_idx=FIT, features=feats, seed=0)
    b = select_neurons(cache, cfg, trial_idx=FIT, features=feats, seed=0)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert np.array_equal(a.trial_idx, FIT) and (a.table.n_fit_trials == 45).all()
    assert (a.funnel.n_fit_trials == 45).all()
    y = cache.labels[FIT]
    assert (a.table.n_fit_left == (y == LEFT).sum()).all() and (a.table.n_fit_ignore == (y == IGNORE).sum()).all()


def test_selection_ignores_non_fit_trials_when_every_unit_is_a_spectral_candidate(cfg):
    """With ``spectral_candidates=all`` every per-trial feature is computed for every unit, so the criteria on
    ``FIT`` cannot depend on what happens in the held-out trials (the property the training runs rely on)."""
    cfg.set_path("selection.spectral_candidates", "all")
    cache = make_cache(4)
    a = select_neurons(cache, cfg, trial_idx=FIT, features=_features(cache, cfg), seed=0)
    cache2 = _with_corrupted_held_out_trials(cache)
    c = select_neurons(cache2, cfg, trial_idx=FIT, features=_features(cache2, cfg), seed=0)
    pd.testing.assert_frame_equal(a.table, c.table)
    for r in REGIONS:
        assert np.array_equal(a.selected[r], c.selected[r])


def test_selection_ignores_non_fit_trials_with_default_screen(cfg):
    assert cfg.selection.get_path("spectral_candidates", "all") == "all"
    cache = make_cache(4)
    a = select_neurons(cache, cfg, trial_idx=FIT, features=_features(cache, cfg), seed=0)
    cache2 = _with_corrupted_held_out_trials(cache)
    c = select_neurons(cache2, cfg, trial_idx=FIT, features=_features(cache2, cfg), seed=0)
    pd.testing.assert_frame_equal(a.table, c.table)


# ----------------------------------------------------------------------------- (e) vectorised statistics vs scipy
def test_vectorised_statistics_match_scipy_on_tied_counts():
    rng = np.random.default_rng(12)
    n, m = 40, 60
    v = rng.poisson(1.5, size=(n, m)).astype(float)            # heavy ties, many zeros
    labels = np.repeat([0, 1, 2], [8, 16, 16])
    a, b = v[labels == 1], v[labels == 2]
    auroc, p = mannwhitney_vectorised(a, b)
    for j in range(m):
        ref = ss.mannwhitneyu(a[:, j], b[:, j], method="asymptotic", use_continuity=False)
        assert auroc[j] * len(a) * len(b) == pytest.approx(ref.statistic, abs=1e-9)
        assert p[j] == pytest.approx(ref.pvalue, abs=1e-12)
    d = (v[:20] - v[20:])
    eff, pw = wilcoxon_vectorised(d)
    for j in range(m):
        dj = d[:, j][d[:, j] != 0]
        if len(dj) < 5:
            assert np.isnan(pw[j])
            continue
        ref = ss.wilcoxon(dj, correction=False, method="approx")
        assert pw[j] == pytest.approx(ref.pvalue, abs=1e-12)
        assert eff[j] == pytest.approx(((dj > 0).sum() - (dj < 0).sum()) / len(dj))
    H, pk = kruskal_vectorised(v, labels)
    for j in range(m):
        ref = ss.kruskal(*[v[labels == g, j] for g in (0, 1, 2)])
        assert H[j] == pytest.approx(ref.statistic, abs=1e-10) and pk[j] == pytest.approx(ref.pvalue, abs=1e-12)
    y = rng.poisson(2.0, size=(n, m)).astype(float)
    rho, ps = spearman_vectorised(v, y)
    for j in range(m):
        ref = ss.spearmanr(v[:, j], y[:, j])
        assert rho[j] == pytest.approx(ref.statistic, abs=1e-12) and ps[j] == pytest.approx(ref.pvalue, abs=1e-10)
    # constant columns are reported as untestable rather than as (spurious) significant
    v[:, 0] = 2.0
    assert np.isnan(mannwhitney_vectorised(v[labels == 1], v[labels == 2])[1][0])
    assert np.isnan(spearman_vectorised(v, y)[1][0])


# ----------------------------------------------------------------------------- (f) class-conditioned coupling
def test_within_class_rank_corr_ignores_class_means_and_detects_coupling():
    rng = np.random.default_rng(13)
    labels = np.repeat([0, 1, 2], 30)
    n = len(labels)
    # correlation created only by class means (both variables shift together across classes)
    shift = np.array([-3.0, 0.0, 3.0])[labels]
    x = shift + rng.normal(size=n)
    y = shift + rng.normal(size=n)
    assert ss.spearmanr(x, y).statistic > 0.5                       # a naive rank correlation would call this coupling
    rho, p = within_class_rank_corr(x[:, None], y[:, None], labels, n_perm=400, rng=np.random.default_rng(0))
    assert abs(rho[0]) < 0.15 and p[0] > 0.05
    # genuine within-class coupling (rho ~ 0.8) on top of the same class means
    z = rng.normal(size=n)
    x2 = shift + z
    y2 = shift + 0.8 * z + np.sqrt(1 - 0.8 ** 2) * rng.normal(size=n)
    rho2, p2 = within_class_rank_corr(x2[:, None], y2[:, None], labels, n_perm=400, rng=np.random.default_rng(0))
    assert rho2[0] > 0.6 and p2[0] < 0.01
    # columns that are constant within every class are untestable (NaN), never significant
    rho3, p3 = within_class_rank_corr(np.ones((n, 1)), y2[:, None], labels, n_perm=50)
    assert np.isnan(rho3[0]) and np.isnan(p3[0])


# ----------------------------------------------------------------------------- held-out sessions never see their test trials
def test_holdout_label_free_selection_uses_adapt_trials_only(cfg):
    """Cross-session / cross-dataset runs select the held-out session's units label-free on its ADAPT trials only."""
    from delaycast.train import fit_trials, make_splits, prepare_sessions
    caches = {"A/S1": make_cache(11), "B/S2": make_cache(12)}
    for c, s in zip(caches.values(), caches):
        c.session = s
    cfg2 = cfg
    cfg2.set_path("data.cache_dir", str(cfg.data.cache_dir) if cfg.data.get_path("cache_dir") else "cache_test")
    splits = make_splits(cfg2, caches, holdout=["B/S2"], seed=0)
    sels, tensors = prepare_sessions(cfg2, caches, mode="criteria", splits=splits, seed=0, holdout=["B/S2"])
    sel = dict(zip(caches, sels))["B/S2"]
    adapt = set(splits["B/S2"]["adapt"].tolist())
    test = set(splits["B/S2"]["test"].tolist())
    assert bool(sel.table.label_free.all())
    assert set(sel.trial_idx.tolist()) == adapt
    assert not (set(sel.trial_idx.tolist()) & test)
    assert int(sel.table.n_fit_trials.iloc[0]) == len(adapt)
    assert len(fit_trials(splits["B/S2"])) == len(adapt)
