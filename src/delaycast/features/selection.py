"""Criterion-based, stability-checked selection of delay-epoch neurons that carry information about
the upcoming action.

Everything is computed per session on the *fit trials* only (``trial_idx``: the training + validation
split during model runs, so test trials never influence which neurons the model sees; all trials for the
descriptive ``delaycast select`` tables).  Statistics are vectorised over units (``features/stats.py``).

Scored criteria (BH-FDR across the floor-passing units of the session, q < ``fdr_q``):

  floor  activity floor          mean delay rate >= ``min_rate_hz`` and spikes on >= ``min_active_trial_frac`` of trials
  S      choice selectivity      Mann-Whitney U, delay spike count Left vs Right (effect size AUROC_LR)
  C      delay->response         class-conditioned rank correlation between the unit's late-delay rate and its own
         coupling                response-epoch rate; p from circular-shift permutations that preserve slow drift
  W      spectro-temporal        Mann-Whitney U Left vs Right of complex-Morlet CWT band power (slow / theta / beta)
         selectivity             after regressing out the unit's spike count (so W is not a rate test in disguise);
                                 Bonferroni over bands, then BH over units
  R      ramping                 Wilcoxon signed-rank across trials of (late - early delay rate) within each class
                                 (Bonferroni over classes): trial-level, class-conditional preparatory build-up

Descriptive statistics (reported, badges in the figure, *not* in the score):

  T      temporal locus          AUROC_LR in sliding windows (``window_ms`` / ``window_step_ms``); cluster-mass
                                 permutation test (label permutations shared across units) -> information onset,
                                 peak window, late fraction, sustained-to-go
  I      no-lick selectivity     Mann-Whitney U Ignore vs lick trials (only when >= ``min_ignore_trials`` Ignore trials)

Eligible units pass the floor and satisfy >= ``min_criteria`` of {S, C, W, R}.  **Stability selection**
(Meinshausen & Buehlmann 2010): the criteria + top-K ranking are recomputed on ``n_subsamples`` stratified
half-subsamples (without replacement) of the fit trials; a unit's *stability* is its selection frequency.
Final selection = eligible units with stability >= ``min_stability`` ranked by (stability, score), top-K per
region; if fewer than K units are stable the region keeps K_eff < K (zero-padded downstream) unless
``fill_unstable`` is set.  The expected number of false selections is bounded by
E[V] <= K^2 / ((2 pi_thr - 1) n_eligible).  Every unit receives a human-readable ``reasons`` string.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES, REGIONS
from ..data.cache import SessionCache
from .spectral import band_power_cwt, smooth_rates
from .stats import (bh_fdr, mannwhitney_vectorised, rankdata, spearman_trend, wilcoxon_vectorised,
                    within_class_rank_corr)

log = logging.getLogger(__name__)

CRITERION_KEYS = ("selectivity", "coupling", "spectral", "ramp")      # scored
CRITERION_LETTERS = {"selectivity": "S", "coupling": "C", "spectral": "W", "ramp": "R", "locus": "T", "ignore": "I"}
LEFT, RIGHT, IGNORE = CLASSES.index("Left"), CLASSES.index("Right"), CLASSES.index("Ignore")


# --------------------------------------------------------------------------- per-trial features
@dataclass
class RegionFeatures:
    """Per-trial, per-unit features of one region, computed once from the cached rasters (all trials)."""

    region: str
    unit_ids: np.ndarray
    counts: np.ndarray        # (n_tr, n_units) delay spike counts
    late_rate: np.ndarray     # (n_tr, n_units) Hz in the last ``late_delay_ms``
    early_rate: np.ndarray    # (n_tr, n_units) Hz in the first ``late_delay_ms``
    resp_rate: np.ndarray     # (n_tr, n_units) Hz in the response window
    win_counts: np.ndarray    # (n_tr, n_units, n_win) counts in sliding windows of the delay
    win_starts_ms: np.ndarray # (n_win,) window start times (ms from delay onset)
    band_power: np.ndarray    # (n_tr, n_cand, n_bands) CWT band power of the spectral candidates
    band_names: list[str]
    cand_idx: np.ndarray      # (n_cand,) unit indices for which band power was computed
    dur_s: float
    bin_ms: float
    window_ms: float
    late_ms: float

    @property
    def n_units(self) -> int:
        return self.counts.shape[1]


def _features_key(cfg, cache: SessionCache | None = None) -> str:
    """Hash of every parameter the cached features depend on (selection + data-cache settings + array shapes)."""
    from ..data.cache import _cache_key
    sel = cfg.selection
    payload = {"late": sel.late_delay_ms, "win": sel.get_path("window_ms", 200), "step": sel.get_path("window_step_ms", 50),
               "bands": sel.bands_hz.to_plain() if hasattr(sel.bands_hz, "to_plain") else dict(sel.bands_hz),
               "wavelet": sel.wavelet, "sigma": cfg.data.smoothing_sigma_ms, "floor": [sel.min_rate_hz, sel.min_active_trial_frac],
               "cand": sel.get_path("spectral_candidates", "all"), "nf": sel.get_path("n_freqs_per_band", 5),
               "fmin": sel.get_path("min_cwt_freq_hz", 2.0), "cache": _cache_key(cfg), "v": 5}
    if cache is not None:
        payload["n_trials"] = int(cache.n_trials)
        payload["n_units"] = {r: int(cache.context[r].shape[1]) for r in REGIONS}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _floor_mask(counts: np.ndarray, dur_s: float, cfg) -> np.ndarray:
    rate = counts.mean(axis=0) / dur_s
    active = (counts > 0).mean(axis=0)
    return (rate >= float(cfg.selection.min_rate_hz)) & (active >= float(cfg.selection.min_active_trial_frac))


def _class_ramp(diff: np.ndarray, labels: np.ndarray, min_trials: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-conditional ramp test on ``diff`` = late - early rate (n_tr, n_units).

    Returns (effect, p, cls) of the class with the smallest Wilcoxon p, Bonferroni-corrected over the classes
    tested (those with >= ``min_trials`` trials). ``effect`` = fraction of positive minus negative differences.
    """
    n_units = diff.shape[1]
    best_p = np.full(n_units, np.nan)
    best_e = np.full(n_units, np.nan)
    best_c = np.full(n_units, -1)
    tested = [c for c in np.unique(labels) if (labels == c).sum() >= min_trials]
    for c in tested:
        e, pv = wilcoxon_vectorised(diff[labels == c])
        pv = pv * len(tested)
        better = np.isfinite(pv) & (~np.isfinite(best_p) | (pv < best_p))
        best_p[better] = pv[better]
        best_e[better] = e[better]
        best_c[better] = c
    return best_e, np.clip(best_p, 0, 1), best_c


def _spectral_candidates(counts, late_rate, early_rate, resp_rate, labels, floor, cfg, rng) -> np.ndarray:
    """Units for which the (expensive) CWT is computed.

    ``all`` (default): every unit; ``floor``: every unit passing the activity floor; ``screened``: floor units
    with a liberal (uncorrected p < 0.05) hint of Left/Right selectivity, coupling or ramping on *all* trials.
    The two restricted modes are kept for very large recordings only: they make the set of W hypotheses depend
    on the whole session (including test trials), which ``all`` avoids.
    """
    mode = cfg.selection.get_path("spectral_candidates", "all")
    idx = np.flatnonzero(floor)
    if mode == "all":
        return np.arange(counts.shape[1])
    if mode == "floor" or idx.size == 0:
        return idx
    li, ri = labels == LEFT, labels == RIGHT
    p_sel = mannwhitney_vectorised(counts[li][:, idx], counts[ri][:, idx])[1] if li.sum() >= 2 and ri.sum() >= 2 else np.full(idx.size, np.nan)
    _, p_cpl = within_class_rank_corr(late_rate[:, idx], resp_rate[:, idx], labels, n_perm=100, rng=rng)
    _, p_ramp, _ = _class_ramp(late_rate[:, idx] - early_rate[:, idx], labels)
    hint = (np.nan_to_num(p_sel, nan=1.0) < 0.05) | (np.nan_to_num(p_cpl, nan=1.0) < 0.05) | (np.nan_to_num(p_ramp, nan=1.0) < 0.05)
    return idx[hint]


def compute_region_features(cache: SessionCache, cfg, region: str) -> RegionFeatures:
    sel = cfg.selection
    bin_ms = float(cache.bin_ms)
    X = cache.context[region]                # (n_tr, n_units, T) uint8
    Y = cache.target[region]                 # (n_tr, n_units, T_tgt)
    n_tr, n_units, T = X.shape
    dur_s = T * bin_ms / 1000.0
    late_ms = float(sel.late_delay_ms)
    counts = X.sum(axis=2).astype(np.float32)
    late_bins = max(1, int(round(late_ms / bin_ms)))
    late_rate = (X[:, :, -late_bins:].sum(axis=2) / (late_bins * bin_ms / 1000.0)).astype(np.float32)
    early_rate = (X[:, :, :late_bins].sum(axis=2) / (late_bins * bin_ms / 1000.0)).astype(np.float32)
    resp_rate = (Y.sum(axis=2) / (Y.shape[2] * float(cache.target_bin_ms) / 1000.0)).astype(np.float32)
    # Sliding windows built from non-overlapping ``step`` sub-bins (cheap, no cumsum over the full raster).
    win_ms = float(sel.get_path("window_ms", 200))
    step_ms = float(sel.get_path("window_step_ms", 50))
    step_bins = max(1, int(round(step_ms / bin_ms)))
    per_win = max(1, int(round(win_ms / step_ms)))
    n_sub = T // step_bins
    sub = X[:, :, : n_sub * step_bins].reshape(n_tr, n_units, n_sub, step_bins).sum(axis=3).astype(np.float32)
    n_win = max(1, n_sub - per_win + 1)
    cs = np.concatenate([np.zeros((n_tr, n_units, 1), np.float32), np.cumsum(sub, axis=2)], axis=2)
    win_counts = cs[:, :, per_win: per_win + n_win] - cs[:, :, :n_win]
    win_starts = np.arange(n_win) * step_bins * bin_ms
    bands = {k: list(v) for k, v in sel.bands_hz.items()}
    if n_units == 0:
        return RegionFeatures(region, cache.unit_ids[region], counts, late_rate, early_rate, resp_rate, win_counts, win_starts,
                              np.zeros((n_tr, 0, len(bands)), np.float32), list(bands), np.zeros(0, int), dur_s, bin_ms, win_ms, late_ms)
    mode = str(sel.get_path("spectral_candidates", "all"))
    if mode == "all":
        # Default: every unit gets the CWT (the impulse-response implementation makes 2000 units x 350 trials a
        # few seconds), so the set of W hypotheses never depends on any trial subset or label.
        cand = np.arange(n_units)
    else:
        floor = _floor_mask(counts, dur_s, cfg)
        cand = _spectral_candidates(counts, late_rate, early_rate, resp_rate, cache.labels, floor, cfg, np.random.default_rng(0))
    names = list(bands)
    bp = np.zeros((n_tr, cand.size, len(bands)), np.float32)
    fmin = float(sel.get_path("min_cwt_freq_hz", 2.0))
    n_freq = int(sel.get_path("n_freqs_per_band", 5))
    block = max(1, int(sel.get_path("cwt_units_per_block", 128)))     # bounds the transient (n_tr * block, T) rate matrix
    for i in range(0, cand.size, block):
        cb = cand[i: i + block]
        rates = smooth_rates(X[:, cb].reshape(n_tr * cb.size, T).astype(np.float32), bin_ms, cfg.data.smoothing_sigma_ms)
        bp_b, names = band_power_cwt(rates, bin_ms, bands, sel.wavelet, n_freqs_per_band=n_freq, min_freq_hz=fmin)
        bp[:, i: i + cb.size] = bp_b.reshape(n_tr, cb.size, -1)
    return RegionFeatures(region, cache.unit_ids[region], counts, late_rate, early_rate, resp_rate, win_counts, win_starts,
                          bp, names, cand, dur_s, bin_ms, win_ms, late_ms)


def compute_features(cache: SessionCache, cfg, use_cache: bool = True) -> dict[str, RegionFeatures]:
    """All-region features, cached on disk under ``data.cache_dir`` (the CWT is the expensive part)."""
    key = _features_key(cfg, cache)
    path = Path(cfg.data.cache_dir) / "features" / f"{cache.session.replace('/', '__')}_{key}.npz"
    if use_cache and path.is_file():
        try:
            return _load_features(path)
        except Exception as e:  # pragma: no cover - corrupted cache
            log.warning("could not read feature cache %s (%s); recomputing", path, e)
    feats = {r: compute_region_features(cache, cfg, r) for r in REGIONS}
    if use_cache:
        _save_features(feats, path)
    return feats


_ARRAY_FIELDS = ("unit_ids", "counts", "late_rate", "early_rate", "resp_rate", "win_counts", "win_starts_ms", "band_power", "cand_idx")


def _save_features(feats: dict[str, RegionFeatures], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for r, f in feats.items():
        for k in _ARRAY_FIELDS:
            arrays[f"{r}__{k}"] = getattr(f, k)
        arrays[f"{r}__meta"] = np.array(json.dumps({"band_names": f.band_names, "dur_s": f.dur_s, "bin_ms": f.bin_ms,
                                                    "window_ms": f.window_ms, "late_ms": f.late_ms}))
    np.savez_compressed(path, **arrays)


def _load_features(path: Path) -> dict[str, RegionFeatures]:
    z = np.load(path, allow_pickle=True)
    out = {}
    for r in REGIONS:
        meta = json.loads(str(z[f"{r}__meta"]))
        out[r] = RegionFeatures(r, *[z[f"{r}__{k}"] for k in _ARRAY_FIELDS[:7]], z[f"{r}__band_power"], list(meta["band_names"]),
                                z[f"{r}__cand_idx"], float(meta["dur_s"]), float(meta["bin_ms"]), float(meta["window_ms"]), float(meta["late_ms"]))
    return out


# --------------------------------------------------------------------------- temporal locus (cluster test)
def _cluster_mass(a: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest cluster mass of a > thr along the last axis (consecutive windows).

    ``a``: (..., n_win) effect sizes (|AUROC - 0.5|). Returns (max_mass, start_of_max_cluster, end_of_max_cluster)."""
    above = a > thr
    cur = np.zeros(a.shape[:-1])
    best = np.zeros(a.shape[:-1])
    cur_start = np.zeros(a.shape[:-1], int)
    best_start = np.zeros(a.shape[:-1], int)
    best_end = np.zeros(a.shape[:-1], int)
    for w in range(a.shape[-1]):
        new = above[..., w] & (cur == 0)
        cur_start = np.where(new, w, cur_start)
        cur = np.where(above[..., w], cur + a[..., w], 0.0)
        better = cur > best
        best = np.where(better, cur, best)
        best_start = np.where(better, cur_start, best_start)
        best_end = np.where(better, w, best_end)
    return best, best_start, best_end


def temporal_locus(win_counts: np.ndarray, labels: np.ndarray, cfg, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Sliding-window AUROC (Left vs Right) time course + cluster-mass permutation test per unit.

    ``win_counts``: (n_tr, n_units, n_win). Returns per-unit arrays: p_locus, auroc_win (n_units, n_win),
    onset_idx / peak_idx / end_idx (window indices, -1 if none), sustained_to_go, max_mass, null_p95.
    """
    sel = cfg.selection
    thr = float(sel.get_path("locus_threshold", 0.1))
    n_perm = int(sel.get_path("locus_n_perm", 200))
    n_tr, n_units, n_win = win_counts.shape
    li, ri = labels == LEFT, labels == RIGHT
    out = {"p_locus": np.full(n_units, np.nan), "auroc_win": np.full((n_units, n_win), np.nan),
           "onset_idx": np.full(n_units, -1), "peak_idx": np.full(n_units, -1), "end_idx": np.full(n_units, -1),
           "sustained_to_go": np.zeros(n_units, bool), "max_mass": np.full(n_units, np.nan), "null_p95": np.full(n_units, np.nan)}
    if li.sum() < 5 or ri.sum() < 5 or n_units == 0:
        return out
    keep = li | ri
    W = win_counts[keep].reshape(int(keep.sum()), n_units * n_win)
    ranks = rankdata(W, axis=0)                              # ranks are label-free: permute only the group masks
    is_left = li[keep]
    n_l, n_r = int(is_left.sum()), int((~is_left).sum())

    def auroc_from(mask_left: np.ndarray) -> np.ndarray:
        u = ranks[mask_left].sum(axis=0) - n_l * (n_l + 1) / 2
        return (u / (n_l * n_r)).reshape(n_units, n_win)

    auroc = auroc_from(is_left)
    a_obs = np.abs(auroc - 0.5)
    mass, start, end = _cluster_mass(a_obs, thr)
    null = np.zeros((n_perm, n_units))
    for b in range(n_perm):
        null[b] = _cluster_mass(np.abs(auroc_from(rng.permutation(is_left)) - 0.5), thr)[0]
    p = (1.0 + (null >= mass[None]).sum(axis=0)) / (1.0 + n_perm)
    sig = mass > 0
    out.update({"p_locus": p, "auroc_win": auroc, "max_mass": mass, "null_p95": np.percentile(null, 95, axis=0),
                "onset_idx": np.where(sig, start, -1), "end_idx": np.where(sig, end, -1),
                "peak_idx": a_obs.argmax(axis=1), "sustained_to_go": sig & (end == n_win - 1)})
    return out


# --------------------------------------------------------------------------- criteria on a trial subset
def _rate_normalised_band_power(bp: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Residual of log band power after regressing on log(count + 1) per unit and band.

    ``bp``: (n_tr, n_cand, n_bands), ``counts``: (n_tr, n_cand). Removes the shot-noise dependence of spectral
    power on firing rate so the W test is not a rate test in disguise."""
    lx = np.log1p(counts)[:, :, None]
    ly = np.log(bp + 1e-6)
    lx_c = lx - lx.mean(axis=0, keepdims=True)
    ly_c = ly - ly.mean(axis=0, keepdims=True)
    var = (lx_c ** 2).sum(axis=0)
    slope = np.where(var > 0, (lx_c * ly_c).sum(axis=0) / np.where(var > 0, var, 1.0), 0.0)
    return ly_c - slope[None] * lx_c


def _criteria_table(feats: dict[str, RegionFeatures], cache: SessionCache, labels: np.ndarray, trial_idx: np.ndarray, cfg,
                    full: bool, rng: np.random.Generator, label_free: bool = False) -> pd.DataFrame:
    """Raw statistics + BH q-values for every unit of every region on the trials ``trial_idx``.

    ``full=False`` (stability subsamples) skips the descriptive statistics (temporal locus, Ignore test, PSTH
    shape) and uses fewer permutations for the coupling test.  ``label_free=True`` (held-out sessions) never
    looks at the class labels: only the floor, the coupling test (all trials as one group) and a
    class-agnostic ramp test are computed; S / W / T / I are reported as not tested."""
    sel = cfg.selection
    y = labels[trial_idx]
    if label_free:
        y = np.zeros(len(trial_idx), dtype=int)     # one group: no label information enters any statistic
    li, ri, ig = y == LEFT, y == RIGHT, y == IGNORE
    n_l, n_r, n_i = int(li.sum()), int(ri.sum()), int(ig.sum())
    n_perm = int(sel.get_path("coupling_n_perm", 500)) if full else int(sel.get_path("subsample_n_perm", 100))
    min_ign = int(sel.get_path("min_ignore_trials", 8))
    frames = []
    for r in REGIONS:
        f = feats[r]
        n_units = f.n_units
        if n_units == 0:
            continue
        counts = f.counts[trial_idx]
        rate = counts.mean(axis=0) / f.dur_s
        active = (counts > 0).mean(axis=0)
        floor = (rate >= float(sel.min_rate_hz)) & (active >= float(sel.min_active_trial_frac))
        nan = np.full(n_units, np.nan)
        # S: Left vs Right
        if n_l >= 2 and n_r >= 2 and not label_free:
            auroc_lr, p_sel = mannwhitney_vectorised(counts[li], counts[ri])
        else:
            auroc_lr, p_sel = np.full(n_units, 0.5), nan.copy()
        # I: Ignore vs lick (descriptive)
        if full and not label_free and n_i >= min_ign and (n_l + n_r) >= 2:
            auroc_ign, p_ign = mannwhitney_vectorised(counts[ig], counts[~ig])
        else:
            auroc_ign, p_ign = np.full(n_units, 0.5), nan.copy()
        means = {c: (counts[y == i].mean(axis=0) / f.dur_s if (y == i).any() else nan.copy()) for i, c in enumerate(CLASSES)}
        pref = np.where(auroc_lr >= 0.5, "Left", "Right")
        # C: class-conditioned coupling with drift-preserving permutation null
        rho_cpl, p_cpl = within_class_rank_corr(f.late_rate[trial_idx], f.resp_rate[trial_idx], y, n_perm=n_perm, rng=rng)
        # R: class-conditional ramp (label-free: one class = all trials, i.e. a net ramp in either direction)
        diff = f.late_rate[trial_idx] - f.early_rate[trial_idx]
        ramp_eff, p_ramp, ramp_cls = _class_ramp(diff, y)
        slope = np.full(n_units, np.nan)
        rho_ramp = np.full(n_units, np.nan)
        dt_s = (f.dur_s * 1000.0 - f.late_ms) / 1000.0        # distance between the early and late window centres
        for c in np.unique(ramp_cls[ramp_cls >= 0]):
            m = ramp_cls == c
            slope[m] = np.median(diff[y == c][:, m], axis=0) / max(dt_s, 1e-3)
            if full:
                psth = smooth_rates(cache.context[r][trial_idx[y == c]].mean(axis=0), f.bin_ms, cfg.data.smoothing_sigma_ms)
                rho_ramp[m] = spearman_trend(psth[m])[0]
        ramp_class = np.array([("all" if label_free else CLASSES[c]) if c >= 0 else "" for c in ramp_cls], dtype=object)
        # W: rate-normalised band power, Left vs Right, Bonferroni over bands
        p_spec = nan.copy()
        best_band = np.array([""] * n_units, dtype=object)
        if f.cand_idx.size and n_l >= 2 and n_r >= 2 and not label_free:
            resid = _rate_normalised_band_power(f.band_power[trial_idx], counts[:, f.cand_idx])   # (n_sub, n_cand, n_bands)
            pb = np.stack([mannwhitney_vectorised(resid[li][:, :, b], resid[ri][:, :, b])[1] for b in range(resid.shape[2])], axis=1)
            pb_f = np.where(np.isfinite(pb), pb, np.inf)
            has = np.isfinite(pb).any(axis=1)
            p_spec[f.cand_idx] = np.where(has, np.minimum(1.0, pb.shape[1] * pb_f.min(axis=1)), np.nan)
            best_band[f.cand_idx] = np.where(has, np.array(f.band_names, dtype=object)[pb_f.argmin(axis=1)], "")
        df = pd.DataFrame({
            "region": r, "unit_index": np.arange(n_units), "unit_id": f.unit_ids,
            "rate_hz": rate, "active_frac": active, "pass_floor": floor,
            "auroc_left_right": auroc_lr, "p_selectivity": p_sel, "preferred_class": pref,
            "auroc_ignore": auroc_ign, "p_ignore": p_ign,
            "rho_coupling": rho_cpl, "p_coupling": p_cpl,
            "p_spectral": p_spec, "spectral_band": best_band,
            "ramp_effect": ramp_eff, "ramp_slope_hz_s": slope, "ramp_class": ramp_class, "rho_ramp": rho_ramp, "p_ramp": p_ramp,
            **{f"mean_rate_{c}": v for c, v in means.items()},
        })
        if full and not label_free:
            tl = temporal_locus(f.win_counts[trial_idx], y, cfg, rng)
            n_win = f.win_counts.shape[2]
            starts = f.win_starts_ms
            on, en, pk = tl["onset_idx"], tl["end_idx"], tl["peak_idx"]
            df["p_locus"] = tl["p_locus"]
            df["onset_ms"] = np.where(on >= 0, starts[np.maximum(on, 0)], np.nan)
            df["locus_end_ms"] = np.where(en >= 0, starts[np.maximum(en, 0)] + f.window_ms, np.nan)
            df["peak_window_ms"] = np.where(pk >= 0, starts[np.maximum(pk, 0)], np.nan)
            a = np.abs(tl["auroc_win"] - 0.5)
            df["peak_auroc"] = np.where(pk >= 0, a[np.arange(n_units), np.maximum(pk, 0)] + 0.5, np.nan)
            late_w = starts >= (f.dur_s * 1000.0 - f.late_ms) - 1e-6
            if late_w.any():
                with np.errstate(invalid="ignore", divide="ignore"):
                    amax = a.max(axis=1)
                    df["late_fraction"] = np.where(amax > 0, a[:, late_w].mean(axis=1) / np.where(amax > 0, amax, 1.0), np.nan)
                df["late_auroc"] = a[:, late_w].mean(axis=1) + 0.5
            else:
                df["late_fraction"], df["late_auroc"] = np.nan, np.nan
            df["sustained_to_go"] = tl["sustained_to_go"]
            for w in range(n_win):
                df[f"auroc_win{w}"] = tl["auroc_win"][:, w]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["n_fit_trials"] = len(trial_idx)
    df["n_fit_left"], df["n_fit_right"], df["n_fit_ignore"] = n_l, n_r, n_i
    df["label_free"] = bool(label_free)
    fl = df.pass_floor.to_numpy()
    for k in ("selectivity", "coupling", "spectral", "ramp", "ignore") + (("locus",) if (full and not label_free) else ()):
        p = df[f"p_{k}"].to_numpy(dtype=float)
        tested = np.where(fl, p, np.nan)
        df[f"q_{k}"] = bh_fdr(tested)                       # floor first, then BH among the floor-passing tested units
        df[f"fdr_family_n_{k}"] = int(np.isfinite(tested).sum())
    df["fdr_family_n"] = int(fl.sum())
    return df


def _flag_and_score(df: pd.DataFrame, cfg) -> pd.DataFrame:
    sel = cfg.selection
    q = float(sel.fdr_q)
    for k in CRITERION_KEYS + ("ignore", "locus"):
        col = f"q_{k}"
        df[f"c_{k}"] = (df[col] < q).fillna(False).astype(bool) if col in df else False
    df["n_criteria"] = df[[f"c_{k}" for k in CRITERION_KEYS]].sum(axis=1)
    # The information onset / end are only meaningful for units whose cluster passes the permutation test.
    if "onset_ms" in df:
        ns = ~df["c_locus"].to_numpy(bool)
        df.loc[ns, ["onset_ms", "locus_end_ms"]] = np.nan
        if "sustained_to_go" in df:
            df.loc[ns, "sustained_to_go"] = False
    w = sel.weights
    score = np.zeros(len(df))
    for k in CRITERION_KEYS:
        qq = df[f"q_{k}"].fillna(1.0).clip(lower=1e-12).to_numpy(dtype=float)
        score += float(w.get(k, 1.0)) * (-np.log10(qq))
    df["score"] = np.where(df.pass_floor, score, -np.inf)
    is_lf = bool(df["label_free"].iloc[0]) if ("label_free" in df and len(df)) else False
    min_crit = int(sel.get_path("min_criteria_label_free", 1)) if is_lf else int(sel.min_criteria)
    df["eligible"] = df.pass_floor & (df.n_criteria >= min_crit)
    return df


def _topk(df: pd.DataFrame, cfg, by: list[str]) -> dict[str, np.ndarray]:
    k = int(cfg.selection.top_k_per_region)
    out = {}
    for r in REGIONS:
        sub = df[(df.region == r) & df.eligible].sort_values(by, ascending=[False] * len(by), kind="mergesort")
        out[r] = sub.unit_index.to_numpy(dtype=int)[:k]
    return out


def _stratified_subsample(labels: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for c in np.unique(labels):
        ii = np.flatnonzero(labels == c)
        n = min(len(ii), max(2, int(round(frac * len(ii)))))
        idx.append(rng.choice(ii, size=n, replace=False))
    return np.sort(np.concatenate(idx))


# --------------------------------------------------------------------------- public API
@dataclass
class SelectionResult:
    session: str
    table: pd.DataFrame                    # one row per unit with every criterion + reasons
    selected: dict[str, np.ndarray]        # region -> indices (into the region's unit axis), rank-sorted
    funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    trial_idx: np.ndarray | None = None    # trials the criteria were computed on (None = all)
    null_stability: np.ndarray | None = None   # median stability of top-K units under label permutation

    def n_selected(self) -> dict[str, int]:
        return {r: len(v) for r, v in self.selected.items()}


def unit_criteria(cache: SessionCache, cfg, trial_idx: np.ndarray | None = None,
                  features: dict[str, RegionFeatures] | None = None, labels: np.ndarray | None = None, seed: int = 0,
                  label_free: bool = False) -> pd.DataFrame:
    """Statistics, q-values, criterion flags and score for every unit (no stability, no ranking)."""
    feats = features or compute_features(cache, cfg)
    idx = np.arange(cache.n_trials) if trial_idx is None else np.asarray(trial_idx, dtype=int)
    labels = cache.labels if labels is None else np.asarray(labels)
    df = _criteria_table(feats, cache, labels, idx, cfg, full=True, rng=np.random.default_rng(seed), label_free=label_free)
    if df.empty:
        return df
    return _flag_and_score(df, cfg)


def stability_frequencies(cache: SessionCache, cfg, feats: dict[str, RegionFeatures], labels: np.ndarray, idx: np.ndarray,
                          key_pos: dict[tuple[str, int], int], rng: np.random.Generator, label_free: bool = False) -> tuple[np.ndarray, int]:
    """Selection frequency of every unit over stratified half-subsamples (without replacement) of ``idx``."""
    sel = cfg.selection
    n_sub = int(sel.get_path("n_subsamples", 50))
    frac = float(sel.get_path("subsample_frac", 0.5))
    hits = np.zeros(len(key_pos))
    n_done = 0
    strat = np.zeros(len(labels), int) if label_free else labels
    for _ in range(n_sub):
        sub = idx[_stratified_subsample(strat[idx], frac, rng)]
        yl = labels[sub]
        if not label_free and ((yl == LEFT).sum() < 2 or (yl == RIGHT).sum() < 2):
            continue
        dfb = _flag_and_score(_criteria_table(feats, cache, labels, sub, cfg, full=False, rng=rng, label_free=label_free), cfg)
        for r, units in _topk(dfb, cfg, ["score"]).items():
            for u in units:
                hits[key_pos[(r, int(u))]] += 1
        n_done += 1
    return hits / max(n_done, 1), n_done


def select_neurons(cache: SessionCache, cfg, trial_idx: np.ndarray | None = None,
                   features: dict[str, RegionFeatures] | None = None, seed: int = 0,
                   labels: np.ndarray | None = None, n_null: int | None = None, label_free: bool = False) -> SelectionResult:
    """Full selection on ``trial_idx`` (default: all trials): criteria -> stability -> ranked top-K per region.

    ``labels`` overrides the session labels (negative controls); ``n_null`` label-permuted repetitions give the
    chance level of the stability statistic (default ``selection.n_null_permutations``); ``label_free`` selects
    without ever reading the labels (held-out sessions: floor + coupling + net ramp, ``min_criteria_label_free``)."""
    sel = cfg.selection
    feats = features or compute_features(cache, cfg)
    idx = np.arange(cache.n_trials) if trial_idx is None else np.asarray(trial_idx, dtype=int)
    labels = cache.labels if labels is None else np.asarray(labels)
    rng = np.random.default_rng(seed)
    df = unit_criteria(cache, cfg, idx, feats, labels=labels, seed=seed, label_free=label_free)
    if df.empty:
        return SelectionResult(cache.session, df, {r: np.zeros(0, int) for r in REGIONS}, trial_idx=idx)
    key_pos = {kk: i for i, kk in enumerate(zip(df.region, df.unit_index))}
    df["stability"], n_done = stability_frequencies(cache, cfg, feats, labels, idx, key_pos, rng, label_free=label_free)
    df["n_subsamples"] = n_done
    min_stab = float(sel.get_path("min_stability", 0.6))
    df["stable"] = df.stability >= min_stab

    k = int(sel.top_k_per_region)
    fill = bool(sel.get_path("fill_unstable", False))
    selected: dict[str, np.ndarray] = {}
    df["selected"] = False
    df["rank"] = np.nan
    df["filled_by_score"] = False
    funnel_rows = []
    for r in REGIONS:
        reg = df[df.region == r]
        sub = reg[reg.eligible]
        chosen = sub[sub.stable].sort_values(["stability", "score"], ascending=False, kind="mergesort").head(k)
        n_fill = 0
        if fill and len(chosen) < k:
            rest = sub[~sub.index.isin(chosen.index)].sort_values("score", ascending=False, kind="mergesort").head(k - len(chosen))
            n_fill = len(rest)
            df.loc[rest.index, "filled_by_score"] = True
            chosen = pd.concat([chosen, rest])
        selected[r] = chosen.unit_index.to_numpy(dtype=int)
        df.loc[chosen.index, "selected"] = True
        df.loc[chosen.index, "rank"] = np.arange(1, len(chosen) + 1)
        n_el = int(reg.eligible.sum())
        # Meinshausen-Buehlmann bound on the expected number of falsely selected units, with q = the number
        # actually selected (K_eff) and p = the eligible candidates; it is informative only when q << p, so it is
        # capped at K_eff (a bound above the number selected says nothing).
        q_sel = int(len(chosen))
        bound = min(q_sel ** 2 / ((2 * min_stab - 1) * n_el), float(q_sel)) if (n_el > 0 and min_stab > 0.5 and q_sel > 0) else np.nan
        funnel_rows.append({"session": cache.session, "region": r, "recorded": int(len(reg)), "pass_floor": int(reg.pass_floor.sum()),
                            "eligible": n_el, "stable": int((reg.eligible & reg.stable).sum()), "selected": int(len(chosen)),
                            "K": k, "filled_by_score": int(n_fill), "expected_false_selections_bound": bound,
                            "n_fit_trials": int(len(idx))})
    # Pairwise agreement between the scored criteria among floor units (phi coefficients): how independent the
    # pieces of evidence are - a reviewer's first question.
    fl = df[df.pass_floor]
    phi = {}
    for i, a in enumerate(CRITERION_KEYS):
        for b in CRITERION_KEYS[i + 1:]:
            x, z = fl[f"c_{a}"].to_numpy(float), fl[f"c_{b}"].to_numpy(float)
            phi[f"phi_{CRITERION_LETTERS[a]}{CRITERION_LETTERS[b]}"] = float(np.corrcoef(x, z)[0, 1]) if (len(fl) > 2 and x.std() > 0 and z.std() > 0) else np.nan
    funnel = pd.DataFrame(funnel_rows)
    for kk, v in phi.items():
        funnel[kk] = v
    df["reasons"] = df.apply(lambda row: _reasons(row, cfg), axis=1)
    df["reason_short"] = df.apply(lambda row: _reason_short(row, cfg), axis=1)
    res = SelectionResult(cache.session, df, selected, funnel, trial_idx=idx)

    n_null = int(sel.get_path("n_null_permutations", 0)) if n_null is None else int(n_null)
    if n_null > 0 and not label_free:
        res.null_stability = null_stability(cache, cfg, feats, idx, n_null, seed + 1)
        funnel["null_median_stability_max"] = float(np.max(res.null_stability)) if res.null_stability.size else np.nan
    return res


def null_stability(cache: SessionCache, cfg, feats: dict[str, RegionFeatures], idx: np.ndarray, n_null: int, seed: int) -> np.ndarray:
    """Median stability of the top-K units when the class labels are permuted (what 'stable' looks like by chance)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        perm = cache.labels.copy()
        perm[idx] = rng.permutation(perm[idx])
        df = _flag_and_score(_criteria_table(feats, cache, perm, idx, cfg, full=False, rng=rng), cfg)
        key_pos = {kk: i for i, kk in enumerate(zip(df.region, df.unit_index))}
        df["stability"], _ = stability_frequencies(cache, cfg, feats, perm, idx, key_pos, rng)
        top = pd.concat([df[(df.region == r) & df.eligible].sort_values(["stability", "score"], ascending=False).head(int(cfg.selection.top_k_per_region))
                         for r in REGIONS])
        out.append(float(top.stability.median()) if len(top) else 0.0)
    return np.asarray(out)


def _fmt_q(q) -> str:
    return f"q={q:.1e}" if np.isfinite(q) else "q=n/a"


_SHORT_REGION = {"ALM_L": "lALM", "ALM_R": "rALM", "STR_L": "lSTR", "STR_R": "rSTR"}


def _reason_short(row: pd.Series, cfg) -> str:
    """Fixed-field one-liner (<= ~110 chars) for figure tables; unmet criteria are shown as an en dash.

    ``lALM u1234 #1/32 stab 96% | S 0.78 q=1e-9 ->Right | C +0.34 q=1e-3 | W theta q=4e-4 | R up@Right q=3e-5 | T 400ms sust``"""
    sel = cfg.selection
    k = int(sel.top_k_per_region)
    reg = _SHORT_REGION.get(row.region, row.region)
    if row.selected:
        status = f"#{int(row['rank'])}/{k} stab {row.stability:.0%}"
    elif not row.pass_floor:
        status = "below floor"
    elif not row.eligible:
        status = f"{int(row.n_criteria)} crit"
    elif not row.get("stable", False):
        status = f"unstable {row.stability:.0%}"
    else:
        status = f">K stab {row.stability:.0%}"
    q = lambda v: f"q={v:.0e}" if np.isfinite(v) else "q=n/a"
    s_ = f"S {row.auroc_left_right:.2f} {q(row.q_selectivity)} ->{row.preferred_class[0]}" if row.c_selectivity else "S –"
    c_ = f"C {row.rho_coupling:+.2f} {q(row.q_coupling)}" if row.c_coupling else "C –"
    w_ = f"W {row.spectral_band} {q(row.q_spectral)}" if row.c_spectral else ("W –" if np.isfinite(row.p_spectral) else "W n/t")
    r_ = f"R {'up' if row.ramp_slope_hz_s > 0 else 'dn'}@{str(row.ramp_class)[0]} {q(row.q_ramp)}" if row.c_ramp else "R –"
    if bool(row.get("c_locus", False)) and np.isfinite(row.get("onset_ms", np.nan)):
        t_ = f"T {int(row.onset_ms)}ms " + ("sust" if row.sustained_to_go else f"-{int(row.locus_end_ms)}ms")
    else:
        t_ = "T –"
    return f"{reg} u{row.unit_id} {status} | {s_} | {c_} | {w_} | {r_} | {t_}"


def _reasons(row: pd.Series, cfg) -> str:
    sel = cfg.selection
    k = int(sel.top_k_per_region)
    head = (f"fit trials n={int(row.n_fit_left)}/{int(row.n_fit_right)}/{int(row.n_fit_ignore)} (L/R/I) | "
            f"delay rate {row.rate_hz:.1f} Hz (L {row.mean_rate_Left:.1f}, R {row.mean_rate_Right:.1f}"
            + (f", I {row.mean_rate_Ignore:.1f}" if np.isfinite(row.mean_rate_Ignore) else "") + ")")
    if not row.pass_floor:
        return head + f" | below activity floor (needs >= {sel.min_rate_hz} Hz and spikes on >= {sel.min_active_trial_frac:.0%} of trials)"
    parts = [head]
    if bool(row.get("label_free", False)):
        parts.append("label-free selection (held-out session: S/W/T/I not tested)")
    if row.c_selectivity:
        parts.append(f"S {row.preferred_class}-preferring delay rate (AUROC_LR={row.auroc_left_right:.2f}, {_fmt_q(row.q_selectivity)})")
    if bool(row.get("c_locus", False)) and np.isfinite(row.get("onset_ms", np.nan)):
        tail = "sustained to go" if row.sustained_to_go else f"until {int(row.locus_end_ms)} ms"
        parts.append(f"T information onset {int(row.onset_ms)} ms after delay start, {tail}; peak {int(row.peak_window_ms)}-"
                     f"{int(row.peak_window_ms + sel.get_path('window_ms', 200))} ms (AUROC {row.peak_auroc:.2f}), late-window AUROC {row.late_auroc:.2f} ({_fmt_q(row.q_locus)})")
    if row.c_coupling:
        parts.append(f"C late-delay ({int(sel.late_delay_ms)} ms) rate predicts own response-epoch rate within class (rho={row.rho_coupling:+.2f}, {_fmt_q(row.q_coupling)})")
    if row.c_spectral:
        parts.append(f"W Left/Right-dependent {row.spectral_band}-band wavelet power after rate normalisation ({_fmt_q(row.q_spectral)})")
    if row.c_ramp:
        parts.append(f"R ramps {'up' if row.ramp_slope_hz_s > 0 else 'down'} {row.ramp_slope_hz_s:+.1f} Hz/s on {row.ramp_class} trials "
                     f"(late {'>' if row.ramp_effect > 0 else '<'} early on {(abs(row.ramp_effect) + 1) / 2:.0%} of them, {_fmt_q(row.q_ramp)})")
    if bool(row.get("c_ignore", False)):
        parts.append(f"I no-lick (Ignore) trials {'higher' if row.auroc_ignore > 0.5 else 'lower'} rate (AUROC={row.auroc_ignore:.2f}, {_fmt_q(row.q_ignore)}; descriptive)")
    n_crit = int(row.n_criteria)
    if "stability" in row.index and np.isfinite(row.stability):
        stab = f"{int(round(row.stability * row.n_subsamples))}/{int(row.n_subsamples)} half-subsamples"
        if row.selected:
            parts.append(f"selected in {stab} (rank {int(row['rank'])}/{k})" + (" [kept to fill K despite low stability]" if bool(row.get("filled_by_score", False)) else ""))
        elif row.eligible and not row.stable:
            parts.append(f"eligible ({n_crit} criteria) but unstable: selected in only {stab} (< {sel.get_path('min_stability', 0.6):.0%})")
        elif row.eligible:
            parts.append(f"eligible and stable ({stab}) but ranked below K={k}")
        else:
            parts.append(f"not eligible: {n_crit} of >= {int(sel.min_criteria)} criteria")
    elif not row.eligible:
        parts.append(f"not eligible: {n_crit} of >= {int(sel.min_criteria)} criteria")
    return " | ".join(parts)


def selection_summary(results: list[SelectionResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        t = res.table
        row = {"session": res.session, "n_units": len(t), "n_floor": int(t.pass_floor.sum()) if len(t) else 0,
               "n_eligible": int(t.eligible.sum()) if len(t) else 0, "n_selected": int(t.selected.sum()) if len(t) else 0,
               "n_trials_used": int(len(res.trial_idx)) if res.trial_idx is not None else -1}
        for r in REGIONS:
            row[f"sel_{r}"] = int(((t.region == r) & t.selected).sum()) if len(t) else 0
            row[f"tot_{r}"] = int((t.region == r).sum()) if len(t) else 0
        for kk in CRITERION_KEYS + ("locus", "ignore"):
            row[f"frac_{kk}"] = float(t.loc[t.pass_floor, f"c_{kk}"].mean()) if len(t) and t.pass_floor.any() else np.nan
        if len(t) and "stability" in t and t.selected.any():
            row["median_stability_selected"] = float(t.loc[t.selected, "stability"].median())
            row["median_onset_ms_selected"] = float(t.loc[t.selected, "onset_ms"].median()) if "onset_ms" in t else np.nan
            row["frac_sustained_to_go_selected"] = float(t.loc[t.selected, "sustained_to_go"].mean()) if "sustained_to_go" in t else np.nan
        if res.null_stability is not None and res.null_stability.size:
            row["null_median_stability_max"] = float(np.max(res.null_stability))
        if len(res.funnel):
            for c in res.funnel.columns:
                if c.startswith("phi_"):
                    row[c] = float(res.funnel[c].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)
