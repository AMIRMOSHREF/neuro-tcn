"""Criterion-based selection of delay-epoch neurons that are informative about the upcoming action.

For every unit of every session we compute, on the delay (context) epoch:

1. **Activity floor**        mean rate >= ``min_rate_hz`` and active on >= ``min_active_trial_frac`` of trials.
2. **Choice selectivity**    Kruskal-Wallis test of delay spike counts across {Ignore, Left, Right}
                             (+ Cohen's d Left vs Right as an effect size); BH-FDR q < ``fdr_q``.
3. **Delay->response coupling**  Spearman correlation across trials between the unit's late-delay
                             rate and its own response-epoch rate; the past of the neuron predicts its future.
4. **Spectro-temporal selectivity**  Kruskal-Wallis across classes of Morlet-CWT band power
                             (slow / theta / beta) of the delay-epoch rate; captures rhythmic or
                             transient structure that mean rate misses.
5. **Ramping**               linear-trend test (Spearman rho of trial-averaged PSTH vs time), the
                             signature of preparatory activity in ALM.

Units that pass the floor and satisfy >= ``min_criteria`` statistical criteria are ranked by a
weighted sum of -log10(q) values; the top-K per region are selected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .. import CLASSES, REGIONS
from ..data.cache import SessionCache
from .spectral import band_power_cwt, smooth_rates


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def _kruskal(values: np.ndarray, labels: np.ndarray) -> float:
    groups = [values[labels == k] for k in range(len(CLASSES)) if (labels == k).sum() >= 2]
    if len(groups) < 2 or all(np.ptp(g) == 0 for g in groups):
        return np.nan
    try:
        return float(stats.kruskal(*groups).pvalue)
    except ValueError:
        return np.nan


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def _eta_sq(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_b = sum(((values[labels == k].mean() - grand) ** 2) * (labels == k).sum() for k in np.unique(labels))
    return float(ss_b / ss_tot)


@dataclass
class SelectionResult:
    session: str
    table: pd.DataFrame                    # one row per unit with every criterion + reasons
    selected: dict[str, np.ndarray]        # region -> indices (into the region's unit axis), score-sorted

    def n_selected(self) -> dict[str, int]:
        return {r: len(v) for r, v in self.selected.items()}


def unit_criteria(cache: SessionCache, cfg) -> pd.DataFrame:
    sel = cfg.selection
    bin_ms = cache.bin_ms
    y = cache.labels
    rows = []
    late_bins = int(round(sel.late_delay_ms / bin_ms))
    bands = {k: list(v) for k, v in sel.bands_hz.items()}
    for r in REGIONS:
        X = cache.context[r]           # (n_trials, n_units, T)
        Y = cache.target[r]            # (n_trials, n_units, T_tgt)
        n_tr, n_units, T = X.shape
        if n_units == 0:
            continue
        dur_s = T * bin_ms / 1000.0
        counts = X.sum(axis=2)                        # (n_trials, n_units)
        rate = counts / dur_s
        active_frac = (counts > 0).mean(axis=0)
        late_rate = X[:, :, -late_bins:].sum(axis=2) / (late_bins * bin_ms / 1000.0)
        resp_rate = Y.sum(axis=2) / (Y.shape[2] * cache.target_bin_ms / 1000.0)
        psth = smooth_rates(X.mean(axis=0), bin_ms, cfg.data.smoothing_sigma_ms)   # (n_units, T)
        tvec = np.arange(T)
        # Spectro-temporal: CWT band power per trial per unit on smoothed single-trial rates.
        rates_tr = smooth_rates(X.reshape(n_tr * n_units, T), bin_ms, cfg.data.smoothing_sigma_ms)
        bp, band_names = band_power_cwt(rates_tr, bin_ms, bands, sel.wavelet)   # (n_tr*n_units, n_bands)
        bp = bp.reshape(n_tr, n_units, -1)
        for u in range(n_units):
            p_sel = _kruskal(counts[:, u], y)
            d_lr = _cohens_d(counts[y == 1, u], counts[y == 2, u])
            eta = _eta_sq(counts[:, u], y)
            if np.ptp(late_rate[:, u]) > 0 and np.ptp(resp_rate[:, u]) > 0:
                rho, p_cpl = stats.spearmanr(late_rate[:, u], resp_rate[:, u])
            else:
                rho, p_cpl = np.nan, np.nan
            p_bands = [_kruskal(bp[:, u, b], y) for b in range(bp.shape[2])]
            p_spec = np.nanmin(p_bands) if np.any(np.isfinite(p_bands)) else np.nan
            best_band = band_names[int(np.nanargmin(p_bands))] if np.isfinite(p_spec) else ""
            if np.ptp(psth[u]) > 0:
                ramp_rho, p_ramp = stats.spearmanr(tvec, psth[u])
            else:
                ramp_rho, p_ramp = np.nan, np.nan
            rows.append({
                "region": r, "unit_index": u, "unit_id": cache.unit_ids[r][u],
                "rate_hz": float(rate[:, u].mean()), "active_frac": float(active_frac[u]),
                "p_selectivity": p_sel, "eta_sq": eta, "d_left_right": d_lr,
                "p_coupling": float(p_cpl) if np.isfinite(p_cpl) else np.nan, "rho_coupling": float(rho) if np.isfinite(rho) else np.nan,
                "p_spectral": p_spec, "spectral_band": best_band,
                "p_ramp": float(p_ramp) if np.isfinite(p_ramp) else np.nan, "ramp_rho": float(ramp_rho) if np.isfinite(ramp_rho) else np.nan,
                **{f"mean_rate_{c}": float(rate[y == i, u].mean()) if (y == i).any() else np.nan for i, c in enumerate(CLASSES)},
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in ("selectivity", "coupling", "spectral", "ramp"):
        df[f"q_{k}"] = bh_fdr(df[f"p_{k}"].to_numpy())
    return df


def select_neurons(cache: SessionCache, cfg) -> SelectionResult:
    sel = cfg.selection
    df = unit_criteria(cache, cfg)
    if df.empty:
        return SelectionResult(cache.session, df, {r: np.zeros(0, int) for r in REGIONS})
    q = sel.fdr_q
    df["pass_floor"] = (df.rate_hz >= sel.min_rate_hz) & (df.active_frac >= sel.min_active_trial_frac)
    df["c_selectivity"] = df.q_selectivity < q
    df["c_coupling"] = df.q_coupling < q
    df["c_spectral"] = df.q_spectral < q
    df["c_ramp"] = df.q_ramp < q
    crit_cols = ["c_selectivity", "c_coupling", "c_spectral", "c_ramp"]
    df["n_criteria"] = df[crit_cols].sum(axis=1)
    w = sel.weights
    score = np.zeros(len(df))
    for k in ("selectivity", "coupling", "spectral", "ramp"):
        qq = df[f"q_{k}"].fillna(1.0).clip(lower=1e-12).to_numpy()
        score += float(w[k]) * (-np.log10(qq))
    df["score"] = np.where(df.pass_floor, score, -np.inf)
    df["eligible"] = df.pass_floor & (df.n_criteria >= sel.min_criteria)

    def reasons(row) -> str:
        out = []
        if not row.pass_floor:
            out.append(f"below activity floor ({row.rate_hz:.1f} Hz, active {row.active_frac:.0%})")
            return "; ".join(out)
        if row.c_selectivity:
            out.append(f"choice-selective delay rate (q={row.q_selectivity:.1e}, eta2={row.eta_sq:.2f}, d(L-R)={row.d_left_right:+.2f})")
        if row.c_coupling:
            out.append(f"late-delay rate predicts own response rate (rho={row.rho_coupling:+.2f}, q={row.q_coupling:.1e})")
        if row.c_spectral:
            out.append(f"class-dependent {row.spectral_band}-band wavelet power (q={row.q_spectral:.1e})")
        if row.c_ramp:
            out.append(f"{'up' if row.ramp_rho > 0 else 'down'}-ramping delay PSTH (rho={row.ramp_rho:+.2f}, q={row.q_ramp:.1e})")
        if not out:
            out.append("no significant criterion")
        return "; ".join(out)

    df["reasons"] = df.apply(reasons, axis=1)
    selected: dict[str, np.ndarray] = {}
    df["selected"] = False
    df["rank"] = np.nan
    for r in REGIONS:
        sub = df[(df.region == r) & df.eligible].sort_values("score", ascending=False).head(int(sel.top_k_per_region))
        selected[r] = sub.unit_index.to_numpy(dtype=int)
        df.loc[sub.index, "selected"] = True
        df.loc[sub.index, "rank"] = np.arange(1, len(sub) + 1)
    return SelectionResult(cache.session, df, selected)


def selection_summary(results: list[SelectionResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        t = res.table
        row = {"session": res.session, "n_units": len(t), "n_eligible": int(t.eligible.sum()) if len(t) else 0,
               "n_selected": int(t.selected.sum()) if len(t) else 0}
        for r in REGIONS:
            row[f"sel_{r}"] = int(((t.region == r) & t.selected).sum()) if len(t) else 0
            row[f"tot_{r}"] = int((t.region == r).sum()) if len(t) else 0
        for k in ("selectivity", "coupling", "spectral", "ramp"):
            row[f"frac_{k}"] = float(t[f"c_{k}"].mean()) if len(t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
