"""Vectorised non-parametric statistics used by the neuron-selection criteria.

All functions operate on every unit at once (units along the last axis) so that thousands of units and
dozens of bootstrap resamples cost milliseconds instead of hundreds of thousands of scipy calls. The
results are numerically identical to ``scipy.stats.kruskal`` / ``scipy.stats.spearmanr`` /
``scipy.stats.mannwhitneyu(method="asymptotic", use_continuity=False)`` /
``scipy.stats.wilcoxon(correction=False, method="approx")`` (all tie-corrected).  The tie term
``sum(t^3 - t)`` is obtained for all columns at once from run lengths of the column-sorted ranks
(``_tie_term``), so no Python loop over units remains.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values); NaNs are ignored and returned as NaN."""
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


def _tie_term(ranks: np.ndarray) -> np.ndarray:
    """``sum(t^3 - t)`` over the tie groups of every column of an (n, m) rank matrix, without a column loop.

    The columns are sorted, a run of equal values is a tie group, and the run lengths are counted with one
    ``bincount`` over ``(column, run id)`` keys where the run id is the cumulative number of value changes.
    NaN entries (used by ``wilcoxon_vectorised`` for zero differences) compare unequal to everything, so
    each forms a run of length 1 and contributes nothing - exactly the "ignore zeros" behaviour wanted.
    Groups of size 1 contribute ``1 - 1 = 0``, so no filtering is required.
    """
    n, m = ranks.shape
    if n < 2 or m == 0:
        return np.zeros(m)
    srt = np.sort(ranks, axis=0)
    change = np.ones((n, m), dtype=bool)
    change[1:] = srt[1:] != srt[:-1]
    run_id = np.cumsum(change, axis=0) - 1                     # 0-based run index inside each column
    key = run_id + np.arange(m, dtype=np.int64)[None, :] * n   # unique (column, run) key
    t = np.bincount(key.ravel(), minlength=n * m).reshape(m, n).astype(np.int64)
    return (t ** 3 - t).sum(axis=1).astype(float)


def _tie_correction(ranks: np.ndarray) -> np.ndarray:
    """1 - sum(t^3 - t) / (n^3 - n) per column of a (n, m) rank matrix (vectorised over columns)."""
    n, m = ranks.shape
    if n < 2:
        return np.ones(m)
    return 1.0 - _tie_term(ranks) / float(n ** 3 - n)


def kruskal_vectorised(values: np.ndarray, labels: np.ndarray, min_per_group: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Kruskal-Wallis H test of ``values`` (n_trials, n_units) across the groups in ``labels``.

    Groups with fewer than ``min_per_group`` trials are dropped. Returns (H, p) per unit; units whose values are
    constant across all trials get NaN (no test possible), matching scipy's behaviour of raising for such input.
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    n_tr, n_units = values.shape
    groups = [g for g in np.unique(labels) if (labels == g).sum() >= min_per_group]
    H = np.full(n_units, np.nan)
    p = np.full(n_units, np.nan)
    if len(groups) < 2 or n_tr < 3:
        return H, p
    keep = np.isin(labels, groups)
    v = values[keep]
    lab = labels[keep]
    n = v.shape[0]
    ranks = rankdata(v, axis=0)  # average ranks, ties handled
    ssbn = np.zeros(n_units)
    for g in groups:
        m = lab == g
        rsum = ranks[m].sum(axis=0)
        ssbn += rsum ** 2 / m.sum()
    h = 12.0 / (n * (n + 1)) * ssbn - 3 * (n + 1)
    corr = _tie_correction(ranks)
    valid = corr > 0
    h = np.where(valid, h / np.where(valid, corr, 1.0), np.nan)
    df = len(groups) - 1
    pv = stats.chi2.sf(h, df)
    H[:] = h
    p[:] = pv
    return H, p


def spearman_vectorised(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman rho and two-sided p-value between columns of x and y ((n, m) each). Constant columns -> NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]
    rho = np.full(x.shape[1], np.nan)
    p = np.full(x.shape[1], np.nan)
    if n < 3:
        return rho, p
    rx = rankdata(x, axis=0)
    ry = rankdata(y, axis=0)
    rx = rx - rx.mean(axis=0, keepdims=True)
    ry = ry - ry.mean(axis=0, keepdims=True)
    sx = np.sqrt((rx ** 2).sum(axis=0))
    sy = np.sqrt((ry ** 2).sum(axis=0))
    ok = (sx > 0) & (sy > 0)
    r = np.full(x.shape[1], np.nan)
    r[ok] = (rx[:, ok] * ry[:, ok]).sum(axis=0) / (sx[ok] * sy[ok])
    r = np.clip(r, -1, 1)
    # t-distribution approximation used by scipy.stats.spearmanr
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-300))
    pv = 2 * stats.t.sf(np.abs(t), n - 2)
    pv = np.where(np.abs(r) >= 1, 0.0, pv)
    rho[ok] = r[ok]
    p[ok] = pv[ok]
    return rho, p


def spearman_trend(psth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic-trend test: Spearman rho of each row of ``psth`` (n_units, T) against time."""
    psth = np.asarray(psth, dtype=float)
    T = psth.shape[1]
    t = np.tile(np.arange(T, dtype=float)[:, None], (1, psth.shape[0]))
    return spearman_vectorised(t, psth.T)


def eta_squared_vectorised(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Effect size eta^2 = SS_between / SS_total per unit (values: (n_trials, n_units))."""
    values = np.asarray(values, dtype=float)
    grand = values.mean(axis=0)
    ss_tot = ((values - grand) ** 2).sum(axis=0)
    ss_b = np.zeros_like(grand)
    for g in np.unique(labels):
        m = labels == g
        ss_b += m.sum() * (values[m].mean(axis=0) - grand) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(ss_tot > 0, ss_b / ss_tot, 0.0)
    return eta


def cohens_d_vectorised(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cohen's d (pooled SD) between two (n_a, m) and (n_b, m) samples per column."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(a.shape[1], np.nan)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return out
    s = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
    d = a.mean(axis=0) - b.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(s > 0, d / s, 0.0)
    return out


def auroc_vectorised(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Area under the ROC curve (Mann-Whitney U / (n_pos n_neg)) per column; 0.5 when either sample is empty."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    m = pos.shape[1]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        return np.full(m, 0.5)
    allv = np.concatenate([pos, neg], axis=0)
    ranks = rankdata(allv, axis=0)
    n_pos = pos.shape[0]
    u = ranks[:n_pos].sum(axis=0) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * neg.shape[0])


def wilcoxon_vectorised(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wilcoxon signed-rank test of paired differences ``d`` (n, m) against zero, per column.

    Normal approximation with tie and zero corrections (Pratt's treatment of zeros is not used; zeros are
    dropped as in scipy's default ``zero_method='wilcox'``). Returns (effect, p) where ``effect`` is the
    fraction of positive minus negative differences (in [-1, 1]); columns with < 5 non-zero differences get
    NaN.

    Fully vectorised: zero differences are turned into NaN before ranking ``|d|`` with
    ``rankdata(nan_policy="omit")`` (NaNs get NaN ranks and do not consume ranks), and the tie term of the
    variance comes from ``_tie_term`` on the same NaN-carrying rank matrix.  Identical to
    ``scipy.stats.wilcoxon(correction=False, method="approx")`` for every column with >= 5 non-zeros.
    """
    d = np.asarray(d, dtype=float)
    n_all, m = d.shape
    effect = np.full(m, np.nan)
    p = np.full(m, np.nan)
    nz = d != 0
    n = nz.sum(axis=0)
    ok = n >= 5
    if not ok.any():
        return effect, p
    absd = np.where(nz, np.abs(d), np.nan)
    ranks = rankdata(absd, axis=0, nan_policy="omit")   # NaN where d == 0, average ranks elsewhere
    r_pos = np.where(d > 0, np.nan_to_num(ranks, nan=0.0), 0.0).sum(axis=0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - _tie_term(ranks) / 48.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (r_pos - mu) / np.sqrt(var)
    pv = 2 * stats.norm.sf(np.abs(z))
    p[ok] = pv[ok]
    effect[ok] = ((d > 0).sum(axis=0) - (d < 0).sum(axis=0))[ok] / n[ok]
    return effect, p


def mannwhitney_vectorised(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U test per column between samples ``a`` (n_a, m) and ``b`` (n_b, m).

    Returns (AUROC, p): AUROC = P(a > b) + 0.5 P(a == b) (= U / n_a n_b); two-sided p from the normal
    approximation with tie correction (identical to ``scipy.stats.mannwhitneyu(method='asymptotic',
    use_continuity=False)``). Constant columns give AUROC 0.5 and p = NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.shape[0], b.shape[0]
    m = a.shape[1]
    auroc = np.full(m, 0.5)
    p = np.full(m, np.nan)
    if n_a < 1 or n_b < 1:
        return auroc, p
    allv = np.concatenate([a, b], axis=0)
    ranks = rankdata(allv, axis=0)
    u = ranks[:n_a].sum(axis=0) - n_a * (n_a + 1) / 2
    auroc = u / (n_a * n_b)
    n = n_a + n_b
    mu = n_a * n_b / 2.0
    corr = _tie_correction(ranks)            # 1 - sum(t^3 - t)/(n^3 - n)
    var = n_a * n_b / 12.0 * ((n + 1) - (1 - corr) * (n ** 3 - n) / (n * (n - 1))) if n > 1 else np.zeros(m)
    ok = np.asarray(var) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (u - mu) / np.sqrt(np.where(ok, var, 1.0))
    pv = 2 * stats.norm.sf(np.abs(z))
    p[ok] = pv[ok]
    return auroc, p


def within_class_rank_corr(x: np.ndarray, y: np.ndarray, labels: np.ndarray, n_perm: int = 500,
                           rng: np.random.Generator | None = None, min_per_class: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditioned rank correlation between columns of x and y, with a drift-preserving permutation p.

    Within every class (with >= ``min_per_class`` trials) the trials are ranked and centred, so a shared class
    variable cannot create a correlation. The null distribution is built from ``n_perm`` *circular shifts* of
    the y-ranks along the trial order inside each class (the same shift for every unit), which keeps the slow
    autocorrelation (drift) of both variables intact and only destroys their trial-by-trial pairing.
    Returns (rho, p) per column; columns constant within all classes give NaN.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = x.shape
    rx = np.zeros_like(x)
    ry = np.zeros_like(y)
    groups = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < min_per_class:
            continue
        groups.append(idx)
        rx[idx] = rankdata(x[idx], axis=0) - (len(idx) + 1) / 2
        ry[idx] = rankdata(y[idx], axis=0) - (len(idx) + 1) / 2
    rho = np.full(m, np.nan)
    p = np.full(m, np.nan)
    if not groups:
        return rho, p
    sx = np.sqrt((rx ** 2).sum(axis=0))
    sy = np.sqrt((ry ** 2).sum(axis=0))
    ok = (sx > 0) & (sy > 0)
    if not ok.any():
        return rho, p
    denom = np.where(ok, sx * sy, 1.0)
    r_obs = (rx * ry).sum(axis=0) / denom
    count = np.zeros(m)
    for _ in range(int(n_perm)):
        ry_p = ry.copy()
        for idx in groups:
            k = int(rng.integers(1, len(idx)))
            ry_p[idx] = np.roll(ry[idx], k, axis=0)
        r_p = (rx * ry_p).sum(axis=0) / denom
        count += np.abs(r_p) >= np.abs(r_obs) - 1e-12
    rho[ok] = r_obs[ok]
    p[ok] = (1.0 + count[ok]) / (1.0 + n_perm)
    return rho, p
