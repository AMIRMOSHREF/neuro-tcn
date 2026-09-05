"""Unit identity recovery for exports that list, per trial, only the units that fired - in one fixed order.

The Data2 trial files carry no unit IDs: every region array holds the spike trains of the units that fired in
that trial, and nothing says which row of one trial is which row of the next.  On the three recordings present
in both trees the rows could be identified by their spike trains, which showed (``twin_unit_order_check``) that
the export writes each session's units in **one fixed order** and merely omits the silent ones (order
consistency 0.998, every row identified).  A trial's row list is therefore a *subsequence* of one master unit
list per region, and identity is recoverable by sequence alignment:

* the master list starts as the trial with the most rows (it holds ~97 % of the units);
* every trial is aligned to it by a monotone dynamic programme (a profile alignment): row i is assigned to a
  master slot k with all assignments strictly increasing in k, a skipped slot costs the log-probability that the
  unit is silent in a trial, an unassignable row (a unit missing from the master) costs an insertion penalty,
  and the match score is a diagonal Gaussian of the row's **fingerprint** against the slot's profile;
* the fingerprint of a row is what a unit keeps from trial to trial: its log firing rate over the trial
  (``rate``), the *shape* of its PSTH - the log rate in six task windows (pre-sample, sample, early / late delay,
  early / late response) relative to the trial rate (``windows``: a delay-ramping unit and a go-responsive unit
  of the same mean rate are told apart, and a trial in which the whole unit fires more moves the rate, not the
  shape) - and two rate-free spike-train statistics, the median-over-mean ISI ratio and the log ISI coefficient
  of variation (``isi``: burstiness and regularity).  A statistic that a row cannot provide (an ISI shape from
  fewer than five spikes, a task window the trial does not cover) is simply left out of that row's score;
* the match score is a Gaussian with per-slot means and sds and one **pooled correlation matrix** shared by all
  slots (the features are not independent: the six shape values are compositional, the two ISI statistics move
  together, and a trial-wide gain would otherwise be counted once per window - with independent features the
  score of a correct row has a far heavier tail than chi-square and rows of the right unit get rejected);
  a row is inserted when its best fit is beyond the 1 - ``p_insert`` chi-square quantile for the features it has;
* after each pass the slot profiles (mean / sd per feature, probability of absence) are re-estimated from the
  assignments - a slot's sd is shrunk toward the typical slot sd of that feature, so a feature is weighted by how
  reliable it is for that unit - and slots are created where at least ``support`` trials inserted a consistent
  row at the same place (never from the first pass, whose master is a single trial); slots that end up with
  fewer than ``support`` rows are pruned, adjacent slots that are never co-assigned and have the same profile are
  merged (a split unit), and one last pass aligns every trial to the final master.

The recovered slots are unit IDs in every sense the cache needs (one row per unit per trial, silent units = zero
rows).  On the twin recordings the recovery is validated against the true IDs (``twin_identity_accuracy``), and
that number is what justifies using it on the sessions that have no twin.  The recovery never looks at the trial
label: an alignment that used the class could route ambiguous rows by label and leak it into the rasters.
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .. import REGIONS
from .rasters import read_epochs, resolve_spike_time_reference, spikes_by_region

log = logging.getLogger(__name__)

FEATURE_SETS = ("rate", "windows", "isi")
WINDOW_NAMES = ("pre_sample", "sample", "delay_early", "delay_late", "response_early", "response_late")
ISI_NAMES = ("isi_median_over_mean", "isi_log_cv")
MIN_ISI_SPIKES = 5       # spikes a row needs before its ISI statistics count
MIN_WINDOW_S = 0.1       # a task window shorter than this gives no feature
P_ABS_MIN, P_ABS_MAX = 0.01, 0.99
FLOOR_RATE, FLOOR_ISI = 0.25, 0.15     # floors of a slot's sd (a unit's rate / ISI shape is not constant across trials)
PRIOR_WEIGHT = 2.0       # pseudo-observations shrinking a slot's sd toward the typical slot sd of the feature
INIT_SD_MULT = 3.0       # first pass: the master is one trial's fingerprints, so a row is expected this far (x floor) from it
NEW_SLOT_MAX_SPREAD = 1.5             # inserted rows at one place must agree (median sd ratio) before a slot is created
SPLIT_MAX_DIST = 1.0                   # standardised profile distance under which never-co-assigned neighbours merge
SPLIT_MAX_DIST_STRONG = 1.5            # ... when their absence from the same trials is far below chance (strong evidence)
SPLIT_MAX_CO = 0.02                    # fraction of trials in which both may be assigned
SPLIT_MIN_EXPECTED_CO = 5.0            # co-assignments expected under independence before the absence counts as strong
Z_CAP = 6.0                            # a single feature cannot reject a row by more than this many sds (heavy tails); above the
                                       # insertion threshold of a one-feature fingerprint plus the largest gap cost
CORR_SHRINK = 0.1                      # pooled correlation matrix shrunk toward the identity


@dataclass
class IdentityMap:
    session: str
    rows: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)   # trial -> region -> slot per row (-1 = none)
    n_slots: dict[str, int] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"session": self.session, "rows": self.rows, "n_slots": self.n_slots, "stats": self.stats}, f)

    @classmethod
    def load(cls, path: Path) -> "IdentityMap":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["session"], d["rows"], d["n_slots"], d["stats"])


# ------------------------------------------------------------------------------------------------ fingerprints
def feature_names(sets=FEATURE_SETS) -> list[str]:
    sets = tuple(str(s).lower() for s in sets)
    unknown = set(sets) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"unknown identity feature set(s) {sorted(unknown)}; choose from {FEATURE_SETS}")
    names: list[str] = []
    if "rate" in sets:
        names.append("log_rate")
    if "windows" in sets:
        names += [(f"shape_{w}" if "rate" in sets else f"log_rate_{w}") for w in WINDOW_NAMES]
    if "isi" in sets:
        names += list(ISI_NAMES)
    if not names:
        raise ValueError("identity recovery needs at least one feature set (rate, windows, isi)")
    return names


def feature_floors(names) -> np.ndarray:
    return np.array([FLOOR_ISI if n.startswith("isi") else FLOOR_RATE for n in names], dtype=float)


def trial_windows(ep: dict) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """The trial window and the six task windows ``(start, stop)`` on the time base of the epoch scalars."""
    ds, go = float(ep["delay_start_times"]), float(ep["go_start_times"])
    t0, t1 = float(ep.get("trial_start", np.nan)), float(ep.get("trial_stop", np.nan))
    ss = float(ep.get("sample_start_times", np.nan))
    if not np.isfinite(ss) or ss >= ds:
        ss = ds - 1.3
    if not np.isfinite(t0) or t0 >= ss:
        t0 = ss - 1.0
    if not np.isfinite(t1) or t1 <= go:
        t1 = go + 2.0
    mid = ds + 0.5 * (go - ds)
    r1 = min(go + 0.6, t1)
    return (t0, t1), [(t0, ss), (ss, ds), (ds, mid), (mid, go), (go, r1), (r1, t1)]


def row_features(spikes: list[np.ndarray], ep: dict, sets=FEATURE_SETS) -> np.ndarray:
    """``(n_rows, D)`` fingerprint of every row of one trial (NaN = statistic unavailable for that row).

    ``spikes`` must be on the time base of the epoch scalars (see ``resolve_spike_time_reference``)."""
    sets = tuple(str(s).lower() for s in sets)
    names = feature_names(sets)
    (t0, t1), wins = trial_windows(ep)
    dur = max(t1 - t0, 1e-3)
    out = np.full((len(spikes), len(names)), np.nan)
    for i, s in enumerate(spikes):
        s = np.sort(np.asarray(s, dtype=float).ravel())
        s = s[(s >= t0) & (s <= t1)]
        col = 0
        total = np.log1p(len(s) / dur)
        if "rate" in sets:
            out[i, col] = total
            col += 1
        if "windows" in sets:
            for a, b in wins:
                if b - a >= MIN_WINDOW_S:
                    c = np.searchsorted(s, b, side="left") - np.searchsorted(s, a, side="left")
                    out[i, col] = np.log1p(c / (b - a)) - (total if "rate" in sets else 0.0)
                col += 1
        if "isi" in sets:
            if len(s) >= MIN_ISI_SPIKES:
                d = np.diff(s)
                m = float(d.mean())
                if m > 0:
                    out[i, col] = np.log((np.median(d) + 1e-3) / (m + 1e-3))
                    out[i, col + 1] = np.log(d.std() / m + 0.05)
            col += 2
    return out


# ------------------------------------------------------------------------------------------------- alignment
def insertion_penalty(n_features: int, p_insert: float) -> float:
    """Log-score below which a row fits no slot: half the chi-square quantile of the feature count, so the rule
    ("a row further than the 1 - ``p_insert`` quantile from every slot belongs to none") does not depend on how
    many features are used."""
    return -0.5 * float(chi2.ppf(1.0 - float(p_insert), max(int(n_features), 1)))


def score_matrix(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, sd_ref: np.ndarray, log_ppres: np.ndarray,
                 R: np.ndarray | None = None, p_insert: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """``(n_rows, N_slots)`` log-score of every row against every slot, and the per-row insertion penalty.

    Per slot the features are Gaussian with the slot's means and sds and the pooled correlation matrix ``R``
    (identity when None); the score is the Mahalanobis form over the features the row has (each z capped at
    ``Z_CAP``), minus ``log(sd / sd_ref)`` per feature (a tight slot is preferred, a wide one costs), a missing
    feature contributing its expected -1/2, plus the slot's log-probability of being present.  The insertion
    penalty of a row is ``insertion_penalty`` for the number of features it has (plus the same -1/2 per missing
    feature), so the rejection rule is the same chi-square quantile whatever the row provides."""
    n, D = X.shape
    N = mu.shape[0]
    S = np.zeros((n, N))
    ins = np.zeros(n)
    valid = np.isfinite(X)
    R = np.eye(D) if R is None else R
    for key in np.unique(valid, axis=0):
        idx = np.flatnonzero((valid == key[None, :]).all(1))
        sub = np.flatnonzero(key)
        d = len(sub)
        miss = -0.5 * (D - d)
        if d == 0:
            S[idx] = miss
            ins[idx] = miss
            continue
        Q = np.linalg.inv(R[np.ix_(sub, sub)])
        z = (X[idx][:, None, sub] - mu[None, :, sub]) / sd[None, :, sub]
        np.clip(z, -Z_CAP, Z_CAP, out=z)
        q = np.einsum("ind,de,ine->in", z, Q, z)
        S[idx] = -0.5 * q - np.log(sd[:, sub] / sd_ref[sub]).sum(1)[None, :] + miss
        ins[idx] = insertion_penalty(d, p_insert) + miss
    return S + log_ppres[None, :], ins


def _pooled_correlation(X, assigns, mu, sd, shrink: float = CORR_SHRINK, min_rows: int = 50) -> np.ndarray:
    """Correlation matrix of the standardised features over all assigned rows that have every feature (shrunk
    toward the identity); the identity when too few rows have all features."""
    D = mu.shape[1]
    zs = []
    for x, a in zip(X, assigns):
        ok = a >= 0
        if ok.any():
            z = (x[ok] - mu[a[ok]]) / sd[a[ok]]
            zs.append(z[np.isfinite(z).all(1)])
    Z = np.concatenate(zs) if zs else np.zeros((0, D))
    if D < 2 or len(Z) < min_rows:
        return np.eye(D)
    R = np.corrcoef(Z.T)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    return (1.0 - shrink) * R + shrink * np.eye(D)


def align_rows(S: np.ndarray, log_pabs: np.ndarray, ins_pen) -> tuple[np.ndarray, float]:
    """Monotone alignment of ``n`` rows to ``N`` master slots given the row-by-slot score matrix ``S``.

    State ``H[k]`` after ``i`` rows = best log-score with master slots ``1..k`` consumed (assigned or skipped).
    Row ``i`` assigned to slot ``k``: ``max_{j<k} H_prev[j] + gaps(j+1..k-1) + S[i, k]``; row inserted:
    ``H_prev[k] + ins_pen[i]`` (``ins_pen``: scalar or one value per row).  Returns the slot index per row
    (``-1`` = inserted) and the final score."""
    n, N = S.shape
    ins_pen = np.broadcast_to(np.asarray(ins_pen, dtype=float), (n,))
    G = np.concatenate([[0.0], np.cumsum(log_pabs)])                 # G[k] = sum of gap costs of slots 1..k
    H = G.copy()                                                     # no row consumed: every slot skipped
    choice = np.zeros((n, N + 1), dtype=np.int8)                     # 0 = assign, 1 = insert
    argj = np.zeros((n, N + 1), dtype=np.int32)
    idx = np.arange(N)
    for i in range(n):
        M = H[:N] - G[:N]
        cm = np.maximum.accumulate(M)
        am = np.maximum.accumulate(np.where(M >= cm, idx, 0))
        A = np.full(N + 1, -np.inf)
        A[1:] = cm + G[:N] + S[i]
        I = H + ins_pen[i]
        ch = (I > A)
        H = np.where(ch, I, A)
        choice[i] = ch
        argj[i, 1:] = am
    assign = np.full(n, -1, dtype=int)
    k = N
    for i in range(n - 1, -1, -1):
        if choice[i, k]:
            continue
        assign[i] = k - 1
        k = int(argj[i, k])
    return assign, float(H[N])


def _slot_profiles(X, assigns, N, mu, sd, floors, n_trials):
    """Mean / sd per feature, absence probability and row count per slot from the current assignments.

    A feature's sd is shrunk toward the typical (median) slot sd of that feature with ``PRIOR_WEIGHT``
    pseudo-observations, so a slot seen in two trials is not trusted to the floor."""
    D = mu.shape[1]
    sums, sq, cnt, rows = np.zeros((N, D)), np.zeros((N, D)), np.zeros((N, D)), np.zeros(N)
    for x, a in zip(X, assigns):
        ok = a >= 0
        if not ok.any():
            continue
        xa, ia = x[ok], a[ok]
        v = np.isfinite(xa)
        xz = np.where(v, xa, 0.0)
        np.add.at(sums, ia, xz)
        np.add.at(sq, ia, xz * xz)
        np.add.at(cnt, ia, v.astype(float))
        np.add.at(rows, ia, 1.0)
    mu_new = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), mu)
    var = np.maximum(sq / np.maximum(cnt, 1.0) - mu_new ** 2, 0.0)
    sd_ref = np.empty(D)
    for d in range(D):
        well = cnt[:, d] >= 5
        sd_ref[d] = float(np.median(np.sqrt(np.maximum(var[well, d], floors[d] ** 2)))) if well.any() else 2.0 * floors[d]
    var_post = (cnt * np.where(cnt > 1, var, 0.0) + PRIOR_WEIGHT * sd_ref[None, :] ** 2) / (cnt + PRIOR_WEIGHT)
    sd_new = np.sqrt(np.maximum(var_post, floors[None, :] ** 2))
    pabs = np.clip(1.0 - rows / n_trials, P_ABS_MIN, P_ABS_MAX)
    return mu_new, sd_new, pabs, rows, sd_ref


def _merge_split_slots(assigns, mu, sd, rows, n_trials, max_co: float = SPLIT_MAX_CO, max_dist: float = SPLIT_MAX_DIST,
                       max_dist_strong: float = SPLIT_MAX_DIST_STRONG):
    """Merge adjacent slots that are (almost) never assigned in the same trial and have the same profile: the
    signature of one unit whose rows were split between a slot and a spurious neighbour.  When the two slots are
    populated enough that independent units would have co-occurred in at least ``SPLIT_MIN_EXPECTED_CO`` trials,
    their absence from the same trials is strong evidence and the profile test is looser (the split biases the
    two profiles apart: one slot collected the rows that deviated).  Returns the slot remapping (old -> new
    index) or None when nothing merges."""
    N = len(mu)
    if N < 2:
        return None
    co = np.zeros(N - 1)
    for a in assigns:
        present = np.zeros(N, bool)
        present[a[a >= 0]] = True
        co += present[:-1] & present[1:]
    d2 = np.mean((mu[1:] - mu[:-1]) ** 2 / (0.5 * (sd[1:] ** 2 + sd[:-1] ** 2)), axis=1)
    expected = rows[:-1] * rows[1:] / max(n_trials, 1)               # co-assignments if the two were independent units
    strong = (expected >= SPLIT_MIN_EXPECTED_CO) & (co <= 0.1 * expected)
    merge = (co / n_trials <= max_co) & (d2 <= np.where(strong, max_dist_strong, max_dist) ** 2)
    if not merge.any():
        return None
    remap = np.zeros(N, dtype=int)
    new = 0
    for k in range(N):
        if k > 0 and merge[k - 1] and remap[k - 1] == new - 1 and not (k > 1 and merge[k - 2] and remap[k - 2] == remap[k - 1]):
            remap[k] = new - 1                                       # join the previous slot (pairs only)
        else:
            remap[k] = new
            new += 1
    return remap


def _apply_remap(assigns, remap):
    return [np.where(a >= 0, remap[np.maximum(a, 0)], -1) for a in assigns]


def recover_region(X: list[np.ndarray], n_iter: int = 5, min_support: int = 3, support_frac: float = 0.03,
                   p_insert: float = 1e-4, prune: bool = True, floors: np.ndarray | None = None) -> tuple[list[np.ndarray], dict]:
    """Recover one region's slots from the per-trial fingerprint matrices ``X`` (rows in file order, ``(n_i, D)``).

    ``support`` = max(``min_support``, ``support_frac`` x trials) is the smallest number of trials that must
    insert a consistent row at the same place before a slot is created there, and the smallest number of rows a
    slot keeps at the end (``prune``): a rare rate outlier must not spawn a slot, and a slot seen in a handful of
    trials is an alignment accident or a unit no analysis can use.  ``p_insert`` sets the insertion penalty
    (see ``insertion_penalty``)."""
    n_trials = len(X)
    lengths = np.array([len(x) for x in X], dtype=int)
    empty = {"n_slots": 0, "n_slots_added": 0, "n_slots_merged": 0, "n_slots_pruned": 0, "n_rows": int(lengths.sum()),
             "frac_rows_assigned": float("nan"), "master_trial": -1, "sd_ref": {}}
    if n_trials == 0 or lengths.max() == 0:
        return [np.full(len(x), -1, dtype=int) for x in X], empty
    D = X[0].shape[1]
    floors = feature_floors([""] * D) if floors is None else np.asarray(floors, dtype=float)
    support = max(int(min_support), int(round(float(support_frac) * n_trials)))
    allx = np.concatenate([x for x in X if len(x)], axis=0)
    with np.errstate(all="ignore"):
        gmean = np.nanmean(allx, axis=0)
    gmean = np.where(np.isfinite(gmean), gmean, 0.0)
    t0 = int(np.argmax(lengths))
    mu = np.where(np.isfinite(X[t0]), X[t0], gmean[None, :]).astype(float)
    sd = np.tile(INIT_SD_MULT * floors, (len(mu), 1))
    sd_ref = INIT_SD_MULT * floors
    pabs = np.full(len(mu), 0.1)
    R = np.eye(D)
    n_added = n_merged = n_pruned = 0

    def run_pass(mu, sd, pabs, sd_ref, R, collect=True):
        log_pabs, log_ppres = np.log(pabs), np.log1p(-pabs)
        assigns, ins_pos, ins_val = [], [], []
        for x in X:
            if len(x) == 0:
                assigns.append(np.zeros(0, dtype=int))
                continue
            S, ins_pen = score_matrix(x, mu, sd, sd_ref, log_ppres, R, p_insert)
            a, _ = align_rows(S, log_pabs, ins_pen)
            assigns.append(a)
            if collect and (a < 0).any():
                last = -1
                for i, s in enumerate(a):
                    if s >= 0:
                        last = int(s)
                    else:
                        ins_pos.append(last + 1)
                        ins_val.append(x[i])
        return assigns, np.asarray(ins_pos, dtype=int), (np.stack(ins_val) if ins_val else np.zeros((0, D)))

    for it in range(n_iter):
        assigns, pos, vals = run_pass(mu, sd, pabs, sd_ref, R)
        mu, sd, pabs, rows, sd_ref = _slot_profiles(X, assigns, len(mu), mu, sd, floors, n_trials)
        R = _pooled_correlation(X, assigns, mu, sd)
        if it == 0 and n_iter > 1:
            continue                     # the first pass matched against one trial's fingerprints: its inserts are noise
        new = []
        for p in np.unique(pos):
            F = vals[pos == p]
            if len(F) < support:
                continue
            v = np.isfinite(F)
            nv = v.sum(0)
            m = np.where(nv > 0, np.where(v, F, 0.0).sum(0) / np.maximum(nv, 1), np.nan)
            s2 = np.where(nv > 1, (np.where(v, F, 0.0) ** 2).sum(0) / np.maximum(nv, 1) - np.where(np.isfinite(m), m, 0.0) ** 2, np.nan)
            okd = nv >= max(2, support // 2)
            if not okd.any():
                continue
            spread = np.median(np.sqrt(np.maximum(s2[okd], 0.0)) / sd_ref[okd])
            if spread > NEW_SLOT_MAX_SPREAD:
                continue
            m_full = np.where(np.isfinite(m), m, gmean)
            var_post = (nv * np.where(nv > 1, np.maximum(np.nan_to_num(s2), 0.0), 0.0) + PRIOR_WEIGHT * sd_ref ** 2) / (nv + PRIOR_WEIGHT)
            new.append((int(p), m_full, np.sqrt(np.maximum(var_post, floors ** 2)),
                        float(np.clip(1.0 - len(F) / n_trials, P_ABS_MIN, P_ABS_MAX))))
        for p, m, s, pa in sorted(new, key=lambda t: -t[0]):          # descending: earlier indices stay valid
            mu, sd, pabs = np.insert(mu, p, m, axis=0), np.insert(sd, p, s, axis=0), np.insert(pabs, p, pa)
            n_added += 1
    # final pass against the grown master, then prune rare slots, merge split units, re-profile and re-align
    assigns, _, _ = run_pass(mu, sd, pabs, sd_ref, R, collect=False)
    mu, sd, pabs, rows, sd_ref = _slot_profiles(X, assigns, len(mu), mu, sd, floors, n_trials)
    R = _pooled_correlation(X, assigns, mu, sd)
    if prune:
        keep = rows >= support
        if not keep.all() and keep.any():
            remap = np.where(keep, np.cumsum(keep) - 1, -1)
            n_pruned = int((~keep).sum())
            assigns = _apply_remap(assigns, remap)
            mu, sd, pabs = mu[keep], sd[keep], pabs[keep]
            mu, sd, pabs, rows, sd_ref = _slot_profiles(X, assigns, len(mu), mu, sd, floors, n_trials)
    remap = _merge_split_slots(assigns, mu, sd, rows, n_trials)
    if remap is not None:
        n_merged = int(len(mu) - remap.max() - 1)
        assigns = _apply_remap(assigns, remap)
        N = int(remap.max() + 1)
        mu, sd, pabs, rows, sd_ref = _slot_profiles(X, assigns, N, np.tile(gmean, (N, 1)), np.tile(2.0 * floors, (N, 1)), floors, n_trials)
    # one more alignment against the final profiles so every trial uses the same master
    assigns, _, _ = run_pass(mu, sd, pabs, sd_ref, R, collect=False)
    mu, sd, pabs, rows, sd_ref = _slot_profiles(X, assigns, len(mu), mu, sd, floors, n_trials)
    n_rows = int(lengths.sum())
    n_assigned = int(sum(int((a >= 0).sum()) for a in assigns))
    stats = {"n_slots": int(len(mu)), "n_slots_added": n_added, "n_slots_merged": n_merged, "n_slots_pruned": n_pruned,
             "n_rows": n_rows, "frac_rows_assigned": n_assigned / max(n_rows, 1), "master_trial": t0,
             "sd_ref": [float(v) for v in sd_ref], "support": int(support),
             "max_abs_corr": float(np.abs(R - np.eye(D)).max()) if D > 1 else 0.0}
    return assigns, stats


def recover_session_identity(recs, n_iter: int = 5, min_support: int = 3, session: str = "", features=FEATURE_SETS,
                             support_frac: float = 0.03, p_insert: float = 1e-4, prune: bool = True) -> IdentityMap:
    """Recover unit identity for every trial of one session (``recs``: TrialRecords sorted by trial number)."""
    t_start = time.perf_counter()
    sets = tuple(str(s).lower() for s in features)
    names = feature_names(sets)
    floors = feature_floors(names)
    feats: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}
    trials = []
    for rec in recs:
        try:
            data = np.load(rec.npz_path, allow_pickle=True)
            by_region = spikes_by_region(data)
            ep = read_epochs(data, rec.csv)
            win = (float(ep["delay_start_times"]) - 3.0, float(ep["go_start_times"]) + 3.0)
            by_region, _ = resolve_spike_time_reference(by_region, ep, win, rec.npz_path)
        except Exception as e:                                       # a corrupt file gets no identity (dropped later)
            log.warning("%s trial %s: unreadable for identity recovery (%s)", session, rec.trial, e)
            continue
        trials.append(int(rec.trial))
        for r in REGIONS:
            feats[r].append(row_features(by_region[r][0], ep, sets))
    ident = IdentityMap(session)
    stats = {}
    for r in REGIONS:
        assigns, st = recover_region(feats[r], n_iter=n_iter, min_support=min_support, support_frac=support_frac,
                                     p_insert=p_insert, prune=prune, floors=floors)
        st["sd_ref"] = dict(zip(names, st["sd_ref"]))
        stats[r] = st
        ident.n_slots[r] = int(st["n_slots"])
        for t, a in zip(trials, assigns):
            ident.rows.setdefault(t, {})[r] = a
    tot_rows = sum(st["n_rows"] for st in stats.values())
    ident.stats = {"per_region": stats, "n_trials": len(trials),
                   "frac_rows_assigned": (sum(st["frac_rows_assigned"] * st["n_rows"] for st in stats.values()) / tot_rows) if tot_rows else float("nan"),
                   "n_slots_total": int(sum(ident.n_slots.values())),
                   "n_slots_added": int(sum(st["n_slots_added"] for st in stats.values())),
                   "n_slots_merged": int(sum(st.get("n_slots_merged", 0) for st in stats.values())),
                   "n_slots_pruned": int(sum(st.get("n_slots_pruned", 0) for st in stats.values())),
                   "features": names,
                   "settings": {"features": list(sets), "n_iter": int(n_iter), "min_support": int(min_support),
                                "support_frac": float(support_frac), "p_insert": float(p_insert), "prune": bool(prune)},
                   "seconds": round(time.perf_counter() - t_start, 1)}
    return ident


def twin_identity_accuracy(ident: IdentityMap, recs_b, recs_a, max_trials: int = 60, tol_s: float = 1e-3) -> dict:
    """How well the recovered slots of a Data2 session agree with the true unit IDs of its Data twin.

    Every Data2 row of a matched trial is identified by its spike train (identical to one Data unit's train once
    both are on the trial's time base); a slot's *label* is the true ID most of its rows carry.  Returns the
    fraction of assigned rows whose true ID equals their slot's label (``row_accuracy``, also by the unit's rate
    tercile), the fraction of identified rows that were assigned to a slot at all (``frac_rows_assigned``), the
    fraction of slots whose rows all carry one ID (``frac_pure_slots``), the fraction of true units that map to
    exactly one slot (``frac_units_one_slot``) and the number of rows identified."""
    by_a = {int(r.trial): r for r in recs_a}
    key = lambda arr: np.round(np.sort(np.asarray(arr, dtype=float)), 3).tobytes()
    slot_votes: dict[str, dict[int, dict[int, int]]] = {r: {} for r in REGIONS}
    unit_counts: dict[tuple[str, int], list] = {}                    # true unit -> spike counts per trial (rate tercile)
    row_log: list[tuple[str, int, int]] = []                         # (region, slot, true id) per identified assigned row
    n_rows = n_ident = n_unassigned = 0
    n_checked = 0
    for rec in sorted(recs_b, key=lambda x: x.trial):
        if n_checked >= max_trials:
            break
        t = int(rec.trial)
        if t not in ident.rows or t not in by_a:
            continue
        try:
            db, da = np.load(rec.npz_path, allow_pickle=True), np.load(by_a[t].npz_path, allow_pickle=True)
            rb, ra = spikes_by_region(db), spikes_by_region(da)
            ep_b, ep_a = read_epochs(db, rec.csv), read_epochs(da, by_a[t].csv)
            win = (float(ep_a["delay_start_times"]) - 3.0, float(ep_a["go_start_times"]) + 3.0)
            rb, _ = resolve_spike_time_reference(rb, ep_b, win)
            ra, _ = resolve_spike_time_reference(ra, ep_a, win)
        except Exception:
            continue
        # the two files may sit on different absolute bases (Data2 relative + start_time): compare relative to
        # each file's own delay onset
        off_a, off_b = float(ep_a["delay_start_times"]), float(ep_b["delay_start_times"])
        n_checked += 1
        for r in REGIONS:
            lookup = {}
            for u, uid in zip(ra[r][0], ra[r][1]):
                u = np.asarray(u, dtype=float)
                if u.size:
                    lookup.setdefault(key(u - off_a), int(uid))
            slots = ident.rows[t].get(r)
            if slots is None:
                continue
            for v, s in zip(rb[r][0], slots):
                n_rows += 1
                tid = lookup.get(key(np.asarray(v, dtype=float) - off_b))
                if tid is None:
                    continue
                n_ident += 1
                unit_counts.setdefault((r, tid), []).append(len(v))
                if s >= 0:
                    d = slot_votes[r].setdefault(int(s), {})
                    d[tid] = d.get(tid, 0) + 1
                    row_log.append((r, int(s), tid))
                else:
                    n_unassigned += 1
    n_assigned = correct = 0
    n_slots = pure = 0
    unit_slots: dict[tuple[str, int], set] = {}
    for r in REGIONS:
        for s, votes in slot_votes[r].items():
            n_slots += 1
            tot = sum(votes.values())
            best = max(votes.values())
            n_assigned += tot
            correct += best
            pure += int(len(votes) == 1)
            label = max(votes, key=votes.get)
            unit_slots.setdefault((r, label), set()).add(s)
    # accuracy by the true unit's rate tercile (the sparsest units are the hardest to align and matter least)
    label = {(r, s): max(v, key=v.get) for r in REGIONS for s, v in slot_votes[r].items()}
    mean_count = {u: float(np.mean(c)) for u, c in unit_counts.items()}
    terc_acc = {}
    if mean_count:
        cuts = np.quantile(list(mean_count.values()), [1 / 3, 2 / 3])
        ok = np.zeros(3); tot = np.zeros(3)
        for r, s, tid in row_log:
            k = int(np.digitize(mean_count[(r, tid)], cuts))
            tot[k] += 1
            ok[k] += int(label[(r, s)] == tid)
        terc_acc = {f"row_accuracy_{name}_rate": (ok[k] / tot[k] if tot[k] else float("nan")) for k, name in enumerate(("low", "mid", "high"))}
    return {"n_trials_checked": n_checked, "n_rows": n_rows, "frac_rows_identified": n_ident / n_rows if n_rows else float("nan"),
            "frac_rows_assigned": (n_ident - n_unassigned) / n_ident if n_ident else float("nan"),
            "row_accuracy": correct / n_assigned if n_assigned else float("nan"), **terc_acc,
            "frac_pure_slots": pure / n_slots if n_slots else float("nan"),
            "frac_units_one_slot": (sum(1 for v in unit_slots.values() if len(v) == 1) / len(unit_slots)) if unit_slots else float("nan"),
            "n_slots_seen": n_slots, "n_true_units_seen": len(unit_slots)}
