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
  unit is silent in a trial, an unassignable row (a unit missing from the master) costs a fixed insertion
  penalty, and the match score is a Gaussian on the row's log firing rate against the slot's rate profile
  (units keep their rate from trial to trial; that is what breaks the ties between neighbouring slots);
* after each pass the slot profiles (mean / sd of the log rate, probability of absence) are re-estimated from
  the assignments and slots are created where at least ``min_support`` trials inserted a row at the same place;
  two or three passes converge.

The recovered slots are unit IDs in every sense the cache needs (one row per unit per trial, silent units = zero
rows).  On the twin recordings the recovery is validated against the true IDs (``twin_identity_accuracy``), and
that number is what justifies using it on the sessions that have no twin.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import REGIONS
from .rasters import read_epochs, spikes_by_region

log = logging.getLogger(__name__)

SD_FLOOR = 0.25          # floor of a slot's log-rate sd (a unit's rate is not constant across trials)
P_ABS_MIN, P_ABS_MAX = 0.01, 0.99
INS_PENALTY = -6.0       # log-probability of a row that belongs to no slot of the current master


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


def row_features(spikes: list[np.ndarray], duration: float) -> np.ndarray:
    """log(1 + rate) per row: the trial-invariant property of a unit that the alignment matches on."""
    counts = np.fromiter((len(s) for s in spikes), dtype=float, count=len(spikes))
    return np.log1p(counts / max(float(duration), 1e-3))


def align_rows(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, log_pabs: np.ndarray, log_ppres: np.ndarray,
               ins_pen: float = INS_PENALTY) -> tuple[np.ndarray, float]:
    """Monotone alignment of ``n`` rows (features ``x``) to ``N`` master slots.

    State ``H[k]`` after ``i`` rows = best log-score with master slots ``1..k`` consumed (assigned or skipped).
    Row ``i`` assigned to slot ``k``: ``max_{j<k} H_prev[j] + gaps(j+1..k-1) + score_k(x_i)``; row inserted:
    ``H_prev[k] + ins_pen``.  Returns the slot index per row (``-1`` = inserted) and the final score."""
    n, N = len(x), len(mu)
    G = np.concatenate([[0.0], np.cumsum(log_pabs)])                 # G[k] = sum of gap costs of slots 1..k
    H = G.copy()                                                     # no row consumed: every slot skipped
    choice = np.zeros((n, N + 1), dtype=np.int8)                     # 0 = assign, 1 = insert
    argj = np.zeros((n, N + 1), dtype=np.int32)
    idx = np.arange(N)
    for i in range(n):
        M = H[:N] - G[:N]
        cm = np.maximum.accumulate(M)
        am = np.maximum.accumulate(np.where(M >= cm, idx, 0))
        score = -0.5 * ((x[i] - mu) / sd) ** 2 - np.log(sd) + log_ppres
        A = np.full(N + 1, -np.inf)
        A[1:] = cm + G[:N] + score
        I = H + ins_pen
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


def _slot_profiles(X, assigns, N, mu, sd):
    """Mean / sd of the feature and absence probability per slot from the current assignments."""
    n_trials = len(X)
    sums, sq, cnt = np.zeros(N), np.zeros(N), np.zeros(N)
    for x, a in zip(X, assigns):
        ok = a >= 0
        np.add.at(sums, a[ok], x[ok])
        np.add.at(sq, a[ok], x[ok] ** 2)
        np.add.at(cnt, a[ok], 1.0)
    mu_new = np.where(cnt > 0, sums / np.maximum(cnt, 1.0), mu)
    var = np.where(cnt > 1, sq / np.maximum(cnt, 1.0) - mu_new ** 2, sd ** 2)
    return mu_new, np.sqrt(np.maximum(var, SD_FLOOR ** 2)), np.clip(1.0 - cnt / n_trials, P_ABS_MIN, P_ABS_MAX), cnt


def _merge_split_slots(X, assigns, mu, sd, cnt, max_co: float = 0.02, max_dmu: float = 1.0):
    """Merge adjacent slots that are (almost) never assigned in the same trial and have similar rates: the
    signature of one unit whose rows were split between a slot and a spurious neighbour.  Returns the slot
    remapping (old -> new index) or None when nothing merges."""
    N = len(mu)
    if N < 2:
        return None
    co = np.zeros(N - 1)
    for a in assigns:
        present = np.zeros(N, bool)
        present[a[a >= 0]] = True
        co += present[:-1] & present[1:]
    n_trials = len(X)
    merge = (co / n_trials <= max_co) & (np.abs(np.diff(mu)) <= max_dmu * np.maximum(sd[:-1], sd[1:]))
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


def recover_region(X: list[np.ndarray], n_iter: int = 3, min_support: int = 3) -> tuple[list[np.ndarray], dict]:
    """Recover one region's slots from the per-trial feature vectors ``X`` (rows in file order).

    ``min_support`` is the smallest number of trials that must insert a row at the same place before a slot is
    created there (raised to 3 % of the trials for long sessions: a rare rate outlier must not spawn a slot)."""
    n_trials = len(X)
    lengths = np.array([len(x) for x in X])
    if n_trials == 0 or lengths.max() == 0:
        return [np.full(len(x), -1, dtype=int) for x in X], {"n_slots": 0, "n_slots_added": 0, "n_slots_merged": 0,
                                                              "n_rows": int(lengths.sum()), "frac_rows_assigned": float("nan"), "master_trial": -1}
    support = max(int(min_support), int(round(0.03 * n_trials)))
    t0 = int(np.argmax(lengths))
    mu = X[t0].astype(float).copy()
    sd = np.full(len(mu), 0.5)
    pabs = np.full(len(mu), 0.1)
    n_added = n_merged = 0
    assigns: list[np.ndarray] = []

    def run_pass(mu, sd, pabs):
        log_pabs, log_ppres = np.log(pabs), np.log1p(-pabs)
        assigns, inserts = [], []
        for x in X:
            a, _ = align_rows(x, mu, sd, log_pabs, log_ppres)
            assigns.append(a)
            last = -1
            for i, s in enumerate(a):
                if s >= 0:
                    last = int(s)
                else:
                    inserts.append((last + 1, float(x[i])))
        return assigns, inserts

    for it in range(n_iter):
        assigns, inserts = run_pass(mu, sd, pabs)
        mu, sd, pabs, cnt = _slot_profiles(X, assigns, len(mu), mu, sd)
        if inserts:
            pos = np.array([p for p, _ in inserts])
            vals = np.array([v for _, v in inserts])
            new = []
            for p in np.unique(pos):
                v = vals[pos == p]
                if len(v) >= support and np.std(v) <= 0.6:
                    new.append((int(p), float(np.median(v)), float(max(np.std(v), SD_FLOOR)),
                                float(np.clip(1.0 - len(v) / n_trials, P_ABS_MIN, P_ABS_MAX))))
            for p, m, s, pa in sorted(new, key=lambda t: -t[0]):      # descending: earlier indices stay valid
                mu, sd, pabs = np.insert(mu, p, m), np.insert(sd, p, s), np.insert(pabs, p, pa)
                n_added += 1
    # final pass, then merge split units and re-profile
    assigns, _ = run_pass(mu, sd, pabs)
    mu, sd, pabs, cnt = _slot_profiles(X, assigns, len(mu), mu, sd)
    remap = _merge_split_slots(X, assigns, mu, sd, cnt)
    if remap is not None:
        n_merged = int(len(mu) - remap.max() - 1)
        assigns = [np.where(a >= 0, remap[np.maximum(a, 0)], -1) for a in assigns]
        N = int(remap.max() + 1)
        mu, sd, pabs, cnt = _slot_profiles(X, assigns, N, np.zeros(N), np.full(N, 0.5))
        # one more alignment against the merged profiles so every trial uses the same final master
        assigns, _ = run_pass(mu, sd, pabs)
    n_rows = int(lengths.sum())
    n_assigned = int(sum(int((a >= 0).sum()) for a in assigns))
    stats = {"n_slots": int(len(mu)), "n_slots_added": n_added, "n_slots_merged": n_merged, "n_rows": n_rows,
             "frac_rows_assigned": n_assigned / max(n_rows, 1), "master_trial": t0}
    return assigns, stats


def recover_session_identity(recs, n_iter: int = 3, min_support: int = 3, session: str = "") -> IdentityMap:
    """Recover unit identity for every trial of one session (``recs``: TrialRecords sorted by trial number)."""
    feats: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}
    trials = []
    for rec in recs:
        try:
            data = np.load(rec.npz_path, allow_pickle=True)
            by_region = spikes_by_region(data)
            ep = read_epochs(data, rec.csv)
        except Exception as e:                                       # a corrupt file gets no identity (dropped later)
            log.warning("%s trial %s: unreadable for identity recovery (%s)", session, rec.trial, e)
            continue
        t0, t1 = float(ep.get("trial_start", np.nan)), float(ep.get("trial_stop", np.nan))
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            duration = t1 - t0
        else:
            duration = float(ep["go_start_times"] - ep["delay_start_times"]) + 4.0
        trials.append(int(rec.trial))
        for r in REGIONS:
            feats[r].append(row_features(by_region[r][0], duration))
    ident = IdentityMap(session)
    stats = {}
    for r in REGIONS:
        assigns, st = recover_region(feats[r], n_iter=n_iter, min_support=min_support)
        stats[r] = st
        ident.n_slots[r] = int(st["n_slots"])
        for t, a in zip(trials, assigns):
            ident.rows.setdefault(t, {})[r] = a
    tot_rows = sum(st["n_rows"] for st in stats.values())
    ident.stats = {"per_region": stats, "n_trials": len(trials),
                   "frac_rows_assigned": (sum(st["frac_rows_assigned"] * st["n_rows"] for st in stats.values()) / tot_rows) if tot_rows else float("nan"),
                   "n_slots_total": int(sum(ident.n_slots.values())),
                   "n_slots_added": int(sum(st["n_slots_added"] for st in stats.values())),
                   "n_slots_merged": int(sum(st.get("n_slots_merged", 0) for st in stats.values()))}
    return ident


def twin_identity_accuracy(ident: IdentityMap, recs_b, recs_a, max_trials: int = 60, tol_s: float = 1e-3) -> dict:
    """How well the recovered slots of a Data2 session agree with the true unit IDs of its Data twin.

    Every Data2 row of a matched trial is identified by its spike train (identical to one Data unit's train once
    both are on the trial's time base); a slot's *label* is the true ID most of its rows carry.  Returns the
    fraction of assigned rows whose true ID equals their slot's label (``row_accuracy``), the fraction of slots
    whose rows all carry one ID (``frac_pure_slots``), the fraction of true units that map to exactly one slot
    (``frac_units_one_slot``) and the number of rows identified."""
    from .rasters import resolve_spike_time_reference
    by_a = {int(r.trial): r for r in recs_a}
    key = lambda arr: np.round(np.sort(np.asarray(arr, dtype=float)), 3).tobytes()
    slot_votes: dict[str, dict[int, dict[int, int]]] = {r: {} for r in REGIONS}
    unit_counts: dict[tuple[str, int], list] = {}                    # true unit -> spike counts per trial (rate tercile)
    row_log: list[tuple[str, int, int]] = []                         # (region, slot, true id) per identified assigned row
    n_rows = n_ident = 0
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
            "row_accuracy": correct / n_assigned if n_assigned else float("nan"), **terc_acc,
            "frac_pure_slots": pure / n_slots if n_slots else float("nan"),
            "frac_units_one_slot": (sum(1 for v in unit_slots.values() if len(v) == 1) / len(unit_slots)) if unit_slots else float("nan"),
            "n_slots_seen": n_slots, "n_true_units_seen": len(unit_slots)}
