"""The claims report: one verdict per prediction of the DelayCAST claim, with the numbers next to it.

``write_report(cfg, out_dir)`` reads every run that :func:`delaycast.runs.load_results` can find under
``out_dir/runs``, the descriptive selection summary (``out_dir/selection/summary.csv``), the per-run
``selection_funnel.csv`` / ``selection_<session>.csv`` tables, and writes ``out_dir/REPORT.md`` (for people)
and ``out_dir/report.json`` (for scripts and figure captions; same verdicts and numbers).

Why the report is built the way it is
-------------------------------------
* **The session is the unit of replication.**  Trials within a session share an animal, a day, a probe
  position and a behavioural state, so trial-level tests would overstate the evidence.  Every comparison
  between two arms is therefore a *paired* comparison across sessions: for each session the metric is first
  averaged over seeds (seeds are re-runs of the same experiment, not new evidence), then the per-session
  difference is tested with a Wilcoxon signed-rank test (exact for n <= 25) and summarised by a
  1000-resample session bootstrap CI of the mean difference.  ``supported`` needs *both* p < 0.05 and a CI
  that excludes zero in the predicted direction; ``inconclusive`` means the test was possible (>= 3
  sessions) but failed; ``not testable`` means fewer than three sessions overlap; ``not run`` means a required
  run is missing.
* **"Not lower than" claims use a non-inferiority margin.**  P1 says the selected-K decoders are *not
  lower* than the all-unit decoder.  A significant difference is the wrong target for such a claim; the rule
  in the contract is that the bootstrap CI lower bound of (selected - all) must exceed -0.02 balanced
  accuracy.  The Wilcoxon test is then run on the shifted differences (diff + 0.02 > 0), which is the
  standard non-inferiority test at that margin, so the "p < 0.05 AND CI" rule keeps the same meaning.
* **Every function tolerates missing runs and keys.**  The report is also run on partial outputs
  (``--quick`` runs, a crashed cross-dataset job); a missing arm yields ``not run`` for that prediction rather
  than an exception, and the header lists exactly which runs and seeds were found so a reader never mistakes
  an absent test for a negative one.
* **Composite predictions combine sub-verdicts conservatively.**  Where a prediction needs several tests
  to hold (P2: above *both* rate-matched and random subsets; P3: occlusion *and* CSI; P8: above chance *and*
  above the random-K null), a failed sub-test makes the whole prediction ``inconclusive`` and a missing one
  makes it ``not run``; ``supported`` needs every part.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import CLASSES, REGIONS
from .runs import load_results

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------------------------------------

CLAIM = (
    "During the 1.2 s delay of the auditory delayed-response task, (i) a criterion-selected subset of at most "
    "K units per region, chosen on training trials only by model-free single-unit statistics that survive "
    "stability selection, supports decoding of the upcoming lick direction with balanced accuracy not lower "
    "than a tuned linear decoder on all recorded units (P1a: linear on selected K >= linear on all units; P1b: "
    "DelayCAST on selected K >= linear on selected K) and above rate-matched and random subsets of the same "
    "size (P2); (ii) the last 500 ms before the go cue retain >= 95 % of full-delay accuracy for both the "
    "selected-unit model and the all-unit linear decoder, and removing the last 400 ms costs more than removing "
    "any earlier 400 ms (P3); (iii) the selected units' late-delay activity forecasts their own response-epoch "
    "activity beyond the units' mean response (P4; persistence and the class-conditional oracle reported); (iv) removing ALM input degrades Left/Right decoding more "
    "than removing striatal input (P5a); striatal involvement in no-lick (Ignore) trials is exploratory (P5b); "
    "(v) model-based importance agrees with the model-free criteria (P6); (vi) the causal spectro-temporal "
    "population branch adds accuracy beyond a matched population-rate control (P7); (vii) a backbone trained on "
    "one dataset decodes the other above the random-K and label-permuted nulls after fitting only session "
    "adapters (P8). Negative control (labels permuted before selection/training) must be at chance (P0)."
)

ALPHA = 0.05                # significance level of every test
NOT_LOWER_MARGIN = 0.02     # non-inferiority margin (balanced accuracy) for "not lower than" claims
N_BOOT = 1000               # session resamples for the CI of the mean paired difference
MIN_SESSIONS = 5            # fewer overlapping sessions -> "not testable" (an exact one-sided Wilcoxon cannot reach p < 0.05 below n = 5)
TAU95_MAX_MS = 500.0        # P3: the last 500 ms must retain 95 % of the accuracy
MIN_IGNORE_TRIALS = 30      # P5b: fewer Ignore test trials -> "not testable"
MIN_AGREEMENT_CELLS = 8     # P6: session x region cells needed for the sign test to be informative
PHI_SW_INDEPENDENCE = 0.7   # phi(S, W) above this: spectral criterion is not independent evidence

VERDICTS = ("supported", "inconclusive", "not testable", "not run")
PREDICTION_IDS = ("P0", "P1a", "P1b", "P2", "P3", "P4", "P5a", "P5b", "P6", "P7", "P8")

# ----------------------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------------------


def _finite(x: Any) -> float:
    """Return ``x`` as a float, or NaN for None / non-numeric / non-finite input."""
    if x is None or isinstance(x, bool):
        return math.nan
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _int(x: Any, default: int = 0) -> int:
    """Integer count from a results field; NaN / None -> ``default`` (note ``int(nan or 0)`` would raise)."""
    v = _finite(x)
    return default if math.isnan(v) else int(v)


def _fmt(x: Any, nd: int = 3) -> str:
    v = _finite(x)
    if math.isnan(v):
        return "n/a"
    if abs(v) < 1e-3 and v != 0 and nd <= 4:
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


def _fmt_ci(ci: Any, nd: int = 3) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]"


def _fmt_p(p: Any) -> str:
    v = _finite(p)
    if math.isnan(v):
        return "n/a"
    return f"{v:.2e}" if v < 1e-3 else f"{v:.3f}"


def _jsonable(o: Any) -> Any:
    """Make numpy / pandas / NaN content JSON-serialisable (NaN -> null so downstream readers see 'missing')."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def _session_tag(session: str) -> str:
    return session.replace("/", "__")


def _seeds(arm: list[dict] | None) -> list[int]:
    return sorted({int(r.get("seed", 0)) for r in arm}) if arm else []


def _mean(vals: list[float]) -> float:
    v = [x for x in (_finite(a) for a in vals) if not math.isnan(x)]
    return float(np.mean(v)) if v else math.nan


# ----------------------------------------------------------------------------------------------------------
# reading per-session values out of results.json
# ----------------------------------------------------------------------------------------------------------


def _baseline_row(results: dict, model: str) -> dict | None:
    for b in results.get("baselines") or []:
        if isinstance(b, dict) and b.get("model") == model:
            return b
    return None


def per_session_values(arm: list[dict] | None, metrics: tuple[str, ...] | str,
                       baseline: str | None = None) -> tuple[dict[str, list[float]], str | None]:
    """session -> list of finite values (one per seed) of the first metric in ``metrics`` that has any data.

    ``baseline`` selects ``results['baselines'][model]['per_session']`` instead of ``results['per_session']``.
    The metric fallback order matters scientifically: Left/Right decoding is what the claim is about, so
    ``balanced_accuracy_lr`` is preferred, but a run whose test split has no Ignore-free metric (e.g. a
    two-class session set) still reports 3-class balanced accuracy rather than nothing.
    """
    if isinstance(metrics, str):
        metrics = (metrics,)
    if not arm:
        return {}, None
    rows_per_seed: list[list[dict]] = []
    for res in arm:
        src = _baseline_row(res, baseline) if baseline else res
        rows = (src or {}).get("per_session") or []
        rows_per_seed.append([r for r in rows if isinstance(r, dict) and "session" in r])
    for metric in metrics:
        out: dict[str, list[float]] = {}
        for rows in rows_per_seed:
            for r in rows:
                v = _finite(r.get(metric))
                if not math.isnan(v):
                    out.setdefault(str(r["session"]), []).append(v)
        if out:
            return out, metric
    return {}, None


def _pooled(arm: list[dict] | None, *path: str) -> list[float]:
    """One pooled number per seed, following ``path`` into the results dict (NaN when absent)."""
    vals = []
    for res in arm or []:
        node: Any = res
        for p in path:
            node = node.get(p) if isinstance(node, dict) else None
        vals.append(_finite(node))
    return vals


def _chance_p95(res: dict) -> float:
    v = _finite((res.get("chance") or {}).get("p95"))
    if math.isnan(v):
        v = _finite((res.get("chance_balanced_accuracy") or {}).get("p95"))
    return v


def empty_criteria_sessions(runs: dict) -> list[str]:
    """Sessions in which the criteria run used no unit at all (K_eff = 0 in every region, in any seed).

    Such a session is a *selection failure* (no unit passed the stability rule, e.g. too few trials), not a
    measurement of what criteria-selected units can do: its criteria-arm accuracy is chance by construction, its
    occlusion / ablation deltas are exactly zero and its forecast is undefined.  Keeping it in a paired
    comparison would count the failure as evidence against (or, for null-shaped deltas, for) the prediction, so
    every criteria-arm comparison excludes it and the report lists it."""
    out: set[str] = set()
    for name in ("criteria", "criteria_popmean", "criteria_nospec"):
        for res in runs.get(name) or []:
            for s, per in (res.get("n_selected") or {}).items():
                if isinstance(per, dict) and sum(_int(v) for v in per.values()) == 0:
                    out.add(str(s))
    return sorted(out)


def k_eff_per_session(runs: dict) -> dict[str, dict[str, float]]:
    """Mean over seeds of the criteria run's units per region and session."""
    acc: dict[str, dict[str, list[int]]] = {}
    for res in runs.get("criteria") or []:
        for s, per in (res.get("n_selected") or {}).items():
            if isinstance(per, dict):
                for r, v in per.items():
                    acc.setdefault(str(s), {}).setdefault(str(r), []).append(_int(v))
    return {s: {r: float(np.mean(v)) for r, v in per.items()} for s, per in acc.items()}


def _bacc_from_preds(y: np.ndarray, yhat: np.ndarray) -> float:
    present = np.unique(y)
    return float(np.mean([np.mean(yhat[y == c] == c) for c in present])) if len(present) else math.nan


def _bacc_lr_from_preds(y: np.ndarray, yhat: np.ndarray) -> float:
    keep = y != 0
    return _bacc_from_preds(y[keep], yhat[keep]) if keep.any() else math.nan


def replication_by_session(arm_a: list[dict] | None, arm_b: list[dict] | None, direction: str, metric: str = "balanced_accuracy",
                           n_boot: int = N_BOOT, seed: int = 0) -> dict | None:
    """Trial-level replication of a two-arm comparison *inside* every session.

    Sessions are the unit of replication for the verdict, but with few sessions the session-level test has no
    power (n = 4: the exact one-sided Wilcoxon cannot go below p = 1/16).  This supplementary statistic asks, per
    session, whether the difference between the arms holds on that session's own test trials: for every seed the
    two arms were evaluated on the same test trials (same split), so the per-trial predictions are paired; the
    trials are resampled ``n_boot`` times, the difference of balanced accuracy is recomputed per seed and averaged
    over seeds, and the session *replicates* the prediction when the bootstrap CI excludes 0 in the predicted
    direction (non-inferiority: the CI lower bound exceeds -NOT_LOWER_MARGIN).  It never changes a verdict; it is
    reported next to it ("replicates in k/n sessions").  Returns None when no run has ``test_predictions.csv``."""
    if not arm_a or not arm_b:
        return None
    stat = _bacc_lr_from_preds if metric == "balanced_accuracy_lr" else _bacc_from_preds
    by_seed_a = {int(r.get("seed", 0)): r for r in arm_a}
    by_seed_b = {int(r.get("seed", 0)): r for r in arm_b}
    per_session: dict[str, list[np.ndarray]] = {}
    point: dict[str, list[float]] = {}
    rng = np.random.default_rng(seed)
    found = False
    for sd in sorted(set(by_seed_a) & set(by_seed_b)):
        pa, pb = by_seed_a[sd].get("_run_dir"), by_seed_b[sd].get("_run_dir")
        if not pa or not pb:
            continue
        fa, fb = Path(pa) / "test_predictions.csv", Path(pb) / "test_predictions.csv"
        if not (fa.is_file() and fb.is_file()):
            continue
        da, db = pd.read_csv(fa), pd.read_csv(fb)
        m = da.merge(db, on=["session", "trial"], suffixes=("_a", "_b"))
        if not len(m):
            continue
        found = True
        for s, g in m.groupby("session"):
            y = g["label_a"].to_numpy(int)
            ya, yb = g["pred_a"].to_numpy(int), g["pred_b"].to_numpy(int)
            n = len(y)
            if n < 10:
                continue
            idx = rng.integers(0, n, size=(n_boot, n))
            diffs = np.array([stat(y[i], ya[i]) - stat(y[i], yb[i]) for i in idx])
            per_session.setdefault(str(s), []).append(diffs)
            point.setdefault(str(s), []).append(stat(y, ya) - stat(y, yb))
    if not found:
        return None
    rows = {}
    for s, boots in per_session.items():
        d = np.nanmean(np.stack(boots), axis=0)          # average the per-seed bootstrap differences
        lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
        pt = float(np.mean(point[s]))
        if direction == ">":
            rep_ = lo > 0
        elif direction == "<":
            rep_ = hi < 0
        else:   # not_lower
            rep_ = lo > -NOT_LOWER_MARGIN
        rows[s] = {"diff": pt, "ci": [lo, hi], "replicates": bool(rep_), "n_seeds": len(boots)}
    return {"per_session": rows, "n_replicating": int(sum(r["replicates"] for r in rows.values())), "n_sessions": len(rows),
            "metric": metric, "direction": direction}


# ----------------------------------------------------------------------------------------------------------
# the statistical rule (identical for every comparison)
# ----------------------------------------------------------------------------------------------------------


def wilcoxon_p(diff: np.ndarray, alternative: str) -> float:
    """Wilcoxon signed-rank p-value across sessions; exact for n <= 25, normal approximation otherwise.

    All-zero differences (a degenerate but possible outcome on tiny synthetic data) give p = 1 instead of an
    exception, so a degenerate arm produces an ``inconclusive`` verdict rather than a crash.
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return math.nan
    method = "exact" if d.size <= 25 else "approx"
    try:
        return float(stats.wilcoxon(d, alternative=alternative, method=method).pvalue)
    except ValueError:
        try:
            return float(stats.wilcoxon(d, alternative=alternative, method="auto").pvalue)
        except ValueError:
            return math.nan


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list[float]:
    """Percentile CI of the mean difference from ``n_boot`` resamples of *sessions* (with replacement)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_test(diff: np.ndarray, direction: str, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Apply the contract's rule to per-session paired differences (already averaged over seeds).

    ``direction``: ``">"`` (A must exceed B), ``"<"`` (A must be below B) or ``"not_lower"`` (non-inferiority
    of A with margin :data:`NOT_LOWER_MARGIN`).  Returns the numbers and the verdict (never ``not run`` --
    that is decided by the caller who knows whether the arms exist).
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    ci = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed) if n else [math.nan, math.nan]
    if direction == ">":
        p = wilcoxon_p(d, "greater")
        passes = (p < ALPHA) and (ci[0] > 0)
        failure = f"p >= {ALPHA} or CI lower bound <= 0"
    elif direction == "<":
        p = wilcoxon_p(d, "less")
        passes = (p < ALPHA) and (ci[1] < 0)
        failure = f"p >= {ALPHA} or CI upper bound >= 0"
    elif direction == "not_lower":
        p = wilcoxon_p(d + NOT_LOWER_MARGIN, "greater")
        passes = (p < ALPHA) and (ci[0] > -NOT_LOWER_MARGIN)
        failure = f"p >= {ALPHA} or CI lower bound <= -{NOT_LOWER_MARGIN}"
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if n < MIN_SESSIONS:
        verdict = "not testable"
    else:
        verdict = "supported" if passes else "inconclusive"
    return {"n_sessions": n, "mean_diff": float(d.mean()) if n else math.nan, "ci": ci, "p": p,
            "direction": direction, "passes": bool(passes) if n else False, "failure_condition": failure,
            "verdict": verdict}


def compare_arms(arm_a: list[dict] | None, arm_b: list[dict] | None, metrics: tuple[str, ...] | str,
                 direction: str, label_a: str, label_b: str, baseline_a: str | None = None,
                 baseline_b: str | None = None, exclude: list[str] | None = None, replication: bool = False) -> dict:
    """Full two-arm comparison: per-session table, seed counts, Wilcoxon, bootstrap CI and verdict.

    ``exclude`` removes sessions (empty criteria set) from the pairing; ``replication`` adds the per-session
    trial-bootstrap statistic (:func:`replication_by_session`) when both arms have per-trial predictions."""
    out: dict[str, Any] = {"arm_a": label_a, "arm_b": label_b, "direction": direction, "table": [],
                           "n_seeds_a": len(_seeds(arm_a)), "n_seeds_b": len(_seeds(arm_b)),
                           "n_seeds": min(len(_seeds(arm_a)), len(_seeds(arm_b))), "excluded_sessions": list(exclude or [])}
    va, ma = per_session_values(arm_a, metrics, baseline_a)
    vb, mb = per_session_values(arm_b, metrics, baseline_b)
    # both arms must be read on the same metric; fall back together if either lacks the preferred one
    if ma and mb and ma != mb:
        common = next((m for m in (metrics if not isinstance(metrics, str) else (metrics,))
                       if per_session_values(arm_a, m, baseline_a)[0] and per_session_values(arm_b, m, baseline_b)[0]), None)
        if common:
            va, ma = per_session_values(arm_a, common, baseline_a)
            vb, mb = per_session_values(arm_b, common, baseline_b)
    out["metric"] = ma if ma == mb else None
    if not va or not vb or out["metric"] is None:
        missing = [lab for lab, v in ((label_a, va), (label_b, vb)) if not v]
        out.update({"verdict": "not run", "n_sessions": 0, "mean_diff": math.nan, "ci": [math.nan, math.nan],
                    "p": math.nan, "passes": False, "failure_condition": "required run missing",
                    "missing": missing or [f"no common metric among {metrics}"]})
        return out
    sessions = sorted((set(va) & set(vb)) - set(exclude or []))
    rows = []
    for s in sessions:
        a, b = float(np.mean(va[s])), float(np.mean(vb[s]))
        rows.append({"session": s, "a": a, "b": b, "diff": a - b, "n_seeds_a": len(va[s]), "n_seeds_b": len(vb[s])})
    out["table"] = rows
    out.update(paired_test(np.array([r["diff"] for r in rows]), direction))
    out["mean_a"] = _mean([r["a"] for r in rows])
    out["mean_b"] = _mean([r["b"] for r in rows])
    if replication and not baseline_a and not baseline_b:
        rp = replication_by_session(arm_a, arm_b, direction, metric=out["metric"] or "balanced_accuracy")
        if rp is not None:
            rp["per_session"] = {s: v for s, v in rp["per_session"].items() if s not in set(exclude or [])}
            rp["n_replicating"] = int(sum(v["replicates"] for v in rp["per_session"].values()))
            rp["n_sessions"] = len(rp["per_session"])
            out["replication"] = rp
    return out


def _combine(parts: dict[str, str]) -> str:
    """Conservative combination of sub-verdicts: a failed part dominates, then a missing part, then an
    untestable part; ``supported`` needs every part."""
    vs = list(parts.values())
    if not vs:
        return "not run"
    if "inconclusive" in vs:
        return "inconclusive"
    if "not run" in vs:
        return "not run"
    if "not testable" in vs:
        return "not testable"
    return "supported"


def _replication_line(c: dict | None) -> str:
    rp = (c or {}).get("replication")
    if not rp or not rp.get("n_sessions"):
        return ""
    return f"; replicates in {rp['n_replicating']}/{rp['n_sessions']} sessions (trial bootstrap)"


def _summary_line(c: dict) -> str:
    """Compact 'numbers next to the verdict' string for the verdict table."""
    if c.get("verdict") == "not run":
        return "missing: " + ", ".join(c.get("missing") or ["required run"])
    return (f"A={_fmt(c.get('mean_a'))} B={_fmt(c.get('mean_b'))} diff={_fmt(c.get('mean_diff'))} "
            f"CI={_fmt_ci(c.get('ci'))} p={_fmt_p(c.get('p'))} n_sessions={c.get('n_sessions', 0)} "
            f"n_seeds={c.get('n_seeds', 0)}")


# ----------------------------------------------------------------------------------------------------------
# the predictions
# ----------------------------------------------------------------------------------------------------------


def _p0(runs: dict) -> dict:
    """Negative control: labels permuted before selection and training must give chance-level accuracy.

    A positive result here would mean the pipeline leaks labels (through selection, splitting or the
    adapters), which would invalidate every other prediction; hence it is checked seed by seed and the
    control passes only if every seed lies at or below its own permutation p95.  "At or below" rather than
    strictly below: a control that collapses to a constant prediction scores exactly 1/3 and so does every
    label shuffle, which is the textbook 'at chance' outcome and must not be reported as a failure."""
    arm = runs.get("negative_control/criteria")
    out: dict[str, Any] = {"id": "P0", "title": "Negative control at chance", "comparator": "negative_control/criteria vs its own label-permutation chance",
                           "statistic": "pooled balanced accuracy <= chance p95 (per seed)",
                           "failure_condition": "any seed with balanced accuracy > chance p95", "per_seed": []}
    if not arm:
        out.update({"verdict": "not run", "result": "missing: negative_control/criteria"})
        return out
    for res in arm:
        bacc = _finite((res.get("classification") or {}).get("balanced_accuracy"))
        p95 = _chance_p95(res)
        out["per_seed"].append({"seed": int(res.get("seed", 0)), "balanced_accuracy": bacc, "chance_p95": p95,
                                "chance_mean": _finite((res.get("chance") or {}).get("mean")),
                                "pass": bool(bacc <= p95) if not (math.isnan(bacc) or math.isnan(p95)) else False})
    passes = [r["pass"] for r in out["per_seed"]]
    out["pass"] = bool(passes) and all(passes)
    out["verdict"] = "supported" if out["pass"] else "inconclusive"
    out["result"] = "; ".join(f"seed{r['seed']}: bacc={_fmt(r['balanced_accuracy'])} p95={_fmt(r['chance_p95'])} "
                              f"{'pass' if r['pass'] else 'FAIL'}" for r in out["per_seed"])
    out["n_seeds"] = len(arm)
    return out


def _p1(runs: dict) -> tuple[dict, dict]:
    crit = runs.get("criteria")
    excl = empty_criteria_sessions(runs)
    metrics = ("balanced_accuracy_lr", "balanced_accuracy")
    a = compare_arms(crit, crit, metrics, "not_lower", "logreg_selected_units", "logreg_all_units",
                     baseline_a="logreg_selected_units", baseline_b="logreg_all_units", exclude=excl)
    b = compare_arms(crit, crit, metrics, "not_lower", "criteria (DelayCAST)", "logreg_selected_units",
                     baseline_b="logreg_selected_units", exclude=excl)
    p1a = {"id": "P1a", "title": "Linear decoder on selected K not lower than linear on all units",
           "comparator": "baseline logreg_selected_units vs logreg_all_units (criteria run)",
           "statistic": f"paired {a.get('metric') or 'balanced_accuracy_lr'} difference, non-inferiority margin {NOT_LOWER_MARGIN}",
           "comparison": a, "verdict": a["verdict"], "failure_condition": a["failure_condition"], "result": _summary_line(a)}
    p1b = {"id": "P1b", "title": "DelayCAST on selected K not lower than linear on selected K",
           "comparator": "run criteria vs baseline logreg_selected_units",
           "statistic": f"paired {b.get('metric') or 'balanced_accuracy_lr'} difference, non-inferiority margin {NOT_LOWER_MARGIN}",
           "comparison": b, "verdict": b["verdict"], "failure_condition": b["failure_condition"], "result": _summary_line(b)}
    # Within-pipeline ablations (reported, no verdict): the full classifier (backbone + linear count read-out) against
    # the linear read-out alone and against the backbone alone, trained with the same splits, selection and augmentation.
    abl = {}
    for name in ("criteria_linonly", "criteria_noskip"):
        if runs.get(name):
            abl[name] = compare_arms(crit, runs.get(name), metrics, ">", "criteria", name, exclude=excl, replication=True)
    if abl:
        p1b["ablations"] = abl
        p1b["result"] += " || " + " || ".join(f"vs {k}: {_summary_line(v)}{_replication_line(v)}" for k, v in abl.items())
    return p1a, p1b


def _p2(runs: dict) -> dict:
    crit = runs.get("criteria")
    excl = empty_criteria_sessions(runs)
    parts = {name: compare_arms(crit, runs.get(name), ("balanced_accuracy_lr", "balanced_accuracy"), ">", "criteria", name, exclude=excl, replication=True)
             for name in ("rate", "random")}
    verdict = "not run" if not crit else _combine({k: v["verdict"] for k, v in parts.items()})
    return {"id": "P2", "title": "Criteria subset above rate-matched and random subsets",
            "comparator": "criteria vs rate; criteria vs random (same K_eff per session x region)", "statistic": "paired Left/Right balanced accuracy difference > 0 (both)",
            "failure_condition": "either comparison with p >= 0.05 or CI lower bound <= 0", "comparisons": parts,
            "verdict": verdict, "result": " || ".join(f"vs {k}: {_summary_line(v)}{_replication_line(v)} -> {v['verdict']}" for k, v in parts.items())}


def _p3(runs: dict) -> dict:
    """Late-delay sufficiency: (i) CSI tau95 CI upper bound <= 500 ms, (ii) occluding the last window costs
    more than occluding an earlier window (paired per session), (iii) the linear decoder's tau95 (reported).

    (ii) follows the claim literally: per session, the last window's delta is compared with the *worst* (most
    negative) earlier window, so the last window has to cost more than any earlier window; the mean of the earlier
    windows is reported alongside. (i) additionally requires the full-context accuracy to be above the chance
    p95 of that seed, otherwise a flat, chance-level sweep would satisfy the tau95 rule trivially."""
    crit = runs.get("criteria")
    out: dict[str, Any] = {"id": "P3", "title": "Last 500 ms retain >= 95 % of accuracy; last window costs most",
                           "comparator": "criteria run: CSI tau95 CI (full-context accuracy above chance p95); temporal occlusion last window vs worst earlier window",
                           "statistic": f"tau95 CI upper <= {TAU95_MAX_MS:.0f} ms (all seeds, full-context accuracy > chance p95) AND paired delta(last) - min delta(earlier) < 0",
                           "failure_condition": f"any seed at chance or with tau95 CI upper > {TAU95_MAX_MS:.0f} ms, or occlusion test p >= 0.05 / CI upper >= 0",
                           "csi_per_seed": [], "linear_tau95_per_seed": [], "argmin_window_per_seed": []}
    if not crit:
        out.update({"verdict": "not run", "result": "missing: criteria", "comparison": {"verdict": "not run"}})
        return out
    # (i) CSI
    for res in crit:
        csi = res.get("csi") or {}
        ci = csi.get("tau95_ci_ms") or [None, None]
        hi = _finite(ci[1] if len(ci) == 2 else None)
        full_acc = _finite((res.get("classification") or {}).get("balanced_accuracy"))
        p95 = _finite((res.get("chance") or {}).get("p95"))
        above_chance = (not math.isnan(full_acc)) and (not math.isnan(p95)) and full_acc > p95
        out["csi_per_seed"].append({"seed": int(res.get("seed", 0)), "tau95_ms": _finite(csi.get("tau95_ms")),
                                    "tau95_ci_ms": [_finite(ci[0] if len(ci) == 2 else None), hi],
                                    "full_balanced_accuracy": full_acc, "chance_p95": p95, "above_chance": above_chance,
                                    "pass": bool(above_chance and hi <= TAU95_MAX_MS) if not math.isnan(hi) else False})
        lin = _finite(res.get("tau95_linear_ms"))
        out["linear_tau95_per_seed"].append({"seed": int(res.get("seed", 0)), "tau95_linear_ms": lin,
                                             "pass": bool(lin <= TAU95_MAX_MS) if not math.isnan(lin) else False})
    csi_pass = bool(out["csi_per_seed"]) and all(r["pass"] for r in out["csi_per_seed"])
    # (ii) temporal occlusion: per session, last window delta minus mean of earlier deltas, averaged over seeds
    last: dict[str, list[float]] = {}
    earlier: dict[str, list[float]] = {}
    for res in crit:
        wins = [w for w in (res.get("temporal_occlusion") or []) if isinstance(w, dict) and "window_end_ms" in w]
        if not wins:
            continue
        end_max = max(_finite(w["window_end_ms"]) for w in wins)
        pooled = [(_finite(w.get("delta_balanced_accuracy")), _finite(w["window_end_ms"])) for w in wins]
        worst = min(pooled, key=lambda t: (t[0] if not math.isnan(t[0]) else math.inf))
        out["argmin_window_per_seed"].append({"seed": int(res.get("seed", 0)), "window_end_ms": worst[1],
                                              "delta_balanced_accuracy": worst[0], "is_last": bool(worst[1] == end_max)})
        last_w = [w for w in wins if _finite(w["window_end_ms"]) == end_max]
        early_w = [w for w in wins if _finite(w["window_end_ms"]) < end_max]
        sessions = set()
        for w in wins:
            sessions |= set((w.get("per_session") or {}).keys())
        for s in sessions:
            lv = [_finite((w.get("per_session") or {}).get(s)) for w in last_w]
            ev = [_finite((w.get("per_session") or {}).get(s)) for w in early_w]
            lv = [v for v in lv if not math.isnan(v)]
            ev = [v for v in ev if not math.isnan(v)]
            if lv and ev:
                last.setdefault(s, []).append(float(np.mean(lv)))
                earlier.setdefault(s, []).append(float(np.min(ev)))      # the worst earlier window
    excl = set(empty_criteria_sessions(runs))
    rows = [{"session": s, "a": float(np.mean(last[s])), "b": float(np.mean(earlier[s])),
             "diff": float(np.mean(last[s]) - np.mean(earlier[s]))} for s in sorted((set(last) & set(earlier)) - excl)]
    comp: dict[str, Any] = {"arm_a": "delta(last window)", "arm_b": "min delta(earlier windows)", "metric": "delta_balanced_accuracy",
                            "table": rows, "n_seeds": len(crit), "n_seeds_a": len(crit), "n_seeds_b": len(crit),
                            "excluded_sessions": sorted(excl)}
    if rows:
        comp.update(paired_test(np.array([r["diff"] for r in rows]), "<"))
        comp["mean_a"], comp["mean_b"] = _mean([r["a"] for r in rows]), _mean([r["b"] for r in rows])
    else:
        comp.update({"verdict": "not run", "missing": ["temporal_occlusion per_session"], "n_sessions": 0,
                     "mean_diff": math.nan, "ci": [math.nan, math.nan], "p": math.nan, "passes": False})
    out["comparison"] = comp
    out["csi_pass"] = csi_pass
    out["linear_tau95_pass"] = bool(out["linear_tau95_per_seed"]) and all(r["pass"] for r in out["linear_tau95_per_seed"])
    parts = {"csi": ("supported" if csi_pass else ("not run" if not out["csi_per_seed"] else "inconclusive")),
             "occlusion": comp["verdict"]}
    out["parts"] = parts
    out["verdict"] = _combine(parts)
    out["result"] = (f"tau95 CI upper per seed: {', '.join(_fmt(r['tau95_ci_ms'][1], 0) for r in out['csi_per_seed'])} ms "
                     f"({'pass' if csi_pass else 'FAIL'}); occlusion {_summary_line(comp)} -> {comp['verdict']}; "
                     f"linear tau95: {', '.join(_fmt(r['tau95_linear_ms'], 0) for r in out['linear_tau95_per_seed']) or 'n/a'} ms")
    return out


def _coupling_enrichment(crit: list[dict]) -> dict:
    """Fraction of C-criterion (coupled) units among selected vs eligible-but-unselected units, per session.

    Fisher's exact test is computed on the first seed only (the same units recur across seeds, so pooling
    counts over seeds would fabricate replication); the sign test across sessions uses the seed-mean fraction
    difference, i.e. the session stays the unit of replication."""
    per_session: dict[str, dict[str, Any]] = {}
    for si, res in enumerate(crit):
        rd = Path(res.get("_run_dir", ""))
        if not rd.is_dir():
            continue
        for f in sorted(rd.glob("selection_*.csv")):
            if f.name == "selection_funnel.csv":
                continue
            try:
                t = pd.read_csv(f)
            except Exception as e:  # pragma: no cover - corrupt table
                log.warning("could not read %s: %s", f, e)
                continue
            if not {"selected", "eligible", "c_coupling"} <= set(t.columns):
                continue
            sel = _as_bool(t["selected"]).to_numpy()
            elig = _as_bool(t["eligible"]).to_numpy()
            cpl = _as_bool(t["c_coupling"]).to_numpy()
            unsel = elig & ~sel
            n_sel, n_uns = int(sel.sum()), int(unsel.sum())
            k_sel, k_uns = int((cpl & sel).sum()), int((cpl & unsel).sum())
            s = f.stem[len("selection_"):].replace("__", "/")
            d = per_session.setdefault(s, {"session": s, "frac_selected": [], "frac_unselected": [], "n_selected": n_sel,
                                           "n_eligible_unselected": n_uns})
            d["frac_selected"].append(k_sel / n_sel if n_sel else math.nan)
            d["frac_unselected"].append(k_uns / n_uns if n_uns else math.nan)
            if si == 0:
                d["fisher_p"] = (float(stats.fisher_exact([[k_sel, n_sel - k_sel], [k_uns, n_uns - k_uns]], alternative="greater")[1])
                                 if n_sel and n_uns else math.nan)
    rows = []
    for s, d in sorted(per_session.items()):
        fs, fu = _mean(d["frac_selected"]), _mean(d["frac_unselected"])
        rows.append({"session": s, "frac_selected": fs, "frac_unselected": fu, "diff": fs - fu, "fisher_p": d.get("fisher_p", math.nan),
                     "n_selected": d["n_selected"], "n_eligible_unselected": d["n_eligible_unselected"]})
    diffs = np.array([r["diff"] for r in rows if not math.isnan(_finite(r["diff"]))])
    nz = diffs[diffs != 0]
    sign_p = float(stats.binomtest(int((nz > 0).sum()), int(nz.size), 0.5, alternative="greater").pvalue) if nz.size else math.nan
    return {"table": rows, "n_sessions": int(diffs.size), "n_positive": int((nz > 0).sum()), "sign_test_p": sign_p,
            "mean_frac_selected": _mean([r["frac_selected"] for r in rows]),
            "mean_frac_unselected": _mean([r["frac_unselected"] for r in rows])}


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0", "yes"))


def _p4(runs: dict) -> dict:
    """Forecast beyond the unit's own mean response: per session the mean over regions of the model's Poisson
    deviance explained relative to the training-PSTH null (0 = the null itself), tested against 0 across sessions.

    Two reference points are reported next to it, neither is the comparator: the *persistence* forecast (the
    late-delay rate carried into the response epoch; on real data it is far below the null because response rates
    differ from delay rates, so "beyond persistence" would be trivially true) and the *class-conditional oracle*
    (the mean response PSTH of the true class; a model above it forecasts trial-specific structure beyond the
    class identity).  Regions without data (no units) are skipped, not counted as zero."""
    crit = runs.get("criteria")
    out: dict[str, Any] = {"id": "P4", "title": "Late-delay activity forecasts response-epoch activity beyond the mean response",
                           "comparator": "criteria forecast: Poisson deviance explained of the model vs the training-PSTH null (0), mean over regions; persistence and class-conditional oracle reported",
                           "statistic": "per-session deviance explained > 0", "failure_condition": "p >= 0.05 or CI lower bound <= 0"}
    if not crit:
        out.update({"verdict": "not run", "result": "missing: criteria", "comparison": {"verdict": "not run"}})
        return out
    excl = set(empty_criteria_sessions(runs))
    model: dict[str, list[float]] = {}
    pers: dict[str, list[float]] = {}
    oracle: dict[str, list[float]] = {}
    for res in crit:
        for row in (res.get("forecast") or {}).get("per_session") or []:
            if not isinstance(row, dict) or "session" not in row or str(row["session"]) in excl:
                continue
            m, p, o = [], [], []
            for r in REGIONS:
                a = _finite(row.get(f"deviance_explained_{r}"))
                if math.isnan(a):
                    continue
                m.append(a)
                p.append(_finite(row.get(f"deviance_explained_persistence_{r}")))
                o.append(_finite(row.get(f"deviance_explained_classmean_{r}")))
            if m:
                model.setdefault(str(row["session"]), []).append(float(np.mean(m)))
                pers.setdefault(str(row["session"]), []).append(float(np.nanmean(p)) if np.isfinite(p).any() else math.nan)
                oracle.setdefault(str(row["session"]), []).append(float(np.nanmean(o)) if np.isfinite(o).any() else math.nan)
    rows = [{"session": s, "a": float(np.mean(model[s])), "b": 0.0, "diff": float(np.mean(model[s])),
             "persistence": float(np.nanmean(pers[s])) if np.isfinite(pers[s]).any() else math.nan,
             "class_oracle": float(np.nanmean(oracle[s])) if np.isfinite(oracle[s]).any() else math.nan}
            for s in sorted(model)]
    comp: dict[str, Any] = {"arm_a": "model deviance explained", "arm_b": "training-PSTH null (0)", "metric": "deviance_explained (mean over regions)",
                            "table": rows, "n_seeds": len(crit), "n_seeds_a": len(crit), "n_seeds_b": len(crit), "excluded_sessions": sorted(excl)}
    if rows:
        comp.update(paired_test(np.array([r["diff"] for r in rows]), ">"))
        comp["mean_a"], comp["mean_b"] = _mean([r["a"] for r in rows]), 0.0
        comp["mean_persistence"] = _mean([r["persistence"] for r in rows if not math.isnan(r["persistence"])])
        comp["mean_class_oracle"] = _mean([r["class_oracle"] for r in rows if not math.isnan(r["class_oracle"])])
        above_oracle = [r["a"] > r["class_oracle"] for r in rows if not math.isnan(r["class_oracle"])]
        comp["n_sessions_above_class_oracle"] = int(sum(above_oracle))
        comp["n_sessions_with_class_oracle"] = len(above_oracle)
    else:
        comp.update({"verdict": "not run", "missing": ["forecast per_session"], "n_sessions": 0, "mean_diff": math.nan,
                     "ci": [math.nan, math.nan], "p": math.nan, "passes": False})
    out["comparison"] = comp
    out["coupling_enrichment"] = _coupling_enrichment(crit)
    out["verdict"] = comp["verdict"]
    ce = out["coupling_enrichment"]
    extra = ""
    if rows:
        extra = (f"; persistence={_fmt(comp.get('mean_persistence'))}, class-conditional oracle={_fmt(comp.get('mean_class_oracle'))} "
                 f"(model above oracle in {comp.get('n_sessions_above_class_oracle', 0)}/{comp.get('n_sessions_with_class_oracle', 0)} sessions)")
    out["result"] = (f"{_summary_line(comp)}{extra}; coupled fraction selected={_fmt(ce['mean_frac_selected'])} vs "
                     f"unselected={_fmt(ce['mean_frac_unselected'])} sign-test p={_fmt_p(ce['sign_test_p'])}")
    return out


def _ablation_delta(entry: dict, session: str) -> float:
    """Per-session ablation delta; the per_session map is a scalar delta of balanced accuracy in the current
    schema, but a nested dict with a Left/Right-only delta is preferred whenever a future evaluate.py writes one."""
    v = (entry.get("per_session") or {}).get(session)
    if isinstance(v, dict):
        x = _finite(v.get("delta_balanced_accuracy_lr"))
        return x if not math.isnan(x) else _finite(v.get("delta_balanced_accuracy"))
    return _finite(v)


def _p5a(runs: dict) -> dict:
    crit = runs.get("criteria")
    out: dict[str, Any] = {"id": "P5a", "title": "Removing ALM hurts Left/Right decoding more than removing striatum",
                           "comparator": "criteria region_ablation (drop; fallback permute): mean delta ALM vs mean delta STR",
                           "statistic": "paired per-session difference (ALM - STR) < 0", "failure_condition": "p >= 0.05 or CI upper bound >= 0"}
    if not crit:
        out.update({"verdict": "not run", "result": "missing: criteria", "comparison": {"verdict": "not run"}})
        return out
    alm: dict[str, list[float]] = {}
    strn: dict[str, list[float]] = {}
    method_used = None
    for res in crit:
        abl = [a for a in (res.get("region_ablation") or []) if isinstance(a, dict)]
        methods = [m for m in ("drop", "permute", "zero") if any(a.get("method") == m for a in abl)]
        if not methods:
            continue
        method_used = method_used or methods[0]
        rows_m = [a for a in abl if a.get("method") == methods[0]]
        sessions = set()
        for a in rows_m:
            sessions |= set((a.get("per_session") or {}).keys())
        for s in sessions:
            da = [_ablation_delta(a, s) for a in rows_m if str(a.get("dropped_region", "")).startswith("ALM")]
            ds = [_ablation_delta(a, s) for a in rows_m if str(a.get("dropped_region", "")).startswith("STR")]
            da = [v for v in da if not math.isnan(v)]
            ds = [v for v in ds if not math.isnan(v)]
            if da and ds:
                alm.setdefault(s, []).append(float(np.mean(da)))
                strn.setdefault(s, []).append(float(np.mean(ds)))
    excl = set(empty_criteria_sessions(runs))
    rows = [{"session": s, "a": float(np.mean(alm[s])), "b": float(np.mean(strn[s])), "diff": float(np.mean(alm[s]) - np.mean(strn[s]))}
            for s in sorted((set(alm) & set(strn)) - excl)]
    comp: dict[str, Any] = {"arm_a": "mean delta (ALM removed)", "arm_b": "mean delta (STR removed)", "metric": f"delta_balanced_accuracy ({method_used})",
                            "table": rows, "n_seeds": len(crit), "n_seeds_a": len(crit), "n_seeds_b": len(crit), "method": method_used,
                            "excluded_sessions": sorted(excl)}
    if rows:
        comp.update(paired_test(np.array([r["diff"] for r in rows]), "<"))
        comp["mean_a"], comp["mean_b"] = _mean([r["a"] for r in rows]), _mean([r["b"] for r in rows])
    else:
        comp.update({"verdict": "not run", "missing": ["region_ablation per_session"], "n_sessions": 0, "mean_diff": math.nan,
                     "ci": [math.nan, math.nan], "p": math.nan, "passes": False})
    # the other ablation method, reported next to the primary one (drop = in-distribution, permute = reliance)
    other_method = {"drop": "permute", "permute": "drop"}.get(method_used or "", None)
    if other_method:
        oa, ob = [], []
        for res in crit:
            rows_o = [a for a in (res.get("region_ablation") or []) if isinstance(a, dict) and a.get("method") == other_method]
            for a in rows_o:
                for s, v in (a.get("per_session") or {}).items():
                    if s in excl:
                        continue
                    (oa if str(a.get("dropped_region", "")).startswith("ALM") else ob).append(_finite(v))
        comp["other_method"] = {"method": other_method, "mean_delta_alm": _mean([v for v in oa if not math.isnan(v)]),
                                "mean_delta_str": _mean([v for v in ob if not math.isnan(v)])}
    out["comparison"] = comp
    out["verdict"] = comp["verdict"]
    om = comp.get("other_method")
    out["result"] = _summary_line(comp) + (f" (method={method_used}" + (f"; {om['method']}: ALM {_fmt(om['mean_delta_alm'])} vs STR {_fmt(om['mean_delta_str'])}" if om else "") + ")" if method_used else "")
    return out


def wilson_ci(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score interval for a binomial proportion (never collapses to zero width at 0 or 1 like Wald)."""
    if n <= 0:
        return [math.nan, math.nan]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _p5b(runs: dict) -> dict:
    """Exploratory: is the no-lick (Ignore) class decodable at all?  Reported as pooled Ignore recall with a
    Wilson CI; ``supported`` (exploratory) if the CI lower bound exceeds the 1/3 uniform-guess recall.  The
    trial-index baseline flags a drift confound: if trial number alone beats chance, Ignore trials (which
    cluster late in a session when the animal disengages) could be decoded from slow drift, not from
    delay-period activity.  ``n`` is the mean Ignore test-trial count per seed, not the sum, because the same
    trials recur across seeds."""
    crit = runs.get("criteria")
    out: dict[str, Any] = {"id": "P5b", "title": "Ignore-trial decodability (exploratory)",
                           "comparator": "criteria pooled Ignore recall vs 1/3 (uniform guess); trial-index drift control",
                           "statistic": "Wilson 95% CI of Ignore recall", "failure_condition": f"n_Ignore < {MIN_IGNORE_TRIALS} (not testable); CI lower bound <= 1/3"}
    if not crit:
        out.update({"verdict": "not run", "result": "missing: criteria"})
        return out
    recalls = _pooled(crit, "classification", "recall", "Ignore")
    ns = _pooled(crit, "classification", "n_per_class", "Ignore")
    recall = _mean(recalls)
    n = int(round(_mean(ns))) if not math.isnan(_mean(ns)) else 0
    k = int(round(recall * n)) if not math.isnan(recall) else 0
    ci = wilson_ci(k, n)
    drift = _mean([_finite((_baseline_row(r, "logreg_trial_index") or {}).get("balanced_accuracy")) for r in crit])
    p95 = _mean([_chance_p95(r) for r in crit])
    confounded = bool(drift > p95) if not (math.isnan(drift) or math.isnan(p95)) else False
    out.update({"ignore_recall": recall, "n_ignore": n, "wilson_ci": ci, "n_seeds": len(crit),
                "trial_index_balanced_accuracy": drift, "chance_p95": p95, "confounded": confounded})
    if n < MIN_IGNORE_TRIALS:
        out["verdict"] = "not testable"
    elif math.isnan(recall):
        out["verdict"] = "not run"
    else:
        out["verdict"] = "supported" if ci[0] > 1.0 / 3.0 else "inconclusive"
    out["result"] = (f"Ignore recall={_fmt(recall)} Wilson CI={_fmt_ci(ci)} n_Ignore={n} n_seeds={len(crit)}; "
                     f"trial-index bacc={_fmt(drift)} vs chance p95={_fmt(p95)}" + (" -> CONFOUNDED" if confounded else ""))
    return out


def _p6(runs: dict) -> dict:
    """Importance / criteria agreement per seed: Spearman rho between occlusion importance and the model-free
    score across selected units, one rho per session x region cell, sign test across cells."""
    crit = runs.get("criteria")
    out: dict[str, Any] = {"id": "P6", "title": "Model-based importance agrees with the model-free criteria",
                           "comparator": "importance_agreement.importance_vs_score (gate_vs_score reported)",
                           "statistic": f"mean_rho > 0 and sign_test_p < {ALPHA} with n_cells >= {MIN_AGREEMENT_CELLS} (every seed)",
                           "failure_condition": f"any seed with mean_rho <= 0 or sign_test_p >= {ALPHA}; n_cells < {MIN_AGREEMENT_CELLS} in all seeds -> not testable",
                           "per_seed": {"importance_vs_score": [], "gate_vs_score": []}}
    if not crit:
        out.update({"verdict": "not run", "result": "missing: criteria"})
        return out
    for key in ("importance_vs_score", "gate_vs_score"):
        for res in crit:
            d = (res.get("importance_agreement") or {}).get(key) or {}
            rho, p, n = _finite(d.get("mean_rho")), _finite(d.get("sign_test_p")), _int(d.get("n_cells"))
            out["per_seed"][key].append({"seed": int(res.get("seed", 0)), "mean_rho": rho, "median_rho": _finite(d.get("median_rho")),
                                         "n_cells": n, "n_positive": _int(d.get("n_positive")), "sign_test_p": p,
                                         "enough_cells": n >= MIN_AGREEMENT_CELLS,
                                         "pass": bool(rho > 0 and p < ALPHA and n >= MIN_AGREEMENT_CELLS)})
    verdicts = {}
    for key, rows in out["per_seed"].items():
        if not rows or not any(r["n_cells"] > 0 for r in rows):
            verdicts[key] = "not run"
        elif not any(r["enough_cells"] for r in rows):
            verdicts[key] = "not testable"
        else:
            verdicts[key] = "supported" if all(r["pass"] for r in rows if r["enough_cells"]) and all(r["enough_cells"] for r in rows) else "inconclusive"
    out["parts"] = verdicts
    out["verdict"] = verdicts["importance_vs_score"]
    out["n_seeds"] = len(crit)
    out["result"] = "; ".join(
        f"{key}: " + ", ".join(f"seed{r['seed']} rho={_fmt(r['mean_rho'])} p={_fmt_p(r['sign_test_p'])} n_cells={r['n_cells']}" for r in rows)
        + f" -> {verdicts[key]}" for key, rows in out["per_seed"].items())
    return out


def _p7(runs: dict) -> dict:
    crit = runs.get("criteria")
    other = "criteria_popmean" if runs.get("criteria_popmean") else ("criteria_nospec" if runs.get("criteria_nospec") else "criteria_popmean")
    comp = compare_arms(crit, runs.get(other), ("balanced_accuracy_lr", "balanced_accuracy"), ">", "criteria", other, exclude=empty_criteria_sessions(runs), replication=True)
    return {"id": "P7", "title": "Spectro-temporal population branch adds accuracy beyond the population-rate control",
            "comparator": f"criteria vs {other}", "statistic": "paired Left/Right balanced accuracy difference > 0",
            "failure_condition": comp["failure_condition"], "comparison": comp, "verdict": comp["verdict"],
            "result": _summary_line(comp) + _replication_line(comp)}


def _p8(runs: dict, population: bool = False) -> dict:
    """``population``: the random-K arm is not applicable (every arm uses the same channels), so the verdict rests on
    the chance part alone and the comparator says so."""
    crit = runs.get("cross_dataset/criteria")
    out: dict[str, Any] = {"id": "P8", "title": "Cross-dataset transfer above chance and above random-K after adapter fitting",
                           "comparator": "cross_dataset/criteria vs chance p95 (per seed) and vs cross_dataset/random (paired per session)",
                           "statistic": "pooled balanced accuracy > chance p95 (all seeds) AND paired difference > 0",
                           "failure_condition": "any seed at/below chance p95, or paired test p >= 0.05 / CI lower bound <= 0", "per_seed": []}
    if population:
        out.update({"comparator": "cross_dataset/criteria vs chance p95 (per seed); the random-K arm is not applicable to population channels",
                    "statistic": "pooled balanced accuracy > chance p95 (all seeds)", "failure_condition": "any seed at/below chance p95"})
    if not crit:
        out.update({"verdict": "not run", "result": "missing: cross_dataset/criteria", "comparison": {"verdict": "not run"}})
        return out
    for res in crit:
        bacc, p95 = _finite((res.get("classification") or {}).get("balanced_accuracy")), _chance_p95(res)
        out["per_seed"].append({"seed": int(res.get("seed", 0)), "balanced_accuracy": bacc, "chance_p95": p95,
                                "pass": bool(bacc > p95) if not (math.isnan(bacc) or math.isnan(p95)) else False})
    chance_pass = bool(out["per_seed"]) and all(r["pass"] for r in out["per_seed"])
    out["chance_pass"] = chance_pass
    seeds_line = "; ".join(f"seed{r['seed']}: bacc={_fmt(r['balanced_accuracy'])} p95={_fmt(r['chance_p95'])} {'pass' if r['pass'] else 'FAIL'}"
                           for r in out["per_seed"])
    if population:
        out["comparison"] = {"verdict": "not run", "missing": ["cross_dataset/random (not applicable to population channels)"]}
        out["parts"] = {"chance": "supported" if chance_pass else "inconclusive"}
        out["verdict"] = _combine(out["parts"])
        out["result"] = seeds_line + " || vs random: not applicable (population channels)"
        return out
    comp = compare_arms(crit, runs.get("cross_dataset/random"), "balanced_accuracy", ">", "cross_dataset/criteria", "cross_dataset/random")
    out["comparison"] = comp
    out["parts"] = {"chance": "supported" if chance_pass else "inconclusive", "random": comp["verdict"]}
    out["verdict"] = _combine(out["parts"])
    out["result"] = seeds_line + f" || vs random: {_summary_line(comp)} -> {comp['verdict']}"
    return out


# ----------------------------------------------------------------------------------------------------------
# header + selection summary
# ----------------------------------------------------------------------------------------------------------


def outcome_diagnostic(cfg, runs: dict) -> dict:
    """Left/Right accuracy of the criteria run split by the audited behavioural outcome (hit = licked the instructed
    side, miss = licked the other side).

    ``Data`` carries no log, but three of its sessions are the same recordings as ``Data2`` sessions, whose audited
    logs have ``outcome`` and ``trial_instruction`` per trial number.  A ``Data`` session is matched to a ``Data2``
    log when the set of trial numbers in its cache equals the set of NPZ trial numbers of that ``Data2`` session.
    A decoder of the *upcoming action* is expected to be less accurate on miss trials (the delay activity partly
    follows the instruction before the animal changes its mind), so the split says how much of the remaining error
    is behavioural rather than a decoding failure.  Reported only; no verdict."""
    out: dict[str, Any] = {"available": False, "sessions": {}, "note": ""}
    crit = runs.get("criteria")
    if not crit or cfg is None:
        return out
    try:
        from .data.cache import _cache_key
        from .data.discovery import discover_dataset_b
        cache_dir = Path(cfg.data.cache_dir) / _cache_key(cfg)
        recs_b = discover_dataset_b(cfg.data.data_b_root) if bool(cfg.data.get_path("use_dataset_b", True)) else []
    except Exception as e:  # pragma: no cover
        out["note"] = f"could not read the Data2 logs: {e}"
        return out
    by_b: dict[str, dict[int, dict]] = {}
    for r in recs_b:
        if r.csv:
            by_b.setdefault(r.session, {})[int(r.trial)] = r.csv
    trials_b = {s: set(d) for s, d in by_b.items()}
    # pooled predictions of every seed of the criteria run
    preds = []
    for res in crit:
        f = Path(res.get("_run_dir", "")) / "test_predictions.csv"
        if f.is_file():
            try:
                df = pd.read_csv(f)
                df["seed"] = int(res.get("seed", 0))
                preds.append(df)
            except Exception:  # pragma: no cover
                continue
    if not preds:
        out["note"] = "no test_predictions.csv"
        return out
    pred = pd.concat(preds, ignore_index=True)
    left, right = CLASSES.index("Left"), CLASSES.index("Right")
    pooled = {"hit": [0, 0], "miss": [0, 0]}
    for sess in sorted(pred.session.unique()):
        meta = cache_dir / (str(sess).replace("/", "__") + ".meta.csv")
        if not meta.is_file():
            continue
        try:
            all_trials = set(pd.read_csv(meta, usecols=["trial"]).trial.astype(int))
        except Exception:  # pragma: no cover
            continue
        if str(sess).startswith("B/"):
            match = str(sess) if str(sess) in by_b else None
        else:
            match = next((sb for sb, ts in trials_b.items() if ts == all_trials), None)
        if match is None:
            continue
        rows = by_b[match]
        p = pred[(pred.session == sess) & pred.label.isin([left, right])].copy()
        p["outcome"] = [str(rows.get(int(t), {}).get("outcome", "")).strip().lower() for t in p.trial]
        p["correct"] = (p.pred == p.label).astype(int)
        entry = {"matched_log": match}
        for oc in ("hit", "miss"):
            q = p[p.outcome == oc]
            entry[oc] = {"n": int(len(q)), "accuracy": _finite(q.correct.mean()) if len(q) else math.nan}
            pooled[oc][0] += int(q.correct.sum())
            pooled[oc][1] += int(len(q))
        out["sessions"][str(sess)] = entry
    if out["sessions"]:
        out["available"] = True
        out["pooled"] = {oc: {"n": n, "accuracy": (k / n if n else math.nan)} for oc, (k, n) in pooled.items()}
    else:
        out["note"] = "no Data session matches a Data2 log by trial numbers"
    return out


def _all_sessions(runs: dict, summary: pd.DataFrame | None) -> list[str]:
    s: set[str] = set()
    for arm in runs.values():
        for res in arm:
            for row in res.get("per_session") or []:
                if isinstance(row, dict) and "session" in row:
                    s.add(str(row["session"]))
    if summary is not None and "session" in summary.columns:
        s |= set(summary["session"].astype(str))
    return sorted(s)


def _animal(session: str) -> str:
    """'A/SessionN' -> 'SessionN' (dataset A has one session per animal); 'B/sub-xxxxxx_ses-...' -> 'sub-xxxxxx'."""
    ds, _, rest = session.partition("/")
    return rest.split("_")[0] if ds == "B" else rest


def _header(runs: dict, summary: pd.DataFrame | None) -> dict:
    sessions = _all_sessions(runs, summary)
    by_ds = {"A": [s for s in sessions if s.startswith("A/")], "B": [s for s in sessions if s.startswith("B/")]}
    other = [s for s in sessions if not (s.startswith("A/") or s.startswith("B/"))]
    animals = {ds: sorted({_animal(s) for s in ss}) for ds, ss in by_ds.items()}
    seeds = sorted({int(r.get("seed", 0)) for arm in runs.values() for r in arm})
    table = []
    for name, arm in sorted(runs.items()):
        n_sess = len(per_session_values(arm, "balanced_accuracy")[0])
        table.append({"run": name, "n_seeds": len(arm), "seeds": _seeds(arm), "n_sessions": n_sess,
                      "balanced_accuracy": _mean(_pooled(arm, "classification", "balanced_accuracy")),
                      "balanced_accuracy_lr": _mean(_pooled(arm, "classification", "balanced_accuracy_lr")),
                      "chance_p95": _mean([_chance_p95(r) for r in arm]),
                      "n_test_trials": _mean(_pooled(arm, "classification", "n"))})
    ref = runs.get("criteria") or next((a for a in runs.values() if a), None)
    per_class = {c: None for c in CLASSES}
    if ref:
        npc = (ref[0].get("classification") or {}).get("n_per_class") or {}
        per_class = {c: (int(npc[c]) if c in npc and npc[c] is not None else None) for c in CLASSES}
    return {"n_sessions": len(sessions), "sessions": sessions, "n_sessions_by_dataset": {k: len(v) for k, v in by_ds.items()},
            "other_sessions": other, "animals": animals, "n_animals": {k: len(v) for k, v in animals.items()},
            "seeds": seeds, "runs": table, "test_trials_per_class": per_class,
            "test_trials_source": (ref[0].get("_name") if ref else None),
            "empty_criteria_sessions": empty_criteria_sessions(runs), "k_eff_per_session": k_eff_per_session(runs)}


def _selection_summary(out_dir: Path, runs: dict) -> dict:
    """Descriptive selection numbers: the all-trial summary table plus the train-split funnels of the criteria
    run (what the model actually saw).  The W-independence flag is the mean phi(S, W) across sessions: when
    selectivity and the spectral criterion co-occur that strongly, W adds no independent evidence for a unit."""
    out: dict[str, Any] = {"summary_csv": None, "per_region": {}, "funnel_per_region": {}, "phi": {}, "w_independence": {}}
    p = Path(out_dir) / "selection" / "summary.csv"
    summary = None
    if p.is_file():
        try:
            summary = pd.read_csv(p)
        except Exception as e:  # pragma: no cover
            log.warning("could not read %s: %s", p, e)
    if summary is not None and len(summary):
        out["summary_csv"] = str(p)
        out["n_sessions"] = int(len(summary))
        for r in REGIONS:
            out["per_region"][r] = {"recorded": int(summary[f"tot_{r}"].sum()) if f"tot_{r}" in summary else None,
                                    "selected": int(summary[f"sel_{r}"].sum()) if f"sel_{r}" in summary else None}
        for k in ("n_units", "n_floor", "n_eligible", "n_selected"):
            if k in summary:
                out[k] = int(summary[k].sum())
        for k in ("median_stability_selected", "null_median_stability_max", "null_n_selected_mean", "null_n_selected_max",
                  "median_onset_ms_selected", "frac_sustained_to_go_selected",
                  "frac_selectivity", "frac_coupling", "frac_spectral", "frac_ramp", "frac_locus", "frac_ignore"):
            if k in summary:
                col = pd.to_numeric(summary[k], errors="coerce")
                out[k] = {"median": _finite(col.median()), "mean": _finite(col.mean()), "max": _finite(col.max()),
                          "per_session": dict(zip(summary["session"].astype(str), [_finite(v) for v in col]))}
        for k in ("phi_SC", "phi_SW", "phi_SR", "phi_CW", "phi_CR", "phi_WR"):
            if k in summary:
                col = pd.to_numeric(summary[k], errors="coerce")
                out["phi"][k] = {"mean": _finite(col.mean()), "max": _finite(col.max()),
                                 "per_session": dict(zip(summary["session"].astype(str), [_finite(v) for v in col]))}
    # train-split funnels of the criteria run
    funnels = []
    for res in runs.get("criteria") or []:
        f = Path(res.get("_run_dir", "")) / "selection_funnel.csv"
        if f.is_file():
            try:
                df = pd.read_csv(f)
                df["seed"] = int(res.get("seed", 0))
                funnels.append(df)
            except Exception as e:  # pragma: no cover
                log.warning("could not read %s: %s", f, e)
    if funnels:
        fun = pd.concat(funnels, ignore_index=True)
        cols = [c for c in ("recorded", "pass_floor", "eligible", "stable", "selected") if c in fun]
        per_seed = fun.groupby(["seed", "region"])[cols].sum().groupby("region").mean()
        out["funnel_per_region"] = {r: {c: _finite(per_seed.loc[r, c]) for c in cols} for r in per_seed.index}
        if "expected_false_selections_bound" in fun:
            b = pd.to_numeric(fun["expected_false_selections_bound"], errors="coerce")
            out["expected_false_selections_bound"] = {"mean_per_region_session": _finite(b.mean()), "max": _finite(b.max())}
        if "K" in fun:
            out["K"] = _finite(pd.to_numeric(fun["K"], errors="coerce").max())
        out["n_seeds_funnel"] = int(fun["seed"].nunique())
        if "phi_SW" in fun and "phi_SW" not in out["phi"]:
            col = pd.to_numeric(fun.groupby("session")["phi_SW"].mean(), errors="coerce")
            out["phi"]["phi_SW"] = {"mean": _finite(col.mean()), "max": _finite(col.max()), "per_session": {str(k): _finite(v) for k, v in col.items()}}
    phi_sw = out["phi"].get("phi_SW", {})
    mean_sw = _finite(phi_sw.get("mean"))
    above = sorted(s for s, v in (phi_sw.get("per_session") or {}).items() if not math.isnan(_finite(v)) and v > PHI_SW_INDEPENDENCE)
    out["w_independence"] = {"phi_SW_mean": mean_sw, "threshold": PHI_SW_INDEPENDENCE, "sessions_above": above,
                             "flag": bool(mean_sw > PHI_SW_INDEPENDENCE) if not math.isnan(mean_sw) else False,
                             "message": ("W is not independent evidence in this dataset" if (not math.isnan(mean_sw) and mean_sw > PHI_SW_INDEPENDENCE)
                                         else ("phi_SW not available" if math.isnan(mean_sw) else "W is independent enough of S"))}
    return out


# ----------------------------------------------------------------------------------------------------------
# assembling + rendering
# ----------------------------------------------------------------------------------------------------------


def evaluate_predictions(runs: dict, population: bool = False) -> dict[str, dict]:
    """All predictions, keyed P0..P8 (P1 and P5 split), each with numbers and a verdict.  ``population``: the runs
    use identity-free population channels (P8 is then tested against chance only, see :func:`_p8`)."""
    p1a, p1b = _p1(runs)
    preds = {"P0": _p0(runs), "P1a": p1a, "P1b": p1b, "P2": _p2(runs), "P3": _p3(runs), "P4": _p4(runs),
             "P5a": _p5a(runs), "P5b": _p5b(runs), "P6": _p6(runs), "P7": _p7(runs), "P8": _p8(runs, population=population)}
    for k, v in preds.items():
        assert v["verdict"] in VERDICTS, (k, v["verdict"])
    return preds


def _comparison_md(c: dict, label_a: str | None = None, label_b: str | None = None) -> str:
    if not c or c.get("verdict") == "not run":
        return f"_not run_ (missing: {', '.join((c or {}).get('missing') or ['required run'])})\n"
    la, lb = label_a or c.get("arm_a", "A"), label_b or c.get("arm_b", "B")
    rp = (c.get("replication") or {}).get("per_session") or {}
    has_oracle = any("class_oracle" in r for r in c.get("table", []))
    rows = []
    for r in c.get("table", []):
        row = [r["session"], _fmt(r["a"]), _fmt(r["b"]), _fmt(r["diff"])]
        if has_oracle:
            row += [_fmt(r.get("persistence")), _fmt(r.get("class_oracle"))]
        if rp:
            v = rp.get(r["session"])
            row.append(f"{_fmt_ci(v['ci'])} {'yes' if v['replicates'] else 'no'}" if v else "n/a")
        rows.append(row)
    headers = ["session", la, lb, "difference (A - B)"] + (["persistence", "class-conditional oracle"] if has_oracle else []) \
        + (["trial-bootstrap CI (replicates?)"] if rp else [])
    txt = _md_table(headers, rows)
    if c.get("excluded_sessions"):
        txt += f"\nexcluded (empty criteria set): {', '.join(c['excluded_sessions'])}\n"
    if c.get("replication"):
        txt += f"\nreplicates in {c['replication']['n_replicating']}/{c['replication']['n_sessions']} sessions (trial bootstrap, metric `{c['replication']['metric']}`)\n"
    txt += (f"\nmetric: `{c.get('metric')}` | mean A = {_fmt(c.get('mean_a'))}, mean B = {_fmt(c.get('mean_b'))}, "
            f"mean difference = {_fmt(c.get('mean_diff'))} | n_sessions = {c.get('n_sessions')} | n_seeds = {c.get('n_seeds')} "
            f"(A {c.get('n_seeds_a')}, B {c.get('n_seeds_b')}) | Wilcoxon p = {_fmt_p(c.get('p'))} "
            f"| bootstrap 95% CI = {_fmt_ci(c.get('ci'))} | direction `{c.get('direction')}`\n")
    return txt


def _verdict_sentence(pid: str, v: str, comp: dict | None = None) -> str:
    base = {"supported": f"**Verdict {pid}: supported.**", "inconclusive": f"**Verdict {pid}: inconclusive** (test possible but the rule was not met).",
            "not testable": f"**Verdict {pid}: not testable** (fewer than {MIN_SESSIONS} sessions or too few trials/cells).",
            "not applicable": f"**Verdict {pid}: not applicable** (population representation: the prediction is about neurons, the channels are not neurons).",
            "not run": f"**Verdict {pid}: not run** (a required run or key is missing)."}[v]
    if v == "not applicable":
        return base
    if comp and comp.get("verdict") not in (None, "not run") and comp.get("n_sessions"):
        base += (f" Mean paired difference {_fmt(comp.get('mean_diff'))}, 95% CI {_fmt_ci(comp.get('ci'))}, "
                 f"Wilcoxon p = {_fmt_p(comp.get('p'))}, n_sessions = {comp.get('n_sessions')}, n_seeds = {comp.get('n_seeds')}.")
    return base + "\n"


def render_markdown(report: dict) -> str:
    h, preds, sel = report["header"], report["predictions"], report["selection"]
    L: list[str] = ["# DelayCAST claims report\n", f"generated from `{report['out_dir']}`\n", "## 1. Data and runs found\n"]
    L.append(f"* sessions: **{h['n_sessions']}** (dataset A: {h['n_sessions_by_dataset']['A']}, dataset B: {h['n_sessions_by_dataset']['B']}"
             + (f", other: {len(h['other_sessions'])}" if h["other_sessions"] else "") + ")")
    L.append(f"* animals: dataset A {h['n_animals']['A']} ({', '.join(h['animals']['A']) or '-'}); dataset B {h['n_animals']['B']} ({', '.join(h['animals']['B']) or '-'})")
    L.append(f"* seeds present: {', '.join(str(s) for s in h['seeds']) or 'none'}")
    if report.get("representation") == "population":
        L.append("* **representation: population** - every region enters as identity-free rate-quantile channels (summed counts of the "
                 "units active in the trial, ranked by delay-epoch count); the Data2 export supports this, so both datasets are in the "
                 "corpus, and the unit-level predictions P1, P2, P4, P6 are marked not applicable")
    tpc = h["test_trials_per_class"]
    L.append("* test trials per class" + (f" ({h['test_trials_source']}, first seed)" if h["test_trials_source"] else "") + ": "
             + ", ".join(f"{c} = {tpc.get(c) if tpc.get(c) is not None else 'n/a'}" for c in CLASSES) + "\n")
    L.append(_md_table(["run", "seeds", "n_sessions", "balanced acc.", "bal. acc. L/R", "chance p95", "n test trials"],
                       [[r["run"], ", ".join(str(s) for s in r["seeds"]), r["n_sessions"], _fmt(r["balanced_accuracy"]), _fmt(r["balanced_accuracy_lr"]),
                         _fmt(r["chance_p95"]), _fmt(r["n_test_trials"], 0)] for r in h["runs"]]))
    keff = h.get("k_eff_per_session") or {}
    if keff:
        L.append("\nunits used by the criteria run (K_eff per region, mean over seeds):\n")
        L.append(_md_table(["session"] + list(REGIONS) + ["total"],
                           [[s] + [_fmt(v.get(r), 1) for r in REGIONS] + [_fmt(sum(v.get(r, 0.0) for r in REGIONS), 1)] for s, v in sorted(keff.items())]))
    if h.get("empty_criteria_sessions"):
        L.append(f"\n**Sessions with an empty criteria set** (no unit passed the stability rule; excluded from every criteria-arm "
                 f"comparison below, listed here instead): {', '.join(h['empty_criteria_sessions'])}\n")
    oc = report.get("outcome") or {}
    if oc.get("available"):
        L.append("\nLeft/Right accuracy of the criteria run by audited behavioural outcome (Data sessions matched to a Data2 log by "
                 "trial numbers; hit = licked the instructed side, miss = licked the other side; pooled over seeds):\n")
        rows = [[s, v["matched_log"], v["hit"]["n"], _fmt(v["hit"]["accuracy"]), v["miss"]["n"], _fmt(v["miss"]["accuracy"])] for s, v in oc["sessions"].items()]
        pl = oc.get("pooled", {})
        rows.append(["**all**", "", pl.get("hit", {}).get("n"), _fmt(pl.get("hit", {}).get("accuracy")), pl.get("miss", {}).get("n"), _fmt(pl.get("miss", {}).get("accuracy"))])
        L.append(_md_table(["session", "Data2 log", "n hit", "accuracy (hit)", "n miss", "accuracy (miss)"], rows))
        L.append("")
    L.append("\n## 2. The claim and the verdicts\n")
    L.append(f"> {report['claim']}\n")
    L.append(f"Statistical rule: per-session paired differences (mean over seeds per session first), Wilcoxon signed-rank across sessions "
             f"(exact for n <= 25), {N_BOOT}-resample session bootstrap CI of the mean difference. 'supported' needs p < {ALPHA} AND a CI "
             f"excluding 0 in the predicted direction (non-inferiority claims: CI lower bound > -{NOT_LOWER_MARGIN}); 'inconclusive' when "
             f">= {MIN_SESSIONS} sessions but the rule fails; 'not testable' when < {MIN_SESSIONS} sessions; 'not run' when a run is missing. "
             f"Where both arms have per-trial predictions, 'replicates in k/n sessions' counts the sessions whose own trial-bootstrap "
             f"({N_BOOT} resamples of matched test trials, averaged over seeds) CI excludes 0 in the predicted direction - supplementary "
             f"evidence that never changes a verdict. Sessions with an empty criteria set are excluded from criteria-arm comparisons.\n")
    L.append(_md_table(["prediction", "comparator", "statistic", "failure condition", "result numbers", "verdict"],
                       [[f"{pid}: {p['title']}", p["comparator"], p["statistic"], p["failure_condition"], p["result"], f"**{p['verdict']}**"]
                        for pid, p in preds.items()]))
    L.append("\n## 3. Predictions in detail\n")
    for pid, p in preds.items():
        L.append(f"### {pid}. {p['title']}\n")
        L.append(f"comparator: {p['comparator']}; statistic: {p['statistic']}; fails if: {p['failure_condition']}\n")
        if pid == "P0":
            L.append(_md_table(["seed", "balanced accuracy", "chance mean", "chance p95", "pass"],
                               [[r["seed"], _fmt(r["balanced_accuracy"]), _fmt(r["chance_mean"]), _fmt(r["chance_p95"]), r["pass"]] for r in p.get("per_seed", [])]))
        elif pid == "P1b" and p.get("ablations"):
            L.append(_comparison_md(p.get("comparison")))
            for k, c in p["ablations"].items():
                L.append(f"\n**within-pipeline ablation: criteria vs {k}** (reported, no verdict; {k} = "
                         + ("the linear count read-out alone" if k.endswith("linonly") else "the backbone classifier alone") + ")\n\n"
                         + _comparison_md(c, "criteria (full)", k))
        elif pid == "P2":
            for k, c in p.get("comparisons", {}).items():
                L.append(f"**criteria vs {k}**\n\n" + _comparison_md(c, "criteria", k))
                L.append(f"sub-verdict: {c['verdict']}\n")
        elif pid == "P3":
            L.append("**(i) context sufficiency index (criteria run)**\n\n" + _md_table(
                ["seed", "tau95 (ms)", "tau95 CI (ms)", f"CI upper <= {TAU95_MAX_MS:.0f}"],
                [[r["seed"], _fmt(r["tau95_ms"], 0), _fmt_ci(r["tau95_ci_ms"], 0), r["pass"]] for r in p.get("csi_per_seed", [])]))
            L.append("\n**(ii) temporal occlusion: last window vs the worst earlier window** (pooled worst window per seed: "
                     + ", ".join(f"seed{r['seed']} end={_fmt(r['window_end_ms'], 0)} ms{' (last)' if r['is_last'] else ''}" for r in p.get("argmin_window_per_seed", [])) + ")\n")
            L.append(_comparison_md(p.get("comparison"), "delta last window", "worst delta earlier window"))
            L.append("\n**(iii) linear decoder tau95 (reported)**: " + ", ".join(f"seed{r['seed']} = {_fmt(r['tau95_linear_ms'], 0)} ms" for r in p.get("linear_tau95_per_seed", [])) + "\n")
        elif pid == "P4":
            L.append(_comparison_md(p.get("comparison"), "model dev. expl.", "null (0)"))
            ce = p.get("coupling_enrichment") or {}
            if ce.get("table"):
                L.append("\n**Coupling (C) enrichment among selected units (reported)**\n\n" + _md_table(
                    ["session", "frac C selected", "frac C eligible-unselected", "difference", "Fisher p (seed 0)"],
                    [[r["session"], _fmt(r["frac_selected"]), _fmt(r["frac_unselected"]), _fmt(r["diff"]), _fmt_p(r["fisher_p"])] for r in ce["table"]]))
                L.append(f"\nsign test across sessions: {ce['n_positive']}/{ce['n_sessions']} positive, p = {_fmt_p(ce['sign_test_p'])}\n")
        elif pid == "P5a":
            L.append(_comparison_md(p.get("comparison"), "mean delta ALM removed", "mean delta STR removed"))
        elif pid == "P5b":
            L.append(f"Ignore recall = {_fmt(p.get('ignore_recall'))}, Wilson 95% CI = {_fmt_ci(p.get('wilson_ci'))}, n_Ignore = {p.get('n_ignore')}, "
                     f"n_seeds = {p.get('n_seeds')}; trial-index baseline balanced accuracy = {_fmt(p.get('trial_index_balanced_accuracy'))} vs chance p95 = "
                     f"{_fmt(p.get('chance_p95'))}" + (" -> **confounded** (drift alone beats chance)" if p.get("confounded") else "") + "\n")
        elif pid == "P6":
            for key, rows in (p.get("per_seed") or {}).items():
                L.append(f"**{key}** (sub-verdict {p.get('parts', {}).get(key)})\n\n" + _md_table(
                    ["seed", "mean rho", "median rho", "n_cells", "n_positive", "sign-test p", "pass"],
                    [[r["seed"], _fmt(r["mean_rho"]), _fmt(r["median_rho"]), r["n_cells"], r["n_positive"], _fmt_p(r["sign_test_p"]), r["pass"]] for r in rows]))
        elif pid == "P8":
            L.append(_md_table(["seed", "balanced accuracy", "chance p95", "pass"],
                               [[r["seed"], _fmt(r["balanced_accuracy"]), _fmt(r["chance_p95"]), r["pass"]] for r in p.get("per_seed", [])]))
            L.append("\n**vs cross_dataset/random**\n\n" + _comparison_md(p.get("comparison")))
        else:
            L.append(_comparison_md(p.get("comparison")))
        L.append(_verdict_sentence(pid, p["verdict"], p.get("comparison")))
    L.append("\n## 4. Selection summary\n")
    if sel.get("summary_csv"):
        L.append(f"from `{sel['summary_csv']}` ({sel.get('n_sessions')} sessions, all trials): units recorded {sel.get('n_units', 'n/a')}, "
                 f"pass floor {sel.get('n_floor', 'n/a')}, eligible {sel.get('n_eligible', 'n/a')}, selected {sel.get('n_selected', 'n/a')}\n")
        L.append(_md_table(["region", "recorded", "selected"], [[r, v["recorded"], v["selected"]] for r, v in sel["per_region"].items()]))
        ms, nl = sel.get("median_stability_selected", {}), sel.get("null_median_stability_max", {})
        L.append(f"\n* median stability of selected units: {_fmt(ms.get('median'))} (median over sessions) vs null median-stability max {_fmt(nl.get('max'))}")
        nn = sel.get("null_n_selected_mean")
        if nn:
            L.append(f"* units that would be selected with permuted labels (empirical false-selection estimate, mean over permutations, "
                     f"summed over regions): mean {_fmt(nn.get('mean'), 1)} per session, max {_fmt(sel.get('null_n_selected_max', {}).get('max'), 1)} "
                     f"- vs {sel.get('n_selected', 'n/a')} selected in total")
        L.append(f"* median onset of selected units: {_fmt(sel.get('median_onset_ms_selected', {}).get('median'), 0)} ms; fraction sustained to go: "
                 f"{_fmt(sel.get('frac_sustained_to_go_selected', {}).get('mean'))}")
        L.append("* criterion fractions among units (mean over sessions): " + ", ".join(
            f"{k[5:]} {_fmt(sel[k]['mean'])}" for k in ("frac_selectivity", "frac_coupling", "frac_spectral", "frac_ramp", "frac_locus", "frac_ignore") if k in sel))
    else:
        L.append("_selection/summary.csv not found_")
    if sel.get("funnel_per_region"):
        L.append(f"\ntrain-split funnel of the criteria run (per region, summed over sessions, mean over {sel.get('n_seeds_funnel')} seed(s); K = {_fmt(sel.get('K'), 0)}):\n")
        cols = ["recorded", "pass_floor", "eligible", "stable", "selected"]
        L.append(_md_table(["region"] + cols, [[r] + [_fmt(v.get(c), 1) for c in cols] for r, v in sel["funnel_per_region"].items()]))
        efb = sel.get("expected_false_selections_bound") or {}
        L.append(f"\n* Meinshausen-Buehlmann false-selection bound: mean per region-session {_fmt(efb.get('mean_per_region_session'))}, max {_fmt(efb.get('max'))} "
                 f"(informative only when K_eff << n_eligible; the label-permutation estimate above is the empirical one)")
    if sel.get("phi"):
        L.append("\n* phi coefficients between criteria (mean over sessions): " + ", ".join(f"{k} = {_fmt(v['mean'])}" for k, v in sel["phi"].items()))
    w = sel.get("w_independence") or {}
    L.append(f"\n**W-independence**: phi_SW mean = {_fmt(w.get('phi_SW_mean'))} (threshold {w.get('threshold')})"
             + (f"; sessions above: {', '.join(w['sessions_above'])}" if w.get("sessions_above") else "")
             + f" -> {'**FLAG: ' + w['message'] + '**' if w.get('flag') else w.get('message', '')}\n")
    return "\n".join(L)


def build_report(cfg, out_dir: Path) -> dict:
    """Everything that goes into report.json (numbers + verdicts); ``write_report`` renders it."""
    out_dir = Path(out_dir)
    runs = load_results(out_dir)
    summary = None
    p = out_dir / "selection" / "summary.csv"
    if p.is_file():
        try:
            summary = pd.read_csv(p)
        except Exception:  # pragma: no cover
            summary = None
    rep_mode = next((str(r.get("representation", "units")) for arm in runs.values() for r in arm if r.get("representation")), "units")
    report = {"out_dir": str(out_dir), "claim": CLAIM, "header": _header(runs, summary),
              "predictions": evaluate_predictions(runs, population=(rep_mode == "population")),
              "selection": _selection_summary(out_dir, runs), "outcome": outcome_diagnostic(cfg, runs),
              "rule": {"alpha": ALPHA, "not_lower_margin": NOT_LOWER_MARGIN, "n_bootstrap": N_BOOT, "min_sessions": MIN_SESSIONS}}
    report["representation"] = rep_mode
    if rep_mode == "population":
        # Channels are rate-quantile groups of the units active in the trial, not neurons: every prediction about
        # *which neurons* (selection vs controls, sparsity, single-unit coupling, per-neuron importance) is
        # meaningless here and is marked as such; the temporal, regional, spectral and transfer predictions stand.
        for pid in ("P1a", "P1b", "P2", "P4", "P6"):
            if pid in report["predictions"]:
                report["predictions"][pid]["verdict"] = "not applicable"
                report["predictions"][pid]["result"] = "population representation (identity-free channels): unit-level prediction not applicable"
    report["verdicts"] = {k: v["verdict"] for k, v in report["predictions"].items()}
    return report


def write_report(cfg, out_dir: Path) -> Path:
    """Write ``REPORT.md`` and ``report.json`` under ``out_dir``; returns the path of REPORT.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(cfg, out_dir)
    md = out_dir / "REPORT.md"
    md.write_text(render_markdown(report), encoding="utf-8")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, indent=1)
    log.info("wrote %s and %s", md, out_dir / "report.json")
    return md
