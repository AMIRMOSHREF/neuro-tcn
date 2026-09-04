"""Evaluation of a trained run.

Everything here is inference-only and works on the *test* trials of every session (held-out trials of a
within-session split, or the non-adapted trials of held-out sessions):

* classification metrics (pooled 3-class, Left-vs-Right, per session) with a session-stratified trial
  bootstrap CI and a within-session label-permutation chance level;
* forecasting: Poisson deviance explained of the model and of a persistence forecast vs the training-PSTH null;
* **context sweep** (only the last tau ms visible) and the **context sufficiency index** tau95 with a
  bootstrap + isotonic-regression CI (balanced-accuracy and log-loss versions);
* **temporal occlusion map**: one window at a time is replaced by the same window of another test trial of
  the same session (permutation occlusion; keeps marginal statistics, destroys trial information), reported
  for the classifier and for the forecaster (backbone-only variant with the persistence input held fixed);
* **region ablation**: permutation occlusion of a whole region and in-distribution region drop
  (the model was trained with region dropout);
* **neuron importance**: permutation occlusion of every selected neuron (one forward pass per neuron and session),
  delta cross-entropy / balanced accuracy / forecast deviance of the *other* neurons, joined with the
  selection statistics and the learned gates -> ``neuron_importance.csv`` and agreement statistics;
* linear baselines: tuned multinomial logistic regression on all units (delay mean + late-delay mean per
  unit), PCA-50, L1 (sparse set overlap with the criteria set), the train-selected K units, per-area
  decoders, and a trial-index drift control; plus the model-free linear context sweep -> tau95_linear.

All quantities that ``report`` compares across arms are also stored **per session**, because the session is
the unit of replication for every claim.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASSES, REGIONS
from .data.dataset import SessionTensors, session_arrays
from .models.delaycast_net import poisson_deviance
from .train import balanced_accuracy

log = logging.getLogger(__name__)
LEFT, RIGHT, IGNORE = CLASSES.index("Left"), CLASSES.index("Right"), CLASSES.index("Ignore")


# ----------------------------------------------------------------------------- metrics
def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, None)))) if len(y) else float("nan")


def _bacc_lr(y: np.ndarray, yhat: np.ndarray) -> float:
    m = (y == LEFT) | (y == RIGHT)
    return balanced_accuracy(y[m], yhat[m], labels=[LEFT, RIGHT]) if m.any() else float("nan")


def metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    yhat = logits.argmax(1)
    prob = _softmax(logits)
    present = [int(c) for c in range(len(CLASSES)) if (y == c).any()]
    return {
        "accuracy": float((y == yhat).mean()) if len(y) else float("nan"),
        "balanced_accuracy": balanced_accuracy(y, yhat, labels=range(len(CLASSES))),
        "balanced_accuracy_lr": _bacc_lr(y, yhat),
        "macro_f1": float(f1_score(y, yhat, average="macro", labels=list(range(len(CLASSES))), zero_division=0)) if len(y) else float("nan"),
        "log_loss": _log_loss(y, prob),
        "n": int(len(y)),
        "n_per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "n_classes_present": len(present),
        "recall": {c: (float((yhat[y == i] == i).mean()) if (y == i).any() else None) for i, c in enumerate(CLASSES)},
    }


def per_session_metrics(y, logits, sessions) -> list[dict]:
    out = []
    for s in sorted(np.unique(sessions)):
        m = sessions == s
        out.append({"session": str(s), **metrics(y[m], logits[m])})
    return out


def bootstrap_ci(y: np.ndarray, logits: np.ndarray, sessions: np.ndarray, n_boot: int, rng: np.random.Generator,
                 stat=lambda y, l: balanced_accuracy(y, l.argmax(1), labels=range(3))) -> list[float]:
    """Trial bootstrap stratified by session x class."""
    groups = [np.flatnonzero((sessions == s) & (y == c)) for s in np.unique(sessions) for c in np.unique(y)]
    groups = [g for g in groups if len(g)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(stat(y[idx], logits[idx]))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def chance_level(y: np.ndarray, yhat: np.ndarray, sessions: np.ndarray, n_shuffles: int, rng: np.random.Generator) -> dict:
    """Within-session label permutations with the predictions fixed."""
    vals = []
    for _ in range(n_shuffles):
        yp = y.copy()
        for s in np.unique(sessions):
            m = np.flatnonzero(sessions == s)
            yp[m] = rng.permutation(y[m])
        vals.append(balanced_accuracy(yp, yhat, labels=range(3)))
    vals = np.asarray(vals)
    return {"mean": float(np.nanmean(vals)), "p95": float(np.nanpercentile(vals, 95)), "p99": float(np.nanpercentile(vals, 99)),
            "analytic": 1.0 / len(CLASSES), "n_shuffles": int(n_shuffles), "scheme": "within_session_label_permutation"}


# ----------------------------------------------------------------------------- forward helpers
@torch.no_grad()
def _forward_chunks(model, x, session, mask, device, pad_mask=None, drop_region=None, late_log_override=None, chunk: int = 256):
    """Run the model on arbitrarily many trials of one session (chunked to bound attention memory)."""
    n = x[REGIONS[0]].shape[0]
    logits, forecast, tattn, rattn = [], {r: [] for r in REGIONS}, {r: [] for r in REGIONS}, []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        out = model({r: x[r][sl].to(device) for r in REGIONS}, session,
                    None if pad_mask is None else pad_mask[sl].to(device),
                    neuron_mask={r: mask[r][sl].to(device) for r in REGIONS}, drop_region=drop_region,
                    late_log_override=None if late_log_override is None else {r: v[sl].to(device) for r, v in late_log_override.items()})
        logits.append(out.logits.cpu())
        rattn.append(out.region_attn.cpu())
        for r in REGIONS:
            forecast[r].append(out.forecast_log_rate[r].cpu())
            tattn[r].append(out.temporal_attn[r].cpu())
    return {"logits": torch.cat(logits), "forecast": {r: torch.cat(v) for r, v in forecast.items()},
            "tattn": {r: torch.cat(v) for r, v in tattn.items()}, "rattn": torch.cat(rattn)}


def _permute_slice(x: dict[str, torch.Tensor], rng: np.random.Generator, labels: np.ndarray, region: str | None = None,
                   bins: slice | None = None, neuron: int | None = None, stratified: bool = False) -> dict[str, torch.Tensor]:
    """Permutation occlusion: replace a slice (region / window / neuron) with the same slice of another trial.

    Donor trials are a derangement of the trial index (class-stratified when requested and possible), so the
    marginal statistics of the slice are preserved while its trial-specific information is destroyed.
    """
    n = x[REGIONS[0]].shape[0]
    donor = np.arange(n)
    if n > 1:
        if stratified:
            for c in np.unique(labels):
                idx = np.flatnonzero(labels == c)
                if len(idx) > 1:
                    donor[idx] = idx[_derangement(len(idx), rng)]
        else:
            donor = _derangement(n, rng)
    donor_t = torch.as_tensor(donor)
    out = {}
    for r in REGIONS:
        xr = x[r]
        if region is not None and r != region:
            out[r] = xr
            continue
        xr = xr.clone()
        src = x[r][donor_t]
        if neuron is not None:
            xr[:, neuron, :] = src[:, neuron, :]
        elif bins is not None:
            xr[:, :, bins] = src[:, :, bins]
        else:
            xr = src
        out[r] = xr
    return out


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    for _ in range(100):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


# ----------------------------------------------------------------------------- test-set assembly
def test_sets(tensors: list[SessionTensors], splits: dict) -> dict[str, dict]:
    return {t.session: session_arrays(t, splits[t.session]["test"]) for t in tensors if len(splits[t.session]["test"])}


def train_null(tensors: list[SessionTensors], splits: dict) -> dict:
    """Per-session, per-region mean response PSTH of the training (or adaptation) trials: (K, T_tgt)."""
    null = {}
    for t in tensors:
        sp = splits[t.session]
        idx = np.r_[sp.get("train", np.zeros(0, int)), sp.get("adapt", np.zeros(0, int))].astype(int)
        if not len(idx):
            idx = np.arange(t.n_trials)
        null[t.session] = {r: torch.from_numpy(t.y[r][idx].mean(axis=0)) for r in REGIONS}
    return null


def train_null_by_class(tensors: list[SessionTensors], splits: dict) -> dict:
    """Per-session, per-region, per-class mean response PSTH of the training (or adaptation) trials: (n_classes, K, T_tgt).

    The *class-conditional oracle*: the best forecast that knows the true class but nothing trial-specific.  A
    forecaster that beats it uses trial-by-trial delay information beyond the class identity.  Classes absent from
    the training trials fall back to the class-free PSTH."""
    null = {}
    for t in tensors:
        sp = splits[t.session]
        idx = np.r_[sp.get("train", np.zeros(0, int)), sp.get("adapt", np.zeros(0, int))].astype(int)
        if not len(idx):
            idx = np.arange(t.n_trials)
        lab = t.labels[idx]
        per = {}
        for r in REGIONS:
            base = t.y[r][idx].mean(axis=0)
            stack = np.stack([t.y[r][idx[lab == c]].mean(axis=0) if (lab == c).sum() else base for c in range(len(CLASSES))])
            per[r] = torch.from_numpy(stack.astype(np.float32))
        null[t.session] = per
    return null


def _dev_expl(dev: float, dev0: float) -> float:
    """1 - D(model)/D(null); NaN when the null deviance is 0 (no real neurons / no spikes), never a fake 1.0."""
    return float(1 - dev / dev0) if dev0 > 1e-8 else float("nan")


def _deviance_terms(model, fc_log_rate, y, mask, null_mu, x, exclude_neuron: int | None = None,
                    class_mu: dict | None = None, labels=None) -> dict[str, tuple]:
    """(model, persistence, null[, class-conditional oracle]) summed deviance per region; optionally excluding one
    neuron slot.  The oracle term is appended only when ``class_mu`` (from :func:`train_null_by_class`) and the
    true ``labels`` are given."""
    out = {}
    for r in REGIONS:
        m = mask[r].clone()
        if exclude_neuron is not None:
            m[:, exclude_neuron] = False
        late = x[r][:, :, -model.late_bins:].mean(-1) * model.count_scale
        persist_mu = late[:, :, None].expand_as(y[r]).clamp(min=1e-3)
        terms = [poisson_deviance(torch.exp(fc_log_rate[r]), y[r], m).item(),
                 poisson_deviance(persist_mu, y[r], m).item(),
                 poisson_deviance(null_mu[r][None].expand_as(y[r]), y[r], m).item()]
        if class_mu is not None and labels is not None:
            lab = torch.as_tensor(np.asarray(labels), dtype=torch.long)
            terms.append(poisson_deviance(class_mu[r][lab], y[r], m).item())
        out[r] = tuple(terms)
    return out


# ----------------------------------------------------------------------------- linear baselines
def _unit_features(cache, cfg, region_filter=None, unit_index: dict | None = None) -> np.ndarray:
    """Delay mean + late-delay mean rate per unit (2 features per unit); optionally a subset of units."""
    late_bins = max(1, int(round(float(cfg.selection.late_delay_ms) / float(cache.bin_ms))))
    feats = []
    for r in REGIONS:
        if region_filter is not None and not any(r.startswith(a) for a in region_filter):
            continue
        X = cache.context[r]
        if unit_index is not None:
            ii = np.asarray(unit_index[r], dtype=int)
            ii = ii[ii >= 0]
            X = X[:, ii]
        if X.shape[1] == 0:
            continue
        feats.append(X.mean(axis=2))
        feats.append(X[:, :, -late_bins:].mean(axis=2))
    return np.concatenate(feats, axis=1).astype(np.float32) if feats else np.zeros((cache.n_trials, 0), np.float32)


def _fit_logreg(Xtr, ytr, kind: str = "cv", C: float | None = None):
    n_min = np.bincount(ytr).min() if len(ytr) else 0
    cv = min(5, max(2, int(n_min))) if n_min >= 2 else None
    if kind == "cv" and cv is not None and Xtr.shape[1] > 0:
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 6), cv=StratifiedKFold(cv, shuffle=True, random_state=0),
                                                                    class_weight="balanced", max_iter=500, n_jobs=1))
    elif kind == "l1":
        clf = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", C=C or 0.1,
                                                                                      class_weight="balanced", max_iter=500)))
    elif kind == "pca":
        n_comp = int(min(50, Xtr.shape[0] - 1, Xtr.shape[1]))
        clf = make_pipeline(StandardScaler(), PCA(n_components=max(n_comp, 1), random_state=0),
                            LogisticRegression(C=C or 1.0, class_weight="balanced", max_iter=500))
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C or 1.0, class_weight="balanced", max_iter=500))
    clf.fit(Xtr, ytr)
    return clf


def _best_c(clf) -> float:
    est = clf[-1]
    if hasattr(est, "C_"):
        return float(np.mean(est.C_))
    return float(getattr(est, "C", 1.0))


def linear_baselines(tensors, splits, caches, cfg, unit_index: dict, rng: np.random.Generator) -> tuple[list[dict], list[dict], dict]:
    """Per-session fits (unit identities differ between sessions), pooled + per-session metrics.

    Returns (baseline rows, linear context sweep rows, extras such as L1-vs-criteria overlap and tuned C)."""
    rows_by_model: dict[str, dict] = {}
    sweep_ms = [int(v) for v in cfg.evaluate.context_sweep_ms]
    sweep_pred: dict[int, list] = {ms: [] for ms in sweep_ms}
    sweep_y: dict[int, list] = {ms: [] for ms in sweep_ms}
    extras = {"l1_overlap": [], "tuned_C": {}}
    specs = [("logreg_all_units", None, None, "cv"), ("logreg_pca50_all_units", None, None, "pca"), ("logreg_l1_all_units", None, None, "l1"),
             ("logreg_selected_units", "sel", None, "cv"), ("logreg_selected_ALM", "sel", ("ALM",), "cv"),
             ("logreg_selected_STR", "sel", ("STR",), "cv"), ("logreg_trial_index", "drift", None, "plain")]
    for t in tensors:
        sp = splits[t.session]
        te = sp["test"]
        tr = np.r_[sp["train"], sp.get("adapt", np.zeros(0, int))].astype(int)
        if not len(te) or not len(tr) or len(np.unique(t.labels[tr])) < 2:
            continue
        cache = caches[t.session]
        y = t.labels
        feats_all = _unit_features(cache, cfg)
        feats_sel = _unit_features(cache, cfg, unit_index=unit_index[t.session])
        trial_index = ((np.arange(cache.n_trials) / max(cache.n_trials - 1, 1))[:, None]).astype(np.float32)
        best_c = None
        for name, source, area, kind in specs:
            if source is None:
                feats = feats_all
            elif source == "sel":
                feats = _unit_features(cache, cfg, region_filter=area, unit_index=unit_index[t.session])
            else:
                feats = trial_index
            if feats.shape[1] == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")      # sklearn penalty/l1_ratio deprecation chatter, convergence notes
                    clf = _fit_logreg(feats[tr], y[tr], kind=kind, C=best_c if kind == "pca" else None)
            except Exception as e:  # pragma: no cover
                log.warning("baseline %s failed for %s: %s", name, t.session, e)
                continue
            if name == "logreg_all_units":
                best_c = _best_c(clf)
                extras["tuned_C"][t.session] = best_c
            prob = clf.predict_proba(feats[te])
            logits = np.log(np.clip(prob, 1e-9, None))
            full = np.full((len(te), len(CLASSES)), -20.0)
            full[:, clf.classes_] = logits
            row = rows_by_model.setdefault(name, {"model": name, "logits": [], "y": [], "sessions": []})
            row["logits"].append(full)
            row["y"].append(y[te])
            row["sessions"].append(np.full(len(te), t.session))
            if name == "logreg_l1_all_units":
                est = clf[-1]
                coef = np.concatenate([e.coef_ for e in est.estimators_], axis=0) if hasattr(est, "estimators_") else est.coef_
                nz = np.flatnonzero(np.abs(coef).sum(axis=0) > 0)
                # map feature index -> (region, unit): features are [mean, late] blocks per region in REGIONS order
                offset = 0
                for r in REGIONS:
                    n_u = cache.context[r].shape[1]
                    l1_units = set()
                    for blk in range(2):
                        sel = nz[(nz >= offset) & (nz < offset + n_u)] - offset
                        l1_units |= set(sel.tolist())
                        offset += n_u
                    crit = set(int(u) for u in unit_index[t.session][r] if u >= 0)
                    union = len(l1_units | crit)
                    extras["l1_overlap"].append({"session": t.session, "region": r, "n_l1": len(l1_units), "n_criteria": len(crit),
                                                 "jaccard": (len(l1_units & crit) / union) if union else float("nan")})
        # model-free context sweep: last tau ms of all units, tuned C from the full-context fit
        if bool(cfg.evaluate.get_path("linear_sweep", True)):
            for ms in sweep_ms:
                keep = max(1, int(round(ms / float(cache.bin_ms))))
                feats = np.concatenate([cache.context[r][:, :, -keep:].mean(axis=2) for r in REGIONS if cache.context[r].shape[1]], axis=1).astype(np.float32)
                try:
                    clf = _fit_logreg(feats[tr], y[tr], kind="plain", C=best_c or 1.0)
                except Exception:  # pragma: no cover
                    continue
                prob = clf.predict_proba(feats[te])
                full = np.full((len(te), len(CLASSES)), -20.0)
                full[:, clf.classes_] = np.log(np.clip(prob, 1e-9, None))
                sweep_pred[ms].append(full)
                sweep_y[ms].append((y[te], np.full(len(te), t.session)))
    rows = []
    for name, d in rows_by_model.items():
        logits = np.concatenate(d["logits"])
        y = np.concatenate(d["y"])
        sessions = np.concatenate(d["sessions"])
        rows.append({"model": name, **metrics(y, logits), "per_session": per_session_metrics(y, logits, sessions)})
    sweep = []
    for ms in sweep_ms:
        if sweep_pred[ms]:
            logits = np.concatenate(sweep_pred[ms])
            y = np.concatenate([v[0] for v in sweep_y[ms]])
            sessions = np.concatenate([v[1] for v in sweep_y[ms]])
            sweep.append({"context_ms": ms, **metrics(y, logits), "per_session": per_session_metrics(y, logits, sessions)})
    return rows, sweep, extras


# ----------------------------------------------------------------------------- context sufficiency index
def csi_from_sweep(sweep_logits: dict[int, np.ndarray], y: np.ndarray, sessions: np.ndarray, frac: float, n_boot: int,
                   rng: np.random.Generator) -> dict:
    """tau95 with a trial-bootstrap + isotonic-regression CI (balanced accuracy up, log-loss down in tau)."""
    taus = np.array(sorted(sweep_logits))
    if len(taus) < 2:
        return {}
    groups = [np.flatnonzero((sessions == s) & (y == c)) for s in np.unique(sessions) for c in np.unique(y)]
    groups = [g for g in groups if len(g)]
    t95_b, t95_ll = [], []
    full = taus.max()

    def _tau95(idx):
        acc = np.array([balanced_accuracy(y[idx], sweep_logits[t][idx].argmax(1), labels=range(3)) for t in taus])
        ll = np.array([_log_loss(y[idx], _softmax(sweep_logits[t][idx])) for t in taus])
        acc_f = IsotonicRegression(increasing=True).fit_transform(taus, np.nan_to_num(acc, nan=np.nanmean(acc)))
        ll_f = IsotonicRegression(increasing=False).fit_transform(taus, ll)
        ok = acc_f >= frac * acc_f[-1]
        ok_ll = ll_f <= (2 - frac) * ll_f[-1]
        return (float(taus[np.argmax(ok)]) if ok.any() else float(full)), (float(taus[np.argmax(ok_ll)]) if ok_ll.any() else float(full))

    t_point, t_point_ll = _tau95(np.arange(len(y)))
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        a, b = _tau95(idx)
        t95_b.append(a)
        t95_ll.append(b)
    pct = lambda v, q: float(np.percentile(v, q, method="closest_observation"))   # CI bounds stay on the sweep grid
    return {"tau95_ms": t_point, "tau95_median_ms": float(np.median(t95_b)), "tau95_ci_ms": [pct(t95_b, 2.5), pct(t95_b, 97.5)],
            "tau95_logloss_ms": t_point_ll, "tau95_logloss_ci_ms": [pct(t95_ll, 2.5), pct(t95_ll, 97.5)],
            "fraction": frac, "n_bootstrap": int(n_boot)}


# ----------------------------------------------------------------------------- main entry point
def evaluate_run(run: dict, cfg, caches: dict | None = None) -> dict:
    model, tensors, splits, out_dir, device = run["model"], run["tensors"], run["splits"], Path(run["out_dir"]), run["device"]
    caches = caches or run.get("caches")
    model.eval()
    rng = np.random.default_rng(int(cfg.train.seed) + 7)
    n_boot = int(cfg.evaluate.get_path("n_bootstrap", 1000))
    occl = str(cfg.evaluate.get_path("occlusion", "permute"))
    tests = test_sets(tensors, splits)
    null = train_null(tensors, splits)
    null_cls = train_null_by_class(tensors, splits)
    t_ctx, bin_ms = model.t_ctx, float(cfg.data.bin_ms)
    unit_index = {t.session: {r: t.unit_index[r] for r in REGIONS} for t in tensors}
    results: dict = {"mode": run.get("mode"), "seed": int(run.get("seed", cfg.train.seed)), "holdout": run.get("holdout", []),
                     "eval_mode": str(cfg.train.eval_mode), "negative_control": bool(run.get("negative_control", False)),
                     "spectral_branch": model.spectral_branch, "adapt_info": run.get("adapt_info", {}), "occlusion": occl,
                     "representation": str(cfg.data.get_path("representation", "units")),
                     # units actually used per session x region (K_eff): the report excludes sessions with an empty set
                     "n_selected": {t.session: {r: int(np.asarray(t.neuron_mask[r]).sum()) for r in REGIONS} for t in tensors}}
    if not tests:
        log.warning("no test trials - nothing to evaluate")
        return results

    # ---- base predictions per session (all analyses reuse the same tensors)
    base, y_all, s_all, tr_all = {}, [], [], []
    for s, b in tests.items():
        base[s] = _forward_chunks(model, b["x"], s, b["mask"], device)
        y_all.append(b["label"].numpy()); s_all.append(np.full(len(b["label"]), s)); tr_all.append(b["trial"].numpy())
    y = np.concatenate(y_all); sessions = np.concatenate(s_all); trials = np.concatenate(tr_all)
    logits = np.concatenate([base[s]["logits"].numpy() for s in tests])
    yhat = logits.argmax(1)
    results["classification"] = metrics(y, logits)
    results["classification_ci"] = {"balanced_accuracy": bootstrap_ci(y, logits, sessions, n_boot, rng),
                                    "balanced_accuracy_lr": bootstrap_ci(y, logits, sessions, n_boot, rng, stat=lambda yy, ll: _bacc_lr(yy, ll.argmax(1)))}
    results["confusion"] = confusion_matrix(y, yhat, labels=list(range(len(CLASSES)))).tolist()
    results["per_session"] = per_session_metrics(y, logits, sessions)
    results["chance"] = chance_level(y, yhat, sessions, int(cfg.evaluate.n_shuffles), rng)
    results["chance_balanced_accuracy"] = {"mean": results["chance"]["mean"], "p95": results["chance"]["p95"]}  # backward compatible
    prob = _softmax(logits)
    pd.DataFrame({"session": sessions, "trial": trials, "label": y, "pred": yhat, **{f"p_{c}": prob[:, i] for i, c in enumerate(CLASSES)}}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)

    # ---- forecasting
    fc_tot = {r: np.zeros(4) for r in REGIONS}
    fc_sess = []
    for s, b in tests.items():
        terms = _deviance_terms(model, base[s]["forecast"], b["y"], b["mask"], null[s], b["x"], class_mu=null_cls[s], labels=b["label"].numpy())
        row = {"session": s}
        for r in REGIONS:
            fc_tot[r] += np.array(terms[r])
            row[f"deviance_explained_{r}"] = _dev_expl(terms[r][0], terms[r][2])
            row[f"deviance_explained_persistence_{r}"] = _dev_expl(terms[r][1], terms[r][2])
            row[f"deviance_explained_classmean_{r}"] = _dev_expl(terms[r][3], terms[r][2])
        fc_sess.append(row)
    results["forecast"] = {**{f"deviance_explained_{r}": _dev_expl(fc_tot[r][0], fc_tot[r][2]) for r in REGIONS},
                           **{f"deviance_explained_persistence_{r}": _dev_expl(fc_tot[r][1], fc_tot[r][2]) for r in REGIONS},
                           **{f"deviance_explained_classmean_{r}": _dev_expl(fc_tot[r][3], fc_tot[r][2]) for r in REGIONS},
                           "per_session": fc_sess}

    # ---- context sweep + CSI
    sweep, sweep_logits = [], {}
    for ms in [int(v) for v in cfg.evaluate.context_sweep_ms]:
        keep = int(round(ms / bin_ms))
        if keep > t_ctx:
            continue
        lg = []
        for s, b in tests.items():
            pm = torch.zeros(len(b["label"]), t_ctx, dtype=torch.bool)
            if keep < t_ctx:
                pm[:, : t_ctx - keep] = True
            lg.append(_forward_chunks(model, b["x"], s, b["mask"], device, pad_mask=pm)["logits"].numpy())
        lg = np.concatenate(lg)
        sweep_logits[ms] = lg
        sweep.append({"context_ms": ms, **metrics(y, lg), "per_session": per_session_metrics(y, lg, sessions)})
    results["context_sweep"] = sweep
    results["csi"] = csi_from_sweep(sweep_logits, y, sessions, float(cfg.evaluate.get_path("csi_fraction", 0.95)), min(n_boot, 500), rng)

    # ---- temporal occlusion map (classifier + forecaster; backbone-only forecaster variant)
    w_bins = max(1, int(round(float(cfg.evaluate.get_path("occlusion_window_ms", 200)) / bin_ms)))
    step = max(1, int(round(float(cfg.evaluate.get_path("occlusion_step_ms", 100)) / bin_ms)))
    base_bacc = results["classification"]["balanced_accuracy"]
    base_ll = results["classification"]["log_loss"]
    tocc = []
    for start in range(0, t_ctx - w_bins + 1, step):
        sl = slice(start, start + w_bins)
        lg, d_fc, d_fc_bb, per_s = [], np.zeros(3), np.zeros(3), {}
        for s, b in tests.items():
            lab = b["label"].numpy()
            if occl == "permute":
                x_o = _permute_slice(b["x"], rng, lab, bins=sl)
                pm = None
            else:
                x_o, pm = b["x"], torch.zeros(len(lab), t_ctx, dtype=torch.bool)
                pm[:, sl] = True
            # persistence input from the un-occluded raster -> backbone-only forecast map
            late = torch.stack([(b["x"][r][:, :, -model.late_bins:].mean(-1)) for r in REGIONS])
            override = {r: torch.log(b["x"][r][:, :, -model.late_bins:].mean(-1) * model.count_scale + 0.05) for r in REGIONS}
            o = _forward_chunks(model, x_o, s, b["mask"], device, pad_mask=pm)
            o_bb = _forward_chunks(model, x_o, s, b["mask"], device, pad_mask=pm, late_log_override=override)
            lg.append(o["logits"].numpy())
            t_base = _deviance_terms(model, base[s]["forecast"], b["y"], b["mask"], null[s], b["x"])
            t_occ = _deviance_terms(model, o["forecast"], b["y"], b["mask"], null[s], b["x"])
            t_bb = _deviance_terms(model, o_bb["forecast"], b["y"], b["mask"], null[s], b["x"])
            for r in REGIONS:
                d_fc += np.array([t_occ[r][0] - t_base[r][0], 0, t_base[r][2]])
                d_fc_bb += np.array([t_bb[r][0] - t_base[r][0], 0, t_base[r][2]])
            m_s = metrics(lab, o["logits"].numpy())
            per_s[s] = m_s["balanced_accuracy"] - [p for p in results["per_session"] if p["session"] == s][0]["balanced_accuracy"]
        lg = np.concatenate(lg)
        m = metrics(y, lg)
        tocc.append({"window_start_ms": start * bin_ms, "window_end_ms": (start + w_bins) * bin_ms,
                     "delta_balanced_accuracy": m["balanced_accuracy"] - base_bacc, "delta_log_loss": m["log_loss"] - base_ll,
                     "delta_balanced_accuracy_lr": m["balanced_accuracy_lr"] - results["classification"]["balanced_accuracy_lr"],
                     "delta_forecast_deviance_explained": float(-d_fc[0] / d_fc[2]) if d_fc[2] > 1e-8 else float("nan"),
                     "delta_forecast_deviance_explained_backbone": float(-d_fc_bb[0] / d_fc_bb[2]) if d_fc_bb[2] > 1e-8 else float("nan"),
                     "per_session": per_s})
    results["temporal_occlusion"] = tocc

    # ---- region ablation: permutation of the whole region and in-distribution region drop
    abl = []
    for r_drop in REGIONS:
        # both are always computed: "drop" is the in-distribution ablation (the network was trained with region
        # dropout, so a missing region is a pattern it has seen - it measures whether the region is *needed*);
        # "permute" replaces the region by another trial's activity and measures how much the trained model
        # *relies* on it.  Robustness training makes the first small even for an informative region, which is why
        # the report shows both and P5a uses "drop" as the in-distribution primary.
        for method in ("permute", "drop"):
            lg, per_s = [], {}
            for s, b in tests.items():
                lab = b["label"].numpy()
                if method == "permute":
                    o = _forward_chunks(model, _permute_slice(b["x"], rng, lab, region=r_drop), s, b["mask"], device)
                else:
                    o = _forward_chunks(model, b["x"], s, b["mask"], device, drop_region=r_drop)
                lg.append(o["logits"].numpy())
                m_s = metrics(lab, o["logits"].numpy())
                base_s = [p for p in results["per_session"] if p["session"] == s][0]
                # both deltas per session: the report's P5a is about Left/Right decoding and prefers the L/R delta
                per_s[s] = {"delta_balanced_accuracy": m_s["balanced_accuracy"] - base_s["balanced_accuracy"],
                            "delta_balanced_accuracy_lr": m_s["balanced_accuracy_lr"] - base_s["balanced_accuracy_lr"]}
            lg = np.concatenate(lg)
            m = metrics(y, lg)
            abl.append({"dropped_region": r_drop, "method": method, **{k: m[k] for k in ("balanced_accuracy", "balanced_accuracy_lr", "macro_f1", "log_loss", "recall")},
                        "delta_balanced_accuracy": m["balanced_accuracy"] - base_bacc,
                        "delta_balanced_accuracy_lr": m["balanced_accuracy_lr"] - results["classification"]["balanced_accuracy_lr"],
                        "per_session": per_s})
    results["region_ablation"] = abl

    # ---- attention export (descriptive)
    att = {"temporal": {r: {} for r in REGIONS}, "region": {}, "gates": {}}
    for c_i, c in enumerate(CLASSES):
        sel = y == c_i
        if sel.any():
            for r in REGIONS:
                att["temporal"][r][c] = np.concatenate([base[s]["tattn"][r].numpy() for s in tests])[sel].mean(0)
            att["region"][c] = np.concatenate([base[s]["rattn"].numpy() for s in tests])[sel].mean(0)
    for s in tests:
        with torch.no_grad():
            ad = model.adapters[s.replace("/", "__").replace(".", "_")]
            att["gates"][s] = {r: ad.gates[r].gates().cpu().numpy() for r in REGIONS}
    np.savez(out_dir / "attention.npz",
             **{f"temporal_{r}_{c}": v for r, d in att["temporal"].items() for c, v in d.items()},
             **{f"region_{c}": v for c, v in att["region"].items()},
             **{f"gates_{s.replace('/', '__')}_{r}": v for s, d in att["gates"].items() for r, v in d.items()})
    cm_t = {}
    for r in REGIONS:
        for c, v in att["temporal"][r].items():
            tt = (np.arange(len(v)) + 0.5) * bin_ms
            cm_t[f"{r}_{c}"] = float((v * tt).sum() / max(v.sum(), 1e-9))
    results["attention_centre_of_mass_ms"] = cm_t

    # ---- neuron importance (permutation occlusion of each selected neuron, batched)
    if bool(cfg.evaluate.get_path("neuron_importance", True)):
        rows = []
        for s, b in tests.items():
            lab = b["label"].numpy()
            base_ll_s = _log_loss(lab, _softmax(base[s]["logits"].numpy()))
            base_bacc_s = metrics(lab, base[s]["logits"].numpy())["balanced_accuracy"]
            terms0 = {}
            for r in REGIONS:
                k_real = int(b["mask"][r][0].sum())
                for k in range(k_real):
                    x_o = _permute_slice(b["x"], rng, lab, region=r, neuron=k)
                    o = _forward_chunks(model, x_o, s, b["mask"], device)
                    m = metrics(lab, o["logits"].numpy())
                    t_base = _deviance_terms(model, base[s]["forecast"], b["y"], b["mask"], null[s], b["x"], exclude_neuron=k)
                    t_occ = _deviance_terms(model, o["forecast"], b["y"], b["mask"], null[s], b["x"], exclude_neuron=k)
                    d_dev = sum(t_occ[rr][0] - t_base[rr][0] for rr in REGIONS) / max(sum(t_base[rr][2] for rr in REGIONS), 1e-8)
                    rows.append({"session": s, "region": r, "k_slot": k, "unit_index": int(unit_index[s][r][k]),
                                 "delta_log_loss": m["log_loss"] - base_ll_s, "delta_balanced_accuracy": m["balanced_accuracy"] - base_bacc_s,
                                 "delta_forecast_deviance_explained_others": float(-d_dev),
                                 "gate": float(att["gates"][s][r][k]), "gate_rel": float(att["gates"][s][r][k] / max(att["gates"][s][r][: k_real].max(), 1e-9))})
        imp = pd.DataFrame(rows, columns=["session", "region", "k_slot", "unit_index", "delta_log_loss", "delta_balanced_accuracy",
                                          "delta_forecast_deviance_explained_others", "gate", "gate_rel"])
        # join with the selection statistics (score, stability, criteria) when they exist
        sel_tabs = {sel.session: sel.table for sel in run.get("selections", []) if sel is not None}
        if len(imp) and sel_tabs:
            parts = []
            for s, g in imp.groupby("session"):
                if s in sel_tabs:
                    cols = [c for c in ("score", "stability", "n_criteria", "rank", "unit_id", "auroc_left_right", "onset_ms", "rho_coupling") if c in sel_tabs[s]]
                    parts.append(g.merge(sel_tabs[s][["region", "unit_index"] + cols], on=["region", "unit_index"], how="left"))
                else:
                    parts.append(g)
            imp = pd.concat(parts, ignore_index=True)
        imp.to_csv(out_dir / "neuron_importance.csv", index=False)
        results["importance_agreement"] = importance_agreement(imp)
    else:
        results["importance_agreement"] = {}

    # ---- linear baselines + model-free tau95
    if caches is not None:
        rows, lin_sweep, extras = linear_baselines(tensors, splits, caches, cfg, unit_index, rng)
        results["baselines"] = rows
        results["linear_sweep"] = lin_sweep
        results["l1_overlap"] = extras["l1_overlap"]
        if lin_sweep:
            lin_logits = {}
            # rebuild per-tau pooled logits for the CSI of the linear decoder from the sweep rows (same trial order as `sweep`)
            results["tau95_linear_ms"] = _linear_tau95(lin_sweep, float(cfg.evaluate.get_path("csi_fraction", 0.95)))

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=_json_default)
    log.info("test balanced accuracy %.3f (chance p95 %.3f) | tau95 %s ms | forecast dev. expl. %s",
             results["classification"]["balanced_accuracy"], results["chance"]["p95"], results["csi"].get("tau95_ms"),
             {k.replace("deviance_explained_", ""): round(v, 3) for k, v in results["forecast"].items() if isinstance(v, float)})
    return results


def _linear_tau95(lin_sweep: list[dict], frac: float) -> float:
    taus = np.array([r["context_ms"] for r in lin_sweep], dtype=float)
    acc = np.array([r["balanced_accuracy"] for r in lin_sweep], dtype=float)
    order = np.argsort(taus)
    taus, acc = taus[order], acc[order]
    fit = IsotonicRegression(increasing=True).fit_transform(taus, np.nan_to_num(acc, nan=np.nanmean(acc)))
    ok = fit >= frac * fit[-1]
    return float(taus[np.argmax(ok)]) if ok.any() else float(taus[-1])


def importance_agreement(imp: pd.DataFrame) -> dict:
    """Within each (session, region) cell: Spearman between permutation importance, gates and the criteria score;
    summarised across cells (mean rho, sign test)."""
    from scipy import stats
    out = {}
    pairs = [("delta_log_loss", "score", "importance_vs_score"), ("gate", "score", "gate_vs_score"), ("gate", "delta_log_loss", "gate_vs_importance"),
             ("delta_log_loss", "stability", "importance_vs_stability")]
    for a, b, name in pairs:
        if a not in imp or b not in imp:
            continue
        rhos = []
        for (s, r), g in imp.groupby(["session", "region"]):
            g = g.dropna(subset=[a, b])
            if len(g) >= 5 and g[a].std() > 0 and g[b].std() > 0:
                rhos.append(float(stats.spearmanr(g[a], g[b]).statistic))
        if rhos:
            rhos = np.asarray(rhos)
            n_pos = int((rhos > 0).sum())
            out[name] = {"mean_rho": float(rhos.mean()), "median_rho": float(np.median(rhos)), "n_cells": int(len(rhos)),
                         "n_positive": n_pos, "sign_test_p": float(stats.binomtest(n_pos, len(rhos), 0.5).pvalue)}
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
