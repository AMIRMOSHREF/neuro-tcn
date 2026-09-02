"""Evaluation: classification metrics, forecasting quality, context-length sweep, region ablation,
label-shuffle chance level, linear baselines and attention/gate export."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASSES, REGIONS
from .data.dataset import TrialDataset
from .models.delaycast_net import poisson_deviance
from .train import balanced_accuracy, make_loader, predict

log = logging.getLogger(__name__)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    return {
        "accuracy": float((y == yhat).mean()),
        "balanced_accuracy": balanced_accuracy(y, yhat),
        "macro_f1": float(f1_score(y, yhat, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
        "n": int(len(y)),
    }


def _context_mask_fn(keep_bins: int, t_ctx: int):
    """Mask (ignore) everything except the last ``keep_bins`` bins of the context window."""
    def fn(batch, device):
        B = len(batch["label"])
        m = torch.zeros(B, t_ctx, dtype=torch.bool, device=device)
        if keep_bins < t_ctx:
            m[:, : t_ctx - keep_bins] = True
        return m
    return fn


def _train_null(tensors, splits) -> dict:
    """Per-session, per-region mean response PSTH of the training (or adaptation) trials: (K, T_tgt)."""
    null = {}
    for t in tensors:
        sp = splits[t.session]
        idx = np.r_[sp.get("train", np.zeros(0, int)), sp.get("adapt", np.zeros(0, int))]
        if not len(idx):
            idx = np.arange(t.n_trials)
        null[t.session] = {r: torch.from_numpy(t.y[r][idx].mean(axis=0)) for r in REGIONS}
    return null


@torch.no_grad()
def forecast_metrics(model, loader, device, null: dict) -> dict:
    """Deviance explained vs the training-set PSTH null, plus the same for a persistence-only forecast
    (each neuron's late-delay rate held constant) so the model's added value can be quantified."""
    model.eval()
    dev = {r: 0.0 for r in REGIONS}
    dev_persist = {r: 0.0 for r in REGIONS}
    dev0 = {r: 0.0 for r in REGIONS}
    for batch in loader:
        x = {r: batch["x"][r].to(device) for r in REGIONS}
        spec = {r: batch["spec"][r].to(device) for r in REGIONS}
        out = model(x, spec, batch["session"])
        for r in REGIONS:
            y = batch["y"][r].to(device)
            m = batch["mask"][r].to(device)
            null_mu = null[batch["session"]][r].to(device)[None].expand_as(y)
            late = x[r][:, :, -model.late_bins:].mean(-1) * model.count_scale  # (B, K) expected counts / target bin
            persist_mu = late[:, :, None].expand_as(y).clamp(min=1e-3)
            dev[r] += poisson_deviance(torch.exp(out.forecast_log_rate[r]), y, m).item()
            dev_persist[r] += poisson_deviance(persist_mu, y, m).item()
            dev0[r] += poisson_deviance(null_mu, y, m).item()
    out = {}
    for r in REGIONS:
        out[f"deviance_explained_{r}"] = 1 - dev[r] / max(dev0[r], 1e-8)
        out[f"deviance_explained_persistence_{r}"] = 1 - dev_persist[r] / max(dev0[r], 1e-8)
    return out


@torch.no_grad()
def export_attention(model, loader, device) -> dict:
    """Class-averaged temporal attention per region, region attention, and neuron gates per session."""
    model.eval()
    tattn = {r: {c: [] for c in range(len(CLASSES))} for r in REGIONS}
    rattn = {c: [] for c in range(len(CLASSES))}
    gates = {}
    for batch in loader:
        x = {r: batch["x"][r].to(device) for r in REGIONS}
        spec = {r: batch["spec"][r].to(device) for r in REGIONS}
        out = model(x, spec, batch["session"])
        lab = batch["label"].numpy()
        for c in np.unique(lab):
            sel = lab == c
            for r in REGIONS:
                tattn[r][int(c)].append(out.temporal_attn[r][sel].cpu().numpy())
            rattn[int(c)].append(out.region_attn[sel].cpu().numpy())
        gates[batch["session"]] = {r: out.gates[r].cpu().numpy() for r in REGIONS}
    return {
        "temporal": {r: {CLASSES[c]: np.concatenate(v).mean(0) for c, v in d.items() if v} for r, d in tattn.items()},
        "region": {CLASSES[c]: np.concatenate(v).mean(0) for c, v in rattn.items() if v},
        "gates": gates,
    }


def linear_baselines(tensors, splits, caches_by_session, cfg) -> pd.DataFrame:
    """Multinomial logistic regression on delay-epoch mean rates: all units vs selected units."""
    rows = []
    for name in ("all_units", "selected_units"):
        Xtr, ytr, Xte, yte = [], [], [], []
        for t in tensors:
            sp = splits[t.session]
            te_idx = sp["test"]
            tr_idx = np.r_[sp["train"], sp.get("adapt", np.zeros(0, int))]
            if not len(te_idx) or not len(tr_idx):
                continue
            if name == "all_units":
                feats = np.concatenate([caches_by_session[t.session].context[r].mean(axis=2) for r in REGIONS], axis=1)
            else:
                feats = np.concatenate([t.x[r][:, t.neuron_mask[r]].mean(axis=2) for r in REGIONS], axis=1)
            # Session-wise fit (unit identities differ between sessions); pooled metrics.
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
            clf.fit(feats[tr_idx], t.labels[tr_idx])
            Xte.append(clf.predict(feats[te_idx]))
            yte.append(t.labels[te_idx])
        if yte:
            rows.append({"model": f"logreg_{name}", **_metrics(np.concatenate(yte), np.concatenate(Xte))})
    return pd.DataFrame(rows)


def evaluate_run(run: dict, cfg, caches_by_session: dict | None = None) -> dict:
    model, tensors, splits, out_dir, device = run["model"], run["tensors"], run["splits"], Path(run["out_dir"]), run["device"]
    test_ds = TrialDataset(tensors, {s: v["test"] for s, v in splits.items() if len(v["test"])})
    loader = make_loader(test_ds, cfg, shuffle=False)
    results: dict = {}

    pr = predict(model, loader, device)
    yhat = pr["logits"].argmax(1)
    results["classification"] = _metrics(pr["labels"], yhat)
    results["confusion"] = confusion_matrix(pr["labels"], yhat, labels=list(range(len(CLASSES)))).tolist()
    per_sess = []
    for s in np.unique(pr["sessions"]):
        m = pr["sessions"] == s
        per_sess.append({"session": s, **_metrics(pr["labels"][m], yhat[m])})
    results["per_session"] = per_sess
    pd.DataFrame({"session": pr["sessions"], "trial": pr["trials"], "label": pr["labels"], "pred": yhat,
                  **{f"p_{c}": torch.softmax(torch.from_numpy(pr["logits"]), 1)[:, i].numpy() for i, c in enumerate(CLASSES)}}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)

    # Chance level from label shuffles.
    rng = np.random.default_rng(0)
    sh = [balanced_accuracy(rng.permutation(pr["labels"]), yhat) for _ in range(int(cfg.evaluate.n_shuffles))]
    results["chance_balanced_accuracy"] = {"mean": float(np.mean(sh)), "p95": float(np.percentile(sh, 95))}

    results["forecast"] = forecast_metrics(model, loader, device, _train_null(tensors, splits))

    # How much past context is needed? Keep only the last tau ms of the delay.
    t_ctx = model.t_ctx
    sweep = []
    for ms in cfg.evaluate.context_sweep_ms:
        keep = int(round(ms / cfg.data.bin_ms))
        if keep > t_ctx:
            continue
        p = predict(model, loader, device, pad_mask_fn=_context_mask_fn(keep, t_ctx))
        sweep.append({"context_ms": ms, **_metrics(p["labels"], p["logits"].argmax(1))})
    results["context_sweep"] = sweep

    # Region ablation: zero one region's input at a time.
    abl = []
    for r_drop in REGIONS:
        preds = []
        model.eval()
        with torch.no_grad():
            labels = []
            for batch in loader:
                x = {r: (torch.zeros_like(batch["x"][r]) if r == r_drop else batch["x"][r]).to(device) for r in REGIONS}
                spec = {r: (torch.zeros_like(batch["spec"][r]) if r == r_drop else batch["spec"][r]).to(device) for r in REGIONS}
                out = model(x, spec, batch["session"])
                preds.append(out.logits.argmax(1).cpu().numpy())
                labels.append(batch["label"].numpy())
        abl.append({"dropped_region": r_drop, **_metrics(np.concatenate(labels), np.concatenate(preds))})
    results["region_ablation"] = abl

    att = export_attention(model, loader, device)
    np.savez(out_dir / "attention.npz",
             **{f"temporal_{r}_{c}": v for r, d in att["temporal"].items() for c, v in d.items()},
             **{f"region_{c}": v for c, v in att["region"].items()},
             **{f"gates_{s.replace('/', '__')}_{r}": v for s, d in att["gates"].items() for r, v in d.items()})

    if caches_by_session is not None:
        results["baselines"] = linear_baselines(tensors, splits, caches_by_session, cfg).to_dict(orient="records")

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("test balanced accuracy %.3f (chance %.3f) | forecast deviance explained %s",
             results["classification"]["balanced_accuracy"], results["chance_balanced_accuracy"]["mean"],
             {k.replace("deviance_explained_", ""): round(v, 3) for k, v in results["forecast"].items()})
    return results
