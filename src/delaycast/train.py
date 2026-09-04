"""Training of DelayCAST-Net on the union of both datasets.

Order of operations inside ``run_training`` (this order is what keeps the evaluation honest):

1. trial splits per session (needs only labels) - train / val / test, or adapt / test for held-out sessions;
2. neuron selection on the *fit* trials only (train + val, or adapt), never on test trials;
3. model tensors for the chosen K units, joint training on all sessions of both datasets with
   context-length, window and region dropout augmentation (so the test-time interventions are in-distribution);
4. optional adapter-only fitting on the held-out sessions (frozen backbone);
5. checkpoint with the exact unit indices, so a run can be re-evaluated without re-selecting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import CLASSES, REGIONS
from .config import Config, dump_config
from .data.cache import SessionCache, load_cache
from .data.dataset import (SessionBatchSampler, SessionTensors, TrialDataset, build_session_tensors, class_weights,
                           collate, stratified_split, tensors_from_indices)
from .features.selection import LEFT, RIGHT, SelectionResult, _features_key, select_neurons
from .models.delaycast_net import DelayCASTNet, poisson_nll

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- utilities
def get_device(cfg: Config) -> torch.device:
    d = cfg.train.device
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _as_list(holdout) -> list[str]:
    if holdout is None:
        return []
    return [holdout] if isinstance(holdout, str) else list(holdout)


def get_caches(cfg: Config, caches: dict[str, SessionCache] | None = None) -> dict[str, SessionCache]:
    """Load every session cache once per process (each CLI command passes the same dict around)."""
    return caches if caches is not None else {c.session: c for c in load_cache(cfg)}


def permute_labels(caches: dict[str, SessionCache], seed: int) -> dict[str, SessionCache]:
    """Negative control: class labels permuted *within session* before selection, training and evaluation."""
    rng = np.random.default_rng(seed)
    out = {}
    for s, c in caches.items():
        cc = copy.copy(c)
        cc.labels = rng.permutation(c.labels)
        out[s] = cc
    return out


# ----------------------------------------------------------------------------- splits + selection
def make_splits(cfg: Config, caches: dict[str, SessionCache], holdout: str | list[str] | None = None,
                seed: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Per-session trial splits.

    * no holdout: train / val / test inside every session (``train.split_scheme``: random | blocked);
    * holdout sessions: ``adapt_frac`` of their trials go to ``adapt`` (session-adapter fitting only), the rest is
      ``test``; the remaining sessions contribute train / val only.
    """
    held = set(_as_list(holdout))
    seed = int(cfg.train.seed) if seed is None else int(seed)
    scheme = str(cfg.train.get_path("split_scheme", "random"))
    splits = {}
    for s, c in caches.items():
        if not held:
            tr, va, te = stratified_split(c.labels, float(cfg.train.val_frac), float(cfg.train.test_frac), seed, scheme=scheme)
            splits[s] = {"train": tr, "val": va, "test": te}
        elif s in held:
            tr, va, te = stratified_split(c.labels, 0.0, 1 - float(cfg.train.adapt_frac), seed, scheme=scheme)
            splits[s] = {"train": np.zeros(0, int), "val": np.zeros(0, int), "test": te, "adapt": np.sort(np.r_[tr, va])}
        else:
            tr, va, _ = stratified_split(c.labels, float(cfg.train.val_frac), 0.0, seed, scheme=scheme)
            splits[s] = {"train": tr, "val": va, "test": np.zeros(0, int)}
    return splits


def fit_trials(split: dict[str, np.ndarray]) -> np.ndarray:
    """Trials that may inform neuron selection: train + val (+ adapt for held-out sessions); never test."""
    return np.sort(np.r_[split.get("train", np.zeros(0, int)), split.get("val", np.zeros(0, int)),
                         split.get("adapt", np.zeros(0, int))].astype(int))


def _selection_cache_path(cfg: Config, cache: SessionCache, idx: np.ndarray, seed: int) -> Path:
    payload = json.dumps(cfg.selection.to_plain(), sort_keys=True, default=str) + _features_key(cfg, cache) + "|stats_v2"
    h = hashlib.md5(payload.encode() + np.asarray(idx, dtype=np.int64).tobytes() + cache.labels.astype(np.int64).tobytes()).hexdigest()[:12]
    return Path(cfg.data.cache_dir) / "selection" / f"{cache.session.replace('/', '__')}_{h}_s{seed}.pkl"


def cached_selection(cfg: Config, cache: SessionCache, idx: np.ndarray | None, seed: int, n_null: int = 0,
                     label_free: bool = False) -> SelectionResult:
    """``select_neurons`` with an on-disk cache keyed by (session, selection config, fit trials, labels, seed).

    The stability procedure costs ~1 min per real session, and the same selection is needed by every
    variant / mode / re-evaluation of the same split, so it is computed once."""
    idx = np.arange(cache.n_trials) if idx is None else np.asarray(idx, dtype=int)
    path = _selection_cache_path(cfg, cache, idx, seed)
    if label_free:
        path = path.with_name(path.stem + "_lf" + path.suffix)
    if path.is_file():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:  # pragma: no cover
            pass
    res = select_neurons(cache, cfg, trial_idx=idx, seed=seed, n_null=n_null, label_free=label_free)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def prepare_sessions(cfg: Config, caches: dict[str, SessionCache], mode: str = "criteria", splits: dict | None = None,
                     seed: int | None = None, holdout: list[str] | None = None):
    """Neuron selection (criteria mode only) and model tensors.

    Sessions with train/val trials: criteria on the fit trials (train + val).  Held-out sessions:
    ``selection.holdout_mode`` = ``label_free`` (default; floor + coupling + net ramp on the *adapt* trials only,
    no label ever read and no test trial ever touched) or ``adapt`` (full criteria on the adapt trials)."""
    seed = int(cfg.train.seed) if seed is None else int(seed)
    held = set(holdout or [])
    holdout_mode = str(cfg.selection.get_path("holdout_mode", "label_free"))
    selections, tensors = [], []
    for s, c in caches.items():
        idx = fit_trials(splits[s]) if splits is not None else None
        if mode == "criteria":
            if s in held and holdout_mode == "label_free":
                sel = cached_selection(cfg, c, idx, seed, label_free=True)
            else:
                sel = cached_selection(cfg, c, idx if bool(cfg.selection.get_path("fit_on_train_only", True)) else None, seed)
        else:
            sel = None
        selections.append(sel)
        tensors.append(build_session_tensors(c, sel, cfg, mode=mode, seed=seed, trial_idx=idx))
    return selections, tensors


# ----------------------------------------------------------------------------- loss / prediction
def compute_loss(model: DelayCASTNet, batch: dict, cfg: Config, cw: torch.Tensor, device, pad_mask=None, drop_region=None):
    x = {r: batch["x"][r].to(device) for r in REGIONS}
    y = {r: batch["y"][r].to(device) for r in REGIONS}
    mask = {r: batch["mask"][r].to(device) for r in REGIONS}
    labels = batch["label"].to(device)
    out = model(x, batch["session"], pad_mask, neuron_mask=mask, drop_region=drop_region)
    ce = F.cross_entropy(out.logits, labels, weight=cw)
    fc = sum(poisson_nll(out.forecast_log_rate[r], y[r], mask[r]) for r in REGIONS) / len(REGIONS)
    loss = float(cfg.model.class_weight) * ce + float(cfg.model.forecast_weight) * fc + model.gate_weight * out.gate_penalty
    return loss, {"ce": ce.item(), "poisson": fc.item(), "gate": out.gate_penalty.item()}, out


@torch.no_grad()
def predict(model: DelayCASTNet, loader: DataLoader, device, pad_mask_fn=None) -> dict:
    model.eval()
    logits, labels, sessions, trials = [], [], [], []
    for batch in loader:
        x = {r: batch["x"][r].to(device) for r in REGIONS}
        mask = {r: batch["mask"][r].to(device) for r in REGIONS}
        pm = pad_mask_fn(batch, device) if pad_mask_fn else None
        out = model(x, batch["session"], pm, neuron_mask=mask)
        logits.append(out.logits.cpu())
        labels.append(batch["label"])
        sessions += [batch["session"]] * len(batch["label"])
        trials.append(batch["trial"])
    return {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy(),
            "sessions": np.array(sessions), "trials": torch.cat(trials).numpy()}


@torch.no_grad()
def validation_loss(model, loader, cfg, cw, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        loss, _, _ = compute_loss(model, batch, cfg, cw, device)
        tot += loss.item() * len(batch["label"])
        n += len(batch["label"])
    return tot / max(n, 1)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """Mean per-class recall. With ``labels`` given, classes absent from ``y_true`` are ignored (NaN-safe)."""
    classes = np.unique(y_true) if labels is None else [c for c in labels if (y_true == c).any()]
    accs = [np.mean(y_pred[y_true == c] == c) for c in classes]
    return float(np.mean(accs)) if accs else float("nan")


def make_loader(ds: TrialDataset, cfg: Config, shuffle: bool) -> DataLoader:
    sampler = SessionBatchSampler(ds, int(cfg.train.batch_size), shuffle=shuffle, seed=int(cfg.train.seed))
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=int(cfg.train.num_workers))


# ----------------------------------------------------------------------------- fitting
def _augment(batch, cfg: Config, rng: np.random.Generator, device):
    """Training-time interventions that make the evaluation interventions in-distribution.

    * prefix masking (``context_aug_prob``): only the last ``keep`` bins of the delay are visible;
    * window dropout (``window_dropout_prob``): one random ``window_dropout_ms`` window of the delay is masked;
    * region dropout (``region_dropout_prob``): one region's input is removed for the whole batch.
    """
    t_ctx = batch["x"][REGIONS[0]].shape[-1]
    B = len(batch["label"])
    bin_ms = float(cfg.data.bin_ms)
    pad_mask = None
    if rng.random() < float(cfg.train.get_path("context_aug_prob", 0.0)):
        min_keep = int(round(float(cfg.train.get_path("context_aug_min_ms", 100)) / bin_ms))
        keep = int(rng.integers(min_keep, t_ctx + 1))
        pad_mask = torch.zeros(B, t_ctx, dtype=torch.bool, device=device)
        pad_mask[:, : t_ctx - keep] = True
    if rng.random() < float(cfg.train.get_path("window_dropout_prob", 0.0)):
        w = max(1, int(round(float(cfg.train.get_path("window_dropout_ms", 200)) / bin_ms)))
        start = int(rng.integers(0, max(t_ctx - w, 0) + 1))
        if pad_mask is None:
            pad_mask = torch.zeros(B, t_ctx, dtype=torch.bool, device=device)
        pad_mask[:, start: start + w] = True
    drop_region = None
    if rng.random() < float(cfg.train.get_path("region_dropout_prob", 0.0)):
        drop_region = REGIONS[int(rng.integers(0, len(REGIONS)))]
    return pad_mask, drop_region


def fit(model: DelayCASTNet, train_ds: TrialDataset, val_ds: TrialDataset, cfg: Config, device, params=None,
        epochs: int | None = None, tag: str = "train", augment: bool = True) -> pd.DataFrame:
    epochs = epochs or int(cfg.train.epochs)
    params = list(params) if params is not None else list(model.parameters())
    # Neuron gates get a larger learning rate and no weight decay so that they can actually move
    # within a short training run (weight decay would pull the logits towards 0.5).
    gate_ids = {id(p) for n, p in model.named_parameters() if ".gates." in n}
    groups = [{"params": [p for p in params if id(p) not in gate_ids], "lr": float(cfg.train.lr), "weight_decay": float(cfg.train.weight_decay)},
              {"params": [p for p in params if id(p) in gate_ids], "lr": float(cfg.train.lr) * float(cfg.train.get_path("gate_lr_mult", 10.0)), "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = class_weights(train_ds.labels(), cap=cfg.train.get_path("class_weight_cap", 5.0)).to(device)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False) if len(val_ds) else None
    best, best_state, bad, history = -np.inf, None, 0, []
    rng = np.random.default_rng(int(cfg.train.seed))
    for ep in range(epochs):
        model.train()
        t0, agg, n = time.time(), {"loss": 0.0, "ce": 0.0, "poisson": 0.0, "gate": 0.0}, 0
        for batch in train_loader:
            pad_mask, drop_region = _augment(batch, cfg, rng, device) if augment else (None, None)
            loss, parts, _ = compute_loss(model, batch, cfg, cw, device, pad_mask, drop_region)
            if not torch.isfinite(loss):
                log.warning("non-finite loss encountered; skipping batch")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            bs = len(batch["label"])
            agg["loss"] += loss.item() * bs
            for k, v in parts.items():
                agg[k] += v * bs
            n += bs
        sched.step()
        row = {"epoch": ep + 1, **{k: v / max(n, 1) for k, v in agg.items()}, "sec": time.time() - t0}
        if val_loader is not None:
            pr = predict(model, val_loader, device)
            row["val_bacc"] = balanced_accuracy(pr["labels"], pr["logits"].argmax(1))
            row["val_acc"] = float((pr["labels"] == pr["logits"].argmax(1)).mean())
            row["val_loss"] = validation_loss(model, val_loader, cfg, cw, device)
            score = row["val_bacc"] if cfg.train.get_path("select_by", "val_loss") == "val_bacc" else -row["val_loss"]
        else:
            score = -row["loss"]
        history.append(row)
        log.info("[%s] ep %3d loss %.3f ce %.3f pois %.3f gate %.3f | val loss %.3f bacc %.3f (%.1fs)", tag, ep + 1, row["loss"],
                 row["ce"], row["poisson"], row["gate"], row.get("val_loss", float("nan")), row.get("val_bacc", float("nan")), row["sec"])
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(cfg.train.patience):
                log.info("[%s] early stopping at epoch %d", tag, ep + 1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def _check_training_set(train_ds: TrialDataset, val_ds: TrialDataset, splits: dict, held: list[str]) -> None:
    """Refuse to train on a degenerate corpus instead of producing a NaN validation curve.

    This is reached when the loader kept (almost) no trials of the training sessions - e.g. every Data2 trial
    failed QC - and it is far cheaper to stop here than to discover it in ``REPORT.md``."""
    n_train, n_val = len(train_ds), len(val_ds)
    train_sessions = [s for s, v in splits.items() if len(v["train"])]
    y = np.asarray(train_ds.labels()) if n_train else np.zeros(0, int)
    n_classes = len(np.unique(y))
    if n_train < 20 or n_classes < 2 or n_val == 0:
        raise RuntimeError(
            f"training set is degenerate: {n_train} training trials over {len(train_sessions)} session(s) "
            f"({n_classes} class(es)), {n_val} validation trials; held-out sessions: {held or 'none'}. "
            "Check `python -m delaycast cache` (columns discovered / dropped / drop_reasons): the training sessions "
            "were almost entirely removed by QC or by data.min_trials_per_session.")
    per_class = np.bincount(y, minlength=len(CLASSES))
    if per_class[LEFT] < 5 or per_class[RIGHT] < 5:
        raise RuntimeError(f"training set has too few lick trials: Left {per_class[LEFT]}, Right {per_class[RIGHT]} "
                           f"over {len(train_sessions)} session(s); see `python -m delaycast cache` drop reasons.")


def build_model(cfg: Config, tensors: list[SessionTensors], device) -> DelayCASTNet:
    t_ctx = tensors[0].x[REGIONS[0]].shape[2]
    t_tgt = tensors[0].y[REGIONS[0]].shape[2]
    model = DelayCASTNet([t.session for t in tensors], int(cfg.selection.top_k_per_region), t_ctx, t_tgt, cfg)
    return model.to(device)


def adapt_holdout(model: DelayCASTNet, tensors: list[SessionTensors], splits: dict, held: list[str], cfg: Config, device) -> dict:
    """Adapter-only fitting on the held-out sessions: frozen backbone, an adapter-validation split for early
    stopping, gates optionally frozen (``train.adapt_fit_gates``)."""
    seed = int(cfg.train.seed)
    adapt_val_frac = float(cfg.train.get_path("adapt_val_frac", 0.2))
    tr_idx, va_idx = {}, {}
    by_session = {t.session: t for t in tensors}
    for h in held:
        idx = splits[h].get("adapt", np.zeros(0, int))
        if not len(idx):
            continue
        labels = by_session[h].labels[idx]
        tr, va, _ = stratified_split(labels, adapt_val_frac, 0.0, seed)
        tr_idx[h], va_idx[h] = idx[tr], idx[va]
    if not tr_idx:
        return {"n_adapt": 0}
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    fit_gates = bool(cfg.train.get_path("adapt_fit_gates", False))
    params = [p for h in tr_idx for p in model.adapter_parameters(h, include_gates=fit_gates)]
    fit(model, TrialDataset(tensors, tr_idx), TrialDataset(tensors, va_idx), cfg, device, params=params,
        epochs=int(cfg.train.adapt_epochs), tag=f"adapt:{'+'.join(held)}", augment=False)
    for p in model.backbone_parameters():
        p.requires_grad_(True)
    return {"n_adapt": int(sum(len(v) for v in tr_idx.values())), "n_adapt_val": int(sum(len(v) for v in va_idx.values())),
            "adapter_parameters": int(sum(p.numel() for p in params)), "gates_fitted": fit_gates}


def run_training(cfg: Config, mode: str = "criteria", out_dir: Path | None = None, holdout: str | list[str] | None = None,
                 caches: dict[str, SessionCache] | None = None, negative_control: bool = False) -> dict:
    """Train + validate.

    ``mode`` selects the neuron set (criteria | rate | random); ``holdout`` is one session key or a list of keys
    for cross-session / cross-dataset evaluation; ``negative_control`` permutes the labels within session
    before everything else (the whole pipeline must then be at chance)."""
    seed = int(cfg.train.seed)
    set_seed(seed)
    device = get_device(cfg)
    out_dir = Path(out_dir or Path(cfg.output_dir) / f"run_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    caches = get_caches(cfg, caches)
    if negative_control:
        caches = permute_labels(caches, seed + 12345)
    splits = make_splits(cfg, caches, holdout, seed)
    selections, tensors = prepare_sessions(cfg, caches, mode=mode, splits=splits, seed=seed, holdout=_as_list(holdout))
    if negative_control and mode == "criteria" and not any(t.neuron_mask[r].any() for t in tensors for r in REGIONS):
        # With permuted labels the label-dependent criteria usually select nothing, which would make the control
        # trivially at chance without exercising the training / evaluation path. Fall back to the label-free
        # ``rate`` units so the network is really trained on permuted labels (this is logged and recorded).
        log.warning("negative control: permuted-label selection is empty in every session; using rate-mode units")
        selections, tensors = prepare_sessions(cfg, caches, mode="rate", splits=splits, seed=seed, holdout=_as_list(holdout))
        mode = "criteria(rate-fallback)"
    for s in selections:
        if s is not None:
            s.table.to_csv(out_dir / f"selection_{s.session.replace('/', '__')}.csv", index=False)
    if any(s is not None for s in selections):
        pd.concat([s.funnel for s in selections if s is not None]).to_csv(out_dir / "selection_funnel.csv", index=False)

    train_ds = TrialDataset(tensors, {s: v["train"] for s, v in splits.items() if len(v["train"])})
    val_ds = TrialDataset(tensors, {s: v["val"] for s, v in splits.items() if len(v["val"])})
    _check_training_set(train_ds, val_ds, splits, held=_as_list(holdout))
    model = build_model(cfg, tensors, device)
    log.info("model parameters: %.2fM (receptive field %d bins, %d transformer blocks, spectral branch: %s)",
             sum(p.numel() for p in model.parameters()) / 1e6, model.tcn.receptive_field, len(model.temporal_attn), model.spectral_branch)
    hist = fit(model, train_ds, val_ds, cfg, device, tag=f"{mode}")
    hist.to_csv(out_dir / "history.csv", index=False)

    held = _as_list(holdout)
    adapt_info = adapt_holdout(model, tensors, splits, held, cfg, device) if held else {}

    unit_index = {t.session: {r: t.unit_index[r].tolist() for r in REGIONS} for t in tensors}
    torch.save({"state_dict": model.state_dict(), "sessions": [t.session for t in tensors], "cfg": cfg.to_plain(),
                "k": model.k, "t_ctx": model.t_ctx, "t_tgt": model.t_tgt, "n_spec": model.n_spec, "mode": mode,
                "unit_index": unit_index, "holdout": held, "seed": seed, "negative_control": negative_control,
                "adapt_info": adapt_info},
               out_dir / "model.pt")
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({s: {k: v.tolist() for k, v in d.items()} for s, d in splits.items()}, f)
    return {"model": model, "tensors": tensors, "selections": selections, "splits": splits, "out_dir": out_dir, "device": device,
            "mode": mode, "holdout": held, "seed": seed, "caches": caches, "negative_control": negative_control, "adapt_info": adapt_info}


def load_run(out_dir: Path, cfg: Config, caches: dict[str, SessionCache] | None = None):
    """Re-create a saved run: the exact unit indices stored in the checkpoint are used (no re-selection)."""
    out_dir = Path(out_dir)
    ck = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    caches = get_caches(cfg, caches)
    if ck.get("negative_control", False):
        caches = permute_labels(caches, int(ck.get("seed", cfg.train.seed)) + 12345)
    tensors = []
    for s in ck["sessions"]:
        idx = {r: np.asarray(v, dtype=int) for r, v in ck["unit_index"][s].items()}
        tensors.append(tensors_from_indices(caches[s], idx, cfg))
    selections = []
    for s in ck["sessions"]:
        p = out_dir / f"selection_{s.replace('/', '__')}.csv"
        if p.is_file():
            tab = pd.read_csv(p)
            sel = {r: tab[(tab.region == r) & tab.selected].sort_values("rank").unit_index.to_numpy(dtype=int) for r in REGIONS}
            selections.append(SelectionResult(s, tab, sel))
        else:
            selections.append(None)
    device = get_device(cfg)
    if ck["t_ctx"] != tensors[0].x[REGIONS[0]].shape[2]:
        raise ValueError(f"checkpoint context length {ck['t_ctx']} differs from the cache ({tensors[0].x[REGIONS[0]].shape[2]} bins)")
    model = DelayCASTNet(ck["sessions"], ck["k"], ck["t_ctx"], ck["t_tgt"], cfg, n_spec=ck.get("n_spec")).to(device)
    model.load_state_dict(ck["state_dict"])
    with open(out_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = {s: {k: np.asarray(v, int) for k, v in d.items()} for s, d in json.load(f).items()}
    return {"model": model, "tensors": tensors, "selections": selections, "splits": splits, "out_dir": out_dir, "device": device,
            "mode": ck.get("mode", "criteria"), "holdout": ck.get("holdout", []), "seed": int(ck.get("seed", cfg.train.seed)),
            "caches": caches, "negative_control": bool(ck.get("negative_control", False)), "adapt_info": ck.get("adapt_info", {})}
