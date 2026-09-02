"""Training of DelayCAST-Net on the union of both datasets."""
from __future__ import annotations

import json
import logging
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
                           collate, stratified_split)
from .features.selection import SelectionResult, select_neurons
from .models.delaycast_net import DelayCASTNet, poisson_nll

log = logging.getLogger(__name__)


def get_device(cfg: Config) -> torch.device:
    d = cfg.train.device
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_sessions(cfg: Config, caches: list[SessionCache], mode: str = "criteria", selections: list[SelectionResult] | None = None):
    selections = selections or [select_neurons(c, cfg) for c in caches]
    tensors = [build_session_tensors(c, s, cfg, mode=mode, seed=int(cfg.train.seed)) for c, s in zip(caches, selections)]
    return selections, tensors


def compute_loss(model: DelayCASTNet, batch: dict, cfg: Config, cw: torch.Tensor, device, pad_mask=None):
    x = {r: batch["x"][r].to(device) for r in REGIONS}
    spec = {r: batch["spec"][r].to(device) for r in REGIONS}
    y = {r: batch["y"][r].to(device) for r in REGIONS}
    mask = {r: batch["mask"][r].to(device) for r in REGIONS}
    labels = batch["label"].to(device)
    out = model(x, spec, batch["session"], pad_mask)
    ce = F.cross_entropy(out.logits, labels, weight=cw)
    fc = sum(poisson_nll(out.forecast_log_rate[r], y[r], mask[r]) for r in REGIONS) / len(REGIONS)
    loss = float(cfg.model.class_weight) * ce + float(cfg.model.forecast_weight) * fc + model.gate_l1 * out.gate_l1
    return loss, {"ce": ce.item(), "poisson": fc.item(), "gate_l1": out.gate_l1.item()}, out


@torch.no_grad()
def predict(model: DelayCASTNet, loader: DataLoader, device, pad_mask_fn=None) -> dict:
    model.eval()
    logits, labels, sessions, trials = [], [], [], []
    for batch in loader:
        x = {r: batch["x"][r].to(device) for r in REGIONS}
        spec = {r: batch["spec"][r].to(device) for r in REGIONS}
        pm = pad_mask_fn(batch, device) if pad_mask_fn else None
        out = model(x, spec, batch["session"], pm)
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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(accs)) if accs else float("nan")


def make_loader(ds: TrialDataset, cfg: Config, shuffle: bool) -> DataLoader:
    sampler = SessionBatchSampler(ds, int(cfg.train.batch_size), shuffle=shuffle, seed=int(cfg.train.seed))
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=int(cfg.train.num_workers))


def fit(model: DelayCASTNet, train_ds: TrialDataset, val_ds: TrialDataset, cfg: Config, device, params=None,
        epochs: int | None = None, tag: str = "train") -> pd.DataFrame:
    epochs = epochs or int(cfg.train.epochs)
    params = list(params) if params is not None else list(model.parameters())
    # Neuron gates get a larger learning rate and no weight decay so that they can actually move
    # towards 0/1 within a short training run (weight decay would pull the logits towards 0.5).
    gate_ids = {id(p) for n, p in model.named_parameters() if ".gates." in n}
    groups = [{"params": [p for p in params if id(p) not in gate_ids], "lr": float(cfg.train.lr), "weight_decay": float(cfg.train.weight_decay)},
              {"params": [p for p in params if id(p) in gate_ids], "lr": float(cfg.train.lr) * float(cfg.train.get_path("gate_lr_mult", 10.0)), "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = class_weights(train_ds.labels()).to(device)
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False) if len(val_ds) else None
    best, best_state, bad, history = -np.inf, None, 0, []
    aug_p = float(cfg.train.get_path("context_aug_prob", 0.0))
    min_keep = int(round(float(cfg.train.get_path("context_aug_min_ms", 100)) / float(cfg.data.bin_ms)))
    rng = np.random.default_rng(int(cfg.train.seed))
    for ep in range(epochs):
        model.train()
        t0, agg, n = time.time(), {"loss": 0.0, "ce": 0.0, "poisson": 0.0, "gate_l1": 0.0}, 0
        for batch in train_loader:
            pad_mask = None
            if aug_p > 0 and rng.random() < aug_p:
                # Context-length augmentation: hide a random prefix of the delay so the network learns
                # to work from any amount of recent history (needed for the context sweep at test time).
                t_ctx = batch["x"][REGIONS[0]].shape[-1]
                keep = int(rng.integers(min_keep, t_ctx + 1))
                pad_mask = torch.zeros(len(batch["label"]), t_ctx, dtype=torch.bool, device=device)
                pad_mask[:, : t_ctx - keep] = True
            loss, parts, _ = compute_loss(model, batch, cfg, cw, device, pad_mask)
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
        log.info("[%s] ep %3d loss %.3f ce %.3f pois %.3f gate %.2f | val loss %.3f bacc %.3f (%.1fs)", tag, ep + 1, row["loss"],
                 row["ce"], row["poisson"], row["gate_l1"], row.get("val_loss", float("nan")), row.get("val_bacc", float("nan")), row["sec"])
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


def build_model(cfg: Config, tensors: list[SessionTensors], device) -> DelayCASTNet:
    t_ctx = tensors[0].x[REGIONS[0]].shape[2]
    t_tgt = tensors[0].y[REGIONS[0]].shape[2]
    n_spec = tensors[0].spec[REGIONS[0]].shape[1]
    model = DelayCASTNet([t.session for t in tensors], int(cfg.selection.top_k_per_region), t_ctx, t_tgt, n_spec, cfg)
    return model.to(device)


def run_training(cfg: Config, mode: str = "criteria", out_dir: Path | None = None, holdout: str | None = None) -> dict:
    """Train + validate. ``mode`` selects the neuron set (criteria | rate | random); ``holdout`` is
    a session key for cross-session evaluation (read-in adaptation on a small fraction of its trials)."""
    set_seed(int(cfg.train.seed))
    device = get_device(cfg)
    out_dir = Path(out_dir or Path(cfg.output_dir) / f"run_{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    caches = load_cache(cfg)
    selections, tensors = prepare_sessions(cfg, caches, mode=mode)
    for s in selections:
        s.table.to_csv(out_dir / f"selection_{s.session.replace('/', '__')}.csv", index=False)

    splits = {}
    if holdout is None:
        for t in tensors:
            tr, va, te = stratified_split(t.labels, float(cfg.train.val_frac), float(cfg.train.test_frac), int(cfg.train.seed))
            splits[t.session] = {"train": tr, "val": va, "test": te}
    else:
        for t in tensors:
            if t.session == holdout:
                # adapt_frac of held-out trials are used to fit the session adapter; the rest is test.
                tr, va, te = stratified_split(t.labels, 0.0, 1 - float(cfg.train.adapt_frac), int(cfg.train.seed))
                splits[t.session] = {"train": np.zeros(0, int), "val": np.zeros(0, int), "test": te, "adapt": np.sort(np.r_[tr, va])}
            else:
                tr, va, _ = stratified_split(t.labels, float(cfg.train.val_frac), 0.0, int(cfg.train.seed))
                splits[t.session] = {"train": tr, "val": va, "test": np.zeros(0, int)}

    train_ds = TrialDataset(tensors, {s: v["train"] for s, v in splits.items() if len(v["train"])})
    val_ds = TrialDataset(tensors, {s: v["val"] for s, v in splits.items() if len(v["val"])})
    model = build_model(cfg, tensors, device)
    log.info("model parameters: %.2fM (receptive field %d bins)", sum(p.numel() for p in model.parameters()) / 1e6, model.tcn.receptive_field)
    hist = fit(model, train_ds, val_ds, cfg, device, tag=f"{mode}")
    hist.to_csv(out_dir / "history.csv", index=False)

    if holdout is not None:
        adapt_ds = TrialDataset(tensors, {holdout: splits[holdout]["adapt"]})
        for p in model.backbone_parameters():
            p.requires_grad_(False)
        fit(model, adapt_ds, TrialDataset(tensors, {}), cfg, device, params=model.adapter_parameters(holdout),
            epochs=int(cfg.train.adapt_epochs), tag=f"adapt:{holdout}")
        for p in model.backbone_parameters():
            p.requires_grad_(True)

    torch.save({"state_dict": model.state_dict(), "sessions": [t.session for t in tensors], "cfg": cfg.to_plain(),
                "k": model.k, "t_ctx": model.t_ctx, "t_tgt": model.t_tgt, "n_spec": model.n_spec, "mode": mode},
               out_dir / "model.pt")
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({s: {k: v.tolist() for k, v in d.items()} for s, d in splits.items()}, f)
    return {"model": model, "tensors": tensors, "selections": selections, "splits": splits, "out_dir": out_dir, "device": device}


def load_run(out_dir: Path, cfg: Config):
    ck = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    caches = load_cache(cfg)
    selections, tensors = prepare_sessions(cfg, caches, mode=ck["mode"])
    device = get_device(cfg)
    model = DelayCASTNet(ck["sessions"], ck["k"], ck["t_ctx"], ck["t_tgt"], ck["n_spec"], cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    with open(out_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = {s: {k: np.asarray(v, int) for k, v in d.items()} for s, d in json.load(f).items()}
    return {"model": model, "tensors": tensors, "selections": selections, "splits": splits, "out_dir": out_dir, "device": device}
