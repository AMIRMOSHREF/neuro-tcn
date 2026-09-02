"""Training loop: lick-raster prediction + 3-class delay-to-action classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data.catalog import discover_trials
from ..data.dataset import DualDatasetRasterDataset, collate_trials
from ..features.spectral import trial_tf_maps
from ..models.spec_tcnn import SPECTCNN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def spec_loss(out: dict, batch: dict, cfg) -> dict[str, torch.Tensor]:
    pred = F.poisson_nll_loss(out["y_lick"], batch["lick"], log_input=False, reduction="mean")
    cls = F.cross_entropy(out["logits"], batch["label"])
    sparse = out["neuron_attn"].mean()
    # encourage peaked (not uniform) neuron attention
    entropy = -(out["neuron_attn"] * (out["neuron_attn"].clamp_min(1e-8).log())).sum(dim=-1).mean()
    total = (
        cfg.train.lambda_pred * pred
        + cfg.train.lambda_cls * cls
        + cfg.train.lambda_sparse * sparse
        + 0.02 * entropy
    )
    return {"total": total, "pred": pred.detach(), "cls": cls.detach(), "sparse": sparse.detach()}


@torch.no_grad()
def _maybe_tf(delay_np: np.ndarray, cfg) -> torch.Tensor:
    tf = trial_tf_maps(delay_np, cfg.epochs.bin_size, cfg.model.n_freq)
    return torch.from_numpy(tf)


def _step_batch(model, batch, cfg, device, compute_tf: bool):
    delay = batch["delay"].to(device)
    batch_dev = {
        "lick": batch["lick"].to(device),
        "label": batch["label"].to(device),
    }
    tf = None
    if compute_tf:
        maps = []
        for i in range(delay.size(0)):
            maps.append(_maybe_tf(delay[i].cpu().numpy(), cfg))
        tf = torch.stack(maps, dim=0).to(device)
    return model(delay, tf), batch_dev


def train_model(cfg, records=None, compute_tf: bool = False) -> dict:
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    if records is None:
        records = discover_trials(cfg)
    dataset = DualDatasetRasterDataset(records, cfg)
    if len(dataset) < 4:
        raise RuntimeError("Need at least 4 valid trials. Run python scripts/prepare_demo.py first.")
    n_val = max(1, int(len(dataset) * cfg.train.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.train.seed))
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_trials)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_trials)

    model = SPECTCNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    history = []
    best = {"val_acc": -1.0, "state": None}

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        run = {"total": 0.0, "pred": 0.0, "cls": 0.0, "n": 0}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}", leave=False):
            out, batch_dev = _step_batch(model, batch, cfg, device, compute_tf)
            losses = spec_loss(out, batch_dev, cfg)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = batch_dev["label"].size(0)
            run["total"] += float(losses["total"].detach()) * bs
            run["pred"] += float(losses["pred"]) * bs
            run["cls"] += float(losses["cls"]) * bs
            run["n"] += bs
        val = evaluate(model, val_loader, cfg, device, compute_tf)
        row = {
            "epoch": epoch,
            "train_loss": run["total"] / max(run["n"], 1),
            "train_pred": run["pred"] / max(run["n"], 1),
            "train_cls": run["cls"] / max(run["n"], 1),
            "val_acc": val["val_acc"],
            "val_pred": val["val_pred"],
            "n_eval": val["n_eval"],
        }
        history.append(row)
        if val["val_acc"] >= best["val_acc"]:
            best = {"val_acc": val["val_acc"], "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        print(
            f"epoch {epoch:02d}  loss={row['train_loss']:.3f}  "
            f"val_acc={val['val_acc']:.3f}  val_pred={val['val_pred']:.3f}"
        )

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": None, "history": history}, ckpt_dir / "spec_tcnn.pt")
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"model": model, "history": history, "n_train": n_train, "n_val": n_val, "device": str(device)}


@torch.no_grad()
def evaluate(model, loader, cfg, device, compute_tf: bool = False) -> dict:
    model.eval()
    correct = 0
    total = 0
    pred_loss = 0.0
    all_true = []
    all_pred = []
    attn_sum = None
    for batch in loader:
        out, batch_dev = _step_batch(model, batch, cfg, device, compute_tf)
        pred_c = out["logits"].argmax(dim=-1)
        correct += int((pred_c == batch_dev["label"]).sum())
        total += int(batch_dev["label"].numel())
        pred_loss += float(F.poisson_nll_loss(out["y_lick"], batch_dev["lick"], log_input=False, reduction="mean")) * batch_dev["label"].size(0)
        all_true.extend(batch_dev["label"].cpu().tolist())
        all_pred.extend(pred_c.cpu().tolist())
        attn = out["neuron_attn"].mean(dim=0).cpu()
        attn_sum = attn if attn_sum is None else attn_sum + attn
    acc = correct / max(total, 1)
    return {
        "val_acc": acc,
        "val_pred": pred_loss / max(total, 1),
        "n_eval": total,
        "y_true": all_true,
        "y_pred": all_pred,
        "mean_neuron_attn": None if attn_sum is None else (attn_sum / max(len(loader), 1)).numpy(),
    }


@torch.no_grad()
def collect_attention(model, dataset, cfg, device) -> dict[str, np.ndarray]:
    from ..constants import REGION_KEYS

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_trials)
    acc = None
    n = 0
    model.eval()
    for batch in loader:
        delay = batch["delay"].to(device)
        out = model(delay, None)
        a = out["neuron_attn"].sum(dim=0).cpu().numpy()
        acc = a if acc is None else acc + a
        n += delay.size(0)
    mean = acc / max(n, 1)
    return {key: mean[i] for i, key in enumerate(REGION_KEYS)}
