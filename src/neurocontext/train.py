from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from .data import RasterDataset, TrialRecord, collate_trials
from .model import ContextForecaster, multitask_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_folds(records: list[TrialRecord], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([record.group for record in records])
    labels = np.asarray([record.label for record in records])
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two independent session groups are required")
    folds = min(n_splits, len(unique_groups))
    return list(GroupKFold(folds).split(np.zeros(len(records)), labels, groups))


@torch.no_grad()
def evaluate(
    model: ContextForecaster, loader: DataLoader, device: torch.device, config: dict
) -> dict:
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    losses: list[float] = []
    forecast_losses: list[float] = []
    for batch in loader:
        delay = batch["delay"].to(device)
        response = batch["response"].to(device)
        mask = batch["unit_mask"].to(device)
        labels = batch["label"].to(device)
        output = model(delay, mask)
        _, parts = multitask_loss(output, response, labels, mask, config)
        losses.append(parts["loss"])
        forecast_losses.append(parts["forecast_loss"])
        truth.extend(labels.cpu().tolist())
        predicted.extend(output["logits"].argmax(dim=1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "poisson_nll": float(np.mean(forecast_losses)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "n_trials": len(truth),
    }


def train_fold(
    records: list[TrialRecord],
    config: dict,
    output_dir: str | Path,
    fold_index: int = 0,
    device_name: str | None = None,
) -> dict:
    seed_everything(int(config["seed"]) + fold_index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx = grouped_folds(records, int(config["training"]["folds"]))[fold_index]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "collate_fn": collate_trials,
    }
    train_loader = DataLoader(
        RasterDataset(train_records, config), shuffle=True, **loader_args
    )
    val_loader = DataLoader(RasterDataset(val_records, config), shuffle=False, **loader_args)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ContextForecaster(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best = float("inf")
    stale = 0
    history: list[dict] = []
    checkpoint = output_dir / f"fold_{fold_index}.pt"

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        epoch_parts: defaultdict[str, list[float]] = defaultdict(list)
        for batch in train_loader:
            delay = batch["delay"].to(device)
            response = batch["response"].to(device)
            mask = batch["unit_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(delay, mask)
            loss, parts = multitask_loss(output, response, labels, mask, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            for key, value in parts.items():
                epoch_parts[key].append(value)
        validation = evaluate(model, val_loader, device, config)
        row = {
            "epoch": epoch,
            **{f"train_{k}": float(np.mean(v)) for k, v in epoch_parts.items()},
            **{f"val_{k}": v for k, v in validation.items()},
        }
        history.append(row)
        if validation["loss"] < best:
            best = validation["loss"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "fold": fold_index,
                    "train_groups": sorted({r.group for r in train_records}),
                    "validation_groups": sorted({r.group for r in val_records}),
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    metrics = evaluate(model, val_loader, device, config)
    metrics.update(
        {
            "fold": fold_index,
            "checkpoint": str(checkpoint),
            "train_groups": state["train_groups"],
            "validation_groups": state["validation_groups"],
        }
    )
    (output_dir / f"fold_{fold_index}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / f"fold_{fold_index}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
