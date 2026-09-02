"""PyTorch dataset: delay rasters → lick rasters + 3-class label."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ExperimentConfig
from ..constants import CLASS_TO_ID, REGION_KEYS
from .catalog import TrialRecord
from .epochs import delay_window, lick_window
from .io_npz import load_trial_npz, pad_or_crop, split_by_region


def encode_trial(record: TrialRecord, cfg: ExperimentConfig) -> dict[str, Any] | None:
    data = load_trial_npz(record.path)
    d0, d1 = delay_window(data)
    l0, l1 = lick_window(data, cfg.epochs.lick_window_s, record.csv_row)
    delay = split_by_region(data, d0, d1, cfg.epochs.delay_bins)
    lick = split_by_region(data, l0, l1, cfg.epochs.lick_bins)
    n = cfg.model.units_per_region
    delay_stack = np.stack([pad_or_crop(delay[k]["raster"], n) for k in REGION_KEYS], axis=0)
    lick_stack = np.stack([pad_or_crop(lick[k]["raster"], n) for k in REGION_KEYS], axis=0)
    if delay_stack.sum() == 0:
        return None
    return {
        "delay": delay_stack.astype(np.float32),
        "lick": lick_stack.astype(np.float32),
        "label": CLASS_TO_ID[record.label],
        "dataset_id": 0 if record.dataset == "Data" else 1,
        "trial": record.trial,
        "session": record.session,
        "path": str(record.path),
        "unit_ids": {k: delay[k]["unit_ids"][:n] for k in REGION_KEYS},
    }


class DualDatasetRasterDataset(Dataset):
    def __init__(self, records: list[TrialRecord], cfg: ExperimentConfig):
        self.cfg = cfg
        self.items: list[dict[str, Any]] = []
        self.records: list[TrialRecord] = []
        for rec in records:
            encoded = encode_trial(rec, cfg)
            if encoded is None:
                continue
            self.items.append(encoded)
            self.records.append(rec)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        return {
            "delay": torch.from_numpy(item["delay"]),
            "lick": torch.from_numpy(item["lick"]),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "dataset_id": torch.tensor(item["dataset_id"], dtype=torch.long),
            "trial": item["trial"],
            "session": item["session"],
            "path": item["path"],
        }


def collate_trials(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "delay": torch.stack([b["delay"] for b in batch], dim=0),
        "lick": torch.stack([b["lick"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "dataset_id": torch.stack([b["dataset_id"] for b in batch], dim=0),
        "trial": [b["trial"] for b in batch],
        "session": [b["session"] for b in batch],
        "path": [b["path"] for b in batch],
    }
