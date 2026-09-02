"""Load trial NPZ files and bin four-region rasters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..constants import REGIONS, REGION_TO_KEY


def _scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _times(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(value, dtype=object)
    if arr.dtype == object:
        flat: list[float] = []
        for item in arr.ravel():
            if item is None or (isinstance(item, float) and np.isnan(item)):
                continue
            if isinstance(item, str):
                item = item.strip()
                if not item or item.lower() in {"nan", "n/a", "none"}:
                    continue
                for part in item.replace(";", ",").split(","):
                    part = part.strip()
                    if part:
                        flat.append(float(part))
            else:
                try:
                    flat.append(float(item))
                except (TypeError, ValueError):
                    continue
        return np.asarray(flat, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).ravel()


def load_trial_npz(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as raw:
        data = {key: raw[key] for key in raw.files}
    data["path"] = str(path)
    data["trial_start"] = _scalar(data.get("trial_start", 0.0))
    data["trial_stop"] = _scalar(data.get("trial_stop", data["trial_start"] + 5.0))
    for key in (
        "presample_start_times",
        "presample_stop_times",
        "sample_start_times",
        "sample_stop_times",
        "delay_start_times",
        "delay_stop_times",
        "go_start_times",
        "go_stop_times",
    ):
        if key in data:
            data[key] = _scalar(data[key])
    data["left_lick_times"] = _times(data.get("left_lick_times"))
    data["right_lick_times"] = _times(data.get("right_lick_times"))
    data["unit_ids"] = np.asarray(data.get("unit_ids", np.arange(len(data.get("brain_region", [])))))
    data["brain_region"] = np.asarray(data.get("brain_region", []))
    spikes = np.asarray(data.get("spike_times", []), dtype=object)
    data["spike_times"] = np.array(
        [np.asarray(s, dtype=np.float64).ravel() if s is not None else np.asarray([], dtype=np.float64) for s in spikes],
        dtype=object,
    )
    return data


def normalize_region(label: str) -> str | None:
    text = str(label).strip()
    if text in REGION_TO_KEY:
        return REGION_TO_KEY[text]
    lowered = text.lower()
    for alias, key in REGION_TO_KEY.items():
        if alias.lower() == lowered:
            return key
    if "unknown" in lowered:
        return None
    if "alm" in lowered and "left" in lowered:
        return "left_ALM"
    if "alm" in lowered and "right" in lowered:
        return "right_ALM"
    if "str" in lowered and "left" in lowered:
        return "left_Striatum"
    if "str" in lowered and "right" in lowered:
        return "right_Striatum"
    return None


def bin_spikes(spike_times: np.ndarray, t0: float, t1: float, n_bins: int) -> np.ndarray:
    if n_bins <= 0 or not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.zeros((0,), dtype=np.float32)
    edges = np.linspace(t0, t1, n_bins + 1)
    spikes = np.asarray(spike_times, dtype=np.float64)
    spikes = spikes[(spikes >= t0) & (spikes < t1)]
    counts, _ = np.histogram(spikes, bins=edges)
    return counts.astype(np.float32)


def split_by_region(
    data: dict[str, Any],
    t0: float,
    t1: float,
    n_bins: int,
) -> dict[str, dict[str, Any]]:
    regions = np.asarray(data["brain_region"])
    spikes = data["spike_times"]
    unit_ids = np.asarray(data["unit_ids"])
    out: dict[str, dict[str, Any]] = {}
    for display in REGIONS:
        key = REGION_TO_KEY[display]
        mask = np.array([normalize_region(r) == key for r in regions], dtype=bool)
        idx = np.where(mask)[0]
        raster = np.zeros((len(idx), n_bins), dtype=np.float32)
        for row, unit_i in enumerate(idx):
            raster[row] = bin_spikes(spikes[unit_i], t0, t1, n_bins)
        out[key] = {
            "display": display,
            "unit_ids": unit_ids[idx] if len(idx) else np.asarray([], dtype=unit_ids.dtype),
            "indices": idx,
            "raster": raster,
        }
    return out


def pad_or_crop(raster: np.ndarray, n_units: int) -> np.ndarray:
    if raster.ndim != 2:
        raise ValueError("raster must be (units, time)")
    have = raster.shape[0]
    if have == n_units:
        return raster
    if have > n_units:
        return raster[:n_units]
    pad = np.zeros((n_units - have, raster.shape[1]), dtype=raster.dtype)
    return np.concatenate([raster, pad], axis=0)
