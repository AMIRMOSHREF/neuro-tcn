from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_TO_INDEX = {"Ignore": 0, "Left": 1, "Right": 2}
TRIAL_RE = re.compile(r"trial[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class TrialRecord:
    npz_path: Path
    dataset: str
    session: str
    subject: str
    trial: int
    label: str

    @property
    def group(self) -> str:
        return f"{self.dataset}:{self.subject}:{self.session}"


def _trial_number(path: Path) -> int:
    match = TRIAL_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse trial number from {path.name}")
    return int(match.group(1))


def _canonical_label(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"ignore": "Ignore", "left": "Left", "right": "Right"}
    if normalized not in aliases:
        raise ValueError(f"Unknown class label: {value}")
    return aliases[normalized]


def _read_metadata(paths: list[Path]) -> dict[tuple[str, int], str]:
    """Map (session basename, trial) to audited behavioral class."""
    result: dict[tuple[str, int], str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                excluded = row.get("excluded", "").strip().lower()
                if excluded == "true":
                    continue
                if row.get("early_lick", "").strip().lower() == "early":
                    continue
                outcome = row.get("outcome", "").strip().lower()
                if outcome == "miss":
                    continue
                if outcome == "ignore":
                    label = "Ignore"
                elif outcome == "hit":
                    instruction = row.get("trial_instruction", "")
                    if instruction.strip().lower() not in {"left", "right"}:
                        continue
                    label = _canonical_label(instruction)
                else:
                    continue
                session = row.get("session_dir") or Path(row.get("session_path", "")).stem
                try:
                    trial = int(float(row["trial"]))
                except (KeyError, TypeError, ValueError):
                    continue
                result[(session, trial)] = label
    return result


def _session_ancestor(npz_path: Path) -> str:
    for parent in npz_path.parents:
        name = parent.name
        if "_ses-" in name or name.lower().startswith("session"):
            return name
    return npz_path.parent.parent.name


def discover_trials(
    data_root: str | Path | None,
    data2_root: str | Path | None,
    metadata_csvs: list[str | Path] | None = None,
) -> list[TrialRecord]:
    """Discover both folder layouts and apply audited Data2 labels when available."""
    records: list[TrialRecord] = []
    metadata = _read_metadata([Path(p) for p in (metadata_csvs or [])])

    if data_root:
        root = Path(data_root)
        for path in sorted(root.glob("Session*/Rasters/*/trial*.npz")):
            label = _canonical_label(path.parent.name)
            session = _session_ancestor(path)
            records.append(
                TrialRecord(path, "Data", session, session, _trial_number(path), label)
            )

    if data2_root:
        root = Path(data2_root)
        patterns = ("sub-*/**/NPZ/*/trial*.npz", "sub-*/**/Rasters/*/trial*.npz")
        seen: set[Path] = set()
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                trial = _trial_number(path)
                session = _session_ancestor(path)
                label = metadata.get((session, trial))
                if label is None:
                    try:
                        label = _canonical_label(path.parent.name)
                    except ValueError:
                        continue
                subject = next(
                    (p.name for p in path.parents if p.name.startswith("sub-")), session
                )
                records.append(TrialRecord(path, "Data2", session, subject, trial, label))
    return records


def _scalar(npz: np.lib.npyio.NpzFile, key: str) -> float:
    value = np.asarray(npz[key]).reshape(-1)
    if not len(value):
        raise ValueError(f"Empty timestamp: {key}")
    return float(value[0])


def _resample_counts(
    spike_times: np.ndarray, start: float, stop: float, bins: int
) -> np.ndarray:
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError(f"Invalid epoch [{start}, {stop}]")
    edges = np.linspace(start, stop, bins + 1)
    return np.histogram(np.asarray(spike_times, dtype=float), bins=edges)[0].astype(np.float32)


def canonical_region(raw: object, aliases: dict[str, str]) -> str | None:
    value = str(raw).strip().lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    direct = {k.lower().replace("-", " "): v for k, v in aliases.items()}
    if value in direct:
        return direct[value]
    hemisphere = "left" if "left" in value else "right" if "right" in value else None
    area = "ALM" if "alm" in value else "Striatum" if ("str" in value) else None
    return f"{hemisphere} {area}" if hemisphere and area else None


def load_trial(record: TrialRecord, config: dict) -> dict[str, torch.Tensor | str]:
    """Load one trial as [region, unit, time], preserving unit masks and IDs."""
    max_units = int(config["max_units_per_region"])
    delay_bins = int(config["delay_bins"])
    response_bins = int(config["response_bins"])
    regions = config["regions"]
    delay = np.zeros((4, max_units, delay_bins), dtype=np.float32)
    response = np.zeros((4, max_units, response_bins), dtype=np.float32)
    mask = np.zeros((4, max_units), dtype=bool)
    ids = np.full((4, max_units), "", dtype="U64")

    with np.load(record.npz_path, allow_pickle=True) as npz:
        d0, d1 = _scalar(npz, "delay_start_times"), _scalar(npz, "delay_stop_times")
        # The response/lick epoch is go_start→go_stop. It is not aligned to the
        # first lick, which would leak class information for Ignore trials.
        r0, r1 = _scalar(npz, "go_start_times"), _scalar(npz, "go_stop_times")
        unit_ids = np.asarray(npz["unit_ids"]).astype(str)
        raw_regions = np.asarray(npz["brain_region"])
        spike_times = np.asarray(npz["spike_times"], dtype=object)

        for r_idx, wanted in enumerate(regions):
            candidates = [
                i
                for i, raw in enumerate(raw_regions)
                if canonical_region(raw, config.get("region_aliases", {})) == wanted
            ][:max_units]
            for row, unit_idx in enumerate(candidates):
                delay[r_idx, row] = _resample_counts(
                    spike_times[unit_idx], d0, d1, delay_bins
                )
                response[r_idx, row] = _resample_counts(
                    spike_times[unit_idx], r0, r1, response_bins
                )
                mask[r_idx, row] = True
                ids[r_idx, row] = unit_ids[unit_idx]

    if not mask.any():
        raise ValueError(f"No recognized units in {record.npz_path}")
    return {
        "delay": torch.from_numpy(delay),
        "response": torch.from_numpy(response),
        "unit_mask": torch.from_numpy(mask),
        "label": torch.tensor(CLASS_TO_INDEX[record.label], dtype=torch.long),
        "unit_ids": ids,
        "path": str(record.npz_path),
        "group": record.group,
    }


class RasterDataset(Dataset):
    def __init__(self, records: list[TrialRecord], config: dict):
        self.records = records
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return load_trial(self.records[index], self.config)


def collate_trials(samples: list[dict]) -> dict:
    tensor_keys = ("delay", "response", "unit_mask", "label")
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    batch["unit_ids"] = [sample["unit_ids"] for sample in samples]
    batch["path"] = [sample["path"] for sample in samples]
    batch["group"] = [sample["group"] for sample in samples]
    return batch


def valid_trials(records: list[TrialRecord], config: dict) -> Iterator[TrialRecord]:
    """Yield readable records; useful for a preflight audit before training."""
    for record in records:
        try:
            load_trial(record, config)
        except (OSError, KeyError, ValueError):
            continue
        yield record
