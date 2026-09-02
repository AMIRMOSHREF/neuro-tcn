"""Discover trials from Data (session/Rasters) and Data2 (sub-*/NPZ) trees."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ExperimentConfig
from .labels import (
    action_class_from_licks,
    folder_class,
    index_data2_rows,
    load_master_logs,
    parse_lick_list,
    trial_should_keep,
)

TRIAL_RE = re.compile(r"trial[_-]?(\d+)\.npz$", re.IGNORECASE)


@dataclass
class TrialRecord:
    path: Path
    dataset: str
    session: str
    subject: str
    trial: int
    label: str
    source: str
    csv_row: dict[str, Any] | None = field(default=None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dataset": self.dataset,
            "session": self.session,
            "subject": self.subject,
            "trial": self.trial,
            "label": self.label,
            "source": self.source,
        }


def _trial_number(path: Path) -> int | None:
    match = TRIAL_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _iter_class_npz(root: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    if not root.exists():
        return found
    for cls in ("Ignore", "Left", "Right"):
        folder = root / cls
        if not folder.is_dir():
            continue
        for npz in sorted(folder.glob("*.npz")):
            found.append((npz, cls))
    if found:
        return found
    for npz in sorted(root.rglob("*.npz")):
        cls = folder_class(npz)
        if cls:
            found.append((npz, cls))
    return found


def _discover_data(root: Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    if not root.exists():
        return records
    sessions = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("session")])
    if not sessions:
        sessions = [root]
    for session in sessions:
        rasters = session / "Rasters"
        search = rasters if rasters.is_dir() else session
        for npz, cls in _iter_class_npz(search):
            trial = _trial_number(npz)
            if trial is None:
                continue
            records.append(
                TrialRecord(
                    path=npz,
                    dataset="Data",
                    session=session.name,
                    subject=session.name,
                    trial=trial,
                    label=cls,
                    source="folder",
                )
            )
    return records


def _session_keys(session_dir: Path) -> list[str]:
    keys = [session_dir.name]
    parent = session_dir.parent.name
    if parent.startswith("sub-"):
        keys.append(parent)
        digits = re.sub(r"[^0-9]", "", parent)
        if digits:
            keys.append(digits)
    return keys


def _discover_data2(root: Path, lookup: dict, cfg: ExperimentConfig) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    if not root.exists():
        return records
    session_dirs = [
        p
        for p in root.rglob("*")
        if p.is_dir() and "ses-" in p.name and (p / "NPZ").is_dir()
    ]
    if not session_dirs:
        session_dirs = [p for p in root.rglob("NPZ") if p.is_dir()]
        session_dirs = [p.parent for p in session_dirs]
    for session_dir in sorted(set(session_dirs)):
        npz_root = session_dir / "NPZ" if (session_dir / "NPZ").is_dir() else session_dir
        subject = session_dir.parent.name if session_dir.parent.name.startswith("sub-") else session_dir.name
        keys = _session_keys(session_dir)
        for npz, folder_lbl in _iter_class_npz(npz_root):
            trial = _trial_number(npz)
            if trial is None:
                continue
            row = None
            for key in keys + [f"{re.sub(r'[^0-9]', '', subject)}:{trial}"]:
                row = lookup.get((key, trial))
                if row is not None:
                    break
            label = folder_lbl
            source = "folder"
            if row is not None:
                if not trial_should_keep(row, cfg):
                    continue
                derived = action_class_from_licks(
                    parse_lick_list(row.get("left_lick_times")),
                    parse_lick_list(row.get("right_lick_times")),
                    outcome=str(row.get("outcome", "")),
                    instruction=str(row.get("trial_instruction", "")),
                    keep_miss_as_ignore=cfg.keep_miss_as_ignore,
                )
                if derived is None:
                    continue
                label = derived
                source = "csv+folder"
            records.append(
                TrialRecord(
                    path=npz,
                    dataset="Data2",
                    session=session_dir.name,
                    subject=subject,
                    trial=trial,
                    label=label,
                    source=source,
                    csv_row=None if row is None else row.to_dict(),
                )
            )
    return records


def discover_trials(cfg: ExperimentConfig) -> list[TrialRecord]:
    data_root = cfg.paths.data_root
    data2_root = cfg.paths.data2_root
    if cfg.prefer_demo_if_missing:
        real_data = data_root.exists() and any(data_root.rglob("*.npz"))
        real_data2 = data2_root.exists() and any(data2_root.rglob("*.npz"))
        if not real_data:
            demo_data = cfg.paths.demo_root / "Data"
            if demo_data.exists():
                data_root = demo_data
        if not real_data2:
            demo_data2 = cfg.paths.demo_root / "Data2"
            if demo_data2.exists():
                data2_root = demo_data2

    meta = cfg.paths.metadata_dir
    csv_candidates = [
        data2_root / "combined_audited_master_log.csv",
        data2_root / "combined_behavioral_master_log.csv",
        meta / "combined_audited_master_log.csv",
        meta / "combined_behavioral_master_log.csv",
        meta / "behavioral_master_log.csv",
    ]
    logs = load_master_logs(csv_candidates)
    lookup = index_data2_rows(logs) if not logs.empty else {}

    records = _discover_data(data_root) + _discover_data2(data2_root, lookup, cfg)
    records.sort(key=lambda r: (r.dataset, r.session, r.label, r.trial))
    return records


def catalog_frame(records: list[TrialRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in records])
