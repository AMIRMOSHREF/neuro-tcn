"""Map folder names and Data2 CSVs onto Ignore / Left / Right actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_lick_list(value: Any) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.asarray([], dtype=np.float64)
    if isinstance(value, (list, tuple, np.ndarray)):
        out = []
        for item in np.asarray(value, dtype=object).ravel():
            out.extend(parse_lick_list(item).tolist())
        return np.asarray(out, dtype=np.float64)
    text = str(value).strip()
    if text.lower() in {"", "nan", "n/a", "none"}:
        return np.asarray([], dtype=np.float64)
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return np.asarray([float(p) for p in parts], dtype=np.float64)


def action_class_from_licks(
    left_licks: np.ndarray,
    right_licks: np.ndarray,
    outcome: str | None = None,
    instruction: str | None = None,
    keep_miss_as_ignore: bool = False,
) -> str | None:
    has_left = len(left_licks) > 0
    has_right = len(right_licks) > 0
    if has_left and has_right:
        return None
    outcome_l = (outcome or "").strip().lower()
    if outcome_l == "ignore" or (not has_left and not has_right):
        return "Ignore"
    if has_left:
        cls = "Left"
    elif has_right:
        cls = "Right"
    else:
        return "Ignore"
    if keep_miss_as_ignore and outcome_l == "miss":
        return "Ignore"
    if instruction and outcome_l == "miss":
        # Miss = instructed side was not licked. Keep observed action if present.
        return cls
    return cls


def folder_class(path: str | Path) -> str | None:
    parts = [p.lower() for p in Path(path).parts]
    for name in ("ignore", "left", "right"):
        if name in parts:
            return name.title()
    return None


def _truthy_excluded(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def trial_should_keep(row: pd.Series, cfg) -> bool:
    if "excluded" in row.index and _truthy_excluded(row.get("excluded")):
        return False
    if cfg.exclude_early_lick:
        early = str(row.get("early_lick", "no early")).strip().lower()
        if early and early not in {"no early", "nan", "none", ""}:
            return False
    if cfg.exclude_photostim:
        onset = str(row.get("photostim_onset", "N/A")).strip().lower()
        if onset not in {"n/a", "nan", "none", ""}:
            return False
        pstart = row.get("photostim_start_times", np.nan)
        if pd.notna(pstart) and str(pstart).strip() not in {"", "nan", "n/a"}:
            return False
    if cfg.exclude_auto_water:
        for col in ("auto_water", "free_water"):
            if col in row.index:
                try:
                    if float(row.get(col, 0) or 0) > 0:
                        return False
                except (TypeError, ValueError):
                    pass
    if cfg.exclude_both_licks:
        left = parse_lick_list(row.get("left_lick_times"))
        right = parse_lick_list(row.get("right_lick_times"))
        if len(left) and len(right):
            return False
    return True


def load_master_logs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        df = pd.read_csv(path)
        df["__source_csv"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def index_data2_rows(df: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    """Map (session_dir or session stem, trial) -> row."""
    lookup: dict[tuple[str, int], pd.Series] = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        trial = row.get("trial", np.nan)
        if pd.isna(trial):
            continue
        trial_i = int(trial)
        keys = []
        session_dir = str(row.get("session_dir", "") or "")
        session_path = str(row.get("session_path", "") or "")
        if session_dir:
            keys.append(session_dir)
        if session_path:
            keys.append(Path(session_path).stem.replace(".nwb", ""))
            keys.append(session_path)
        subject = str(row.get("subject_id", "") or "")
        if subject:
            keys.append(f"{subject}:{trial_i}")
        for key in keys:
            lookup[(key, trial_i)] = row
    return lookup
