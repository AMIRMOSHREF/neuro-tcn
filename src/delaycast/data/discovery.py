"""Locate trials in the two on-disk layouts and join them with the behavioral CSV logs.

Dataset A  (``Data``):
    <root>/Session1/Rasters/{Ignore,Left,Right}/trial_32.npz
    <root>/Session1/Videos/{Ignore,Left,Right}/trial_0032_lick_left.avi
    No CSV log; every epoch timestamp is read from the NPZ itself.

Dataset B  (``Data2``):
    <root>/sub-440957/sub-440957_ses-20190211T143614_behavior+ecephys+image+ogen/NPZ/{Ignore,Left,Right}/trial2.npz
    <root>/sub-440957/sub-440957_ses-.../behavioral_master_log_audited.csv     (per-session, preferred)
    <root>/combined_audited_master_log.csv                                     (fallback, has ``session_dir``)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import CLASSES

_CLASS_DIRS = {c.lower(): c for c in CLASSES}
_TRIAL_RE = re.compile(r"trial[_-]?0*(\d+)", re.IGNORECASE)


@dataclass
class TrialRecord:
    dataset: str            # "A" or "B"
    session: str            # unique session key, e.g. "A/Session1" or "B/sub-440957_ses-20190211T143614"
    subject: str
    trial: int
    label: str              # Ignore | Left | Right (from folder name)
    npz_path: Path
    video_path: Path | None = None
    csv: dict = field(default_factory=dict)   # matching behavioral log row (Dataset B), may be empty


def parse_trial_number(name: str) -> int | None:
    m = _TRIAL_RE.search(name)
    return int(m.group(1)) if m else None


def _class_dirs(parent: Path) -> dict[str, Path]:
    out = {}
    if not parent.is_dir():
        return out
    for d in parent.iterdir():
        if d.is_dir() and d.name.lower() in _CLASS_DIRS:
            out[_CLASS_DIRS[d.name.lower()]] = d
    return out


def _index_videos(video_root: Path) -> dict[tuple[str, int], Path]:
    idx: dict[tuple[str, int], Path] = {}
    for label, d in _class_dirs(video_root).items():
        for p in d.iterdir():
            if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv"):
                n = parse_trial_number(p.stem)
                if n is not None:
                    idx[(label, n)] = p
    return idx


# --------------------------------------------------------------------------- Dataset A
def discover_dataset_a(root: str | Path) -> list[TrialRecord]:
    root = Path(root)
    records: list[TrialRecord] = []
    if not root.is_dir():
        return records
    for sess_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        raster_root = sess_dir / "Rasters"
        if not raster_root.is_dir():
            continue
        videos = _index_videos(sess_dir / "Videos")
        for label, cdir in _class_dirs(raster_root).items():
            for npz in sorted(cdir.glob("*.npz")):
                n = parse_trial_number(npz.stem)
                if n is None:
                    continue
                records.append(
                    TrialRecord(
                        dataset="A",
                        session=f"A/{sess_dir.name}",
                        subject=sess_dir.name,
                        trial=n,
                        label=label,
                        npz_path=npz,
                        video_path=videos.get((label, n)),
                    )
                )
    return records


# --------------------------------------------------------------------------- Dataset B
def _load_session_csv(sess_dir: Path, combined: pd.DataFrame | None) -> pd.DataFrame | None:
    for name in ("behavioral_master_log_audited.csv", "behavioral_master_log.csv"):
        p = sess_dir / name
        if p.is_file():
            return pd.read_csv(p, low_memory=False)
    if combined is not None and "session_dir" in combined.columns:
        sub = combined[combined["session_dir"] == sess_dir.name]
        if len(sub):
            return sub.copy()
    return None


def _load_combined(root: Path) -> pd.DataFrame | None:
    for name in ("combined_audited_master_log.csv", "combined_behavioral_master_log.csv"):
        p = root / name
        if p.is_file():
            return pd.read_csv(p, low_memory=False)
    return None


def discover_dataset_b(root: str | Path) -> list[TrialRecord]:
    root = Path(root)
    records: list[TrialRecord] = []
    if not root.is_dir():
        return records
    combined = _load_combined(root)
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("sub-")):
        for sess_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir() and "ses-" in p.name):
            npz_root = sess_dir / "NPZ"
            if not npz_root.is_dir():
                continue
            csv = _load_session_csv(sess_dir, combined)
            csv_by_trial: dict[int, dict] = {}
            if csv is not None and "trial" in csv.columns:
                for _, row in csv.iterrows():
                    try:
                        csv_by_trial[int(row["trial"])] = row.to_dict()
                    except (TypeError, ValueError):
                        continue
            videos = _index_videos(sess_dir / "Video") | _index_videos(sess_dir / "Videos")
            short = sess_dir.name.split("_behavior")[0]
            for label, cdir in _class_dirs(npz_root).items():
                for npz in sorted(cdir.glob("*.npz")):
                    n = parse_trial_number(npz.stem)
                    if n is None:
                        continue
                    records.append(
                        TrialRecord(
                            dataset="B",
                            session=f"B/{short}",
                            subject=subj_dir.name,
                            trial=n,
                            label=label,
                            npz_path=npz,
                            video_path=videos.get((label, n)),
                            csv=csv_by_trial.get(n, {}),
                        )
                    )
    return records


def discover_all(cfg) -> list[TrialRecord]:
    recs: list[TrialRecord] = []
    if cfg.data.use_dataset_a:
        recs += discover_dataset_a(cfg.data.data_a_root)
    if cfg.data.use_dataset_b:
        recs += discover_dataset_b(cfg.data.data_b_root)
    return recs


def summarize(records: Iterable[TrialRecord]) -> pd.DataFrame:
    rows = [
        {"dataset": r.dataset, "session": r.session, "label": r.label, "has_csv": bool(r.csv), "has_video": r.video_path is not None}
        for r in records
    ]
    if not rows:
        return pd.DataFrame(columns=["dataset", "session", "Ignore", "Left", "Right", "n_trials"])
    df = pd.DataFrame(rows)
    tab = df.pivot_table(index=["dataset", "session"], columns="label", values="has_csv", aggfunc="size", fill_value=0)
    for c in CLASSES:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[list(CLASSES)]
    tab["n_trials"] = tab.sum(axis=1)
    tab["with_csv"] = df.groupby(["dataset", "session"])["has_csv"].sum()
    tab["with_video"] = df.groupby(["dataset", "session"])["has_video"].sum()
    return tab.reset_index()
