#!/usr/bin/env python
"""Export per-trial NPZ files (the `Data` schema: unit_ids + brain_region + spike_times for ALL units) from NWB.

Why this exists
---------------
The `Data2` NPZ export (`left_ALM_spikes`, ... object arrays) lists only the units that fired in each trial and
carries no unit IDs, so unit identity across trials is lost and no per-unit analysis is possible on it.  The `Data`
export (`Data/Session*/Rasters/<Class>/trial_<n>.npz`) keeps every unit of the session with its ID in every trial.
This script writes that schema for any NWB file with a `units` table and a `trials` table, into the Data2 folder
layout so that `python -m delaycast` picks it up unchanged:

    <out>/sub-<id>/<nwb stem>/NPZ/{Ignore,Left,Right}/trial<n>.npz

Labels come from the audited behavioural log when given (`--log behavioral_master_log_audited.csv`: observed lick side
after the go cue, Ignore when no lick, rows flagged `excluded` are skipped), otherwise from the NWB lick columns /
outcome.  Epoch scalars are copied from the trials table (plural or singular column names), lick times from the
trials table or the log row, region labels from the units table (`location` / `brain_region` / `unit_location` /
`electrode_group.location` / `electrodes -> location`).

Usage
-----
    python scripts/export_nwb_trials.py --nwb path/to/sub-440958_ses-20190214T123412_behavior+ecephys+image+ogen.nwb \
        --out C:/PythonProject/Rodent/Data2 --log path/to/behavioral_master_log_audited.csv
    python scripts/export_nwb_trials.py --nwb-dir path/with/many/nwb --out C:/PythonProject/Rodent/Data2_reexport

Requires `pip install pynwb`.  Run `python -m delaycast inspect --npz-detail` afterwards: the Dataset B block must
say "constant" unit counts (or "aligned by unit ID") and show lick times.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = ["presample_start_times", "presample_stop_times", "sample_start_times", "sample_stop_times",
          "delay_start_times", "delay_stop_times", "go_start_times", "go_stop_times"]
ALIASES = {"trial_start": ["start_time", "trial_start"], "trial_stop": ["stop_time", "trial_stop"]}
for _k in EPOCHS:
    ALIASES[_k] = [_k, _k[:-1], _k.replace("_times", "")]          # plural, singular, bare
REGION_COLUMNS = ["brain_region", "location", "unit_location", "anatomical_location", "region", "area"]


def _norm_region(label) -> str | None:
    s = str(label).strip().lower().replace("-", " ").replace("_", " ")
    area = "ALM" if "alm" in s else ("Striatum" if "str" in s else None)
    side = "left" if re.search(r"\bleft\b|\bl\b", s) else ("right" if re.search(r"\bright\b|\br\b", s) else None)
    return f"{side} {area}" if area and side else None


def _to_float(v) -> float:
    try:
        a = np.asarray(v, dtype=float).ravel()
        return float(a[0]) if a.size else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _times(v) -> np.ndarray:
    if v is None:
        return np.empty(0)
    if isinstance(v, str):
        parts = [p for p in re.split(r"[,;\s\[\]()]+", v.strip()) if p]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                pass
        return np.asarray(out, dtype=float)
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.empty(0)
    return a[np.isfinite(a)]


def unit_regions(nwb) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """(unit_ids, region label per unit, spike_times per unit) from the NWB units table."""
    units = nwb.units
    if units is None:
        raise SystemExit("this NWB file has no units table")
    df = units.to_dataframe()
    ids = np.asarray(df.index, dtype=np.int64)
    spikes = [np.asarray(st, dtype=float).ravel() for st in df["spike_times"]]
    labels = None
    for col in REGION_COLUMNS:
        if col in df.columns:
            labels = df[col].astype(str).to_numpy()
            break
    if labels is None and "electrode_group" in df.columns:
        labels = np.array([getattr(g, "location", "") for g in df["electrode_group"]], dtype=object)
    if labels is None and "electrodes" in df.columns:
        lab = []
        for e in df["electrodes"]:
            try:
                lab.append(str(e["location"].iloc[0]) if hasattr(e, "iloc") else str(e.location[0]))
            except Exception:
                lab.append("")
        labels = np.array(lab, dtype=object)
    if labels is None:
        raise SystemExit(f"no region column on the units table (looked for {REGION_COLUMNS}, electrode_group, electrodes)")
    canon = np.array([_norm_region(x) or "unknown" for x in labels], dtype=object)
    n_unknown = int((canon == "unknown").sum())
    if n_unknown:
        print(f"  {n_unknown} unit(s) with a region label that is not left/right ALM/Striatum: {sorted(set(labels[canon == 'unknown']))}")
    return ids, canon, spikes


def trial_table(nwb) -> pd.DataFrame:
    tr = nwb.trials if nwb.trials is not None else nwb.intervals.get("trials")
    if tr is None:
        raise SystemExit("this NWB file has no trials table")
    df = tr.to_dataframe().reset_index().rename(columns={"id": "trial_id"})
    df["trial"] = np.arange(1, len(df) + 1)     # 1-based like the behavioural logs / NPZ names
    return df


def epoch_scalars(row: pd.Series) -> dict[str, float]:
    out = {}
    for key, names in ALIASES.items():
        val = float("nan")
        for n in names:
            if n in row.index:
                val = _to_float(row[n])
                if np.isfinite(val):
                    break
        out[key] = val
    if np.isnan(out["go_start_times"]) and np.isfinite(out["delay_stop_times"]):
        out["go_start_times"] = out["delay_stop_times"]
    return out


def label_for(row: pd.Series, log_row: pd.Series | None, go: float) -> tuple[str | None, np.ndarray, np.ndarray, str]:
    """(class, left licks, right licks, note).  None = skip this trial."""
    src = log_row if log_row is not None else row
    if log_row is not None and str(log_row.get("excluded", "False")).lower() == "true":
        return None, np.empty(0), np.empty(0), f"excluded:{log_row.get('exclusion_reason', '')}"
    left = _times(src.get("left_lick_times")) if "left_lick_times" in src.index else np.empty(0)
    right = _times(src.get("right_lick_times")) if "right_lick_times" in src.index else np.empty(0)
    if left.size + right.size == 0 and log_row is None and "left_lick_times" not in row.index:
        outcome = str(row.get("outcome", "")).lower()
        instr = str(row.get("trial_instruction", "")).lower()
        if outcome == "ignore":
            return "Ignore", left, right, "label from outcome"
        if outcome == "hit" and instr in ("left", "right"):
            return instr.capitalize(), left, right, "label from outcome+instruction"
        if outcome == "miss" and instr in ("left", "right"):
            return ("Right" if instr == "left" else "Left"), left, right, "label from outcome+instruction"
        return None, left, right, "no lick record and no usable outcome"
    l_post, r_post = bool(np.any(left >= go)), bool(np.any(right >= go))
    if l_post and r_post:
        return None, left, right, "licked both sides"
    return ("Left" if l_post else "Right" if r_post else "Ignore"), left, right, "label from licks"


def export_one(nwb_path: Path, out_root: Path, log_path: Path | None, window_pre: float, window_post: float) -> None:
    from pynwb import NWBHDF5IO
    print(f"{nwb_path.name}")
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        ids, regions, spikes = unit_regions(nwb)
        trials = trial_table(nwb)
        subject = getattr(nwb.subject, "subject_id", None) or "unknown"
    log = None
    if log_path is not None and log_path.is_file():
        log = pd.read_csv(log_path, low_memory=False)
        if "trial" not in log.columns:
            log = None
            print("  log has no `trial` column; labels come from the NWB")
        else:
            log = log.set_index("trial")
    stem = nwb_path.stem
    sess_dir = out_root / f"sub-{subject}" / stem
    keep = regions != "unknown"
    ids_k, regions_k, spikes_k = ids[keep], regions[keep], [s for s, k in zip(spikes, keep) if k]
    counts = {"Ignore": 0, "Left": 0, "Right": 0}
    skipped: dict[str, int] = {}
    for _, row in trials.iterrows():
        ep = epoch_scalars(row)
        if not (np.isfinite(ep["delay_start_times"]) and np.isfinite(ep["go_start_times"])):
            skipped["no delay/go epoch"] = skipped.get("no delay/go epoch", 0) + 1
            continue
        log_row = log.loc[int(row["trial"])] if (log is not None and int(row["trial"]) in log.index) else None
        if isinstance(log_row, pd.DataFrame):
            log_row = log_row.iloc[0]
        cls, left, right, note = label_for(row, log_row, ep["go_start_times"])
        if cls is None:
            skipped[note] = skipped.get(note, 0) + 1
            continue
        t0 = ep["trial_start"] if np.isfinite(ep["trial_start"]) else ep["delay_start_times"] - window_pre
        t1 = ep["trial_stop"] if np.isfinite(ep["trial_stop"]) else ep["go_start_times"] + window_post
        t0, t1 = min(t0, ep["delay_start_times"] - window_pre), max(t1, ep["go_start_times"] + window_post)
        st = np.empty(len(spikes_k), dtype=object)
        for j, s in enumerate(spikes_k):
            st[j] = s[(s >= t0) & (s < t1)]
        payload = {"unit_ids": ids_k, "brain_region": regions_k.astype(str), "spike_times": st,
                   "trial_start": np.float64(t0), "trial_stop": np.float64(t1),
                   **{k: np.float64(v) for k, v in ep.items() if k not in ("trial_start", "trial_stop")},
                   "left_lick_times": left, "right_lick_times": right,
                   "trial_uid": np.int64(row["trial"]), "source_nwb": str(nwb_path.name)}
        d = sess_dir / "NPZ" / cls
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"trial{int(row['trial'])}.npz", **payload)
        counts[cls] += 1
    if log is not None:
        log.reset_index().to_csv(sess_dir / "behavioral_master_log_audited.csv", index=False)
    print(f"  -> {sess_dir}: {counts} ({len(ids_k)} units in every trial; skipped {skipped or 'none'})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nwb", type=Path, nargs="*", default=[], help="NWB file(s)")
    ap.add_argument("--nwb-dir", type=Path, default=None, help="export every *.nwb under this folder")
    ap.add_argument("--out", type=Path, required=True, help="output root (the Data2-style tree is created inside)")
    ap.add_argument("--log", type=Path, default=None, help="audited behavioural log CSV for a single --nwb (labels + lick times)")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="folder with <nwb stem>/behavioral_master_log_audited.csv per session (e.g. the existing Data2 root)")
    ap.add_argument("--window-pre", type=float, default=2.0, help="seconds before delay onset kept in spike_times (default 2)")
    ap.add_argument("--window-post", type=float, default=2.0, help="seconds after the go cue kept (default 2)")
    args = ap.parse_args(argv)
    try:
        import pynwb  # noqa: F401
    except ImportError:
        sys.exit("pynwb is required: pip install pynwb")
    files = list(args.nwb)
    if args.nwb_dir is not None:
        files += sorted(args.nwb_dir.rglob("*.nwb"))
    if not files:
        sys.exit("no NWB files given (--nwb or --nwb-dir)")
    for f in files:
        log = args.log
        if log is None and args.log_dir is not None:
            hits = list(args.log_dir.rglob(f"{f.stem}/behavioral_master_log_audited.csv")) + \
                   list(args.log_dir.rglob(f"{f.stem}/behavioral_master_log.csv"))
            log = hits[0] if hits else None
        export_one(f, args.out, log, args.window_pre, args.window_post)


if __name__ == "__main__":
    main()
