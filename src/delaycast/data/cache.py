"""Per-session tensor cache: bin every trial once and store aligned arrays on disk."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import CLASSES, CLASS_TO_IDX, REGIONS
from .discovery import TrialRecord, discover_all
from .rasters import label_from_licks, load_trial_rasters

log = logging.getLogger(__name__)


@dataclass
class SessionCache:
    session: str
    dataset: str
    subject: str
    context: dict[str, np.ndarray]   # region -> (n_trials, n_units, T_ctx) uint8 spike counts
    target: dict[str, np.ndarray]    # region -> (n_trials, n_units, T_tgt) uint8 spike counts
    unit_ids: dict[str, np.ndarray]  # region -> (n_units,)
    labels: np.ndarray               # (n_trials,) int class index
    trials: np.ndarray               # (n_trials,) trial numbers
    meta: pd.DataFrame               # one row per trial (paths, qc flags, csv info)
    bin_ms: float
    target_bin_ms: float

    @property
    def n_trials(self) -> int:
        return len(self.labels)

    def class_counts(self) -> dict[str, int]:
        return {c: int((self.labels == i).sum()) for i, c in enumerate(CLASSES)}

    @property
    def n_units(self) -> dict[str, int]:
        return {r: int(self.context[r].shape[1]) for r in REGIONS}

    def nbytes(self) -> int:
        return int(sum(self.context[r].nbytes + self.target[r].nbytes for r in REGIONS))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"labels": self.labels, "trials": self.trials}
        for r in REGIONS:
            arrays[f"ctx_{r}"] = as_counts_u8(self.context[r])
            arrays[f"tgt_{r}"] = as_counts_u8(self.target[r])
            arrays[f"uid_{r}"] = self.unit_ids[r]
        np.savez_compressed(path, **arrays)
        self.meta.to_csv(path.with_suffix(".meta.csv"), index=False)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"session": self.session, "dataset": self.dataset, "subject": self.subject,
                 "bin_ms": self.bin_ms, "target_bin_ms": self.target_bin_ms},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "SessionCache":
        z = np.load(path, allow_pickle=True)
        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        meta = pd.read_csv(path.with_suffix(".meta.csv"))
        return cls(
            session=info["session"], dataset=info["dataset"], subject=info["subject"],
            context={r: as_counts_u8(z[f"ctx_{r}"]) for r in REGIONS},
            target={r: as_counts_u8(z[f"tgt_{r}"]) for r in REGIONS},
            unit_ids={r: z[f"uid_{r}"] for r in REGIONS},
            labels=z["labels"], trials=z["trials"], meta=meta,
            bin_ms=info["bin_ms"], target_bin_ms=info["target_bin_ms"],
        )


def as_counts_u8(a: np.ndarray) -> np.ndarray:
    """Spike counts per bin as uint8 (a 10 ms / 50 ms bin never holds > 255 spikes); 4x less RAM than float32.

    Every consumer does arithmetic that promotes (sum -> uint64, mean / division -> float64), so the dtype is
    transparent; model tensors are cast to float32 for the selected K units only.
    """
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _csv_flags(rec: TrialRecord, cfg) -> tuple[bool, str]:
    """Apply the behavioral-log exclusion rules of Dataset B. Returns (keep, reason)."""
    row = rec.csv
    if not row:
        if cfg.data.qc.require_csv_row:
            return False, "no_csv_row"
        return True, ""
    if str(row.get("excluded", "False")).lower() == "true":
        return False, f"csv_excluded:{row.get('exclusion_reason', '')}"
    if str(row.get("early_lick", "no early")).strip().lower() == "early":
        return False, "csv_early_lick"
    if cfg.data.qc.csv_exclude_photostim:
        ps = str(row.get("photostim_onset", "nan")).strip().lower()
        if ps not in ("nan", "n/a", "", "none"):
            return False, "csv_photostim"
    if cfg.data.qc.csv_exclude_auto_water:
        try:
            if float(row.get("auto_water", 0) or 0) > 0 or float(row.get("free_water", 0) or 0) > 0:
                return False, "csv_auto_or_free_water"
        except (TypeError, ValueError):
            pass
    outcome = str(row.get("outcome", "")).strip().lower()
    if outcome not in ("hit", "ignore", ""):
        return False, f"csv_outcome_{outcome}"
    return True, ""


def _cache_key(cfg) -> str:
    c = cfg.data
    return f"bin{c.bin_ms}_tbin{c.target_bin_ms}_ctx{int(c.context.include_sample)}_{c.context.pre_delay_ms}_resp{c.target.response_ms}"


def build_cache(cfg, force: bool = False) -> list[SessionCache]:
    """Discover, QC and bin every trial; one compressed NPZ per session under ``cache_dir``."""
    cache_dir = Path(cfg.data.cache_dir) / _cache_key(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = discover_all(cfg)
    if not records:
        raise FileNotFoundError(
            f"No trials found under data_a_root={cfg.data.data_a_root!r} / data_b_root={cfg.data.data_b_root!r}"
        )
    by_session: dict[str, list[TrialRecord]] = {}
    for r in records:
        by_session.setdefault(r.session, []).append(r)

    caches: list[SessionCache] = []
    qc_rows = []
    for sess, recs in by_session.items():
        out_path = cache_dir / (sess.replace("/", "__") + ".npz")
        if out_path.exists() and not force:
            caches.append(SessionCache.load(out_path))
            continue
        ctx: dict[str, list] = {r: [] for r in REGIONS}
        tgt: dict[str, list] = {r: [] for r in REGIONS}
        uids: dict[str, np.ndarray | None] = {r: None for r in REGIONS}
        labels, trials, meta = [], [], []
        n_ctx = n_tgt = None
        for rec in tqdm(sorted(recs, key=lambda x: x.trial), desc=f"binning {sess}", leave=False):
            keep, reason = _csv_flags(rec, cfg)
            try:
                tr = load_trial_rasters(rec.npz_path, cfg, metadata=rec.csv)
            except Exception as e:  # corrupted / incomplete NPZ
                qc_rows.append({"session": sess, "trial": rec.trial, "label": rec.label, "kept": False, "reason": f"load_error:{e}"})
                continue
            if keep and cfg.data.qc.drop_early_lick and tr.qc["early_lick"]:
                keep, reason = False, "npz_early_lick"
            implied = label_from_licks(tr.qc)
            if keep and cfg.data.qc.drop_label_mismatch and implied is not None and implied != rec.label:
                keep, reason = False, f"label_mismatch(folder={rec.label},licks={implied})"
            if keep and implied is None and cfg.data.qc.drop_label_mismatch:
                keep, reason = False, "licked_both_sides"
            qc_rows.append({"session": sess, "trial": rec.trial, "label": rec.label, "kept": keep, "reason": reason,
                            "n_unknown_region": tr.qc["n_unknown_region"], "delay_len_s": tr.qc["delay_len_s"]})
            if not keep:
                continue
            if n_ctx is None:
                n_ctx = tr.context[REGIONS[0]].shape[1]
                n_tgt = tr.target[REGIONS[0]].shape[1]
            for r in REGIONS:
                cx, tg = tr.context[r], tr.target[r]
                # Guard against +-1 bin differences caused by float epoch times.
                cx = _fit_len(cx, n_ctx)
                tg = _fit_len(tg, n_tgt)
                if uids[r] is None:
                    uids[r] = tr.unit_ids[r]
                elif len(tr.unit_ids[r]) != len(uids[r]):
                    raise ValueError(f"{sess}: unit count changed within session for {r} ({rec.npz_path})")
                ctx[r].append(cx)
                tgt[r].append(tg)
            labels.append(CLASS_TO_IDX[rec.label])
            trials.append(rec.trial)
            meta.append({"trial": rec.trial, "label": rec.label, "npz_path": str(rec.npz_path),
                         "video_path": str(rec.video_path) if rec.video_path else "",
                         "first_lick_s": _first_lick(tr), **{f"ep_{k}": v for k, v in tr.epochs.items()}})
        if not labels:
            log.warning("session %s has no trials after QC", sess)
            continue
        sc = SessionCache(
            session=sess, dataset=recs[0].dataset, subject=recs[0].subject,
            context={r: as_counts_u8(np.stack(ctx[r])) if ctx[r] else np.zeros((len(labels), 0, n_ctx), np.uint8) for r in REGIONS},
            target={r: as_counts_u8(np.stack(tgt[r])) if tgt[r] else np.zeros((len(labels), 0, n_tgt), np.uint8) for r in REGIONS},
            unit_ids={r: (uids[r] if uids[r] is not None else np.zeros(0)) for r in REGIONS},
            labels=np.asarray(labels), trials=np.asarray(trials), meta=pd.DataFrame(meta),
            bin_ms=cfg.data.bin_ms, target_bin_ms=cfg.data.target_bin_ms,
        )
        sc.save(out_path)
        caches.append(sc)
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(cache_dir / "qc_log.csv", index=False)
    return caches


def _fit_len(a: np.ndarray, n: int) -> np.ndarray:
    if a.shape[1] == n:
        return a
    if a.shape[1] > n:
        return a[:, :n]
    return np.pad(a, ((0, 0), (0, n - a.shape[1])))


def _first_lick(tr) -> float:
    go = tr.epochs["go_start_times"]
    licks = np.concatenate([tr.lick_left, tr.lick_right])
    licks = licks[licks >= go]
    return float(licks.min() - go) if licks.size else float("nan")


def load_cache(cfg) -> list[SessionCache]:
    cache_dir = Path(cfg.data.cache_dir) / _cache_key(cfg)
    paths = sorted(cache_dir.glob("*.npz"))
    if not paths:
        return build_cache(cfg)
    return [SessionCache.load(p) for p in paths]


def cache_summary(caches: list[SessionCache]) -> pd.DataFrame:
    rows = []
    for c in caches:
        row = {"session": c.session, "dataset": c.dataset, "n_trials": c.n_trials, **c.class_counts()}
        row.update({f"units_{r}": c.context[r].shape[1] for r in REGIONS})
        row["T_ctx"] = c.context[REGIONS[0]].shape[2]
        row["T_tgt"] = c.target[REGIONS[0]].shape[2]
        row["MB"] = round(c.nbytes() / 1e6, 1)
        rows.append(row)
    return pd.DataFrame(rows)
