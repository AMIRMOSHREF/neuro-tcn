"""Per-session tensor cache: bin every trial once and store aligned arrays on disk."""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import CLASSES, CLASS_TO_IDX, REGIONS
from .discovery import TrialRecord, discover_all
from .rasters import label_from_licks, load_trial_rasters, population_rasters, unit_ids_by_region

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
    qc_info: dict = field(default_factory=dict)  # build statistics stored next to the arrays (length fixes, drops)

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
                 "bin_ms": self.bin_ms, "target_bin_ms": self.target_bin_ms, **self.qc_info},
                f, indent=2, default=_json_default,
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
            qc_info={k: v for k, v in info.items() if k not in _JSON_CORE_KEYS},
        )


_JSON_CORE_KEYS = ("session", "dataset", "subject", "bin_ms", "target_bin_ms")


def _json_default(o):
    """Serialise numpy scalars that end up in the QC statistics."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def as_counts_u8(a: np.ndarray) -> np.ndarray:
    """Spike counts per bin as uint8 (a 10 ms / 50 ms bin never holds > 255 spikes); 4x less RAM than float32.

    Every consumer does arithmetic that promotes (sum -> uint64, mean / division -> float64), so the dtype is
    transparent; model tensors are cast to float32 for the selected K units only.
    """
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if a.size and np.nanmax(a) > 255:
        # 255 spikes in one 10 ms bin is a 25 kHz rate: physically impossible for one unit, so this only
        # happens when a raster is not a single-unit count (e.g. a population sum) - flag it, do not silently clip.
        log.warning("as_counts_u8: %d bin(s) exceed 255 spikes (max %.0f) and are clipped to 255",
                    int((a > 255).sum()), float(np.nanmax(a)))
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
    keep_outcomes = cfg.data.qc.get_path("csv_keep_outcomes", ["hit", "miss", "ignore"])
    keep_outcomes = {str(o).strip().lower() for o in (keep_outcomes if isinstance(keep_outcomes, (list, tuple)) else str(keep_outcomes).split(","))}
    if outcome not in keep_outcomes and outcome not in ("", "nan"):
        return False, f"csv_outcome_{outcome}"
    return True, ""


LOADER_VERSION = 6
"""Bumped on EVERY change of the trial loader / QC rules: it is part of the cache key, so an old cache can never be
read by a newer loader (v2: unit alignment by ID, log lick times, miss trials kept; v3: empty NPZ lick arrays defer to
the log, sessions without unit identity excluded)."""


def _identity_path(cfg, session: str) -> Path:
    return Path(cfg.data.cache_dir) / _cache_key(cfg) / "identity" / (session.replace("/", "__") + ".pkl")


def identity_settings(cfg) -> dict:
    """Keyword arguments of ``recover_session_identity`` from ``data.identity_*`` (features: rate / windows / isi)."""
    from .identity import FEATURE_SETS
    feats = cfg.data.get_path("identity_features", list(FEATURE_SETS))
    if isinstance(feats, str):
        feats = [f.strip() for f in feats.replace("+", ",").split(",") if f.strip()]
    return {"features": tuple(str(f).lower() for f in feats),
            "n_iter": int(cfg.data.get_path("identity_iterations", 5)),
            "min_support": int(cfg.data.get_path("identity_min_support", 3)),
            "support_frac": float(cfg.data.get_path("identity_support_frac", 0.03)),
            "p_insert": float(cfg.data.get_path("identity_p_insert", 1e-4)),
            "prune": bool(cfg.data.get_path("identity_prune", True))}


def identity_tag(cfg) -> str:
    """Short cache-key tag of the identity settings (a different recovery is a different cache)."""
    st = identity_settings(cfg)
    return (f"id{''.join(f[0] for f in st['features'])}s{100 * st['support_frac']:g}p{-np.log10(st['p_insert']):g}"
            + ("" if st["prune"] else "np"))


def _recover_identity(cfg, session: str, recs):
    """Recovered identity map of one session, from disk when it exists (keyed with the cache)."""
    from .identity import IdentityMap, recover_session_identity
    path = _identity_path(cfg, session)
    if path.is_file():
        try:
            return IdentityMap.load(path)
        except Exception:  # pragma: no cover
            pass
    st = identity_settings(cfg)
    log.info("%s: recovering unit identity by sequence alignment over %d trials (features %s) ...", session, len(recs),
             "+".join(st["features"]))
    ident = recover_session_identity(recs, session=session, **st)
    ident.save(path)
    return ident


def twin_identity_validation(caches: list, dup: pd.DataFrame, cfg, records=None) -> pd.DataFrame:
    """Recovered-identity accuracy of every Data2 twin against its Data copy (see identity.twin_identity_accuracy)."""
    from .identity import IdentityMap, twin_identity_accuracy
    if records is None:
        records = discover_all(cfg)
    by_session: dict[str, list] = {}
    for r in records:
        by_session.setdefault(r.session, []).append(r)
    rows = []
    for _, d in dup.iterrows():
        path = _identity_path(cfg, d.session_b)
        if not path.is_file() or d.session_a not in by_session or d.session_b not in by_session:
            continue
        ident = IdentityMap.load(path)
        acc = twin_identity_accuracy(ident, by_session[d.session_b], by_session[d.session_a])
        rows.append({"session_b": d.session_b, "n_slots": ident.stats.get("n_slots_total"),
                     "n_slots_added": ident.stats.get("n_slots_added"), "n_slots_pruned": ident.stats.get("n_slots_pruned", 0),
                     "n_slots_merged": ident.stats.get("n_slots_merged", 0), **acc})
    return pd.DataFrame(rows)


# Recovery settings compared by `delaycast identity` on the twin recordings (label -> overrides of identity_settings).
IDENTITY_SWEEP = [
    ("rate", {"features": ("rate",)}),
    ("rate+windows", {"features": ("rate", "windows")}),
    ("rate+windows+isi", {"features": ("rate", "windows", "isi")}),
    ("rate+windows+isi, support 5%", {"features": ("rate", "windows", "isi"), "support_frac": 0.05}),
    ("rate+windows+isi, p_insert 1e-3", {"features": ("rate", "windows", "isi"), "p_insert": 1e-3}),
    ("rate+windows+isi, no prune", {"features": ("rate", "windows", "isi"), "prune": False}),
]


def identity_sweep(caches: list, dup: pd.DataFrame, cfg, records=None, settings=None, sessions=None) -> pd.DataFrame:
    """Recover the identity of every Data2 twin under several settings and score each against the true IDs of the
    Data copy (``twin_identity_accuracy``) - the table that decides which fingerprint the recovery should use.

    ``settings``: list of ``(label, overrides)`` (default ``IDENTITY_SWEEP`` plus the configured setting);
    ``sessions``: substrings selecting the twin sessions (default all).  Nothing is written to the cache."""
    from .identity import recover_session_identity, twin_identity_accuracy
    if records is None:
        records = discover_all(cfg)
    by_session: dict[str, list] = {}
    for r in records:
        by_session.setdefault(r.session, []).append(r)
    base = identity_settings(cfg)
    if settings is None:
        settings = list(IDENTITY_SWEEP)
        if all(dict(base, **ov) != base for _, ov in settings):
            settings.append(("configured", {}))
    rows = []
    for _, d in dup.iterrows():
        if d.session_a not in by_session or d.session_b not in by_session:
            continue
        if sessions and not any(s in d.session_b for s in sessions):
            continue
        recs_b = sorted(by_session[d.session_b], key=lambda x: x.trial)
        for label, ov in settings:
            st = dict(base, **ov)
            log.info("%s: identity sweep '%s' (%s) ...", d.session_b, label, ", ".join(f"{k}={v}" for k, v in st.items()))
            ident = recover_session_identity(recs_b, session=d.session_b, **st)
            acc = twin_identity_accuracy(ident, recs_b, by_session[d.session_a])
            per = ident.stats.get("per_region", {})
            sd_ref = {}
            for r in REGIONS:
                for k, v in (per.get(r, {}).get("sd_ref") or {}).items():
                    sd_ref.setdefault(k, []).append(v)
            rows.append({"setting": label, "session_b": d.session_b, "n_slots": ident.stats.get("n_slots_total"),
                         "n_slots_added": ident.stats.get("n_slots_added"), "n_slots_pruned": ident.stats.get("n_slots_pruned", 0),
                         "n_slots_merged": ident.stats.get("n_slots_merged", 0), "seconds": ident.stats.get("seconds"), **acc,
                         "sd_ref": " ".join(f"{k.replace('log_rate', 'rate').replace('shape_', 's_').replace('isi_', 'i_')}={np.mean(v):.2f}" for k, v in sd_ref.items())})
    return pd.DataFrame(rows)


def representation(cfg) -> str:
    """``units`` (default: one row per recorded unit, identity by unit_ids) or ``population`` (identity-free
    rate-quantile channels per region, see :func:`delaycast.data.rasters.population_rasters`)."""
    rep = str(cfg.data.get_path("representation", "units")).lower()
    if rep not in ("units", "population"):
        raise ValueError(f"data.representation must be 'units' or 'population', got {rep!r}")
    return rep


def _cache_key(cfg) -> str:
    c = cfg.data
    key = (f"bin{c.bin_ms}_tbin{c.target_bin_ms}_ctx{int(c.context.include_sample)}_{c.context.pre_delay_ms}"
           f"_resp{c.target.response_ms}_v{LOADER_VERSION}")
    if representation(cfg) == "population":
        key += f"_pop{int(c.get_path('population_groups', 8))}"
    elif bool(c.get_path("recover_identity", True)):
        key += "_" + identity_tag(cfg)
    return key


# A session whose NPZs carry no unit identity (pre-split arrays without IDs) and whose unit count changes from
# trial to trial cannot be used: nothing says which row of trial t is which row of trial t+1, and every per-unit
# statistic would silently mix units.  Above this fraction of mismatching trials the session is excluded at cache time.
MAX_POSITIONAL_MISMATCH_FRAC = 0.2
UNIT_IDENTITY_FIX = ("the NPZ export lists only the units that fired in each trial and carries no unit_ids, so unit "
                     "identity across trials is lost; re-export the session from its NWB file with `unit_ids` + "
                     "`brain_region` + `spike_times` for ALL units (scripts/export_nwb_trials.py, or the exporter that "
                     "produced Data/Session*), then delete the cache")


def _unit_universe(recs: list[TrialRecord]) -> tuple[dict[str, np.ndarray] | None, dict]:
    """First pass over a session: the union of unit IDs per region across all its trials (order of first appearance).

    Reads only ``unit_ids`` / ``brain_region`` (cheap).  Returns ``(None, info)`` when the NPZs carry no usable
    IDs (pre-split schema, or duplicated IDs), in which case rows are aligned by position and a trial whose unit
    count differs from the session's is dropped.  ``info`` reports how many units each trial contributed versus
    the union, which is how an "active units only" export (Data2) shows up: the union is larger than any
    single trial.
    """
    universe: dict[str, list] = {r: [] for r in REGIONS}
    seen: dict[str, set] = {r: set() for r in REGIONS}
    presence: dict[str, dict] = {r: {} for r in REGIONS}
    per_trial = []
    for rec in recs:
        try:
            ids, note = unit_ids_by_region(rec.npz_path)
        except Exception:  # unreadable file: reported again (with the reason) by the binning pass
            continue
        if ids is None:
            log.warning("%s: units cannot be aligned by ID (%s in %s); using position, trials whose unit count "
                        "differs from the first kept trial will be dropped", recs[0].session, note, rec.npz_path.name)
            return None, {"unit_alignment": "positional", "unit_alignment_note": note}
        n_tr = 0
        for r in REGIONS:
            for u in np.asarray(ids[r]).ravel().tolist():
                if u not in seen[r]:
                    seen[r].add(u)
                    universe[r].append(u)
                presence[r][u] = presence[r].get(u, 0) + 1
            n_tr += len(ids[r])
        per_trial.append(n_tr)
    if not per_trial:
        return None, {"unit_alignment": "positional", "unit_alignment_note": "no readable NPZ"}
    out = {r: np.asarray(universe[r]) for r in REGIONS}
    n_union = int(sum(len(v) for v in out.values()))
    info = {"unit_alignment": "id", "unit_alignment_note": "ok", "units_union": n_union, "units_per_trial_median": float(np.median(per_trial)),
            "units_per_trial_min": int(np.min(per_trial)), "units_per_trial_max": int(np.max(per_trial)),
            "unit_presence_median": {r: (float(np.median(list(presence[r].values())) / len(recs)) if presence[r] else np.nan) for r in REGIONS}}
    return out, info


def build_cache(cfg, force: bool = False) -> list[SessionCache]:
    """Discover, QC and bin every trial; one compressed NPZ per session under ``cache_dir``.

    Memory model: after the first kept trial of a session the per-region uint8 tensors are allocated once for
    all trials of the session (upper bound = number of discovered trials, trimmed at the end) and every later
    trial is written into its row.  No float32 intermediate and no ``np.stack`` copy is ever made, so a
    2000-unit x 350-trial session costs ~100 MB while it is being built instead of ~4x that.

    Trials are dropped with a reason in ``qc_log.csv`` instead of aborting the whole cache: a delay whose
    length deviates from ``data.context.delay_ms`` by more than ``data.qc.max_delay_dev_ms``
    (``delay_len_<x>s``; the behavioural logs guarantee 1.2 s, anything else is an epoch-extraction error),
    and a trial whose unit count in any region differs from the first kept trial of the session
    (``unit_count_mismatch:<region>:<n>!=<n_ref>``; unit identity is positional inside a session, so such a
    trial cannot be aligned to the others).  Rasters that are +-1 bin off the session length because of float
    epoch times are fixed by ``_fit_len`` and counted in ``qc_info['length_fixes']`` of the cache JSON.
    """
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
    delay_ms = float(cfg.data.context.get_path("delay_ms", 1200))
    max_dev_ms = float(cfg.data.qc.get_path("max_delay_dev_ms", 15))

    caches: list[SessionCache] = []
    qc_rows = []
    excluded_rows = []
    for sess, recs in by_session.items():
        out_path = cache_dir / (sess.replace("/", "__") + ".npz")
        if out_path.exists() and not force:
            caches.append(SessionCache.load(out_path))
            continue
        recs = sorted(recs, key=lambda x: x.trial)
        pop = representation(cfg) == "population"
        n_groups = int(cfg.data.get_path("population_groups", 8))
        ident = None
        if pop:   # identity-free channels: no unit universe, no positional identity, nothing to align
            unit_index, align_info = None, {"unit_alignment": "population", "unit_alignment_note": f"{n_groups} rate-quantile channels per region"}
        else:
            unit_index, align_info = _unit_universe(recs)
            if unit_index is None and bool(cfg.data.get_path("recover_identity", True)) and len(recs) >= 10:
                # No usable unit IDs (the Data2 export): recover them by sequence alignment - the export lists
                # every session's units in one fixed order minus the silent ones (see data/identity.py).
                ident = _recover_identity(cfg, sess, recs)
                unit_index = {r: np.arange(ident.n_slots[r], dtype=np.int64) for r in REGIONS}
                align_info = {"unit_alignment": "recovered", "units_union": int(ident.stats["n_slots_total"]),
                              "units_per_trial_min": int(min(len(v[REGIONS[0]]) for v in ident.rows.values())) if ident.rows else 0,
                              "units_per_trial_max": int(max(len(v[REGIONS[0]]) for v in ident.rows.values())) if ident.rows else 0,
                              "unit_alignment_note": (f"identity recovered by sequence alignment: {ident.stats['n_slots_total']} slots "
                                                      f"({ident.stats['n_slots_added']} added beyond the longest trial), "
                                                      f"{100 * ident.stats['frac_rows_assigned']:.1f}% of rows assigned"),
                              "identity_stats": ident.stats}
                log.info("%s: %s", sess, align_info["unit_alignment_note"])
        if unit_index is not None and align_info["units_per_trial_min"] < align_info["units_union"]:
            log.info("%s: units aligned by ID; %d units in the session, %d-%d present per trial (absent units = silent, zero rows)",
                     sess, align_info["units_union"], align_info["units_per_trial_min"], align_info["units_per_trial_max"])
        ctx: dict[str, np.ndarray | None] = {r: None for r in REGIONS}
        tgt: dict[str, np.ndarray | None] = {r: None for r in REGIONS}
        uids: dict[str, np.ndarray | None] = {r: None for r in REGIONS}
        labels, trials, meta = [], [], []
        n_ctx = n_tgt = None
        n_kept = 0
        len_fix = {"context": 0, "target": 0, "context_max_abs_bins": 0, "target_max_abs_bins": 0}
        drop_reasons: dict[str, int] = {}
        lick_sources: dict[str, int] = {}
        lick_labels: dict[str, int] = {}
        for rec in tqdm(recs, desc=f"binning {sess}", leave=False):
            keep, reason = _csv_flags(rec, cfg)
            try:
                if ident is not None and int(rec.trial) not in ident.rows:
                    raise ValueError("no recovered identity for this trial (unreadable during recovery)")
                tr = load_trial_rasters(rec.npz_path, cfg, metadata=rec.csv, unit_index=unit_index,
                                        row_ids=ident.rows[int(rec.trial)] if ident is not None else None)
                if pop:
                    tr = population_rasters(tr, n_groups)
            except Exception as e:  # corrupted / incomplete NPZ
                reason = f"load_error:{e}"
                qc_rows.append({"session": sess, "trial": rec.trial, "label": rec.label, "kept": False, "reason": reason})
                drop_reasons[reason.split(":")[0]] = drop_reasons.get(reason.split(":")[0], 0) + 1
                continue
            delay_len_s = float(tr.qc["delay_len_s"])
            if keep and abs(delay_len_s * 1000.0 - delay_ms) > max_dev_ms:
                keep, reason = False, f"delay_len_{delay_len_s:.2f}s"
            if keep and cfg.data.qc.drop_early_lick and tr.qc["early_lick"]:
                keep, reason = False, f"early_lick_{tr.qc.get('lick_source', 'npz')}"
            implied = label_from_licks(tr.qc)   # None = no lick record anywhere (nothing to check the folder against)
            lick_sources[tr.qc.get("lick_source", "?")] = lick_sources.get(tr.qc.get("lick_source", "?"), 0) + 1
            lick_labels[implied or "unknown"] = lick_labels.get(implied or "unknown", 0) + 1
            if keep and cfg.data.qc.drop_label_mismatch and implied == "Both":
                keep, reason = False, "licked_both_sides"
            if keep and cfg.data.qc.drop_label_mismatch and implied is not None and implied != rec.label:
                keep, reason = False, f"label_mismatch(folder={rec.label},licks={implied})"
            if keep and unit_index is None and n_kept > 0:
                # Unit identity is positional inside a session: a trial with a different unit count in any
                # region cannot be aligned to the tensors already filled, so it is dropped (not the session).
                for r in REGIONS:
                    n_r, n_ref = len(tr.unit_ids[r]), len(uids[r])
                    if n_r != n_ref:
                        keep, reason = False, f"unit_count_mismatch:{r}:{n_r}!={n_ref}"
                        if drop_reasons.get("unit_count_mismatch", 0) < 3:
                            log.warning("%s trial %s: %s (trial dropped; further mismatches of this session are only counted)", sess, rec.trial, reason)
                        break
            qc_rows.append({"session": sess, "trial": rec.trial, "label": rec.label, "kept": keep, "reason": reason,
                            "n_unknown_region": tr.qc["n_unknown_region"], "delay_len_s": delay_len_s,
                            "lick_source": tr.qc.get("lick_source", ""), "lick_label": implied or "",
                            "csv_outcome": str(rec.csv.get("outcome", "")) if rec.csv else ""})
            if not keep:
                key = reason.split(":")[0].split("(")[0]
                key = "delay_len" if key.startswith("delay_len_") else key
                drop_reasons[key] = drop_reasons.get(key, 0) + 1
                continue
            if n_kept == 0:
                # First kept trial fixes the session geometry; allocate the uint8 tensors once (upper bound
                # on the trial count, trimmed below) and write every later trial straight into its row.
                n_ctx = tr.context[REGIONS[0]].shape[1]
                n_tgt = tr.target[REGIONS[0]].shape[1]
                for r in REGIONS:
                    uids[r] = tr.unit_ids[r]
                    ctx[r] = np.zeros((len(recs), len(tr.unit_ids[r]), n_ctx), dtype=np.uint8)
                    tgt[r] = np.zeros((len(recs), len(tr.unit_ids[r]), n_tgt), dtype=np.uint8)
            fixed_ctx = fixed_tgt = False
            for r in REGIONS:
                cx, tg = tr.context[r], tr.target[r]
                # Guard against +-1 bin differences caused by float epoch times.  The context is anchored at
                # the go cue (its last bin), the target at the go cue (its first bin).
                if cx.shape[1] != n_ctx:
                    fixed_ctx = True
                    len_fix["context_max_abs_bins"] = max(len_fix["context_max_abs_bins"], abs(cx.shape[1] - n_ctx))
                    cx = _fit_len(cx, n_ctx, align="right")
                if tg.shape[1] != n_tgt:
                    fixed_tgt = True
                    len_fix["target_max_abs_bins"] = max(len_fix["target_max_abs_bins"], abs(tg.shape[1] - n_tgt))
                    tg = _fit_len(tg, n_tgt, align="left")
                ctx[r][n_kept] = as_counts_u8(cx)
                tgt[r][n_kept] = as_counts_u8(tg)
            len_fix["context"] += int(fixed_ctx)
            len_fix["target"] += int(fixed_tgt)
            n_kept += 1
            labels.append(CLASS_TO_IDX[rec.label])
            trials.append(rec.trial)
            meta.append({"trial": rec.trial, "label": rec.label, "npz_path": str(rec.npz_path),
                         "video_path": str(rec.video_path) if rec.video_path else "",
                         "first_lick_s": _first_lick(tr), "lick_source": tr.qc.get("lick_source", ""),
                         "spike_ref": tr.qc.get("spike_time_reference", ""),
                         "csv_outcome": str(rec.csv.get("outcome", "")) if rec.csv else "",
                         "csv_instruction": str(rec.csv.get("trial_instruction", "")) if rec.csv else "",
                         **({f"pooled_{r}": v for r, v in tr.qc.get("n_units_pooled", {}).items()} if pop else {}),
                         **{f"ep_{k}": v for k, v in tr.epochs.items()}})
        n_ctx_spikes = int(sum(int(ctx[r][:n_kept].astype(np.int64).sum()) for r in REGIONS if ctx[r] is not None)) if n_kept else 0
        if n_kept and n_ctx_spikes == 0:
            # Every kept trial is empty in the delay window: the spikes were on a time base the loader could not
            # resolve, or the files hold no spikes.  Training on silence would be meaningless.
            refs = dict(Counter(m.get("spike_ref", "") for m in meta))
            log.error("session %s EXCLUDED: no spike in the context window of any of its %d kept trials (spike time "
                      "references seen: %s) - see `inspect --npz-detail` (spike range vs epochs)", sess, n_kept, refs)
            excluded_rows.append({"session": sess, "n_discovered": len(recs), "n_unit_count_mismatch": 0,
                                  "reason": "no_spikes_in_window",
                                  "fix": "the spike times of the trial files are not on the time base of the epoch scalars "
                                         "(absolute seconds); `inspect --npz-detail` prints the spike range and the epochs",
                                  "alignment_note": f"spike time references: {refs}"})
            continue
        n_mismatch = drop_reasons.get("unit_count_mismatch", 0)
        if unit_index is None and not pop and len(recs) >= 10 and n_mismatch > MAX_POSITIONAL_MISMATCH_FRAC * len(recs):
            log.error("session %s EXCLUDED: unit count differs from the first trial's in %d of %d trials and %s",
                      sess, n_mismatch, len(recs), UNIT_IDENTITY_FIX)
            excluded_rows.append({"session": sess, "n_discovered": len(recs), "n_unit_count_mismatch": n_mismatch,
                                  "reason": "unit_identity_unavailable", "fix": UNIT_IDENTITY_FIX,
                                  "alignment_note": align_info.get("unit_alignment_note", "")})
            continue
        if not labels:
            log.warning("session %s has no trials after QC (%s)", sess,
                        ", ".join(f"{k}: {v}" for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1])) or "no trials discovered")
            continue
        if n_kept < 0.5 * len(recs):
            log.warning("session %s: only %d of %d discovered trials survive QC (%s); lick record %s implies %s", sess, n_kept, len(recs),
                        ", ".join(f"{k}: {v}" for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1])), lick_sources, lick_labels)
        if len_fix["context"] or len_fix["target"]:
            log.info("%s: raster length fixed for %d context / %d target trial(s) (max |delta| %d / %d bins)", sess,
                     len_fix["context"], len_fix["target"], len_fix["context_max_abs_bins"], len_fix["target_max_abs_bins"])
        sc = SessionCache(
            session=sess, dataset=recs[0].dataset, subject=recs[0].subject,
            context={r: np.ascontiguousarray(ctx[r][:n_kept]) for r in REGIONS},
            target={r: np.ascontiguousarray(tgt[r][:n_kept]) for r in REGIONS},
            unit_ids={r: (uids[r] if uids[r] is not None else np.zeros(0)) for r in REGIONS},
            labels=np.asarray(labels), trials=np.asarray(trials), meta=pd.DataFrame(meta),
            bin_ms=cfg.data.bin_ms, target_bin_ms=cfg.data.target_bin_ms,
            qc_info={"length_fixes": len_fix, "n_discovered": len(recs), "n_kept": n_kept,
                     "n_dropped": len(recs) - n_kept, "drop_reasons": drop_reasons,
                     "delay_ms": delay_ms, "max_delay_dev_ms": max_dev_ms, **align_info,
                     "representation": "population" if pop else "units",
                     "spike_time_references": dict(Counter(m.get("spike_ref", "") for m in meta)),
                     "units_pooled": ({r: int(np.median([m.get(f"pooled_{r}", 0) for m in meta])) for r in REGIONS} if pop else None),
                     "lick_sources": lick_sources, "lick_labels": lick_labels},
        )
        sc.save(out_path)
        caches.append(sc)
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(cache_dir / "qc_log.csv", index=False)
    if excluded_rows or not (cache_dir / "excluded_sessions.csv").exists():
        pd.DataFrame(excluded_rows, columns=["session", "n_discovered", "n_unit_count_mismatch", "reason", "fix", "alignment_note"]) \
            .to_csv(cache_dir / "excluded_sessions.csv", index=False)
    return caches


def excluded_sessions(cfg) -> pd.DataFrame:
    """Sessions the cache builder refused (with the reason and the fix); empty when none."""
    p = Path(cfg.data.cache_dir) / _cache_key(cfg) / "excluded_sessions.csv"
    if not p.is_file():
        return pd.DataFrame(columns=["session", "n_discovered", "n_unit_count_mismatch", "reason", "fix", "alignment_note"])
    return pd.read_csv(p)


def _fit_len(a: np.ndarray, n: int, align: str = "right") -> np.ndarray:
    """Force the time axis (axis 1) of a raster to exactly ``n`` bins.

    ``align="right"`` (context window): keep the LAST ``n`` bins when longer, left-pad zeros when shorter,
    so that the final bin always ends at the go cue - the anchor every criterion (late-delay windows, the
    causal model, the forecasting target) is defined against.  ``align="left"`` (target window): keep the
    FIRST ``n`` bins / right-pad, because the response epoch starts at the go cue.
    """
    L = a.shape[1]
    if L == n:
        return a
    if align == "right":
        return a[:, L - n:] if L > n else np.pad(a, ((0, 0), (n - L, 0)))
    if align == "left":
        return a[:, :n] if L > n else np.pad(a, ((0, 0), (0, n - L)))
    raise ValueError(f"align must be 'right' or 'left', got {align!r}")


def _first_lick(tr) -> float:
    go = tr.epochs["go_start_times"]
    licks = np.concatenate([tr.lick_left, tr.lick_right])
    licks = licks[licks >= go]
    return float(licks.min() - go) if licks.size else float("nan")


def find_duplicate_sessions(caches: list[SessionCache], tol_s: float = 0.002, min_overlap: float = 0.9, keep: str = "A") -> pd.DataFrame:
    """Sessions that appear in both ``Data`` (dataset A) and ``Data2`` (dataset B).

    The two on-disk trees were extracted from the same NWB recordings, so a session can be present twice.
    Two sessions are the same recording when their trials' absolute delay-onset timestamps coincide (within
    ``tol_s``) for at least ``min_overlap`` of the trials of the smaller session - a fingerprint that cannot
    match by chance. Training on both copies would leak trials between the training and test sets, and
    "cross-dataset transfer" would be tested on the training recordings, so ``load_cache`` drops one copy
    (``keep`` = ``data.duplicate_keep``: ``A`` keeps the Data copy - complete unit table with IDs and NPZ lick times,
    identical class labels - ``B`` the Data2 copy with the audited behavioural log) unless
    ``data.drop_duplicate_sessions`` is false.  The table lists the unit counts of both copies because the two exports
    curate units differently.
    """
    rows = []
    a_caches = [c for c in caches if c.dataset == "A"]
    b_caches = [c for c in caches if c.dataset == "B"]
    for a in a_caches:
        ta = _delay_onsets(a)
        for b in b_caches:
            tb = _delay_onsets(b)
            if ta.size == 0 or tb.size == 0:
                continue
            if abs(a.n_trials - b.n_trials) > 0.1 * max(a.n_trials, b.n_trials):
                continue
            tb_sorted = np.sort(tb)
            pos = np.searchsorted(tb_sorted, ta)
            near = np.minimum(np.abs(ta - tb_sorted[np.clip(pos, 0, len(tb_sorted) - 1)]),
                              np.abs(ta - tb_sorted[np.clip(pos - 1, 0, len(tb_sorted) - 1)]))
            overlap = float((near <= tol_s).mean())
            if overlap >= min_overlap:
                rows.append({"session_a": a.session, "session_b": b.session, "n_trials_a": a.n_trials, "n_trials_b": b.n_trials,
                             "units_a": int(sum(a.n_units.values())), "units_b": int(sum(b.n_units.values())),
                             "overlap": overlap, "dropped": b.session if keep == "A" else a.session})
    return pd.DataFrame(rows, columns=["session_a", "session_b", "n_trials_a", "n_trials_b", "units_a", "units_b", "overlap", "dropped"])


def duplicate_channel_agreement(caches: list[SessionCache], dup: pd.DataFrame, tol_s: float = 0.002) -> pd.DataFrame:
    """Population representation only: for every duplicate pair, the fraction of matched trials (same delay-onset
    timestamp) whose context and target channels are *identical* in the ``Data`` and the ``Data2`` export.

    The two exports list different unit sets (every unit vs. the units that fired), in different orders, with
    different epoch key names; the rate-quantile channels are designed to be a function of the multiset of non-zero
    unit rows only, so a real recording present in both trees is the direct check that the representation - and the
    binning behind it - agrees across the two exports.  Anything below 1.0 points at a curation difference between
    the exports (units dropped by one of them), not at a bug in the channels."""
    by = {c.session: c for c in caches}
    rows = []
    for _, d in dup.iterrows():
        a, b = by.get(d.session_a), by.get(d.session_b)
        if a is None or b is None:
            continue
        ta = pd.to_numeric(a.meta.get("ep_delay_start_times"), errors="coerce").to_numpy(dtype=float)
        tb = pd.to_numeric(b.meta.get("ep_delay_start_times"), errors="coerce").to_numpy(dtype=float)
        n_match, n_same_ctx, n_same_tgt, n_same_label = 0, 0, 0, 0
        for i, t in enumerate(ta):
            if not np.isfinite(t):
                continue
            j = np.flatnonzero(np.abs(tb - t) <= tol_s)
            if j.size != 1:
                continue
            j = int(j[0])
            n_match += 1
            n_same_ctx += all(np.array_equal(a.context[r][i], b.context[r][j]) for r in REGIONS)
            n_same_tgt += all(np.array_equal(a.target[r][i], b.target[r][j]) for r in REGIONS)
            n_same_label += int(a.labels[i] == b.labels[j])
        rows.append({"session_a": d.session_a, "session_b": d.session_b, "n_matched_trials": n_match,
                     "frac_identical_context": n_same_ctx / n_match if n_match else float("nan"),
                     "frac_identical_target": n_same_tgt / n_match if n_match else float("nan"),
                     "frac_same_label": n_same_label / n_match if n_match else float("nan")})
    return pd.DataFrame(rows, columns=["session_a", "session_b", "n_matched_trials", "frac_identical_context",
                                       "frac_identical_target", "frac_same_label"])


def twin_unit_order_check(caches: list[SessionCache], dup: pd.DataFrame, cfg, max_trials: int = 30, tol_s: float = 1e-3) -> pd.DataFrame:
    """Is unit identity recoverable for the Data2-only sessions from the *order* of their rows?

    The Data2 files carry no unit IDs.  On the recordings present in both trees every Data2 row can be identified
    by its spike train (identical to one Data unit's train once the time base is shared), which gives, per trial,
    the sequence of true unit IDs in Data2 row order.  Three things are then measured, per twin pair, over up to
    ``max_trials`` matched trials:

    * ``frac_identical_order``: the Data2 rows are the Data units minus the silent ones **in Data order**
      (identity = position after dropping silent units);
    * ``order_consistency``: across consecutive trials, the units present in both are in the same relative order
      (mean fraction of adjacent pairs that keep their order).  1.0 means the export writes units in one fixed
      order per session (Data's or another) and identity is recoverable by sequence alignment; a value near 0.5
      means the order is re-drawn per trial (e.g. sorted by something trial-dependent) and nothing can be aligned;
    * ``rho_pos_vs_{unit_id, rate, first_spike}``: Spearman correlation of a row's position with the unit's ID,
      its spike count in the trial and its first spike time - which of them the export sorted by.
    """
    from scipy.stats import spearmanr
    from .rasters import read_epochs, resolve_spike_time_reference, spikes_by_region
    by = {c.session: c for c in caches}
    rows = []
    for _, d in dup.iterrows():
        a, b = by.get(d.session_a), by.get(d.session_b)
        if a is None or b is None:
            continue
        ta = pd.to_numeric(a.meta.get("ep_delay_start_times"), errors="coerce").to_numpy(dtype=float)
        tb = pd.to_numeric(b.meta.get("ep_delay_start_times"), errors="coerce").to_numpy(dtype=float)
        n = n_counts = n_order = 0
        n_rows = n_ident = 0
        prev_seq: dict[str, np.ndarray] = {}
        consist, rho_id, rho_rate, rho_first = [], [], [], []
        for i, t in enumerate(ta):
            if n >= max_trials or not np.isfinite(t):
                continue
            j = np.flatnonzero(np.abs(tb - t) <= 2e-3)
            if j.size != 1:
                continue
            j = int(j[0])
            try:
                da = np.load(a.meta.npz_path.iloc[i], allow_pickle=True)
                db = np.load(b.meta.npz_path.iloc[j], allow_pickle=True)
                ra, rb = spikes_by_region(da), spikes_by_region(db)
                ep_a, ep_b = read_epochs(da), read_epochs(db)
                win = (float(ep_a["delay_start_times"]) - 3.0, float(ep_a["go_start_times"]) + 3.0)
                ra, _ = resolve_spike_time_reference(ra, ep_a, win)
                rb, _ = resolve_spike_time_reference(rb, ep_b, win)
            except Exception:
                continue
            n += 1
            same_counts, same_order = True, True
            for r in REGIONS:
                ua_all = [(int(uid), np.asarray(u, dtype=float)) for u, uid in zip(ra[r][0], ra[r][1])]
                ua = [(uid, u) for uid, u in ua_all if u.size]
                ub = [np.asarray(v, dtype=float) for v in rb[r][0]]
                if len(ua) != len(ub):
                    same_counts = same_order = False
                key = lambda arr: np.round(np.sort(arr), 3).tobytes()
                lookup: dict[bytes, tuple[int, int]] = {}
                for k_pos, (uid, u) in enumerate(ua):
                    lookup.setdefault(key(u), (uid, k_pos))
                ids, pos_in_data = [], []
                for v in ub:
                    hit = lookup.get(key(v))
                    ids.append(hit[0] if hit else -1)
                    pos_in_data.append(hit[1] if hit else -1)
                ids, pos_in_data = np.asarray(ids), np.asarray(pos_in_data)
                n_rows += len(ub)
                n_ident += int((ids >= 0).sum())
                if not (len(ua) == len(ub) and (pos_in_data == np.arange(len(ub))).all()):
                    same_order = False
                ok = ids >= 0
                if ok.sum() >= 3:
                    pos = np.arange(len(ub))[ok]
                    rates = np.array([ub[k].size for k in np.flatnonzero(ok)], dtype=float)
                    firsts = np.array([ub[k].min() for k in np.flatnonzero(ok)], dtype=float)
                    for store, val in ((rho_id, ids[ok]), (rho_rate, rates), (rho_first, firsts)):
                        rho = spearmanr(pos, val).correlation if np.unique(val).size > 1 else np.nan
                        if np.isfinite(rho):
                            store.append(float(rho))
                    seq = ids[ok]
                    if r in prev_seq:
                        common = np.intersect1d(seq, prev_seq[r])
                        if common.size >= 3:
                            pa = {u: k for k, u in enumerate(prev_seq[r])}
                            order_now = [u for u in seq if u in pa]                    # common units in this trial's order
                            prev_pos = np.array([pa[u] for u in order_now])
                            consist.append(float(np.mean(np.diff(prev_pos) > 0)))
                    prev_seq[r] = seq
            n_counts += int(same_counts)
            n_order += int(same_order)
        rows.append({"session_a": d.session_a, "session_b": d.session_b, "n_trials_checked": n,
                     "frac_rows_identified": n_ident / n_rows if n_rows else float("nan"),
                     "frac_same_active_counts": n_counts / n if n else float("nan"),
                     "frac_identical_order": n_order / n if n else float("nan"),
                     "order_consistency": float(np.mean(consist)) if consist else float("nan"),
                     "rho_pos_vs_unit_id": float(np.mean(rho_id)) if rho_id else float("nan"),
                     "rho_pos_vs_rate": float(np.mean(rho_rate)) if rho_rate else float("nan"),
                     "rho_pos_vs_first_spike": float(np.mean(rho_first)) if rho_first else float("nan")})
    return pd.DataFrame(rows, columns=["session_a", "session_b", "n_trials_checked", "frac_rows_identified", "frac_same_active_counts",
                                       "frac_identical_order", "order_consistency", "rho_pos_vs_unit_id", "rho_pos_vs_rate",
                                       "rho_pos_vs_first_spike"])


def _delay_onsets(c: SessionCache) -> np.ndarray:
    col = "ep_delay_start_times"
    if col not in c.meta:
        return np.zeros(0)
    return pd.to_numeric(c.meta[col], errors="coerce").dropna().to_numpy(dtype=float)


def drop_duplicate_sessions(caches: list[SessionCache], cfg, report_path: Path | None = None) -> list[SessionCache]:
    dup = find_duplicate_sessions(caches, keep=str(cfg.data.get_path("duplicate_keep", "A")).upper())
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        dup.to_csv(report_path, index=False)
    if not len(dup):
        return caches
    for _, r in dup.iterrows():
        log.warning("%s is the same recording as %s (%.0f%% of delay onsets coincide; %d vs %d units); dropping %s",
                    r.session_a, r.session_b, 100 * r.overlap, r.units_a, r.units_b, r.dropped)
    if not bool(cfg.data.get_path("drop_duplicate_sessions", True)):
        log.warning("data.drop_duplicate_sessions is false: keeping both copies (trials will leak between them)")
        return caches
    dropped = set(dup.dropped)
    return [c for c in caches if c.session not in dropped]


def drop_small_sessions(caches: list[SessionCache], cfg) -> list[SessionCache]:
    """Sessions too small to train, select or test on are excluded from every command (with a warning).

    ``data.min_trials_per_session`` (default 30) and ``data.min_trials_per_lick_class`` (default 5 Left and 5
    Right): below that, a stratified split has no validation trials, stability subsamples have no strata and the
    per-session statistics of the report are meaningless.  A session that drops here is almost always a loading
    problem (see the ``drop_reasons`` of ``cache``), not a small recording.
    """
    min_trials = int(cfg.data.get_path("min_trials_per_session", 30))
    min_lr = int(cfg.data.get_path("min_trials_per_lick_class", 5))
    kept = []
    for c in caches:
        cc = c.class_counts()
        if c.n_trials < min_trials or cc.get("Left", 0) < min_lr or cc.get("Right", 0) < min_lr:
            log.warning("session %s excluded: %d trials (%s) - below data.min_trials_per_session=%d / min_trials_per_lick_class=%d; "
                        "drop reasons at cache time: %s", c.session, c.n_trials, cc, min_trials, min_lr,
                        c.qc_info.get("drop_reasons", {}))
            continue
        kept.append(c)
    return kept


def load_cache(cfg) -> list[SessionCache]:
    cache_dir = Path(cfg.data.cache_dir) / _cache_key(cfg)
    paths = sorted(cache_dir.glob("*.npz"))
    caches = build_cache(cfg) if not paths else [SessionCache.load(p) for p in paths]
    caches = drop_duplicate_sessions(caches, cfg, cache_dir / "duplicate_sessions.csv")
    return drop_small_sessions(caches, cfg)


def cache_summary(caches: list[SessionCache]) -> pd.DataFrame:
    rows = []
    for c in caches:
        row = {"session": c.session, "dataset": c.dataset, "n_trials": c.n_trials, **c.class_counts()}
        row.update({f"units_{r}": c.context[r].shape[1] for r in REGIONS})
        row["T_ctx"] = c.context[REGIONS[0]].shape[2]
        row["T_tgt"] = c.target[REGIONS[0]].shape[2]
        row["MB"] = round(c.nbytes() / 1e6, 1)
        qi = c.qc_info or {}
        row["discovered"] = int(qi.get("n_discovered", c.n_trials))
        row["dropped"] = int(qi.get("n_dropped", 0))
        reasons = qi.get("drop_reasons", {}) or {}
        row["drop_reasons"] = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "-"
        row["align"] = str(qi.get("unit_alignment", "?"))
        if qi.get("unit_alignment") == "recovered" and qi.get("identity_stats"):
            st = qi["identity_stats"]
            row["align"] += f" ({st.get('n_slots_total')} slots, {100 * float(st.get('frac_rows_assigned', float('nan'))):.1f}% rows)"
        if qi.get("units_pooled"):
            row["align"] += " (" + "/".join(str(v) for v in qi["units_pooled"].values()) + " units pooled)"
        refs = qi.get("spike_time_references", {}) or {}
        row["spike_ref"] = "/".join(f"{k}:{v}" for k, v in refs.items()) or "?"
        srcs = qi.get("lick_sources", {}) or {}
        row["licks"] = "/".join(f"{k}:{v}" for k, v in srcs.items()) or "?"
        labs = qi.get("lick_labels", {}) or {}
        row["lick_labels"] = "/".join(f"{k[0]}:{v}" for k, v in labs.items()) or "?"
        rows.append(row)
    return pd.DataFrame(rows)
