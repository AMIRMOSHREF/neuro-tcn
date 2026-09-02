"""Turn one trial NPZ into four region rasters aligned to the delay and response epochs."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .. import REGIONS

_SIDE_RE = re.compile(r"\b(left|right|l|r)\b", re.IGNORECASE)


def normalize_region(label: str) -> str | None:
    """Map free-text region labels ("left ALM", "Right Striatum", "ALM_L", ...) onto the four canonical keys."""
    s = str(label).strip().lower().replace("-", " ").replace("_", " ")
    if "alm" in s:
        area = "ALM"
    elif "str" in s:  # striatum / STR
        area = "STR"
    else:
        return None
    if re.search(r"\bleft\b|\bl\b", s):
        side = "L"
    elif re.search(r"\bright\b|\br\b", s):
        side = "R"
    else:
        return None
    return f"{area}_{side}"


def _scalar(data, key: str, default=np.nan) -> float:
    if key not in data.files:
        return default
    v = np.asarray(data[key]).ravel()
    if v.size == 0:
        return default
    try:
        return float(v[0])
    except (TypeError, ValueError):
        return default


def _times(data, key: str) -> np.ndarray:
    if key not in data.files:
        return np.empty(0)
    v = np.asarray(data[key], dtype=object).ravel()
    out = []
    for x in v:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            pass
    return np.asarray(out, dtype=float)


@dataclass
class TrialRasters:
    """Rasters for one trial. ``context[r]``: (n_units_r, T_ctx) counts, ``target[r]``: (n_units_r, T_tgt)."""

    context: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    unit_ids: dict[str, np.ndarray]
    epochs: dict[str, float]
    ctx_edges: np.ndarray
    tgt_edges: np.ndarray
    lick_left: np.ndarray
    lick_right: np.ndarray
    qc: dict = field(default_factory=dict)

    @property
    def n_units(self) -> dict[str, int]:
        return {r: self.context[r].shape[0] for r in REGIONS}


def read_epochs(data) -> dict[str, float]:
    keys = [
        "trial_start", "trial_stop",
        "presample_start_times", "presample_stop_times",
        "sample_start_times", "sample_stop_times",
        "delay_start_times", "delay_stop_times",
        "go_start_times", "go_stop_times",
    ]
    ep = {k: _scalar(data, k) for k in keys}
    # ``go_start`` is the moment the response window opens; fall back to delay_stop if missing.
    if np.isnan(ep["go_start_times"]) and not np.isnan(ep["delay_stop_times"]):
        ep["go_start_times"] = ep["delay_stop_times"]
    return ep


def bin_spikes(spike_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    st = np.asarray(spike_times, dtype=float)
    if st.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.float32)
    counts, _ = np.histogram(st, bins=edges)
    return counts.astype(np.float32)


def load_trial_rasters(npz_path, cfg) -> TrialRasters:
    """Bin spikes into the context (delay) and target (response) windows defined by the config."""
    data = np.load(npz_path, allow_pickle=True)
    ep = read_epochs(data)
    bin_s = cfg.data.bin_ms / 1000.0
    tbin_s = cfg.data.target_bin_ms / 1000.0

    delay_start = ep["delay_start_times"]
    go_start = ep["go_start_times"]
    if np.isnan(delay_start) or np.isnan(go_start):
        raise ValueError(f"{npz_path}: missing delay_start_times / go_start_times")

    if cfg.data.context.include_sample and not np.isnan(ep["sample_start_times"]):
        ctx_start = ep["sample_start_times"]
    else:
        ctx_start = delay_start - cfg.data.context.pre_delay_ms / 1000.0
    ctx_stop = go_start
    n_ctx = int(round((ctx_stop - ctx_start) / bin_s))
    ctx_edges = ctx_start + np.arange(n_ctx + 1) * bin_s

    tgt_stop = go_start + cfg.data.target.response_ms / 1000.0
    n_tgt = int(round((tgt_stop - go_start) / tbin_s))
    tgt_edges = go_start + np.arange(n_tgt + 1) * tbin_s

    regions_raw = np.asarray(data["brain_region"]).astype(str)
    spike_times = data["spike_times"]
    unit_ids = np.asarray(data["unit_ids"]) if "unit_ids" in data.files else np.arange(len(regions_raw))
    canon = np.array([normalize_region(r) or "unknown" for r in regions_raw])

    context, target, uids = {}, {}, {}
    for r in REGIONS:
        idx = np.where(canon == r)[0]
        cx = np.zeros((len(idx), n_ctx), dtype=np.float32)
        tg = np.zeros((len(idx), n_tgt), dtype=np.float32)
        for row, ui in enumerate(idx):
            st = np.asarray(spike_times[ui], dtype=float).ravel()
            cx[row] = bin_spikes(st, ctx_edges)
            tg[row] = bin_spikes(st, tgt_edges)
        context[r], target[r], uids[r] = cx, tg, unit_ids[idx]

    lick_left = _times(data, "left_lick_times")
    lick_right = _times(data, "right_lick_times")
    qc = {
        "n_unknown_region": int((canon == "unknown").sum()),
        "early_lick": bool(np.any(np.concatenate([lick_left, lick_right]) < go_start)) if (lick_left.size + lick_right.size) else False,
        "licked_left": bool(np.any(lick_left >= go_start)),
        "licked_right": bool(np.any(lick_right >= go_start)),
        "delay_len_s": float(go_start - delay_start),
    }
    return TrialRasters(context, target, uids, ep, ctx_edges, tgt_edges, lick_left, lick_right, qc)


def label_from_licks(qc: dict) -> str | None:
    """Behavioural label implied by the lick times (None when ambiguous, e.g. licked both sides)."""
    l, r = qc.get("licked_left", False), qc.get("licked_right", False)
    if l and not r:
        return "Left"
    if r and not l:
        return "Right"
    if not l and not r:
        return "Ignore"
    return None
