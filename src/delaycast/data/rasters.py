"""Turn one trial NPZ into four region rasters aligned to the delay and response epochs.

Design notes
------------
* Spike times arrive in three on-disk shapes (a ragged object array of per-unit arrays, a 2-D NaN-padded
  float matrix with one row per unit, or a bare 1-D float array for a single unit).  ``_as_unit_list`` turns
  all of them into one canonical ``list[np.ndarray]`` so that every downstream step has a single code path.
* Binning is done for all units of a region at once with ``np.bincount`` (``bin_units``).  For the real
  data volume (~2000 units x 350 trials x ~15 sessions) the old per-unit ``np.histogram`` loop was the
  dominant cost of the cache build; a flat bincount over the concatenated spike times is ~10x faster and
  returns uint8 directly, so no float32 raster is ever materialised (measured: 39 ms -> 3 ms per
  2000-unit trial).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .. import REGIONS

_SIDE_RE = re.compile(r"\b(left|right|l|r)\b", re.IGNORECASE)

_EMPTY = np.empty(0, dtype=float)


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


def _to_times(x) -> np.ndarray:
    """Flatten any (possibly nested) numeric container into a 1-D float array without NaN / inf.

    NaNs are the padding value of the 2-D matrix schema and ``inf`` can never be a spike time, so both are
    dropped instead of being propagated into the binning arithmetic.  Non-numeric entries (strings, None)
    are ignored rather than raising, because a single malformed unit must not lose the whole trial.
    """
    if x is None:
        return _EMPTY
    a = np.asarray(x)
    if a.dtype == object:
        parts = [_to_times(el) for el in a.ravel()]
        return np.concatenate(parts) if parts else _EMPTY
    try:
        a = a.astype(float, copy=False).ravel()
    except (TypeError, ValueError):
        return _EMPTY
    return a[np.isfinite(a)]


def _as_unit_list(arr) -> list[np.ndarray]:
    """Canonicalise a spike-time container into a list with one flat float array per unit.

    Accepted layouts (all found in the exported NPZ files):

    * object array / list of per-unit arrays (ragged) -> one entry per element; a 2-D object array is
      read row-wise (MATLAB-style cell exports of shape ``(n_units, 1)``);
    * 2-D float matrix, one row per unit, NaN-padded to the longest unit -> NaNs are dropped per row.
      A matrix of shape ``(n_units, 0)`` still denotes ``n_units`` silent units;
    * 1-D (or 0-D) float array -> a single unit;
    * empty arrays (size 0 and at most one non-empty axis) -> no units.
    """
    if isinstance(arr, np.ndarray):
        a = arr
    else:
        try:
            a = np.asarray(arr)
        except ValueError:  # ragged Python list: numpy >= 1.24 refuses the implicit object array
            a = np.empty(len(arr), dtype=object)
            for i, x in enumerate(arr):
                a[i] = x
    if a.dtype == object:
        if a.ndim == 0:
            return [_to_times(a.item())]
        if a.shape[0] == 0:
            return []
        rows = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a
        return [_to_times(row) for row in rows]
    if a.ndim >= 2:
        if a.shape[0] == 0:
            return []
        rows = a.reshape(a.shape[0], -1)
        return [_to_times(row) for row in rows]
    if a.size == 0:
        return []
    return [_to_times(a)]


def _times(data, key: str) -> np.ndarray:
    if key not in data.files:
        return _EMPTY
    return _to_times(data[key])


def parse_time_list(value) -> np.ndarray:
    """Lick times as written in the behavioural logs: a float, a "t1, t2; t3" string, a list, or NaN / "N/A" / ""."""
    if value is None:
        return _EMPTY
    if isinstance(value, (list, tuple, np.ndarray)):
        return _to_times(np.asarray(value, dtype=object))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _EMPTY if not np.isfinite(float(value)) else np.asarray([float(value)])
    text = str(value).strip().strip("[]()")
    if text.lower() in ("", "nan", "n/a", "none", "null"):
        return _EMPTY
    out = []
    for part in re.split(r"[,;\s]+", text):
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return np.asarray(out, dtype=float) if out else _EMPTY


def unit_ids_by_region(npz_path) -> tuple[dict[str, np.ndarray] | None, str]:
    """Unit IDs per canonical region of one combined-schema NPZ, reading only the two small arrays.

    ``np.load`` of an NPZ is lazy per member, so this never decompresses ``spike_times``; it is what the cache
    builder uses in its first pass to learn the unit universe of a session.  Returns ``(ids, "ok")`` or
    ``(None, why)`` when the file offers no usable identity: pre-split schema (no IDs), ``unit_ids`` and
    ``brain_region`` of different length, or an ID that occurs twice *within one region* (IDs only have to be
    unique per region - probe-local cluster numbers that repeat across hemispheres are fine).
    """
    data = np.load(npz_path, allow_pickle=True)
    for key in ("brain_region", "unit_ids"):
        if key not in data.files:
            return None, f"no {key} key"
    regions_raw = np.asarray(data["brain_region"]).astype(str).ravel()
    ids = np.asarray(data["unit_ids"]).ravel()
    if ids.size != regions_raw.size:
        return None, f"unit_ids has {ids.size} entries but brain_region {regions_raw.size}"
    canon = np.array([normalize_region(region) or "unknown" for region in regions_raw])
    out = {}
    for r in REGIONS:
        rid = ids[canon == r]
        if len(np.unique(rid)) != rid.size:
            return None, f"duplicate unit_ids within {r} ({rid.size - len(np.unique(rid))} repeats)"
        out[r] = rid
    return out, "ok"


@dataclass
class TrialRasters:
    """Rasters for one trial. ``context[r]``: (n_units_r, T_ctx) uint8 counts, ``target[r]``: (n_units_r, T_tgt)."""

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


def _metadata_scalar(metadata: dict | None, key: str) -> float:
    if not metadata:
        return float("nan")
    try:
        value = float(metadata.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def read_epochs(data, metadata: dict | None = None) -> dict[str, float]:
    keys = [
        "trial_start", "trial_stop",
        "presample_start_times", "presample_stop_times",
        "sample_start_times", "sample_stop_times",
        "delay_start_times", "delay_stop_times",
        "go_start_times", "go_stop_times",
    ]
    aliases = {
        "trial_start": "start_time",
        "trial_stop": "stop_time",
        "presample_start_times": "presample_start_time",
        "presample_stop_times": "presample_stop_time",
        "sample_start_times": "sample_start_time",
        "sample_stop_times": "sample_stop_time",
        "delay_start_times": "delay_start_time",
        "delay_stop_times": "delay_stop_time",
        "go_start_times": "go_start_time",
        "go_stop_times": "go_stop_time",
    }
    ep = {
        key: _scalar(data, key, _scalar(data, aliases[key]))
        for key in keys
    }
    # Some Data2 extractions retained spikes and unit metadata but wrote missing
    # epoch scalars into the NPZ. The audited behavioral row is authoritative
    # for those trials, so use it only where the NPZ value is absent/non-finite.
    for key in keys:
        if not np.isfinite(ep[key]):
            ep[key] = _metadata_scalar(metadata, key)
    # ``go_start`` is the moment the response window opens; fall back to delay_stop if missing.
    if np.isnan(ep["go_start_times"]) and not np.isnan(ep["delay_stop_times"]):
        ep["go_start_times"] = ep["delay_stop_times"]
    return ep


def bin_spikes(spike_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Reference single-unit binning with ``np.histogram`` (kept for tests and ad-hoc use).

    Note the histogram convention: the last bin is closed on the right, so a spike exactly at ``edges[-1]``
    is counted; ``bin_units`` (the production path) excludes it.
    """
    st = np.asarray(spike_times, dtype=float)
    if st.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.float32)
    counts, _ = np.histogram(st, bins=edges)
    return counts.astype(np.float32)


def bin_units(spike_list: list[np.ndarray], start: float, n_bins: int, bin_s: float) -> np.ndarray:
    """Bin the spikes of every unit into ``n_bins`` bins of ``bin_s`` seconds starting at ``start``.

    Returns a ``(n_units, n_bins)`` uint8 count matrix.  Bin ``k`` covers ``[start + k*bin_s, start + (k+1)*bin_s)``
    (floor binning, half-open on the right).  Consequently a spike exactly at the window end
    ``start + n_bins*bin_s`` is *excluded*, whereas ``np.histogram`` closes its last bin and would count it;
    apart from that end point the counts are identical to the per-unit histogram path because the bin index is
    corrected against the explicit edge array in the same way ``np.histogram`` compares against its edges
    (a spike sitting exactly on an edge always goes to the bin that starts there, regardless of rounding in
    the division).

    Implementation: all spikes of the region are concatenated once, each spike is tagged with its unit row and
    the (row, bin) pairs are counted with a single ``np.bincount``, which is ~10x faster than one
    ``np.histogram`` per unit for ~2000 units.  Counts are clipped to 255 (a 10 ms bin can never hold 255
    spikes of one unit) so the result fits the uint8 cache without a float intermediate.
    """
    n_units = len(spike_list)
    n_bins = int(n_bins)
    out = np.zeros((n_units, max(n_bins, 0)), dtype=np.uint8)
    if n_units == 0 or n_bins <= 0:
        return out
    lengths = np.fromiter((len(s) for s in spike_list), dtype=np.int64, count=n_units)
    if lengths.sum() == 0:
        return out
    t = np.concatenate([np.asarray(s, dtype=float).ravel() for s in spike_list])
    rows = np.repeat(np.arange(n_units, dtype=np.int64), lengths)
    edges = float(start) + np.arange(n_bins + 1, dtype=float) * float(bin_s)
    with np.errstate(invalid="ignore"):
        idx = np.floor((t - float(start)) / float(bin_s))
    idx = np.where(np.isfinite(idx), idx, -1).astype(np.int64)
    # Exact edge convention: floor() of the scaled time can be off by one where ``t`` lies on (or within
    # rounding of) an edge; compare against the same edge values np.histogram would use and correct.
    probe = np.clip(idx, 0, n_bins - 1)
    idx -= (t < edges[probe])
    idx += (t >= edges[probe + 1])
    keep = (idx >= 0) & (idx < n_bins)
    flat = rows[keep] * n_bins + idx[keep]
    counts = np.bincount(flat, minlength=n_units * n_bins)
    return np.minimum(counts, 255).astype(np.uint8).reshape(n_units, n_bins)


_SPLIT_KEYS = {
    "ALM_L": "left_ALM_spikes",
    "ALM_R": "right_ALM_spikes",
    "STR_L": "left_Striatum_spikes",
    "STR_R": "right_Striatum_spikes",
}


def spikes_by_region(data) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    """Per-region spike times from either NPZ schema.

    Returns ``{region: (list_of_spike_arrays, unit_ids)}`` for all four canonical regions (empty for regions
    absent from the file).  Two schemas are recognised:

    * combined (Dataset A): ``brain_region`` (one label per unit) + ``spike_times`` (+ optional ``unit_ids``);
      units whose label cannot be normalised are dropped (counted separately by ``load_trial_rasters``);
    * pre-split (Dataset B): ``left_ALM_spikes``, ``right_ALM_spikes``, ``left_Striatum_spikes``,
      ``right_Striatum_spikes``.  Data2 has no explicit unit IDs; array position is stable within one session,
      so the positional index is used as the ID but must never be compared across sessions.

    Each spike container may be a ragged object array, a NaN-padded 2-D matrix or a single 1-D array (see
    ``_as_unit_list``).
    """
    if "brain_region" in data.files and "spike_times" in data.files:
        regions_raw = np.asarray(data["brain_region"]).astype(str).ravel()
        units = _as_unit_list(data["spike_times"])
        if len(units) != len(regions_raw):
            raise ValueError(
                f"spike_times holds {len(units)} units but brain_region has {len(regions_raw)} labels"
            )
        if "unit_ids" in data.files and np.asarray(data["unit_ids"]).size == len(units):
            unit_ids = np.asarray(data["unit_ids"]).ravel()
        else:
            unit_ids = np.arange(len(units), dtype=np.int64)
        canon = np.array([normalize_region(region) or "unknown" for region in regions_raw])
        out = {}
        for region in REGIONS:
            idx = np.flatnonzero(canon == region)
            out[region] = ([units[i] for i in idx], unit_ids[idx])
        return out

    missing = [key for key in _SPLIT_KEYS.values() if key not in data.files]
    if missing:
        raise KeyError(
            "NPZ has neither combined brain_region/spike_times nor all split "
            f"region arrays; missing {missing}"
        )
    result = {}
    for region, key in _SPLIT_KEYS.items():
        units = _as_unit_list(data[key])
        result[region] = (units, np.arange(len(units), dtype=np.int64))
    return result


# Backwards-compatible alias (older modules imported the private name).
_spikes_by_region = spikes_by_region


def load_trial_rasters(npz_path, cfg, metadata: dict | None = None,
                       unit_index: dict[str, np.ndarray] | None = None) -> TrialRasters:
    """Bin spikes into the context (delay) and target (response) windows defined by the config.

    Both rasters are uint8 count matrices produced by ``bin_units``; the context window ends at the go cue
    and the target window starts at it, so the two never overlap (a spike exactly at go belongs to the target).

    ``unit_index`` (region -> array of unit IDs, the session's unit universe) aligns the rows of every trial by
    unit ID instead of by position: a unit of the universe that is absent from this trial's NPZ gets a row of
    zeros (the Data2 export omits units without spikes in the trial), and its absence is counted in
    ``qc['n_units_absent']``.  Lick times come from the NPZ when it has them and from the behavioural-log row
    (``metadata['left_lick_times']`` / ``['right_lick_times']``) otherwise; ``qc['lick_source']`` records which
    (``npz`` | ``csv`` | ``none``), because without any lick record the folder label cannot be verified.
    """
    data = np.load(npz_path, allow_pickle=True)
    ep = read_epochs(data, metadata)
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
    # Anchor the grid at the go cue: the last context edge is exactly ``go_start`` on every trial, so a spike
    # fired at or after go can never enter the context (it belongs to the target). A +-5 ms mismatch between the
    # nominal delay and the bin grid therefore shows up at the delay-onset end, where nothing depends on it.
    ctx_start = go_start - n_ctx * bin_s
    ctx_edges = ctx_start + np.arange(n_ctx + 1) * bin_s

    tgt_stop = go_start + cfg.data.target.response_ms / 1000.0
    n_tgt = int(round((tgt_stop - go_start) / tbin_s))
    tgt_edges = go_start + np.arange(n_tgt + 1) * tbin_s

    n_unknown_region = (
        sum(normalize_region(region) is None for region in np.asarray(data["brain_region"]).astype(str).ravel())
        if "brain_region" in data.files
        else 0
    )
    region_data = spikes_by_region(data)
    context, target, uids = {}, {}, {}
    n_absent, n_extra = {}, {}
    for r in REGIONS:
        spikes, region_ids = region_data[r]
        cx = bin_units(spikes, ctx_start, n_ctx, bin_s)
        tg = bin_units(spikes, go_start, n_tgt, tbin_s)
        if unit_index is not None:
            universe = np.asarray(unit_index[r]).ravel()
            pos = {int(u): i for i, u in enumerate(universe.tolist())}
            rows = np.array([pos.get(int(u), -1) for u in np.asarray(region_ids).ravel().tolist()], dtype=int)
            cx_al = np.zeros((len(universe), cx.shape[1]), dtype=cx.dtype)
            tg_al = np.zeros((len(universe), tg.shape[1]), dtype=tg.dtype)
            ok = rows >= 0
            cx_al[rows[ok]] = cx[ok]
            tg_al[rows[ok]] = tg[ok]
            n_extra[r] = int((~ok).sum())
            n_absent[r] = int(len(universe) - ok.sum())
            cx, tg, region_ids = cx_al, tg_al, universe
        context[r] = cx
        target[r] = tg
        uids[r] = region_ids

    # Lick record priority: NPZ arrays that actually contain licks > the behavioural-log row (which the audit
    # used to define the class) > NPZ arrays that exist but are empty > nothing.  The Data2 export writes the
    # lick keys into every NPZ but leaves them empty on almost every lick trial, so "key present" is not
    # evidence of "no lick"; the log row is authoritative there.
    has_npz_licks = "left_lick_times" in data.files or "right_lick_times" in data.files
    lick_left = _times(data, "left_lick_times")
    lick_right = _times(data, "right_lick_times")
    has_csv_licks = bool(metadata) and any(k in metadata for k in ("left_lick_times", "right_lick_times"))
    if lick_left.size + lick_right.size > 0:
        lick_source = "npz"
    elif has_csv_licks:
        lick_left = parse_time_list(metadata.get("left_lick_times"))
        lick_right = parse_time_list(metadata.get("right_lick_times"))
        lick_source = "csv"
    elif has_npz_licks:
        lick_source = "npz"
    else:
        lick_source = "none"
    qc = {
        "n_unknown_region": int(n_unknown_region),
        "early_lick": bool(np.any(np.concatenate([lick_left, lick_right]) < go_start)) if (lick_left.size + lick_right.size) else False,
        "licked_left": bool(np.any(lick_left >= go_start)),
        "licked_right": bool(np.any(lick_right >= go_start)),
        "lick_source": lick_source,
        "delay_len_s": float(go_start - delay_start),
        "n_units_absent": n_absent,
        "n_units_extra": n_extra,
    }
    return TrialRasters(context, target, uids, ep, ctx_edges, tgt_edges, lick_left, lick_right, qc)


def label_from_licks(qc: dict) -> str | None:
    """Behavioural label implied by the lick times.

    ``None`` when no lick record exists at all (``lick_source == 'none'``: nothing to check the folder label
    against); ``"Both"`` when the animal licked both sides after the go cue (ambiguous action).
    """
    if qc.get("lick_source", "npz") == "none":
        return None
    l, r = qc.get("licked_left", False), qc.get("licked_right", False)
    if l and not r:
        return "Left"
    if r and not l:
        return "Right"
    if not l and not r:
        return "Ignore"
    return "Both"
