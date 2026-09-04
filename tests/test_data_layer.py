"""Data-layer tests: NPZ schema readers, vectorised binning, cache QC, CWT kernel and vectorised statistics.

All inputs are small synthetic NPZ files written with ``np.savez`` into ``tmp_path`` so that both on-disk
schemas (Dataset A: combined ``brain_region``/``spike_times``; Dataset B: pre-split ``left_ALM_spikes`` ...)
and every spike-container layout (ragged object array, NaN-padded 2-D matrix, single 1-D unit, empty) are
exercised without the real recordings.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pywt
from scipy import stats as ss

from delaycast import REGION_LABELS, REGIONS
from delaycast.config import load_config
from delaycast.data.cache import SessionCache, _fit_len, as_counts_u8, build_cache
from delaycast.data.rasters import (_as_unit_list, _spikes_by_region, bin_spikes, bin_units, load_trial_rasters,
                                    spikes_by_region)
from delaycast.features.spectral import (_cwt_power, band_frequencies, band_power_cwt, band_power_stft, cwt_scales,
                                         cwt_scalogram, smooth_rates)
from delaycast.features.stats import (_tie_correction, _tie_term, kruskal_vectorised, mannwhitney_vectorised,
                                      wilcoxon_vectorised)

SPLIT_KEYS = {"ALM_L": "left_ALM_spikes", "ALM_R": "right_ALM_spikes",
              "STR_L": "left_Striatum_spikes", "STR_R": "right_Striatum_spikes"}


# ----------------------------------------------------------------------------- synthetic NPZ helpers
def _unit_spikes(rng: np.random.Generator, n_units: int, t0: float, t1: float, rate_hz: float = 8.0) -> list[np.ndarray]:
    return [np.sort(rng.uniform(t0, t1, size=rng.poisson(rate_hz * (t1 - t0)))) for _ in range(n_units)]


def _nan_pad(units: list[np.ndarray]) -> np.ndarray:
    """(n_units, max_len) float matrix padded with NaN - the second spike-time layout found in exports."""
    width = max([len(u) for u in units] + [1])
    out = np.full((len(units), width), np.nan)
    for i, u in enumerate(units):
        out[i, :len(u)] = u
    return out


def _object_array(units: list[np.ndarray]) -> np.ndarray:
    out = np.empty(len(units), dtype=object)
    for i, u in enumerate(units):
        out[i] = u
    return out


def _payload(rng, n_units: dict[str, int], cls: str = "Left", t0: float = 10.0, delay_s: float = 1.2) -> dict:
    """Combined-schema (Dataset A) trial payload with self-consistent epochs and lick times."""
    sample_start = t0 + 0.7
    delay_start = sample_start + 0.65
    go_start = delay_start + delay_s
    go_stop = go_start + 1.5
    regions, spikes, uids = [], [], []
    for r in REGIONS:
        for u in _unit_spikes(rng, n_units[r], t0, go_stop + 0.2):
            spikes.append(u)
            regions.append(REGION_LABELS[r])
            uids.append(len(uids))
    left = right = np.empty(0)
    if cls in ("Left", "Right"):
        licks = go_start + 0.2 + np.arange(6) * 0.13
        left, right = (licks, right) if cls == "Left" else (left, licks)
    return {
        "unit_ids": np.asarray(uids), "brain_region": np.asarray(regions), "spike_times": _object_array(spikes),
        "trial_start": np.float64(t0), "trial_stop": np.float64(go_stop + 0.2),
        "presample_start_times": np.float64(t0 + 0.2), "presample_stop_times": np.float64(sample_start),
        "sample_start_times": np.float64(sample_start), "sample_stop_times": np.float64(delay_start),
        "delay_start_times": np.float64(delay_start), "delay_stop_times": np.float64(go_start),
        "go_start_times": np.float64(go_start), "go_stop_times": np.float64(go_stop),
        "left_lick_times": left, "right_lick_times": right,
    }


def _to_split(payload: dict, layouts: dict[str, str] | None = None) -> dict:
    """Dataset B schema: four region arrays (+ singular epoch names) with a chosen container layout per region.

    ``layouts[r]`` in {"object", "matrix", "single", "empty"}: ``single`` keeps only the first unit as a bare
    1-D array, ``empty`` removes the region entirely.
    """
    layouts = layouts or {}
    out = {k: v for k, v in payload.items() if k not in ("unit_ids", "brain_region", "spike_times")}
    regions = np.asarray(payload["brain_region"]).astype(str)
    for r, key in SPLIT_KEYS.items():
        units = [payload["spike_times"][i] for i in np.flatnonzero(regions == REGION_LABELS[r])]
        layout = layouts.get(r, "object")
        if layout == "matrix":
            out[key] = _nan_pad(units)
        elif layout == "single":
            out[key] = np.asarray(units[0], dtype=float)
        elif layout == "empty":
            out[key] = np.empty(0)
        else:
            out[key] = _object_array(units)
    for plural, singular in {"trial_start": "start_time", "trial_stop": "stop_time",
                             "delay_start_times": "delay_start_time", "delay_stop_times": "delay_stop_time",
                             "go_start_times": "go_start_time", "go_stop_times": "go_stop_time"}.items():
        out[singular] = out.pop(plural)
    return out


def _cfg(tmp_path: Path, **overrides):
    cfg = load_config(None)
    cfg.set_path("data.data_a_root", str(tmp_path / "Data"))
    cfg.set_path("data.data_b_root", str(tmp_path / "Data2"))
    cfg.set_path("data.cache_dir", str(tmp_path / "cache"))
    cfg.set_path("data.use_dataset_a", (tmp_path / "Data").is_dir())
    cfg.set_path("data.use_dataset_b", (tmp_path / "Data2").is_dir())
    for k, v in overrides.items():
        cfg.set_path(k, v)
    return cfg


def _write_a(tmp_path: Path, session: str, trial: int, cls: str, payload: dict) -> Path:
    d = tmp_path / "Data" / session / "Rasters" / cls
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"trial_{trial}.npz", **payload)
    return d / f"trial_{trial}.npz"


def _write_b(tmp_path: Path, trial: int, cls: str, payload: dict) -> Path:
    d = tmp_path / "Data2" / "sub-1" / "sub-1_ses-20190301T120000_behavior+ecephys" / "NPZ" / cls
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"trial{trial}.npz", **payload)
    return d / f"trial{trial}.npz"


def _write_b_csv(tmp_path: Path, rows: list[dict]) -> Path:
    sess = tmp_path / "Data2" / "sub-1" / "sub-1_ses-20190301T120000_behavior+ecephys"
    sess.mkdir(parents=True, exist_ok=True)
    p = sess / "behavioral_master_log_audited.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _without_units(payload: dict, drop_uids: list[int]) -> dict:
    """The same trial with some units missing from the NPZ (how the Data2 export treats units without spikes)."""
    keep = ~np.isin(np.asarray(payload["unit_ids"]), drop_uids)
    out = dict(payload)
    out["unit_ids"] = np.asarray(payload["unit_ids"])[keep]
    out["brain_region"] = np.asarray(payload["brain_region"])[keep]
    out["spike_times"] = _object_array([s for s, k in zip(payload["spike_times"], keep) if k])
    return out


def _strip_licks(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("left_lick_times", "right_lick_times")}


# ----------------------------------------------------------------------------- rasters.py
def test_as_unit_list_layouts():
    a, b = np.array([0.5, 1.0, 2.5]), np.array([0.1])
    # ragged object array / list
    assert [u.tolist() for u in _as_unit_list(_object_array([a, b, np.empty(0)]))] == [a.tolist(), b.tolist(), []]
    assert [u.tolist() for u in _as_unit_list([a, b])] == [a.tolist(), b.tolist()]
    # 2-D NaN-padded matrix: one row per unit, NaNs dropped
    mat = _nan_pad([a, b])
    units = _as_unit_list(mat)
    assert len(units) == 2 and units[0].tolist() == a.tolist() and units[1].tolist() == b.tolist()
    assert all(np.isfinite(u).all() for u in units)
    # (n_units, 0) matrix = n silent units; (0, k) = nothing
    assert [len(u) for u in _as_unit_list(np.empty((3, 0)))] == [0, 0, 0]
    assert _as_unit_list(np.empty((0, 4))) == []
    # 1-D float array = a single unit; NaN / inf inside it are dropped
    single = _as_unit_list(np.array([0.2, np.nan, 0.4, np.inf]))
    assert len(single) == 1 and single[0].tolist() == [0.2, 0.4]
    # empties
    assert _as_unit_list(np.empty(0)) == []
    assert _as_unit_list(np.empty(0, dtype=object)) == []
    # MATLAB-style (n, 1) object cell and nested wrappers
    cell = np.empty((2, 1), dtype=object)
    cell[0, 0], cell[1, 0] = a, [b]
    assert [u.tolist() for u in _as_unit_list(cell)] == [a.tolist(), b.tolist()]


def test_spikes_by_region_both_schemas(tmp_path):
    rng = np.random.default_rng(0)
    n_units = {"ALM_L": 3, "ALM_R": 2, "STR_L": 1, "STR_R": 2}
    payload = _payload(rng, n_units)
    payload["brain_region"][-1] = "cerebellum"  # unknown label -> unit is dropped, not the trial
    np.savez(tmp_path / "a.npz", **payload)
    by_r = spikes_by_region(np.load(tmp_path / "a.npz", allow_pickle=True))
    assert {r: len(u) for r, (u, _) in by_r.items()} == {"ALM_L": 3, "ALM_R": 2, "STR_L": 1, "STR_R": 1}
    assert by_r["ALM_R"][1].tolist() == [3, 4]                         # unit_ids follow the file
    assert by_r["ALM_L"][0][1].tolist() == payload["spike_times"][1].tolist()
    assert _spikes_by_region is spikes_by_region                        # alias kept for old imports

    split = _to_split(payload, {"ALM_L": "matrix", "ALM_R": "object", "STR_L": "single", "STR_R": "empty"})
    np.savez(tmp_path / "b.npz", **split)
    by_r = spikes_by_region(np.load(tmp_path / "b.npz", allow_pickle=True))
    assert {r: len(u) for r, (u, _) in by_r.items()} == {"ALM_L": 3, "ALM_R": 2, "STR_L": 1, "STR_R": 0}
    assert by_r["ALM_L"][1].tolist() == [0, 1, 2]                      # positional ids
    for i in range(3):
        assert by_r["ALM_L"][0][i].tolist() == payload["spike_times"][i].tolist()
    assert by_r["STR_L"][0][0].tolist() == payload["spike_times"][5].tolist()

    bad = dict(payload)
    bad["brain_region"] = payload["brain_region"][:-1]
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="brain_region"):
        spikes_by_region(np.load(tmp_path / "bad.npz", allow_pickle=True))
    with pytest.raises(KeyError, match="neither"):   # neither schema present
        spikes_by_region(np.load(_savez(tmp_path / "none.npz", trial_start=np.float64(0)), allow_pickle=True))


def _savez(path: Path, **arrays) -> Path:
    np.savez(path, **arrays)
    return path


def test_bin_units_matches_histogram_except_endpoint():
    rng = np.random.default_rng(1)
    start, n_bins, bin_s = 12.345, 120, 0.01
    edges = start + np.arange(n_bins + 1) * bin_s
    units = _unit_spikes(rng, 300, start - 0.5, start + 2.0, rate_hz=30.0) + [np.empty(0)]
    ref = np.stack([bin_spikes(u, edges) for u in units])
    got = bin_units(units, start, n_bins, bin_s)
    assert got.dtype == np.uint8 and got.shape == (301, n_bins)
    assert np.array_equal(got, ref.astype(np.uint8))
    # edges: a spike exactly on an interior edge goes to the bin starting there (like np.histogram); a spike
    # exactly at the window end is excluded (np.histogram closes its last bin and would count it).
    sp = [np.array([start, start + 3 * bin_s, start + n_bins * bin_s, start + n_bins * bin_s - 1e-9, start - 1e-9])]
    got = bin_units(sp, start, n_bins, bin_s)[0]
    hist = np.histogram(sp[0], edges)[0]
    assert got[0] == 1 and got[3] == 1 and got[n_bins - 1] == 1
    assert hist[n_bins - 1] == 2 and np.array_equal(got[:-1], hist[:-1])
    # degenerate inputs
    assert bin_units([], start, n_bins, bin_s).shape == (0, n_bins)
    assert bin_units([np.empty(0)], start, n_bins, bin_s).sum() == 0
    assert bin_units([np.array([1.0])], start, 0, bin_s).shape == (1, 0)
    # timing (informational): 2000 units at ~20 spikes each
    big = [np.sort(rng.uniform(start - 1, start + 3.5, size=20)) for _ in range(2000)]
    t0 = time.perf_counter(); bin_units(big, start, n_bins, bin_s); t_new = time.perf_counter() - t0
    t0 = time.perf_counter(); [bin_spikes(u, edges) for u in big]; t_old = time.perf_counter() - t0
    print(f"\nbin_units: {t_new * 1e3:.2f} ms vs histogram loop {t_old * 1e3:.1f} ms per 2000-unit trial")


def test_load_trial_rasters_uint8_both_schemas(tmp_path):
    rng = np.random.default_rng(2)
    n_units = {"ALM_L": 4, "ALM_R": 3, "STR_L": 1, "STR_R": 2}
    payload = _payload(rng, n_units)
    cfg = load_config(None)
    pa = _savez(tmp_path / "a.npz", **payload)
    tr = load_trial_rasters(pa, cfg)
    for r in REGIONS:
        assert tr.context[r].dtype == np.uint8 and tr.context[r].shape == (n_units[r], 120)
        assert tr.target[r].dtype == np.uint8 and tr.target[r].shape == (n_units[r], 30)
    # counts agree with the reference histogram binning on the same edges
    for i, u in enumerate(payload["spike_times"][:4]):
        assert np.array_equal(tr.context["ALM_L"][i], bin_spikes(u, tr.ctx_edges).astype(np.uint8))
        assert np.array_equal(tr.target["ALM_L"][i], bin_spikes(u, tr.tgt_edges).astype(np.uint8))
    assert tr.qc["delay_len_s"] == pytest.approx(1.2) and tr.qc["licked_left"] and not tr.qc["early_lick"]

    pb = _savez(tmp_path / "b.npz", **_to_split(payload, {"ALM_L": "matrix", "STR_L": "single", "STR_R": "empty"}))
    trb = load_trial_rasters(pb, cfg)
    assert np.array_equal(trb.context["ALM_L"], tr.context["ALM_L"])
    assert np.array_equal(trb.target["ALM_R"], tr.target["ALM_R"])
    assert trb.context["STR_L"].shape == (1, 120) and np.array_equal(trb.context["STR_L"], tr.context["STR_L"])
    assert trb.context["STR_R"].shape == (0, 120) and trb.unit_ids["STR_R"].size == 0


# ----------------------------------------------------------------------------- cache.py
def test_fit_len_alignment():
    a = np.arange(1, 6, dtype=np.uint8)[None].repeat(2, axis=0)  # (2, 5) = 1..5
    assert _fit_len(a, 5) is a
    assert _fit_len(a, 3, align="right").tolist()[0] == [3, 4, 5]        # context: keep the LAST bins (go anchored)
    assert _fit_len(a, 7, align="right").tolist()[0] == [0, 0, 1, 2, 3, 4, 5]
    assert _fit_len(a, 3, align="left").tolist()[0] == [1, 2, 3]         # target: keep the FIRST bins
    assert _fit_len(a, 7, align="left").tolist()[0] == [1, 2, 3, 4, 5, 0, 0]
    with pytest.raises(ValueError):
        _fit_len(a, 3, align="centre")


def test_as_counts_u8_warns_above_255(caplog):
    with caplog.at_level(logging.WARNING, logger="delaycast.data.cache"):
        out = as_counts_u8(np.array([[1.0, 300.0]]))
    assert out.dtype == np.uint8 and out.tolist() == [[1, 255]]
    assert any("exceed 255" in rec.message for rec in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="delaycast.data.cache"):
        as_counts_u8(np.array([[1.0, 2.0]]))
    assert not caplog.records


def test_build_cache_qc_drops_and_length_fixes(tmp_path, caplog):
    rng = np.random.default_rng(3)
    n_units = {"ALM_L": 5, "ALM_R": 4, "STR_L": 3, "STR_R": 2}
    # Session1: two good trials, one 1.3 s delay (dropped), one 1.208 s delay (kept, 121-bin context fixed)
    _write_a(tmp_path, "Session1", 1, "Left", _payload(rng, n_units, "Left", t0=0.0))
    _write_a(tmp_path, "Session1", 2, "Right", _payload(rng, n_units, "Right", t0=10.0))
    _write_a(tmp_path, "Session1", 3, "Left", _payload(rng, n_units, "Left", t0=20.0, delay_s=1.3))
    long_payload = _payload(rng, n_units, "Ignore", t0=30.0, delay_s=1.208)
    _write_a(tmp_path, "Session1", 4, "Ignore", long_payload)
    # Session2: the second trial lacks one STR_L unit (uid 10; the Data2 export omits units without spikes) ->
    # rows are aligned by unit ID, the absent unit gets a zero row and the trial is kept
    _write_a(tmp_path, "Session2", 1, "Left", _payload(rng, n_units, "Left", t0=0.0))
    _write_a(tmp_path, "Session2", 2, "Right", _without_units(_payload(rng, n_units, "Right", t0=10.0), [10]))
    _write_a(tmp_path, "Session2", 3, "Right", _payload(rng, n_units, "Right", t0=20.0))
    cfg = _cfg(tmp_path)
    assert float(cfg.data.qc.max_delay_dev_ms) == 15.0

    with caplog.at_level(logging.WARNING, logger="delaycast.data.cache"):
        caches = {c.session: c for c in build_cache(cfg, force=True)}
    assert set(caches) == {"A/Session1", "A/Session2"}
    s1, s2 = caches["A/Session1"], caches["A/Session2"]
    assert s1.trials.tolist() == [1, 2, 4] and s1.labels.tolist() == [1, 2, 0]
    assert s2.trials.tolist() == [1, 2, 3] and s2.unit_ids["STR_L"].tolist() == [9, 10, 11]
    assert not s2.context["STR_L"][1, 1].any() and not s2.target["STR_L"][1, 1].any()   # absent unit = silent
    assert s2.context["STR_L"][1, 0].any() or s2.context["STR_L"][1, 2].any()
    assert s2.qc_info["unit_alignment"] == "id" and s2.qc_info["units_union"] == 14 and s2.qc_info["units_per_trial_min"] == 13
    assert s1.qc_info["unit_presence_median"]["STR_L"] == 1.0
    for c in (s1, s2):
        for r in REGIONS:
            assert c.context[r].dtype == np.uint8 and c.context[r].shape == (c.n_trials, n_units[r], 120)
            assert c.target[r].dtype == np.uint8 and c.target[r].shape == (c.n_trials, n_units[r], 30)
            assert c.context[r].flags["C_CONTIGUOUS"]

    cache_sub = next(p for p in (tmp_path / "cache").iterdir() if p.is_dir())
    qc = pd.read_csv(cache_sub / "qc_log.csv")
    r3 = qc[(qc.session == "A/Session1") & (qc.trial == 3)].iloc[0]
    assert not r3.kept and r3.reason.startswith("delay_len_") and r3.reason == "delay_len_1.30s"
    r2 = qc[(qc.session == "A/Session2") & (qc.trial == 2)].iloc[0]
    assert r2.kept and r2.lick_source == "npz" and r2.lick_label == "Right"
    assert qc[qc.kept].shape[0] == 6

    # The 1.208 s trial produced a 121-bin context; the cache keeps its LAST 120 bins (go-cue anchored) and
    # the length fix is counted in the JSON next to the arrays.
    raw = load_trial_rasters(s1.meta.npz_path[2], cfg)
    assert raw.context["ALM_L"].shape[1] == 121 and raw.target["ALM_L"].shape[1] == 30
    assert np.array_equal(s1.context["ALM_L"][2], raw.context["ALM_L"][:, 1:])
    assert np.array_equal(s1.context["ALM_L"][0], load_trial_rasters(s1.meta.npz_path[0], cfg).context["ALM_L"])
    assert s1.qc_info["length_fixes"] == {"context": 1, "target": 0, "context_max_abs_bins": 1, "target_max_abs_bins": 0}
    assert s1.qc_info["n_kept"] == 3 and s1.qc_info["n_dropped"] == 1 and s1.qc_info["drop_reasons"] == {"delay_len": 1}
    assert s2.qc_info["drop_reasons"] == {}
    info = json.loads((cache_sub / "A__Session1.json").read_text())
    assert info["length_fixes"]["context"] == 1 and info["max_delay_dev_ms"] == 15.0
    # round trip through disk keeps everything (including qc_info)
    again = {c.session: c for c in build_cache(cfg, force=False)}["A/Session1"]   # loads from disk
    assert isinstance(again, SessionCache)
    assert again.qc_info["length_fixes"] == s1.qc_info["length_fixes"]
    assert np.array_equal(again.context["STR_R"], s1.context["STR_R"])

    # A stricter tolerance turns the 1.208 s trial into a drop as well.
    cfg_strict = _cfg(tmp_path, **{"data.qc.max_delay_dev_ms": 5, "data.cache_dir": str(tmp_path / "cache_strict")})
    strict = {c.session: c for c in build_cache(cfg_strict, force=True)}["A/Session1"]
    assert strict.trials.tolist() == [1, 2] and strict.qc_info["drop_reasons"] == {"delay_len": 2}


def test_build_cache_split_schema_dataset_b(tmp_path):
    rng = np.random.default_rng(4)
    n_units = {"ALM_L": 3, "ALM_R": 2, "STR_L": 1, "STR_R": 2}
    layouts = {"ALM_L": "matrix", "ALM_R": "object", "STR_L": "single", "STR_R": "empty"}
    payloads = [_payload(rng, n_units, cls, t0=10.0 * i) for i, cls in enumerate(["Left", "Right", "Left", "Ignore"], start=1)]
    for i, (p, cls) in enumerate(zip(payloads, ["Left", "Right", "Left", "Ignore"]), start=1):
        _write_b(tmp_path, i, cls, _to_split(p, layouts))
    cfg = _cfg(tmp_path)
    (c,) = build_cache(cfg, force=True)
    assert c.dataset == "B" and c.n_trials == 4 and c.labels.tolist() == [1, 2, 1, 0]
    assert c.n_units == {"ALM_L": 3, "ALM_R": 2, "STR_L": 1, "STR_R": 0}
    assert c.context["STR_R"].shape == (4, 0, 120) and c.target["STR_R"].shape == (4, 0, 30)
    # the NaN-padded and single-unit layouts bin exactly like the ragged one
    ref = load_trial_rasters(_savez(tmp_path / "ref.npz", **payloads[0]), cfg)
    assert np.array_equal(c.context["ALM_L"][0], ref.context["ALM_L"])
    assert np.array_equal(c.context["STR_L"][0], ref.context["STR_L"])
    assert np.array_equal(c.target["ALM_R"][0], ref.target["ALM_R"])


def test_cmd_inspect_routes_through_spikes_by_region(tmp_path, capsys):
    """``inspect --npz-detail`` on a split-schema file (no ``brain_region`` key) must not raise."""
    from delaycast.cli import main
    rng = np.random.default_rng(5)
    payload = _payload(rng, {"ALM_L": 2, "ALM_R": 1, "STR_L": 1, "STR_R": 1})
    _write_b(tmp_path, 1, "Left", _to_split(payload, {"ALM_L": "matrix", "STR_L": "single"}))
    main(["inspect", "--npz-detail", "--set", f"data.data_b_root={tmp_path / 'Data2'}", "--set", "data.use_dataset_a=false"])
    out = capsys.readouterr().out
    assert "regions:" in out and "'ALM_L': 2" in out and "'STR_R': 1" in out
    assert "spikes:" in out


# ----------------------------------------------------------------------------- spectral.py
def _pywt_power(X: np.ndarray, bin_ms: float, freqs: np.ndarray, wavelet: str) -> np.ndarray:
    coefs, _ = pywt.cwt(X, cwt_scales(freqs, bin_ms, wavelet), wavelet, sampling_period=bin_ms / 1000.0, axis=-1, method="fft")
    return np.moveaxis(np.abs(coefs) ** 2, 0, 1)  # (N, F, T)


def test_cwt_kernel_matches_pywt():
    rng = np.random.default_rng(6)
    T, bin_ms, wavelet = 120, 10.0, "cmor1.5-1.0"
    bands = {"slow": [1, 4], "theta": [4, 12], "beta": [12, 30]}
    freqs, band_idx, names = band_frequencies(bands, 5, 2.0)
    N = 2048
    rates = rng.gamma(2.0, 5.0, size=(N, T)).astype(np.float32)
    X = rates - rates.mean(axis=-1, keepdims=True)

    # (a) float64: the impulse-response matmul reproduces pywt to round-off
    X64 = X.astype(np.float64)
    assert np.allclose(_cwt_power(X64, bin_ms, freqs, wavelet), _pywt_power(X64, bin_ms, freqs, wavelet), rtol=1e-4)
    # (b) public float32 path (pywt itself works in complex64 here): allclose at rtol 1e-4 with a tiny absolute
    #     floor relative to the largest power, since near-zero coefficients are dominated by float32 cancellation
    t0 = time.perf_counter(); ref = _pywt_power(X, bin_ms, freqs, wavelet); t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); got = cwt_scalogram(rates, bin_ms, freqs, wavelet, chunk_rows=512); t_new = time.perf_counter() - t0
    assert got.dtype == np.float32 and got.shape == (N, len(freqs), T)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5 * ref.max())
    assert np.allclose(cwt_scalogram(rates[0], bin_ms, freqs, wavelet), ref[0], rtol=1e-4, atol=1e-5 * ref.max())
    # (c) band power: mean of the pywt power over the cone-of-influence-trimmed window
    lo, hi = int(0.1 * T), int(0.9 * T)
    bp_ref = np.stack([ref[:, band_idx == b, lo:hi].mean(axis=(1, 2)) for b in range(len(names))], axis=1)
    t0 = time.perf_counter(); bp, got_names = band_power_cwt(rates, bin_ms, bands, wavelet, chunk_rows=512); t_bp = time.perf_counter() - t0
    assert got_names == names and bp.dtype == np.float32
    assert np.allclose(bp, bp_ref, rtol=1e-4)
    # rows that are integer counts (uint8 cache) are accepted directly
    assert np.allclose(band_power_cwt(rates.astype(np.uint8), bin_ms, bands, wavelet)[0],
                       band_power_cwt(rates.astype(np.uint8).astype(np.float32), bin_ms, bands, wavelet)[0])
    print(f"\nCWT rows/s: pywt {N / t_ref:.0f}, kernel matmul {N / t_new:.0f} (x{t_ref / t_new:.1f}); "
          f"band_power_cwt {N / t_bp:.0f} rows/s")


def test_band_frequencies_min_freq():
    bands = {"slow": [1, 4], "theta": [4, 12]}
    f, idx, names = band_frequencies(bands, 5)               # default min 2 Hz
    assert names == ["slow", "theta"] and f[0] == pytest.approx(2.0) and f[4] == pytest.approx(4.0)
    assert np.allclose(f[:5], np.geomspace(2.0, 4.0, 5)) and idx.tolist() == [0] * 5 + [1] * 5
    f, _, _ = band_frequencies(bands, 5, min_freq_hz=0.5)   # explicit lower limit -> band edge wins
    assert f[0] == pytest.approx(1.0)
    f, _, _ = band_frequencies(bands, 3, min_freq_hz=6.0)   # limit above the band top -> degenerate, not descending
    assert f[:3].tolist() == [6.0, 6.0, 6.0]
    # band_power_cwt forwards the keyword (default 2 Hz) and changes the slow-band value accordingly
    rng = np.random.default_rng(7)
    rates = rng.gamma(2.0, 5.0, size=(16, 120)).astype(np.float32)
    bp2, _ = band_power_cwt(rates, 10.0, bands)
    bp05, _ = band_power_cwt(rates, 10.0, bands, min_freq_hz=0.5)
    assert not np.allclose(bp2[:, 0], bp05[:, 0]) and np.allclose(bp2[:, 1], bp05[:, 1])


def test_float32_guards_on_integer_input():
    counts = np.random.default_rng(8).poisson(1.0, size=(6, 120)).astype(np.uint8)
    bands = {"theta": [4, 12], "beta": [12, 30]}
    r = smooth_rates(counts, 10.0, 20.0)
    assert r.dtype == np.float32 and np.allclose(r, smooth_rates(counts.astype(np.float32), 10.0, 20.0))
    assert smooth_rates(counts, 10.0, 0.0).dtype == np.float32
    assert smooth_rates(counts.astype(np.float64), 10.0, 20.0).dtype == np.float64
    bp = band_power_stft(counts, 10.0, bands)
    assert bp.shape == (6, 2, 120) and np.allclose(bp, band_power_stft(counts.astype(np.float32), 10.0, bands))
    bpc, _ = band_power_cwt(counts, 10.0, bands)
    assert bpc.dtype == np.float32 and bpc.shape == (6, 2)


# ----------------------------------------------------------------------------- stats.py
def _tie_correction_loop(ranks: np.ndarray) -> np.ndarray:
    """The previous (per-column Python loop) implementation, kept as the reference."""
    n, m = ranks.shape
    corr = np.ones(m)
    if n < 2:
        return corr
    srt = np.sort(ranks, axis=0)
    for j in range(m):
        col = srt[:, j]
        change = np.r_[True, col[1:] != col[:-1]]
        counts = np.diff(np.r_[np.flatnonzero(change), n])
        t = counts[counts > 1]
        if t.size:
            corr[j] = 1.0 - float(np.sum(t ** 3 - t)) / float(n ** 3 - n)
    return corr


def test_tie_correction_vectorised_matches_loop():
    rng = np.random.default_rng(9)
    for n, m in [(2, 1), (7, 5), (60, 300)]:
        v = rng.poisson(2.0, size=(n, m)).astype(float)
        v[:, 0] = 1.0                                    # fully tied column
        ranks = ss.rankdata(v, axis=0)
        assert np.array_equal(_tie_correction(ranks), _tie_correction_loop(ranks))
    assert _tie_correction(np.ones((1, 4))).tolist() == [1.0] * 4
    # NaN entries form runs of length 1 and never contribute to the tie term
    r = np.array([[1.5, np.nan], [1.5, np.nan], [3.0, np.nan], [np.nan, 1.0]])
    assert _tie_term(r).tolist() == [6.0, 0.0]


def test_kruskal_mannwhitney_wilcoxon_match_scipy():
    rng = np.random.default_rng(10)
    n, m = 48, 150
    v = rng.poisson(2.5, size=(n, m)).astype(float)       # heavy ties
    v[:, 0] = 3.0                                          # constant column -> NaN p
    labels = rng.integers(0, 3, size=n)

    H, p = kruskal_vectorised(v, labels)
    for j in range(1, m):
        r = ss.kruskal(*[v[labels == g, j] for g in np.unique(labels)])
        assert H[j] == pytest.approx(r.statistic, rel=0, abs=1e-10) and p[j] == pytest.approx(r.pvalue, rel=0, abs=1e-12)
    assert np.isnan(p[0])

    a, b = v[:20], v[20:]
    auroc, p_mw = mannwhitney_vectorised(a, b)
    for j in range(1, m):
        r = ss.mannwhitneyu(a[:, j], b[:, j], method="asymptotic", use_continuity=False)
        assert auroc[j] * a.shape[0] * b.shape[0] == pytest.approx(r.statistic, abs=1e-9)
        assert p_mw[j] == pytest.approx(r.pvalue, rel=0, abs=1e-12)
    assert auroc[0] == 0.5 and np.isnan(p_mw[0])

    d = rng.integers(-3, 4, size=(n, m)).astype(float)   # many zeros and ties
    d[:, 0] = 0.0                                          # all zero -> NaN
    d[:4, 1] = 1.0; d[4:, 1] = 0.0                         # only 4 non-zero -> NaN (below the n >= 5 floor)
    effect, p_w = wilcoxon_vectorised(d)
    assert np.isnan(p_w[0]) and np.isnan(effect[0]) and np.isnan(p_w[1])
    for j in range(2, m):
        dj = d[:, j][d[:, j] != 0]
        r = ss.wilcoxon(dj, correction=False, method="approx")
        assert p_w[j] == pytest.approx(r.pvalue, rel=0, abs=1e-12)
        assert effect[j] == pytest.approx(((dj > 0).sum() - (dj < 0).sum()) / len(dj))
    # no non-zero column at all -> early return with NaNs
    e0, p0 = wilcoxon_vectorised(np.zeros((6, 3)))
    assert np.isnan(e0).all() and np.isnan(p0).all()


def test_context_grid_is_anchored_at_go(tmp_path):
    """The last context edge is the go cue even when the delay is not a whole number of bins; a spike just
    after go must fall in the target, never in the last context bin."""
    import numpy as np
    from delaycast.config import load_config
    from delaycast.data.rasters import load_trial_rasters
    from rodent_tcnn.data.synthetic import generate_trial

    payload = generate_trial("Left", 6, np.random.default_rng(0), 1)
    go = float(payload["delay_start_times"]) + 1.2057          # 120.57 bins -> rounds to 121 bins of 10 ms
    payload["delay_stop_times"] = np.asarray(go)
    payload["go_start_times"] = np.asarray(go)
    payload["go_stop_times"] = np.asarray(go + 1.5)
    st = np.asarray(payload["spike_times"], dtype=object)
    st[0] = np.asarray([go + 0.002])                            # a spike 2 ms after the go cue
    payload["spike_times"] = st
    p = tmp_path / "t.npz"
    np.savez(p, **payload)
    tr = load_trial_rasters(p, load_config(None))
    assert tr.ctx_edges[-1] == pytest.approx(go, abs=1e-9)
    region = [r for r in tr.context if tr.context[r].shape[0] > 0][0]
    assert tr.context[region][0, -1] == 0
    assert tr.target[region][0, 0] == 1


def test_duplicate_sessions_are_detected_and_dropped():
    """Two caches whose trials share absolute delay-onset timestamps are the same recording (Data vs Data2)."""
    import numpy as np
    import pandas as pd
    from delaycast import REGIONS
    from delaycast.config import load_config
    from delaycast.data.cache import SessionCache, drop_duplicate_sessions, find_duplicate_sessions

    def make(session, dataset, onsets):
        n = len(onsets)
        return SessionCache(session=session, dataset=dataset, subject=session,
                            context={r: np.zeros((n, 3, 120), np.uint8) for r in REGIONS},
                            target={r: np.zeros((n, 3, 30), np.uint8) for r in REGIONS},
                            unit_ids={r: np.arange(3) for r in REGIONS}, labels=np.zeros(n, int), trials=np.arange(n),
                            meta=pd.DataFrame({"ep_delay_start_times": onsets}), bin_ms=10.0, target_bin_ms=50.0)

    onsets = np.cumsum(np.random.default_rng(0).uniform(6, 9, size=100)) + 12.3
    a_dup = make("A/Session2", "A", onsets)
    a_other = make("A/Session1", "A", onsets + 1000.0)
    b = make("B/sub-1_ses-x", "B", onsets)
    dup = find_duplicate_sessions([a_dup, a_other, b])
    assert list(dup.session_a) == ["A/Session2"] and list(dup.session_b) == ["B/sub-1_ses-x"]
    kept = drop_duplicate_sessions([a_dup, a_other, b], load_config(None))
    assert sorted(c.session for c in kept) == ["A/Session1", "B/sub-1_ses-x"]


def test_positional_alignment_without_unit_ids_drops_mismatched_trials(tmp_path, caplog):
    """Pre-split schema (no IDs): a trial whose unit count differs cannot be aligned and is dropped."""
    rng = np.random.default_rng(6)
    n_units = {"ALM_L": 3, "ALM_R": 2, "STR_L": 2, "STR_R": 1}
    _write_b(tmp_path, 1, "Left", _to_split(_payload(rng, n_units, "Left", t0=0.0)))
    _write_b(tmp_path, 2, "Right", _to_split(_payload(rng, {**n_units, "STR_L": 3}, "Right", t0=10.0)))
    _write_b(tmp_path, 3, "Right", _to_split(_payload(rng, n_units, "Right", t0=20.0)))
    cfg = _cfg(tmp_path)
    with caplog.at_level(logging.WARNING, logger="delaycast.data.cache"):
        (c,) = build_cache(cfg, force=True)
    assert c.trials.tolist() == [1, 3] and c.qc_info["unit_alignment"] == "positional"
    assert c.qc_info["drop_reasons"] == {"unit_count_mismatch": 1}
    assert any("unit_count_mismatch:STR_L:3!=2" in rec.message for rec in caplog.records)


def test_data2_lick_times_from_csv_and_outcome_rule(tmp_path):
    """Data2 NPZs carry no lick arrays: the behavioural-log row supplies them (string lists), and without any lick
    record the folder label is kept unverified instead of being called 'Ignore'."""
    from delaycast.data.rasters import parse_time_list

    assert parse_time_list("2.80, 2.93; 3.1").tolist() == [2.8, 2.93, 3.1]
    assert parse_time_list(2.5).tolist() == [2.5] and parse_time_list(float("nan")).size == 0
    assert parse_time_list("N/A").size == 0 and parse_time_list("").size == 0 and parse_time_list(None).size == 0

    rng = np.random.default_rng(7)
    n_units = {"ALM_L": 3, "ALM_R": 2, "STR_L": 2, "STR_R": 1}
    plan = {  # trial: (folder, t0, csv row or None)   go cue = t0 + 2.55
        1: ("Left", 0.0, {"outcome": "hit", "early_lick": "no early", "left_lick_times": "2.80, 2.93, 3.05", "right_lick_times": ""}),
        2: ("Right", 10.0, {"outcome": "hit", "early_lick": "early", "left_lick_times": "", "right_lick_times": "12.8"}),
        3: ("Right", 20.0, {"outcome": "hit", "early_lick": "no early", "left_lick_times": "21.9", "right_lick_times": "22.8"}),
        4: ("Left", 30.0, {"outcome": "hit", "early_lick": "no early", "left_lick_times": "", "right_lick_times": "32.8, 32.9"}),
        5: ("Right", 40.0, {"outcome": "miss", "early_lick": "no early", "left_lick_times": "", "right_lick_times": "42.8"}),
        6: ("Ignore", 50.0, None),
        7: ("Left", 60.0, {"outcome": "hit", "early_lick": "no early", "left_lick_times": "62.8", "right_lick_times": "62.9"}),
        8: ("Ignore", 70.0, {"outcome": "ignore", "early_lick": "no early", "left_lick_times": "", "right_lick_times": ""}),
    }
    rows = []
    for tr, (cls, t0, row) in plan.items():
        _write_b(tmp_path, tr, cls, _strip_licks(_payload(rng, n_units, cls, t0=t0)))
        if row is not None:
            rows.append({"trial": tr, "trial_instruction": "left" if cls == "Left" else "right", **row})
    _write_b_csv(tmp_path, rows)

    cfg = _cfg(tmp_path)
    (c,) = build_cache(cfg, force=True)
    assert c.trials.tolist() == [1, 5, 6, 8] and c.labels.tolist() == [1, 2, 0, 0]
    assert c.meta.lick_source.tolist() == ["csv", "csv", "none", "csv"]
    assert c.meta.csv_outcome.tolist() == ["hit", "miss", "", "ignore"]
    assert c.qc_info["lick_sources"] == {"csv": 3, "none": 1}
    cache_sub = next(p for p in (tmp_path / "cache").iterdir() if p.is_dir())
    qc = pd.read_csv(cache_sub / "qc_log.csv").set_index("trial")
    assert qc.loc[2, "reason"] == "csv_early_lick"
    assert qc.loc[3, "reason"] == "early_lick_csv"
    assert qc.loc[4, "reason"] == "label_mismatch(folder=Left,licks=Right)"
    assert qc.loc[7, "reason"] == "licked_both_sides"
    assert qc.loc[6, "kept"] and qc.loc[6, "lick_source"] == "none" and pd.isna(qc.loc[6, "lick_label"])
    assert qc.loc[1, "lick_label"] == "Left"

    # instruction-only analysis: error trials can be excluded through the config
    cfg2 = _cfg(tmp_path, **{"data.qc.csv_keep_outcomes": ["hit", "ignore"], "data.cache_dir": str(tmp_path / "cache2")})
    (c2,) = build_cache(cfg2, force=True)
    assert c2.trials.tolist() == [1, 6, 8] and c2.qc_info["drop_reasons"]["csv_outcome_miss"] == 1


def test_small_sessions_are_excluded_and_training_set_is_checked():
    from delaycast.data.cache import drop_small_sessions
    from delaycast.data.dataset import TrialDataset, tensors_from_indices
    from delaycast.features.selection import _stratified_subsample
    from delaycast.train import _check_training_set

    def make(session, labels):
        n = len(labels)
        return SessionCache(session=session, dataset="B", subject=session,
                            context={r: np.zeros((n, 2, 120), np.uint8) for r in REGIONS},
                            target={r: np.zeros((n, 2, 30), np.uint8) for r in REGIONS},
                            unit_ids={r: np.arange(2) for r in REGIONS}, labels=np.asarray(labels, int), trials=np.arange(n),
                            meta=pd.DataFrame({"trial": np.arange(n)}), bin_ms=10.0, target_bin_ms=50.0)

    cfg = load_config(None)
    big, tiny, lopsided = make("B/big", [1, 2] * 20), make("B/tiny", [0]), make("B/lopsided", [1] * 40 + [2] * 3)
    assert [c.session for c in drop_small_sessions([big, tiny, lopsided], cfg)] == ["B/big"]
    cfg.set_path("data.min_trials_per_session", 1)
    cfg.set_path("data.min_trials_per_lick_class", 0)
    assert len(drop_small_sessions([big, tiny, lopsided], cfg)) == 3

    assert _stratified_subsample(np.zeros(0, int), 0.5, np.random.default_rng(0)).size == 0
    assert len(_stratified_subsample(np.array([1, 1, 1, 2, 2, 2]), 0.5, np.random.default_rng(0))) == 4

    idx = {r: np.arange(2) for r in REGIONS}
    t_big, t_tiny = tensors_from_indices(big, idx, cfg), tensors_from_indices(tiny, idx, cfg)
    ok_train = TrialDataset([t_big], {"B/big": np.arange(30)})
    ok_val = TrialDataset([t_big], {"B/big": np.arange(30, 40)})
    _check_training_set(ok_train, ok_val, {"B/big": {"train": np.arange(30)}}, held=[])
    with pytest.raises(RuntimeError, match="degenerate"):
        _check_training_set(TrialDataset([t_tiny], {"B/tiny": np.arange(1)}), ok_val, {"B/tiny": {"train": np.arange(1)}}, held=["A/x"])
    with pytest.raises(RuntimeError, match="degenerate"):
        _check_training_set(ok_train, TrialDataset([t_big], {}), {"B/big": {"train": np.arange(30)}}, held=[])
