"""Generate a small synthetic dataset that mirrors the on-disk layout of ``Data`` and ``Data2``.

The generator is only meant for smoke-testing the pipeline end-to-end on a machine without
the real recordings. Spike trains are inhomogeneous Poisson processes with:
  * a random fraction of "informative" units per region that ramp during the delay in a
    choice-dependent way (Left / Right / Ignore),
  * a per-trial excitability gain shared by the late delay and the response epoch of "coupled" units, so
    that a unit's late-delay rate predicts its own response-epoch rate *within* a class (criterion C),
  * a subset of units with choice-dependent 4-12 Hz modulation (spectro-temporal information).
Epoch timings follow the audited behavioral logs: presample ~0.5-0.9 s, sample 0.65 s,
delay 1.2 s, response window 1.5 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES, REGION_LABELS, REGIONS

SAMPLE_S, DELAY_S, GO_S = 0.65, 1.2, 1.5


def _poisson_times(rate_hz: np.ndarray, t0: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(np.clip(rate_hz, 0, None) * dt)
    idx = np.repeat(np.arange(len(rate_hz)), counts)
    return t0 + (idx + rng.random(len(idx))) * dt


def _session_units(rng, unit_counts: dict[str, int], info_frac=0.35, spec_frac=0.2, coupled_frac=0.5) -> dict[str, dict]:
    """Draw the fixed properties of every unit of a session once (identity is stable across trials)."""
    units = {}
    for r in REGIONS:
        n = unit_counts[r]
        units[r] = {
            "base": rng.gamma(2.0, 2.5, size=n),            # baseline Hz
            "informative": rng.random(n) < info_frac,
            "spectral": rng.random(n) < spec_frac,
            "coupled": rng.random(n) < coupled_frac,        # late-delay <-> response trial-gain coupling
            "pref": rng.integers(0, 3, size=n),             # preferred class of informative units
            "amp": rng.uniform(6, 14, size=n),
        }
    return units


def _unit_profile(units: dict, cls_idx: int, rng: np.random.Generator | None = None):
    """Return (n_units, T) rate profiles (context + target) for one region and one class.

    ``rng`` draws the per-trial excitability gain of coupled units (log-normal, sd 0.35) that multiplies the
    late-delay and response-epoch rates together."""
    dt = 0.005
    t_pre, t_sample, t_delay, t_go = 0.7, SAMPLE_S, DELAY_S, GO_S
    n = int(round((t_pre + t_sample + t_delay + t_go) / dt))
    t = np.arange(n) * dt
    base = units["base"]
    n_units = len(base)
    rates = np.tile(base[:, None], (1, n))
    d0, d1 = t_pre + t_sample, t_pre + t_sample + t_delay
    in_delay = (t >= d0) & (t < d1)
    in_go = t >= d1
    ramp = np.clip((t - d0) / t_delay, 0, 1)
    late_win = (t >= d1 - 0.5) & (t < d1)
    trial_gain = np.exp(rng.normal(0.0, 0.35, size=n_units)) if rng is not None else np.ones(n_units)
    for u in range(n_units):
        if units["informative"][u]:
            gain = 1.0 if units["pref"][u] == cls_idx else -0.4
            rates[u, in_delay] += gain * units["amp"][u] * ramp[in_delay]
            late = rates[u, in_delay][-40:].mean()
            rates[u, in_go] += 0.8 * (late - base[u]) * np.exp(-(t[in_go] - d1) / 0.6) + (
                6.0 * (units["pref"][u] == cls_idx) * np.exp(-((t[in_go] - d1 - 0.25) ** 2) / 0.02)
            )
        if units["spectral"][u]:
            f = 6.0 + 2.0 * cls_idx
            rates[u, in_delay] += 4.0 * (1 + np.sin(2 * np.pi * f * t[in_delay]))
        if units.get("coupled", np.zeros(n_units, bool))[u]:
            rates[u, late_win] *= trial_gain[u]
            rates[u, in_go] *= trial_gain[u]
    return np.clip(rates, 0.2, None), dt, (t_pre, t_sample, t_delay, t_go)


def _make_trial(rng, units: dict[str, dict], cls: str, t_start: float):
    cls_idx = CLASSES.index(cls)
    regions, spikes, uids = [], [], []
    uid = 0
    t_pre = t_sample = t_delay = t_go = None
    for r in REGIONS:
        rates, dt, (t_pre, t_sample, t_delay, t_go) = _unit_profile(units[r], cls_idx, rng)
        for u in range(rates.shape[0]):
            spikes.append(_poisson_times(rates[u], t_start, dt, rng))
            regions.append(REGION_LABELS[r])
            uids.append(uid)
            uid += 1
    presample_start = t_start + 0.2
    sample_start = t_start + t_pre
    delay_start = sample_start + t_sample
    go_start = delay_start + t_delay
    go_stop = go_start + t_go
    trial_stop = go_stop + 0.2
    left, right = np.empty(0), np.empty(0)
    if cls in ("Left", "Right"):
        first = go_start + rng.uniform(0.15, 0.3)
        licks = first + np.cumsum(np.r_[0, rng.uniform(0.11, 0.17, size=rng.integers(6, 12))])
        licks = licks[licks < go_stop]
        if cls == "Left":
            left = licks
        else:
            right = licks
    payload = {
        "unit_ids": np.asarray(uids),
        "brain_region": np.asarray(regions),
        "spike_times": np.asarray(spikes, dtype=object),
        "trial_start": np.float64(t_start), "trial_stop": np.float64(trial_stop),
        "presample_start_times": np.float64(presample_start), "presample_stop_times": np.float64(sample_start),
        "sample_start_times": np.float64(sample_start), "sample_stop_times": np.float64(delay_start),
        "delay_start_times": np.float64(delay_start), "delay_stop_times": np.float64(go_start),
        "go_start_times": np.float64(go_start), "go_stop_times": np.float64(go_stop),
        "left_lick_times": left, "right_lick_times": right,
    }
    return payload, trial_stop


def _csv_row(session_dir: str, subject: str, trial: int, payload: dict, cls: str) -> dict:
    fmt = lambda a: ", ".join(f"{x:.4f}" for x in a) if len(a) else ""
    return {
        "session_path": f"{subject}/{session_dir}.nwb", "subject_id": subject.replace("sub-", ""), "session_id": "",
        "trial": trial, "trial_uid": trial, "id": trial - 1, "photostim_onset": "N/A", "photostim_power": "N/A",
        "photostim_duration": "N/A", "task": "audio delay", "task_protocol": 1,
        "trial_instruction": "left" if cls == "Left" else "right", "early_lick": "no early",
        "outcome": "ignore" if cls == "Ignore" else "hit", "auto_water": 0, "free_water": 0,
        "start_time": float(payload["trial_start"]), "stop_time": float(payload["trial_stop"]),
        "presample_start_times": float(payload["presample_start_times"]), "presample_stop_times": float(payload["presample_stop_times"]),
        "sample_start_times": float(payload["sample_start_times"]), "sample_stop_times": float(payload["sample_stop_times"]),
        "delay_start_times": float(payload["delay_start_times"]), "delay_stop_times": float(payload["delay_stop_times"]),
        "go_start_times": float(payload["go_start_times"]), "go_stop_times": float(payload["go_stop_times"]),
        "left_lick_times": fmt(payload["left_lick_times"]), "right_lick_times": fmt(payload["right_lick_times"]),
        "photostim_start_times": "", "photostim_stop_times": "",
        "trial_time_total": float(payload["trial_stop"] - payload["trial_start"]),
        "trialend_start_times": "", "trialend_stop_times": "", "excluded": False, "exclusion_reason": "",
        "session_dir": session_dir,
    }


def make_synthetic(root: str | Path, n_sessions_a=2, n_sessions_b=2, trials_per_class=(12, 40, 40),
                   units_per_region=(14, 22), seed=0) -> tuple[Path, Path]:
    """Create ``<root>/Data`` and ``<root>/Data2``. Returns their paths."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    data_a, data_b = root / "Data", root / "Data2"

    # ---- Dataset A: Session*/Rasters/<Class>/trial_<n>.npz
    for s in range(1, n_sessions_a + 1):
        sess = data_a / f"Session{s}"
        units = _session_units(rng, {r: int(rng.integers(*units_per_region)) for r in REGIONS})
        t = 0.0
        order = [c for c, n in zip(CLASSES, trials_per_class) for _ in range(n)]
        rng.shuffle(order)
        for i, cls in enumerate(order, start=1):
            payload, t_stop = _make_trial(rng, units, cls, t)
            d = sess / "Rasters" / cls
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"trial_{i}.npz", **payload)
            (sess / "Videos" / cls).mkdir(parents=True, exist_ok=True)
            (sess / "Videos" / cls / f"trial_{i:04d}_{'ignore' if cls == 'Ignore' else 'lick_' + cls.lower()}.avi").touch()
            t = t_stop + rng.uniform(1.5, 3.0)

    # ---- Dataset B: sub-*/sub-*_ses-*/NPZ/<Class>/trial<n>.npz + CSV logs
    combined = []
    for s in range(n_sessions_b):
        subject = f"sub-99{s:04d}"
        sess_dir = f"{subject}_ses-2019030{s + 1}T120000_behavior+ecephys+image+ogen"
        sess = data_b / subject / sess_dir
        units = _session_units(rng, {r: int(rng.integers(*units_per_region)) for r in REGIONS})
        t = 0.0
        order = [c for c, n in zip(CLASSES, trials_per_class) for _ in range(n)]
        rng.shuffle(order)
        rows = []
        for i, cls in enumerate(order, start=1):
            payload, t_stop = _make_trial(rng, units, cls, t)
            d = sess / "NPZ" / cls
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"trial{i}.npz", **payload)
            rows.append(_csv_row(sess_dir, subject, i, payload, cls))
            t = t_stop + rng.uniform(1.5, 3.0)
        df = pd.DataFrame(rows)
        df.drop(columns=["session_dir"]).to_csv(sess / "behavioral_master_log_audited.csv", index=False)
        combined.append(df)
    data_b.mkdir(parents=True, exist_ok=True)
    pd.concat(combined).to_csv(data_b / "combined_audited_master_log.csv", index=False)
    return data_a, data_b
