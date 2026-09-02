"""Biophysically motivated synthetic trials matching the real NPZ schema.

Delay-choice and ramping ALM cells, plus lick-locked striatal cells, are
injected so neuron selection and the TCNN have a recoverable ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import REGIONS, REGION_TO_KEY


NEURON_TYPES = (
    "delay_choice",
    "delay_ramp",
    "lick_burst",
    "sample_sensory",
    "tonic",
    "delay_suppress",
    "sparse",
)


def _inhomogeneous_poisson(rate_hz: np.ndarray, t0: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Generate spike times from a piecewise-constant rate vector (Hz)."""
    spikes: list[float] = []
    for i, r in enumerate(rate_hz):
        lam = max(float(r), 0.0) * dt
        n = rng.poisson(lam)
        if n:
            spikes.extend((t0 + (i + rng.random(n)) * dt).tolist())
    if not spikes:
        return np.asarray([], dtype=np.float64)
    return np.sort(np.asarray(spikes, dtype=np.float64))


def _region_preference(region: str, label: str) -> float:
    """Contralateral ALM bias; striatum more execution-aligned."""
    if label == "Ignore":
        return 0.0
    if region == "left ALM":
        return 0.85 if label == "Right" else -0.35
    if region == "right ALM":
        return 0.85 if label == "Left" else -0.35
    if region == "left Striatum":
        return 0.55 if label == "Right" else -0.15
    if region == "right Striatum":
        return 0.55 if label == "Left" else -0.15
    return 0.0


def _assign_types(n: int, region: str, rng: np.random.Generator) -> np.ndarray:
    if "ALM" in region:
        p = np.array([0.22, 0.16, 0.10, 0.10, 0.22, 0.08, 0.12])
    else:
        p = np.array([0.10, 0.10, 0.28, 0.08, 0.22, 0.08, 0.14])
    p = p / p.sum()
    return rng.choice(NEURON_TYPES, size=n, p=p)


def trial_timeline(rng: np.random.Generator, label: str) -> dict[str, float | np.ndarray]:
    start = 0.0
    presample_start = 0.50
    sample_start = presample_start + rng.uniform(0.65, 0.90)
    sample_stop = sample_start + 0.65
    delay_start = sample_stop
    delay_stop = delay_start + 1.20
    go_start = delay_stop
    go_stop = go_start + 1.50
    stop = go_stop + 0.20
    left = np.asarray([], dtype=np.float64)
    right = np.asarray([], dtype=np.float64)
    if label in {"Left", "Right"}:
        first = go_start + rng.uniform(0.12, 0.32)
        n_licks = int(rng.integers(6, 13))
        isi = rng.uniform(0.11, 0.16, size=n_licks)
        times = first + np.cumsum(np.concatenate([[0.0], isi[:-1]]))
        times = times[times < stop]
        if label == "Left":
            left = times
        else:
            right = times
    return {
        "trial_start": start,
        "trial_stop": stop,
        "presample_start_times": presample_start,
        "presample_stop_times": sample_start,
        "sample_start_times": sample_start,
        "sample_stop_times": sample_stop,
        "delay_start_times": delay_start,
        "delay_stop_times": delay_stop,
        "go_start_times": go_start,
        "go_stop_times": go_stop,
        "left_lick_times": left,
        "right_lick_times": right,
    }


def _rate_profile(
    ntype: str,
    times: np.ndarray,
    tl: dict,
    label: str,
    pref: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rate = np.full(times.shape, rng.uniform(1.2, 3.5), dtype=np.float64)
    sample0, sample1 = tl["sample_start_times"], tl["sample_stop_times"]
    d0, d1 = tl["delay_start_times"], tl["delay_stop_times"]
    g0 = tl["go_start_times"]
    lick0 = None
    if label == "Left" and len(tl["left_lick_times"]):
        lick0 = float(tl["left_lick_times"][0])
    if label == "Right" and len(tl["right_lick_times"]):
        lick0 = float(tl["right_lick_times"][0])

    if ntype == "delay_choice":
        gain = 8.0 + 10.0 * max(pref, 0.0)
        if label == "Ignore":
            gain *= 0.15
        elif pref < 0:
            gain *= 0.25
        mask = (times >= d0) & (times < d1)
        rate[mask] += gain
        # persist slightly into go
        rate[(times >= d1) & (times < d1 + 0.25)] += 0.45 * gain
    elif ntype == "delay_ramp":
        mask = (times >= d0) & (times < d1)
        frac = np.clip((times[mask] - d0) / max(d1 - d0, 1e-6), 0, 1)
        amp = 6.0 + 9.0 * max(pref, 0.0)
        if label == "Ignore":
            amp *= 0.2
        elif pref < 0:
            amp *= 0.3
        rate[mask] += amp * (frac**1.4)
        rate[(times >= d1) & (times < d1 + 0.15)] += 0.7 * amp
    elif ntype == "lick_burst":
        if lick0 is not None and pref >= 0:
            burst = np.exp(-0.5 * ((times - lick0) / 0.07) ** 2)
            rate += (18.0 + 8.0 * max(pref, 0)) * burst
            prelude = (times >= d1 - 0.25) & (times < lick0)
            rate[prelude] += 2.5 * max(pref, 0)
        else:
            rate += 0.2
    elif ntype == "sample_sensory":
        mask = (times >= sample0) & (times < sample1)
        rate[mask] += 9.0
    elif ntype == "delay_suppress":
        mask = (times >= d0) & (times < d1)
        rate[mask] *= 0.25
    elif ntype == "sparse":
        rate = np.full(times.shape, rng.uniform(0.15, 0.7))
    # tonic: baseline only
    rate += rng.normal(0, 0.15, size=rate.shape)
    return np.clip(rate, 0.05, 80.0)


def generate_trial(
    label: str,
    n_per_region: int,
    rng: np.random.Generator,
    trial_id: int,
    figure_dense: bool = False,
) -> dict:
    tl = trial_timeline(rng, label)
    t0, t1 = float(tl["trial_start"]), float(tl["trial_stop"])
    dt = 0.002
    times = np.arange(t0, t1, dt)
    unit_ids = []
    regions = []
    spikes = []
    types = []
    prefs = []
    uid = 1
    for region in REGIONS:
        assigned = _assign_types(n_per_region, region, rng)
        # Guarantee a few ground-truth selectable cells per ALM hemisphere.
        if "ALM" in region:
            assigned[: max(3, n_per_region // 8)] = "delay_choice"
            assigned[max(3, n_per_region // 8) : max(5, n_per_region // 6)] = "delay_ramp"
        for ntype in assigned:
            pref = _region_preference(region, label)
            pref = float(np.clip(pref + rng.normal(0, 0.12), -1.0, 1.0))
            rate = _rate_profile(ntype, times, tl, label, pref, rng)
            unit_ids.append(uid)
            regions.append(region)
            spikes.append(_inhomogeneous_poisson(rate, t0, dt, rng))
            types.append(ntype)
            prefs.append(pref)
            uid += 1
    payload = {
        "unit_ids": np.asarray(unit_ids, dtype=np.int32),
        "brain_region": np.asarray(regions, dtype=object),
        "spike_times": np.array(spikes, dtype=object),
        "neuron_type": np.asarray(types, dtype=object),
        "laterality_pref": np.asarray(prefs, dtype=np.float32),
        "label": np.asarray(label),
        "trial_id": np.asarray(trial_id, dtype=np.int32),
        **{k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in tl.items()},
    }
    if figure_dense:
        payload["figure_dense"] = np.asarray(True)
    return payload


def generate_demo_tree(
    root: Path,
    n_per_region: int = 32,
    trials_per_class: int = 8,
    seed: int = 7,
    figure_units: int = 72,
) -> dict:
    """Write Data/ and Data2/ trees plus a high-density figure trial."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    data_root = root / "Data"
    data2_root = root / "Data2"
    rows = []

    def write_npz(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    # Dataset 1: Session1, Session2
    trial_counter = 1
    for session_i, session in enumerate(("Session1", "Session2"), start=1):
        for cls in ("Ignore", "Left", "Right"):
            for _ in range(trials_per_class):
                payload = generate_trial(cls, n_per_region, rng, trial_counter)
                write_npz(data_root / session / "Rasters" / cls / f"trial_{trial_counter}.npz", payload)
                trial_counter += 1

    # Dataset 2: two sessions under two subjects
    sessions = [
        ("sub-440957", "sub-440957_ses-20190211T143614_behavior+ecephys+image+ogen"),
        ("sub-442571x", "sub-442571_ses-20190227T134351_behavior+ecephys+image+ogen"),
    ]
    csv_rows = []
    for subject, ses in sessions:
        for cls in ("Ignore", "Left", "Right"):
            for _ in range(trials_per_class):
                payload = generate_trial(cls, n_per_region, rng, trial_counter)
                write_npz(data2_root / subject / ses / "NPZ" / cls / f"trial{trial_counter}.npz", payload)
                left = payload["left_lick_times"]
                right = payload["right_lick_times"]
                csv_rows.append(
                    {
                        "session_path": f"{subject.replace('x', '')}/{ses}.nwb",
                        "subject_id": re_subject(subject),
                        "session_id": "",
                        "trial": trial_counter,
                        "trial_uid": trial_counter,
                        "id": trial_counter - 1,
                        "photostim_onset": "N/A",
                        "photostim_power": "N/A",
                        "photostim_duration": "N/A",
                        "task": "audio delay",
                        "task_protocol": 1,
                        "trial_instruction": "left" if cls == "Left" else ("right" if cls == "Right" else "left"),
                        "early_lick": "no early",
                        "outcome": "ignore" if cls == "Ignore" else "hit",
                        "auto_water": 0,
                        "free_water": 0,
                        "start_time": float(payload["trial_start"]),
                        "stop_time": float(payload["trial_stop"]),
                        "presample_start_times": float(payload["presample_start_times"]),
                        "presample_stop_times": float(payload["presample_stop_times"]),
                        "sample_start_times": float(payload["sample_start_times"]),
                        "sample_stop_times": float(payload["sample_stop_times"]),
                        "delay_start_times": float(payload["delay_start_times"]),
                        "delay_stop_times": float(payload["delay_stop_times"]),
                        "go_start_times": float(payload["go_start_times"]),
                        "go_stop_times": float(payload["go_stop_times"]),
                        "left_lick_times": ",".join(f"{x:.4f}" for x in left),
                        "right_lick_times": ",".join(f"{x:.4f}" for x in right),
                        "photostim_start_times": "",
                        "photostim_stop_times": "",
                        "trial_time_total": float(payload["trial_stop"] - payload["trial_start"]),
                        "trialend_start_times": "",
                        "trialend_stop_times": "",
                        "excluded": False,
                        "exclusion_reason": "",
                        "session_dir": ses,
                    }
                )
                trial_counter += 1

    df = pd.DataFrame(csv_rows)
    df.to_csv(data2_root / "combined_audited_master_log.csv", index=False)
    df.to_csv(data2_root / "combined_behavioral_master_log.csv", index=False)
    for subject, ses in sessions:
        sub = df[df["session_dir"] == ses]
        dest = data2_root / subject / ses / "behavioral_master_log.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(dest, index=False)

    # High-density figure trial (Right lick — classic contralateral ALM story)
    fig_payload = generate_trial("Right", figure_units, np.random.default_rng(seed + 99), 9999, figure_dense=True)
    fig_path = root / "figure_trial" / "trial_figure_right.npz"
    write_npz(fig_path, fig_payload)

    # Second figure trial for Left, used in appendix-style panel
    fig_left = generate_trial("Left", figure_units, np.random.default_rng(seed + 101), 9998, figure_dense=True)
    write_npz(root / "figure_trial" / "trial_figure_left.npz", fig_left)
    fig_ign = generate_trial("Ignore", figure_units, np.random.default_rng(seed + 103), 9997, figure_dense=True)
    write_npz(root / "figure_trial" / "trial_figure_ignore.npz", fig_ign)

    return {
        "root": str(root),
        "figure_trial": str(fig_path),
        "n_data_trials": 2 * 3 * trials_per_class,
        "n_data2_trials": 2 * 3 * trials_per_class,
        "n_per_region": n_per_region,
    }


def re_subject(name: str) -> str:
    return "".join(ch for ch in name if ch.isdigit()) or name


def generate_matched_population(
    template: dict,
    n_per_class: int = 6,
    seed: int = 21,
) -> list[dict]:
    """Re-simulate trials that share the template's unit types and laterality."""
    rng = np.random.default_rng(seed)
    regions = np.asarray(template["brain_region"])
    types = np.asarray(template["neuron_type"])
    unit_ids = np.asarray(template["unit_ids"])
    trials = []
    dt = 0.002
    for cls in ("Ignore", "Left", "Right"):
        for _ in range(n_per_class):
            tl = trial_timeline(rng, cls)
            t0, t1 = float(tl["trial_start"]), float(tl["trial_stop"])
            times = np.arange(t0, t1, dt)
            spikes = []
            prefs = []
            for i, region in enumerate(regions):
                pref = float(np.clip(_region_preference(str(region), cls) + rng.normal(0, 0.08), -1, 1))
                # keep type-driven identity; laterality follows this trial's class
                rate = _rate_profile(str(types[i]), times, tl, cls, pref, rng)
                spikes.append(_inhomogeneous_poisson(rate, t0, dt, rng))
                prefs.append(pref)
            payload = {
                "unit_ids": unit_ids,
                "brain_region": regions,
                "spike_times": np.array(spikes, dtype=object),
                "neuron_type": types,
                "laterality_pref": np.asarray(prefs, dtype=np.float32),
                "label": np.asarray(cls),
                **{k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in tl.items()},
            }
            trials.append({"payload": payload, "label": cls})
    return trials
