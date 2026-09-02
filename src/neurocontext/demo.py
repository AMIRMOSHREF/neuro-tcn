from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

REGIONS = ["left ALM", "right ALM", "left Striatum", "right Striatum"]


def _trial_npz(path: Path, trial: int, label: str, rng: np.random.Generator) -> None:
    units_per_region = 8
    regions = np.repeat(REGIONS, units_per_region)
    unit_ids = np.asarray([f"{r[:2].replace(' ', '')}_{i:02d}" for r in REGIONS for i in range(units_per_region)])
    trial_start, delay_start, delay_stop, go_start, go_stop = 0.0, 1.4, 2.6, 2.6, 4.1
    response_preferences = {"Ignore": 0, "Left": 1, "Right": 2}
    class_idx = response_preferences[label]
    spike_times = np.empty(len(regions), dtype=object)
    for unit_idx in range(len(regions)):
        region_idx = unit_idx // units_per_region
        preferred = unit_idx % 3
        base_rate = 2.5 + 0.3 * region_idx
        delay_rate = base_rate + (7.0 if preferred == class_idx else 0.5)
        response_rate = base_rate + (10.0 if preferred == class_idx else 0.7)
        epochs = [
            rng.uniform(trial_start, delay_start, rng.poisson(base_rate * 1.4)),
            rng.uniform(delay_start, delay_stop, rng.poisson(delay_rate * 1.2)),
            rng.uniform(go_start, go_stop, rng.poisson(response_rate * 1.5)),
        ]
        spike_times[unit_idx] = np.sort(np.concatenate(epochs)).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        unit_ids=unit_ids,
        brain_region=regions,
        spike_times=spike_times,
        trial_start=trial_start,
        trial_stop=go_stop,
        presample_start_times=0.2,
        presample_stop_times=0.8,
        sample_start_times=0.8,
        sample_stop_times=1.4,
        delay_start_times=delay_start,
        delay_stop_times=delay_stop,
        go_start_times=go_start,
        go_stop_times=go_stop,
        left_lick_times=np.asarray([3.0]) if label == "Left" else np.asarray([]),
        right_lick_times=np.asarray([3.0]) if label == "Right" else np.asarray([]),
    )


def generate_demo(root: str | Path, base_config: dict, seed: int = 17) -> dict[str, Path]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    labels = ["Ignore", "Left", "Right"]
    data_root = root / "Data"
    data2_root = root / "Data2"
    metadata_path = root / "audited_metadata.csv"
    metadata_rows = []

    for session_idx in range(1, 4):
        for trial in range(1, 19):
            label = labels[(trial + session_idx) % 3]
            _trial_npz(
                data_root / f"Session{session_idx}" / "Rasters" / label / f"trial_{trial}.npz",
                trial, label, rng,
            )

    for subject_idx in range(1, 4):
        subject = f"sub-demo{subject_idx}"
        session = f"{subject}_ses-2026010{subject_idx}T120000_behavior+ecephys"
        for trial in range(1, 19):
            label = labels[(trial + subject_idx + 1) % 3]
            _trial_npz(
                data2_root / subject / session / "NPZ" / label / f"trial{trial}.npz",
                trial, label, rng,
            )
            metadata_rows.append(
                {
                    "session_path": f"{subject}/{session}.nwb",
                    "session_dir": session,
                    "subject_id": subject,
                    "trial": trial,
                    "trial_instruction": label.lower() if label != "Ignore" else "left",
                    "early_lick": "no early",
                    "outcome": "ignore" if label == "Ignore" else "hit",
                    "excluded": "False",
                }
            )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    demo_config = dict(base_config)
    demo_config["max_units_per_region"] = 8
    demo_config["delay_bins"] = 32
    demo_config["response_bins"] = 36
    demo_config["model"] = {**base_config["model"], "hidden_dim": 16, "tcn_layers": 3}
    demo_config["training"] = {
        **base_config["training"],
        "batch_size": 12,
        "epochs": 4,
        "patience": 4,
        "folds": 3,
    }
    config_path = root / "demo_config.yaml"
    config_path.write_text(yaml.safe_dump(demo_config, sort_keys=False), encoding="utf-8")
    return {
        "data_root": data_root,
        "data2_root": data2_root,
        "metadata": metadata_path,
        "config": config_path,
    }
