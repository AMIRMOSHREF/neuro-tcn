from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import TrialRecord, load_trial
from .model import ContextForecaster


def load_model(
    checkpoint: str | Path, device: torch.device
) -> tuple[ContextForecaster, dict, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = ContextForecaster(config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, config, state


def _eta_squared(values: list[float], labels: list[int]) -> float:
    x = np.asarray(values)
    y = np.asarray(labels)
    if len(x) < 3 or np.var(x) <= 1e-12:
        return 0.0
    grand = x.mean()
    between = sum(np.sum(y == group) * (x[y == group].mean() - grand) ** 2 for group in np.unique(y))
    total = np.sum((x - grand) ** 2)
    return float(between / total) if total > 0 else 0.0


@torch.no_grad()
def rank_neurons(
    checkpoint: str | Path,
    records: list[TrialRecord],
    output_csv: str | Path,
    device_name: str | None = None,
) -> list[dict]:
    """Aggregate learned gates by session-unit without mixing unit identities."""
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, state = load_model(checkpoint, device)
    training_groups = set(state.get("train_groups", []))
    if training_groups:
        records = [record for record in records if record.group in training_groups]
    observations: defaultdict[tuple, dict[str, list]] = defaultdict(
        lambda: {"gate": [], "rate": [], "label": [], "peak_bin": [], "top": []}
    )
    top_fraction = float(config["selection"]["top_fraction"])

    for record in records:
        sample = load_trial(record, config)
        delay = sample["delay"].unsqueeze(0).to(device)
        mask = sample["unit_mask"].unsqueeze(0).to(device)
        output = model(delay, mask)
        gates = output["neuron_gate"][0].cpu().numpy()
        temporal = output["temporal_attention"][0].cpu().numpy()
        unit_ids = sample["unit_ids"]
        for region_idx, region in enumerate(config["regions"]):
            active_indices = np.flatnonzero(sample["unit_mask"][region_idx].numpy())
            active_scores = gates[region_idx, active_indices]
            threshold = (
                np.quantile(active_scores, 1 - top_fraction)
                if len(active_scores)
                else np.inf
            )
            for unit_idx in active_indices:
                key = (record.group, region, str(unit_ids[region_idx, unit_idx]))
                slot = observations[key]
                score = float(gates[region_idx, unit_idx])
                slot["gate"].append(score)
                slot["rate"].append(float(delay[0, region_idx, unit_idx].sum().cpu()))
                slot["label"].append(int(sample["label"]))
                slot["peak_bin"].append(int(temporal[region_idx, unit_idx].argmax()))
                slot["top"].append(score >= threshold)

    rows: list[dict] = []
    for (group, region, unit_id), values in observations.items():
        labels = np.asarray(values["label"])
        top_membership = np.asarray(values["top"])
        class_stabilities = {
            int(label): float(top_membership[labels == label].mean())
            for label in np.unique(labels)
        }
        preferred_class, stability = max(
            class_stabilities.items(), key=lambda item: item[1]
        )
        modulation = _eta_squared(values["rate"], values["label"])
        mean_gate = float(np.mean(values["gate"]))
        reasons = []
        if stability >= float(config["selection"]["stability_threshold"]):
            reasons.append("stable top-attention membership")
        if modulation >= 0.05:
            reasons.append("class-modulated delay firing")
        if np.mean(values["rate"]) >= 1.0:
            reasons.append("reliable delay activity")
        is_stable = stability >= float(config["selection"]["stability_threshold"])
        has_activity = np.mean(values["rate"]) >= 1.0
        has_support = len(values["gate"]) >= 3
        rows.append(
            {
                "group": group,
                "region": region,
                "unit_id": unit_id,
                "n_trials": len(values["gate"]),
                "mean_gate": mean_gate,
                "selection_stability": stability,
                "preferred_class": ["Ignore", "Left", "Right"][preferred_class],
                "class_eta_squared": modulation,
                "mean_delay_spikes": float(np.mean(values["rate"])),
                "preferred_context_bin": int(np.median(values["peak_bin"])),
                "selected": bool(is_stable and has_activity and has_support),
                "reasons": "; ".join(reasons) if reasons else "not selected",
            }
        )
    rows.sort(key=lambda row: (row["region"], -row["selection_stability"], -row["mean_gate"]))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows
