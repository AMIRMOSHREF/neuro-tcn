from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import canonical_region

REGION_COLORS = {
    "left ALM": "#176B87",
    "right ALM": "#D97706",
    "left Striatum": "#2E7D32",
    "right Striatum": "#A23B72",
}


def _read_selection(
    path: str | Path | None, session: str
) -> dict[tuple[str, str], dict]:
    if not path:
        return {}
    selected = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_group = row.get("group", "")
            if (
                row.get("selected", "").lower() == "true"
                and (not row_group or row_group.endswith(f":{session}"))
            ):
                selected[(row["region"], str(row["unit_id"]))] = row
    return selected


def make_selection_figure(
    npz_path: str | Path,
    output_path: str | Path,
    config: dict,
    selection_csv: str | Path | None = None,
) -> Path:
    """Create a publication-ready all-units versus selected-units raster figure."""
    npz_path = Path(npz_path)
    session = next(
        (parent.name for parent in npz_path.parents if "_ses-" in parent.name or parent.name.lower().startswith("session")),
        npz_path.parent.parent.name,
    )
    selected_rows = _read_selection(selection_csv, session)
    with np.load(npz_path, allow_pickle=True) as npz:
        regions_raw = np.asarray(npz["brain_region"])
        spikes = np.asarray(npz["spike_times"], dtype=object)
        unit_ids = np.asarray(npz["unit_ids"]).astype(str)
        delay_start = float(np.asarray(npz["delay_start_times"]).reshape(-1)[0])
        delay_stop = float(np.asarray(npz["delay_stop_times"]).reshape(-1)[0])
        go_start = float(np.asarray(npz["go_start_times"]).reshape(-1)[0])
        go_stop = float(np.asarray(npz["go_stop_times"]).reshape(-1)[0])
        trial_start = float(np.asarray(npz["trial_start"]).reshape(-1)[0])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8.5,
        }
    )
    fig, axes = plt.subplots(
        4, 2, figsize=(12.2, 9.2), sharex=True,
        gridspec_kw={"width_ratios": [1.65, 1.0], "hspace": 0.22, "wspace": 0.16},
    )
    any_selected = False
    summary_lines: list[str] = []
    for region_idx, region in enumerate(config["regions"]):
        color = REGION_COLORS[region]
        indices = [
            idx
            for idx, raw in enumerate(regions_raw)
            if canonical_region(raw, config.get("region_aliases", {})) == region
        ]
        selected_indices = [
            idx for idx in indices if (region, str(unit_ids[idx])) in selected_rows
        ]
        # A visible QA fallback is useful before a trained checkpoint exists, but is
        # deliberately not presented as model-based selection.
        if not selected_indices and not selection_csv and indices:
            activity = [
                np.sum(
                    (np.asarray(spikes[idx], dtype=float) >= delay_start)
                    & (np.asarray(spikes[idx], dtype=float) <= delay_stop)
                )
                for idx in indices
            ]
            count = max(1, int(np.ceil(len(indices) * float(config["selection"]["top_fraction"]))))
            selected_indices = [indices[i] for i in np.argsort(activity)[-count:]]
        any_selected |= bool(selected_indices)

        for column, shown, title in [
            (0, indices, "All recorded units"),
            (1, selected_indices, "Selected delay-encoding units"),
        ]:
            ax = axes[region_idx, column]
            for row, unit_idx in enumerate(shown):
                times = np.asarray(spikes[unit_idx], dtype=float) - delay_start
                valid = times[(times >= trial_start - delay_start) & (times <= go_stop - delay_start)]
                ax.vlines(valid, row - 0.42, row + 0.42, color=color, linewidth=0.48)
            ax.axvspan(0, delay_stop - delay_start, color="#CBD5E1", alpha=0.42, lw=0)
            ax.axvspan(
                go_start - delay_start, go_stop - delay_start,
                color="#FDE68A", alpha=0.36, lw=0,
            )
            ax.axvline(0, color="#475569", lw=0.8, ls="--")
            ax.axvline(go_start - delay_start, color="#92400E", lw=0.8, ls="--")
            ax.set_ylim(-1, max(len(shown), 1))
            ax.set_ylabel(f"{region}\nNeuron", color=color, fontweight="bold")
            if region_idx == 0:
                ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
            if column == 1:
                ax.text(
                    0.98, 0.92, f"{len(shown)}/{len(indices)} units",
                    transform=ax.transAxes, ha="right", va="top", color=color,
                    fontsize=8, fontweight="bold",
                )
        if selected_indices:
            selected_details = [
                selected_rows.get((region, str(unit_ids[idx])), {}) for idx in selected_indices
            ]
            stable = [float(row["selection_stability"]) for row in selected_details if row]
            summary_lines.append(
                f"{region}: {len(selected_indices)}/{len(indices)}"
                + (f" (median stability {np.median(stable):.2f})" if stable else "")
            )

    for ax in axes[-1]:
        ax.set_xlabel("Time from delay onset (s)")
    fig.suptitle(
        "Delay-period ensemble selection across the bilateral ALM–striatum circuit",
        x=0.08, y=0.985, ha="left", fontsize=15, fontweight="bold", color="#172033",
    )
    fig.text(
        0.08, 0.947,
        "Gray shading: delay context supplied to the model  •  amber shading: held-out response epoch forecast",
        ha="left", color="#526071", fontsize=9,
    )
    criteria = (
        "Selection criteria\n"
        "• learned sparse gate ranks a neuron within the top fraction of its region\n"
        "• rank is stable within its preferred class across training trials (default ≥70%)\n"
        "• neuron has measurable delay activity; class modulation (η²) is reported\n"
        "• unit identities are never pooled across recording sessions"
    )
    if selection_csv:
        criteria += "\n\n" + "\n".join(summary_lines)
    else:
        criteria += "\n\nQA mode: highlighted units are the top delay spike-count fraction,\nnot model-selected scientific results."
    fig.text(
        0.08, 0.012, criteria, ha="left", va="bottom", fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    if not any_selected:
        raise ValueError("No units were selected; check unit IDs and selection CSV")
    fig.subplots_adjust(top=0.91, bottom=0.19, left=0.10, right=0.98)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
