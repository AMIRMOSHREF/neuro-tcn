"""Figure 2: time-frequency structure of the delay epoch (Morlet CWT scalograms of the population
rate per region for one trial, class-averaged wavelet band power of selected units, STFT band-power
time courses per class)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import CLASSES, REGION_COLORS, REGION_LABELS, REGIONS
from ..data.cache import SessionCache
from ..features.spectral import band_power_cwt, band_power_stft, cwt_scalogram, smooth_rates
from .style import CLASS_COLORS, apply_style, panel_label


def plot_time_frequency(cache: SessionCache, table: pd.DataFrame, cfg, trial_idx: int, out_path: Path) -> Path:
    apply_style()
    bands = {k: list(v) for k, v in cfg.selection.bands_hz.items()}
    freqs = np.geomspace(1, 30, 40)
    bin_ms = cache.bin_ms
    T = cache.context[REGIONS[0]].shape[2]
    tvec = (np.arange(T) + 0.5) * bin_ms / 1000.0
    y = cache.labels

    fig, axes = plt.subplots(3, 4, figsize=(15, 9), gridspec_kw={"height_ratios": [1.2, 1, 1], "hspace": 0.5, "wspace": 0.35})
    for j, r in enumerate(REGIONS):
        sel = table[(table.region == r) & table.selected].unit_index.to_numpy(dtype=int)
        units = sel if len(sel) else np.arange(cache.context[r].shape[1])
        # Row 1: scalogram of this trial's population rate of the selected units.
        pop = smooth_rates(cache.context[r][trial_idx, units].mean(0), bin_ms, cfg.data.smoothing_sigma_ms)
        ax = axes[0, j]
        if len(units):
            P = cwt_scalogram(pop, bin_ms, freqs, cfg.selection.wavelet)
            P = P / (P.mean(axis=1, keepdims=True) + 1e-9)  # normalise per frequency
            im = ax.pcolormesh(tvec, freqs, P, shading="auto", cmap="magma")
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 4, 8, 16, 30])
            ax.set_yticklabels(["1", "2", "4", "8", "16", "30"])
            plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, label="rel. power")
        ax.set_title(f"{REGION_LABELS[r]} — CWT, trial {cache.trials[trial_idx]} ({CLASSES[y[trial_idx]]})", color=REGION_COLORS[r], loc="left")
        ax.set_ylabel("frequency (Hz)")
        ax.set_xlabel("time from delay onset (s)")
        # Row 2: STFT band power over time per class (population of selected units).
        ax = axes[1, j]
        pops = smooth_rates(cache.context[r][:, units].mean(1), bin_ms, cfg.data.smoothing_sigma_ms)
        bp = band_power_stft(pops, bin_ms, bands)  # (n_trials, n_bands, T)
        for b_i, bname in enumerate(bands):
            for c_i, c in enumerate(CLASSES):
                m = y == c_i
                if not m.any():
                    continue
                ax.plot(tvec, np.log1p(bp[m, b_i].mean(0)), color=CLASS_COLORS[c], lw=1.0, alpha=0.4 + 0.3 * b_i,
                        ls=["-", "--", ":"][b_i % 3], label=f"{bname} · {c}" if j == 0 else None)
        ax.set_title("STFT band power (log), class means", loc="left")
        ax.set_xlabel("time from delay onset (s)")
        if j == 0:
            ax.legend(ncol=3, fontsize=6, loc="upper left")
        # Row 3: per-unit CWT band power (delay mean) by class for the selected units.
        ax = axes[2, j]
        if len(units):
            n_tr = cache.n_trials
            rates = smooth_rates(cache.context[r][:, units].reshape(n_tr * len(units), T), bin_ms, cfg.data.smoothing_sigma_ms)
            bpc, names = band_power_cwt(rates, bin_ms, bands, cfg.selection.wavelet)
            bpc = bpc.reshape(n_tr, len(units), -1)
            w = 0.25
            for c_i, c in enumerate(CLASSES):
                m = y == c_i
                if not m.any():
                    continue
                vals = np.log1p(bpc[m].mean(0))  # (n_units, n_bands)
                ax.bar(np.arange(len(names)) + (c_i - 1) * w, vals.mean(0), width=w, yerr=vals.std(0) / np.sqrt(max(len(units), 1)),
                       color=CLASS_COLORS[c], label=c if j == 0 else None, capsize=2)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels([f"{n}\n{bands[n][0]}-{bands[n][1]} Hz" for n in names])
        ax.set_title("Wavelet band power of selected units (delay mean)", loc="left")
        ax.set_ylabel("log(1+power)")
        if j == 0:
            ax.legend(fontsize=6.5)
    panel_label(axes[0, 0], "A"); panel_label(axes[1, 0], "B"); panel_label(axes[2, 0], "C")
    fig.suptitle(f"Time-frequency features of the delay epoch — session {cache.session}", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(cfg.figures.dpi))
    plt.close(fig)
    return out_path
