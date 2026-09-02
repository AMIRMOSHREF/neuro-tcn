"""Multi-criterion neuron selection on delay-period population activity.

Criteria (all z-scored within region, then weighted):
1. class d' on delay firing rate (Left vs Right, plus Ignore contrast)
2. delay→lick coupling (Pearson of delay PSTH vs lick PSTH)
3. time-frequency selectivity (beta/gamma class modulation)
4. model neuron-attention (if provided)
5. prediction gain (occlusion ΔMSE if provided)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..constants import CLASSES, REGION_KEYS, KEY_TO_REGION
from .spectral import band_energy, wavelet_power


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or b.size < 4:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def _dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    md = float(np.mean(pos) - np.mean(neg))
    v = 0.5 * (float(np.var(pos)) + float(np.var(neg)))
    if v < 1e-8:
        return 0.0
    return md / np.sqrt(v)


@dataclass
class NeuronScore:
    region: str
    region_key: str
    local_index: int
    unit_id: int
    selected: bool
    score: float
    dprime: float
    delay_lick_coupling: float
    tf_selectivity: float
    attention: float
    prediction_gain: float
    delay_rate_hz: float
    preferred_class: str
    reasons: list[str]
    neuron_type: str | None = None


def collect_population(
    items: list[dict[str, Any]],
    bin_size: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Stack delay/lick rasters per region across trials.

    Returns region_key -> {
        delay: (trials, units, T), lick: (trials, units, T), labels: (trials,)
    }
    """
    out: dict[str, dict[str, list]] = {k: {"delay": [], "lick": [], "labels": []} for k in REGION_KEYS}
    for item in items:
        delay = item["delay"]
        lick = item["lick"]
        label = item["label"]
        for r, key in enumerate(REGION_KEYS):
            out[key]["delay"].append(delay[r])
            out[key]["lick"].append(lick[r])
            out[key]["labels"].append(label)
    packed = {}
    for key in REGION_KEYS:
        packed[key] = {
            "delay": np.stack(out[key]["delay"], axis=0) if out[key]["delay"] else np.zeros((0, 0, 0)),
            "lick": np.stack(out[key]["lick"], axis=0) if out[key]["lick"] else np.zeros((0, 0, 0)),
            "labels": np.asarray(out[key]["labels"], dtype=np.int64),
            "bin_size": bin_size,
        }
    return packed


def _criterion_matrix(
    delay: np.ndarray,
    lick: np.ndarray,
    labels: np.ndarray,
    bin_size: float,
    attention: np.ndarray | None,
    pred_gain: np.ndarray | None,
) -> dict[str, np.ndarray]:
    n_trials, n_units, t_d = delay.shape
    delay_rate = delay.sum(axis=2) / max(t_d * bin_size, 1e-6)
    dprime = np.zeros(n_units, dtype=np.float64)
    pref = np.full(n_units, "Ignore", dtype=object)
    coupling = np.zeros(n_units, dtype=np.float64)
    tf_sel = np.zeros(n_units, dtype=np.float64)

    left = labels == 1
    right = labels == 2
    ignore = labels == 0
    for u in range(n_units):
        fr = delay_rate[:, u]
        d_lr = abs(_dprime(fr[left], fr[right])) if left.any() and right.any() else 0.0
        d_ig = abs(_dprime(fr[~ignore], fr[ignore])) if ignore.any() and (~ignore).any() else 0.0
        dprime[u] = max(d_lr, 0.65 * d_ig)
        means = [float(fr[labels == c].mean()) if np.any(labels == c) else -np.inf for c in range(3)]
        pref[u] = ("Ignore", "Left", "Right")[int(np.argmax(means))]
        # PSTH coupling
        delay_psth = delay[:, u, :].mean(axis=0)
        lick_psth = lick[:, u, :].mean(axis=0)
        # resample lick onto delay length for correlation of shape
        lick_rs = np.interp(np.linspace(0, 1, t_d), np.linspace(0, 1, max(lick_psth.size, 1)), lick_psth)
        coupling[u] = max(_safe_corr(delay_psth, lick_rs), 0.0)
        # TF selectivity: beta/gamma energy difference Left vs Right
        try:
            tf = wavelet_power(delay[:, u, :].mean(axis=0, keepdims=True), bin_size, n_freq=12)
            bands = []
            for mask in (left, right, ignore):
                if not mask.any():
                    bands.append(0.0)
                    continue
                tf_c = wavelet_power(delay[mask, u, :].mean(axis=0, keepdims=True), bin_size, n_freq=12)
                bands.append(float(band_energy(tf_c, bin_size, 12.0, 45.0)[0]))
            tf_sel[u] = abs(bands[1] - bands[2]) + 0.4 * abs(bands[0] - 0.5 * (bands[1] + bands[2]))
        except Exception:
            tf_sel[u] = 0.0

    attn = np.zeros(n_units) if attention is None else np.asarray(attention, dtype=np.float64)
    if attn.size != n_units:
        attn = np.zeros(n_units)
    gain = np.zeros(n_units) if pred_gain is None else np.asarray(pred_gain, dtype=np.float64)
    if gain.size != n_units:
        gain = np.zeros(n_units)

    return {
        "dprime": dprime,
        "delay_lick_coupling": coupling,
        "tf_selectivity": tf_sel,
        "attention": attn,
        "prediction_gain": gain,
        "delay_rate_hz": delay_rate.mean(axis=0),
        "preferred_class": pref,
    }


def _reasons(row: dict[str, float], weights: dict[str, float], ntype: str | None) -> list[str]:
    bullets: list[str] = []
    if row["dprime"] > 0.8:
        bullets.append(
            f"Choice-discriminative delay rate (d′={row['dprime']:.2f} vs other actions; prefers {row['preferred_class']})."
        )
    if row["delay_lick_coupling"] > 0.25:
        bullets.append(
            f"Delay PSTH shape predicts this unit’s lick-period PSTH (r={row['delay_lick_coupling']:.2f})."
        )
    if row["tf_selectivity"] > 0.02:
        bullets.append(
            f"Delay-period β/low-γ wavelet power is class-modulated (TF score={row['tf_selectivity']:.2f})."
        )
    if row["attention"] > 0:
        bullets.append(
            f"SPEC-TCNN neuron attention is high (a={row['attention']:.3f}); the unit is used as past context."
        )
    if row["prediction_gain"] > 0:
        bullets.append(
            f"Occluding this unit raises lick-raster prediction error (ΔMSE gain={row['prediction_gain']:.3f})."
        )
    if ntype in {"delay_choice", "delay_ramp"}:
        pretty = "persistent delay-choice cell" if ntype == "delay_choice" else "delay-ramping preparatory cell"
        bullets.append(f"Ground-truth type: {pretty} — the ALM preparatory motif of Li/Inagaki/Svoboda.")
    if row["delay_rate_hz"] < 0.4:
        bullets.append("Low delay rate; kept only if other scores remain high.")
    if not bullets:
        bullets.append("Composite score above the regional threshold from weaker combined cues.")
    return bullets


def select_neurons(
    population: dict[str, dict[str, np.ndarray]],
    cfg,
    attention_by_region: dict[str, np.ndarray] | None = None,
    pred_gain_by_region: dict[str, np.ndarray] | None = None,
    unit_ids_by_region: dict[str, np.ndarray] | None = None,
    neuron_types_by_region: dict[str, np.ndarray] | None = None,
) -> list[NeuronScore]:
    weights = cfg.selection.weights
    results: list[NeuronScore] = []
    for key in REGION_KEYS:
        pack = population[key]
        delay, lick, labels = pack["delay"], pack["lick"], pack["labels"]
        if delay.size == 0:
            continue
        n_units = delay.shape[1]
        attn = None if attention_by_region is None else attention_by_region.get(key)
        gain = None if pred_gain_by_region is None else pred_gain_by_region.get(key)
        crit = _criterion_matrix(delay, lick, labels, pack["bin_size"], attn, gain)
        composite = np.zeros(n_units)
        for name, w in weights.items():
            composite += w * _zscore(crit[name])
        # drop near-silent units unless extremely discriminative
        silent = crit["delay_rate_hz"] < cfg.selection.min_rate_hz
        composite[silent] -= 1.5
        k = max(1, int(np.ceil(cfg.selection.top_fraction * n_units)))
        order = np.argsort(-composite)
        selected_idx = set(order[:k].tolist())
        ids = np.arange(n_units) if unit_ids_by_region is None else np.asarray(unit_ids_by_region.get(key, np.arange(n_units)))
        types = None if neuron_types_by_region is None else neuron_types_by_region.get(key)
        for i in range(n_units):
            row = {name: float(crit[name][i]) for name in ("dprime", "delay_lick_coupling", "tf_selectivity", "attention", "prediction_gain", "delay_rate_hz")}
            row["preferred_class"] = str(crit["preferred_class"][i])
            ntype = None if types is None else str(types[i])
            uid = int(ids[i]) if i < len(ids) else i
            results.append(
                NeuronScore(
                    region=KEY_TO_REGION[key],
                    region_key=key,
                    local_index=i,
                    unit_id=uid,
                    selected=i in selected_idx,
                    score=float(composite[i]),
                    dprime=row["dprime"],
                    delay_lick_coupling=row["delay_lick_coupling"],
                    tf_selectivity=row["tf_selectivity"],
                    attention=row["attention"],
                    prediction_gain=row["prediction_gain"],
                    delay_rate_hz=row["delay_rate_hz"],
                    preferred_class=row["preferred_class"],
                    reasons=_reasons(row, weights, ntype),
                    neuron_type=ntype,
                )
            )
    results.sort(key=lambda s: (-int(s.selected), -s.score))
    return results


def selection_table(scores: list[NeuronScore]) -> pd.DataFrame:
    rows = []
    for s in scores:
        rows.append(
            {
                "region": s.region,
                "unit_id": s.unit_id,
                "local_index": s.local_index,
                "selected": s.selected,
                "score": s.score,
                "dprime": s.dprime,
                "delay_lick_coupling": s.delay_lick_coupling,
                "tf_selectivity": s.tf_selectivity,
                "attention": s.attention,
                "prediction_gain": s.prediction_gain,
                "delay_rate_hz": s.delay_rate_hz,
                "preferred_class": s.preferred_class,
                "neuron_type": s.neuron_type,
                "reasons": " | ".join(s.reasons),
            }
        )
    return pd.DataFrame(rows)
