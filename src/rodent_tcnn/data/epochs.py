"""Delay vs lick-time windows from NPZ timestamps and CSV lick lists."""

from __future__ import annotations

from typing import Any

import numpy as np

from .labels import parse_lick_list


def first_lick_time(data: dict[str, Any], csv_row=None) -> tuple[float, str | None]:
    left = np.asarray(data.get("left_lick_times", []), dtype=np.float64)
    right = np.asarray(data.get("right_lick_times", []), dtype=np.float64)
    if csv_row is not None:
        if len(left) == 0:
            left = parse_lick_list(csv_row.get("left_lick_times"))
        if len(right) == 0:
            right = parse_lick_list(csv_row.get("right_lick_times"))
    candidates: list[tuple[float, str]] = []
    if len(left):
        candidates.append((float(np.min(left)), "Left"))
    if len(right):
        candidates.append((float(np.min(right)), "Right"))
    if not candidates:
        return float("nan"), None
    candidates.sort()
    return candidates[0]


def delay_window(data: dict[str, Any]) -> tuple[float, float]:
    t0 = float(data.get("delay_start_times", np.nan))
    t1 = float(data.get("delay_stop_times", np.nan))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        sample_stop = float(data.get("sample_stop_times", data.get("trial_start", 0.0)))
        go = float(data.get("go_start_times", sample_stop + 1.2))
        return sample_stop, go
    return t0, t1


def lick_window(data: dict[str, Any], window_s: float, csv_row=None) -> tuple[float, float]:
    lick_t, _ = first_lick_time(data, csv_row)
    go = float(data.get("go_start_times", np.nan))
    stop = float(data.get("trial_stop", np.nan))
    if np.isfinite(lick_t):
        t0 = lick_t
    elif np.isfinite(go):
        t0 = go
    else:
        _, delay_t1 = delay_window(data)
        t0 = delay_t1
    t1 = t0 + float(window_s)
    if np.isfinite(stop):
        t1 = min(t1, stop)
    if t1 <= t0:
        t1 = t0 + window_s
    return t0, t1


def epoch_span(data: dict[str, Any]) -> dict[str, tuple[float, float]]:
    start = float(data.get("trial_start", 0.0))
    stop = float(data.get("trial_stop", start + 5.0))
    delay = delay_window(data)
    lick = lick_window(data, 0.8)
    return {
        "trial": (start, stop),
        "presample": (
            float(data.get("presample_start_times", start)),
            float(data.get("presample_stop_times", start)),
        ),
        "sample": (
            float(data.get("sample_start_times", start)),
            float(data.get("sample_stop_times", start)),
        ),
        "delay": delay,
        "go": (
            float(data.get("go_start_times", delay[1])),
            float(data.get("go_stop_times", stop)),
        ),
        "lick": lick,
    }
