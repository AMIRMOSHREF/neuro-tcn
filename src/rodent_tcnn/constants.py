"""Shared labels, region maps, and epoch names used across both datasets."""

from __future__ import annotations

REGIONS: tuple[str, ...] = (
    "left ALM",
    "right ALM",
    "left Striatum",
    "right Striatum",
)

REGION_KEYS: tuple[str, ...] = (
    "left_ALM",
    "right_ALM",
    "left_Striatum",
    "right_Striatum",
)

REGION_TO_KEY: dict[str, str] = {
    "left ALM": "left_ALM",
    "right ALM": "right_ALM",
    "left Striatum": "left_Striatum",
    "right Striatum": "right_Striatum",
    "left alm": "left_ALM",
    "right alm": "right_ALM",
    "left striatum": "left_Striatum",
    "right striatum": "right_Striatum",
    "ALM-Left": "left_ALM",
    "ALM-Right": "right_ALM",
    "STR-Left": "left_Striatum",
    "STR-Right": "right_Striatum",
}

KEY_TO_REGION: dict[str, str] = {v: k for k, v in REGION_TO_KEY.items() if k in REGIONS}

CLASSES: tuple[str, ...] = ("Ignore", "Left", "Right")
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_ID.items()}

REGION_COLORS: dict[str, str] = {
    "left ALM": "#1f4e79",
    "right ALM": "#c45c26",
    "left Striatum": "#2e7d4f",
    "right Striatum": "#8b1e3f",
}

CLASS_COLORS: dict[str, str] = {
    "Ignore": "#6b7280",
    "Left": "#2563eb",
    "Right": "#dc2626",
}

EPOCH_KEYS: tuple[str, ...] = (
    "presample_start_times",
    "presample_stop_times",
    "sample_start_times",
    "sample_stop_times",
    "delay_start_times",
    "delay_stop_times",
    "go_start_times",
    "go_stop_times",
    "trial_start",
    "trial_stop",
)
