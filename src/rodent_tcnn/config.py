"""YAML + CLI configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _p(value: Any) -> Path:
    return Path(str(value)).expanduser()


@dataclass
class Paths:
    data_root: Path = Path(r"C:\PythonProject\Rodent\Data")
    data2_root: Path = Path(r"C:\PythonProject\Rodent\Data2")
    demo_root: Path = Path("data/demo")
    metadata_dir: Path = Path("data/metadata")
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("figures")
    checkpoint_dir: Path = Path("outputs/checkpoints")


@dataclass
class EpochWindows:
    bin_size: float = 0.01
    delay_bins: int = 120
    lick_bins: int = 80
    lick_window_s: float = 0.80
    ignore_align: str = "go"  # go-aligned window when no lick occurs


@dataclass
class ModelConfig:
    units_per_region: int = 32
    d_model: int = 64
    n_dcc_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.15
    n_freq: int = 16
    n_heads: int = 4
    sparsity_l1: float = 0.02


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 12
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    lambda_pred: float = 1.0
    lambda_cls: float = 0.7
    lambda_sparse: float = 0.05
    seed: int = 7
    device: str = "cpu"
    val_fraction: float = 0.2


@dataclass
class SelectionConfig:
    top_fraction: float = 0.18
    min_rate_hz: float = 0.4
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "attention": 0.28,
            "prediction_gain": 0.22,
            "dprime": 0.22,
            "delay_lick_coupling": 0.16,
            "tf_selectivity": 0.12,
        }
    )


@dataclass
class ExperimentConfig:
    paths: Paths = field(default_factory=Paths)
    epochs: EpochWindows = field(default_factory=EpochWindows)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    prefer_demo_if_missing: bool = True
    exclude_photostim: bool = True
    exclude_early_lick: bool = True
    exclude_auto_water: bool = True
    exclude_both_licks: bool = True
    keep_miss_as_ignore: bool = False


def _merge(dc, blob: dict[str, Any]) -> None:
    for key, value in blob.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, Path):
            setattr(dc, key, _p(value))
        else:
            setattr(dc, key, value)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        default = Path("configs/default.yaml")
        path = default if default.exists() else None
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _merge(cfg, raw)
    return cfg
