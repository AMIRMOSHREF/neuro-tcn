"""Assemble model-ready tensors from the session caches and the neuron-selection results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from .. import CLASSES, REGIONS
from ..features.selection import SelectionResult
from ..features.spectral import band_power_stft, smooth_rates
from .cache import SessionCache


@dataclass
class SessionTensors:
    session: str
    dataset: str
    x: dict[str, np.ndarray]        # region -> (n_trials, K, T_ctx) counts of selected neurons (zero-padded)
    spec: dict[str, np.ndarray]     # region -> (n_trials, n_bands, T_ctx) population STFT band power
    y: dict[str, np.ndarray]        # region -> (n_trials, K, T_tgt) response-epoch counts
    neuron_mask: dict[str, np.ndarray]  # region -> (K,) True for real neurons
    unit_index: dict[str, np.ndarray]   # region -> (K,) original unit index (-1 for padding)
    labels: np.ndarray
    trials: np.ndarray

    @property
    def n_trials(self) -> int:
        return len(self.labels)


def choose_indices(sel: SelectionResult, cache: SessionCache, cfg, mode: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """``criteria`` (default), ``rate`` (top-K most active, no selectivity) or ``random`` (K random units)."""
    k = int(cfg.selection.top_k_per_region)
    out = {}
    for r in REGIONS:
        n = cache.context[r].shape[1]
        if mode == "criteria":
            out[r] = sel.selected[r][:k]
        elif mode == "rate":
            rates = cache.context[r].sum(axis=(0, 2))
            out[r] = np.argsort(-rates)[:k]
        elif mode == "random":
            out[r] = rng.permutation(n)[:k]
        else:
            raise ValueError(mode)
    return out


def build_session_tensors(cache: SessionCache, sel: SelectionResult, cfg, mode: str = "criteria", seed: int = 0) -> SessionTensors:
    k = int(cfg.selection.top_k_per_region)
    rng = np.random.default_rng(seed)
    idx = choose_indices(sel, cache, cfg, mode, rng)
    bands = {kk: list(v) for kk, v in cfg.selection.bands_hz.items()}
    x, spec, y, mask, uidx = {}, {}, {}, {}, {}
    n_tr = cache.n_trials
    for r in REGIONS:
        ii = idx[r]
        T, Tt = cache.context[r].shape[2], cache.target[r].shape[2]
        xr = np.zeros((n_tr, k, T), np.float32)
        yr = np.zeros((n_tr, k, Tt), np.float32)
        m = np.zeros(k, bool)
        ui = np.full(k, -1, int)
        if len(ii):
            xr[:, : len(ii)] = cache.context[r][:, ii]
            yr[:, : len(ii)] = cache.target[r][:, ii]
            m[: len(ii)] = True
            ui[: len(ii)] = ii
            pop = smooth_rates(cache.context[r][:, ii].mean(axis=1), cache.bin_ms, cfg.data.smoothing_sigma_ms)  # (n_tr, T)
            sp = band_power_stft(pop, cache.bin_ms, bands).astype(np.float32)   # (n_tr, n_bands, T)
        else:
            sp = np.zeros((n_tr, len(bands), T), np.float32)
        x[r], y[r], mask[r], uidx[r], spec[r] = xr, yr, m, ui, sp
    return SessionTensors(cache.session, cache.dataset, x, spec, y, mask, uidx, cache.labels.copy(), cache.trials.copy())


class TrialDataset(Dataset):
    """Flat index over (session, trial) pairs restricted to a subset of trials per session."""

    def __init__(self, sessions: list[SessionTensors], subset: dict[str, np.ndarray]):
        self.sessions = {s.session: s for s in sessions}
        self.items: list[tuple[str, int]] = [(s, int(i)) for s, ids in subset.items() for i in ids]
        self.session_of_item = np.array([s for s, _ in self.items])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        s, t = self.items[i]
        st = self.sessions[s]
        return {
            "session": s,
            "x": {r: torch.from_numpy(st.x[r][t]) for r in REGIONS},
            "spec": {r: torch.from_numpy(st.spec[r][t]) for r in REGIONS},
            "y": {r: torch.from_numpy(st.y[r][t]) for r in REGIONS},
            "mask": {r: torch.from_numpy(st.neuron_mask[r]) for r in REGIONS},
            "label": int(st.labels[t]),
            "trial": int(st.trials[t]),
        }

    def labels(self) -> np.ndarray:
        return np.array([self.sessions[s].labels[t] for s, t in self.items])


def collate(batch: list[dict]) -> dict:
    assert len({b["session"] for b in batch}) == 1, "batches must come from a single session"
    out = {"session": batch[0]["session"], "label": torch.tensor([b["label"] for b in batch]),
           "trial": torch.tensor([b["trial"] for b in batch])}
    for key in ("x", "spec", "y", "mask"):
        out[key] = {r: torch.stack([b[key][r] for b in batch]) for r in REGIONS}
    return out


class SessionBatchSampler(Sampler):
    """Yields batches whose trials all belong to the same session (required by the session adapters)."""

    def __init__(self, dataset: TrialDataset, batch_size: int, shuffle: bool, seed: int = 0):
        self.ds, self.bs, self.shuffle, self.seed, self.epoch = dataset, batch_size, shuffle, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for s in np.unique(self.ds.session_of_item):
            idx = np.where(self.ds.session_of_item == s)[0]
            if self.shuffle:
                idx = rng.permutation(idx)
            batches += [idx[i: i + self.bs].tolist() for i in range(0, len(idx), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(int(np.ceil((self.ds.session_of_item == s).sum() / self.bs)) for s in np.unique(self.ds.session_of_item))


def _can_stratify(labels: np.ndarray, min_per_class: int) -> bool:
    counts = np.bincount(labels, minlength=len(CLASSES))
    return bool(np.all(counts[np.unique(labels)] >= min_per_class))


def _split(idx: np.ndarray, labels: np.ndarray, frac: float, seed: int):
    n_out = int(round(frac * len(idx)))
    if n_out <= 0:
        return idx, np.zeros(0, int)
    if n_out >= len(idx):
        return np.zeros(0, int), idx
    strat = labels[idx] if _can_stratify(labels[idx], 2) and n_out >= len(np.unique(labels[idx])) else None
    a, b = train_test_split(idx, test_size=n_out, random_state=seed, stratify=strat)
    return a, b


def stratified_split(labels: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Per-session stratified train/val/test split that tolerates tiny (or empty) classes/fractions."""
    idx = np.arange(len(labels))
    rest, te = _split(idx, labels, test_frac, seed)
    tr, va = _split(rest, labels, val_frac / max(1 - test_frac, 1e-9), seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    w[counts == 0] = 0.0
    return torch.tensor(w, dtype=torch.float32)
