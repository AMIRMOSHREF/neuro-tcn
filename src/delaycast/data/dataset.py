"""Assemble model-ready tensors from the session caches and the neuron-selection results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

from .. import CLASSES, REGIONS
from ..features.selection import SelectionResult
from .cache import SessionCache


@dataclass
class SessionTensors:
    session: str
    dataset: str
    x: dict[str, np.ndarray]        # region -> (n_trials, K, T_ctx) counts of selected neurons (zero-padded)
    y: dict[str, np.ndarray]        # region -> (n_trials, K, T_tgt) response-epoch counts
    neuron_mask: dict[str, np.ndarray]  # region -> (K,) True for real neurons
    unit_index: dict[str, np.ndarray]   # region -> (K,) original unit index (-1 for padding)
    labels: np.ndarray
    trials: np.ndarray

    @property
    def n_trials(self) -> int:
        return len(self.labels)


def choose_indices(sel: SelectionResult | None, cache: SessionCache, cfg, mode: str, rng: np.random.Generator,
                   trial_idx: np.ndarray | None = None, k_per_region: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """``criteria`` (default), ``rate`` (the most active units on the fit trials, no selectivity) or ``random``.

    With ``data.representation: population`` every mode returns all channels of the region (see
    :func:`delaycast.data.rasters.population_rasters`).

    ``k_per_region`` sets how many units the ``rate`` / ``random`` sets take per region: the claim compares the
    criteria set with rate-matched and random subsets *of the same size*, so the control arms are matched to the
    number of units the criteria selection actually produced in that session and region (K_eff <= K), not to K.
    """
    k = int(cfg.selection.top_k_per_region)
    out = {}
    idx = np.arange(cache.n_trials) if trial_idx is None else np.asarray(trial_idx, dtype=int)
    population = str(cfg.data.get_path("representation", "units")).lower() == "population"
    for r in REGIONS:
        n = cache.context[r].shape[1]
        kr = k if k_per_region is None else min(k, max(0, int(k_per_region.get(r, k))))
        if population:
            # Identity-free rate-quantile channels are not neurons: there is nothing to select, so every arm
            # uses all of them (the selection-vs-controls predictions are reported as not applicable).
            out[r] = np.arange(min(n, k), dtype=int)
        elif mode == "criteria":
            if sel is None:
                raise ValueError("mode='criteria' needs a SelectionResult")
            out[r] = np.asarray(sel.selected[r][:k], dtype=int)
        elif mode == "rate":
            rates = cache.context[r][idx].sum(axis=(0, 2))
            out[r] = np.argsort(-rates, kind="mergesort")[:kr]
        elif mode == "random":
            out[r] = rng.permutation(n)[:kr]
        else:
            raise ValueError(mode)
    return out


def build_session_tensors(cache: SessionCache, sel: SelectionResult | None, cfg, mode: str = "criteria", seed: int = 0,
                          trial_idx: np.ndarray | None = None, k_per_region: dict[str, int] | None = None) -> SessionTensors:
    rng = np.random.default_rng(seed)
    idx = choose_indices(sel, cache, cfg, mode, rng, trial_idx, k_per_region=k_per_region)
    return tensors_from_indices(cache, idx, cfg)


def tensors_from_indices(cache: SessionCache, idx: dict[str, np.ndarray], cfg) -> SessionTensors:
    """Model-ready tensors for an explicit choice of unit indices per region (zero-padded to K)."""
    k = int(cfg.selection.top_k_per_region)
    x, y, mask, uidx = {}, {}, {}, {}
    n_tr = cache.n_trials
    for r in REGIONS:
        ii = np.asarray(idx[r], dtype=int)
        ii = ii[ii >= 0][:k]          # -1 marks an empty (padded) slot in saved checkpoints; never an index
        T, Tt = cache.context[r].shape[2], cache.target[r].shape[2]
        xr = np.zeros((n_tr, k, T), np.float32)
        yr = np.zeros((n_tr, k, Tt), np.float32)
        m = np.zeros(k, bool)
        ui = np.full(k, -1, int)
        if len(ii):
            xr[:, : len(ii)] = cache.context[r][:, ii].astype(np.float32)
            yr[:, : len(ii)] = cache.target[r][:, ii].astype(np.float32)
            m[: len(ii)] = True
            ui[: len(ii)] = ii
        x[r], y[r], mask[r], uidx[r] = xr, yr, m, ui
    return SessionTensors(cache.session, cache.dataset, x, y, mask, uidx, cache.labels.copy(), cache.trials.copy())


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
    for key in ("x", "y", "mask"):
        out[key] = {r: torch.stack([b[key][r] for b in batch]) for r in REGIONS}
    return out


def session_arrays(st: SessionTensors, idx: np.ndarray, device=None) -> dict:
    """All trials ``idx`` of one session as one batch dict (used by the evaluation analyses)."""
    idx = np.asarray(idx, dtype=int)
    out = {"session": st.session, "label": torch.as_tensor(st.labels[idx]), "trial": torch.as_tensor(st.trials[idx])}
    out["x"] = {r: torch.from_numpy(st.x[r][idx]) for r in REGIONS}
    out["y"] = {r: torch.from_numpy(st.y[r][idx]) for r in REGIONS}
    out["mask"] = {r: torch.from_numpy(np.tile(st.neuron_mask[r][None], (len(idx), 1))) for r in REGIONS}
    if device is not None:
        for key in ("x", "y", "mask"):
            out[key] = {r: v.to(device) for r, v in out[key].items()}
        out["label"] = out["label"].to(device)
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


def stratified_split(labels: np.ndarray, val_frac: float, test_frac: float, seed: int, scheme: str = "random",
                     n_blocks: int = 5, guard: int = 3):
    """Per-session train/val/test split that tolerates tiny (or empty) classes/fractions.

    ``scheme='random'``: stratified random trials.  ``scheme='blocked'``: the test set is ``n_blocks``
    contiguous blocks spread through the session (neighbouring trials share slow drift, so a random split
    is optimistic); ``guard`` trials on either side of every block are dropped from train/val.
    """
    idx = np.arange(len(labels))
    if scheme == "blocked" and test_frac > 0 and len(labels) >= 4 * n_blocks:
        n = len(labels)
        n_test = int(round(test_frac * n))
        per_block = max(1, n_test // n_blocks)
        starts = np.linspace(0, n - per_block, n_blocks + 2)[1:-1].astype(int)
        # deterministic jitter so different seeds see different blocks
        rng = np.random.default_rng(seed)
        starts = np.clip(starts + rng.integers(-per_block // 2, per_block // 2 + 1, size=n_blocks), 0, n - per_block)
        te = np.unique(np.concatenate([np.arange(s0, s0 + per_block) for s0 in starts]))
        excluded = np.unique(np.concatenate([np.arange(max(s0 - guard, 0), min(s0 + per_block + guard, n)) for s0 in starts]))
        rest = np.setdiff1d(idx, excluded)
        tr, va = _split(rest, labels, val_frac / max(1 - test_frac, 1e-9), seed)
        return np.sort(tr), np.sort(va), np.sort(te)
    rest, te = _split(idx, labels, test_frac, seed)
    tr, va = _split(rest, labels, val_frac / max(1 - test_frac, 1e-9), seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def class_weights(labels: np.ndarray, cap: float | None = 5.0) -> torch.Tensor:
    """Inverse-frequency class weights, capped (a handful of Ignore trials would otherwise get weight ~20)."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    if cap is not None:
        w = np.minimum(w, float(cap))
    w[counts == 0] = 0.0
    return torch.tensor(w, dtype=torch.float32)
