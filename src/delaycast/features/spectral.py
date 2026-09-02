"""Time-frequency features of spike-count rasters (STFT band power and Morlet CWT)."""
from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft


def smooth_rates(counts: np.ndarray, bin_ms: float, sigma_ms: float) -> np.ndarray:
    """Gaussian-smoothed firing rate (Hz) along the last axis of a count array."""
    rate = counts / (bin_ms / 1000.0)
    if sigma_ms <= 0:
        return rate
    return gaussian_filter1d(rate, sigma=sigma_ms / bin_ms, axis=-1, mode="nearest")


def band_power_stft(rates: np.ndarray, bin_ms: float, bands: dict[str, list[float]],
                    win_ms: float = 300.0, hop_bins: int = 1) -> np.ndarray:
    """STFT band power along time. ``rates``: (..., T) -> (..., n_bands, T) (interpolated to T bins).

    A 300 ms Hann window at 100 Hz gives ~3.3 Hz resolution, which is sufficient to separate the
    slow (1-4 Hz ramp/offset), theta (4-12 Hz) and beta (12-30 Hz) bands used here.
    """
    fs = 1000.0 / bin_ms
    nper = max(8, int(round(win_ms / bin_ms)))
    x = rates - rates.mean(axis=-1, keepdims=True)
    f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nper - hop_bins, boundary="even", padded=False, axis=-1)
    P = np.abs(Z) ** 2  # (..., F, Tst)
    T = rates.shape[-1]
    out = []
    for lo, hi in bands.values():
        m = (f >= lo) & (f < hi)
        bp = P[..., m, :].mean(axis=-2) if m.any() else np.zeros(P.shape[:-2] + (P.shape[-1],))
        # STFT time axis has hop=1 bin so the length is T (+/-1); resample linearly to exactly T.
        src = np.linspace(0, 1, bp.shape[-1])
        dst = np.linspace(0, 1, T)
        bp_r = np.apply_along_axis(lambda v: np.interp(dst, src, v), -1, bp)
        out.append(bp_r)
    return np.stack(out, axis=-2)


def cwt_scalogram(rates: np.ndarray, bin_ms: float, freqs_hz: np.ndarray, wavelet: str = "cmor1.5-1.0") -> np.ndarray:
    """Complex Morlet CWT power. ``rates``: (T,) or (N, T) -> (N, F, T) (or (F, T))."""
    fs = 1000.0 / bin_ms
    dt = 1.0 / fs
    fc = pywt.central_frequency(wavelet)
    scales = fc / (np.asarray(freqs_hz) * dt)
    single = rates.ndim == 1
    X = rates[None] if single else rates
    X = X - X.mean(axis=-1, keepdims=True)
    coefs, _ = pywt.cwt(X, scales, wavelet, sampling_period=dt, axis=-1)  # (F, N, T)
    power = np.abs(coefs) ** 2
    power = np.moveaxis(power, 0, 1)  # (N, F, T)
    return power[0] if single else power


def band_power_cwt(rates: np.ndarray, bin_ms: float, bands: dict[str, list[float]], wavelet: str = "cmor1.5-1.0",
                   n_freqs_per_band: int = 6) -> tuple[np.ndarray, list[str]]:
    """Mean CWT power inside each band, averaged over time. ``rates``: (N, T) -> (N, n_bands)."""
    names, freqs, band_idx = [], [], []
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        f = np.geomspace(max(lo, 0.5), hi, n_freqs_per_band)
        freqs.append(f)
        band_idx += [i] * len(f)
        names.append(name)
    freqs = np.concatenate(freqs)
    band_idx = np.asarray(band_idx)
    power = cwt_scalogram(rates, bin_ms, freqs, wavelet)  # (N, F, T)
    # Trim the cone of influence: drop the first/last 10% of bins where edge effects dominate.
    T = power.shape[-1]
    lo, hi = int(0.1 * T), int(0.9 * T)
    mean_p = power[..., lo:hi].mean(axis=-1)  # (N, F)
    out = np.stack([mean_p[:, band_idx == i].mean(axis=1) for i in range(len(names))], axis=1)
    return out, names
