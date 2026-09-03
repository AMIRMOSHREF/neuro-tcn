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


def cwt_scalogram(rates: np.ndarray, bin_ms: float, freqs_hz: np.ndarray, wavelet: str = "cmor1.5-1.0",
                  chunk_rows: int = 2048) -> np.ndarray:
    """Complex Morlet CWT power. ``rates``: (T,) or (N, T) -> (N, F, T) (or (F, T)).

    Rows are processed in chunks of ``chunk_rows`` so that hundreds of thousands of single-trial rate traces
    (2000 units x 350 trials) never materialise a (N, F, T) complex array at once; the FFT-based
    convolution is used because it is ~1.5x faster than direct convolution for these lengths.
    """
    fs = 1000.0 / bin_ms
    dt = 1.0 / fs
    fc = pywt.central_frequency(wavelet)
    scales = fc / (np.asarray(freqs_hz, dtype=float) * dt)
    single = rates.ndim == 1
    X = rates[None] if single else rates
    X = (X - X.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
    out = np.empty((X.shape[0], len(scales), X.shape[-1]), dtype=np.float32)
    for i in range(0, X.shape[0], max(int(chunk_rows), 1)):
        coefs, _ = pywt.cwt(X[i:i + chunk_rows], scales, wavelet, sampling_period=dt, axis=-1, method="fft")  # (F, n, T)
        out[i:i + chunk_rows] = np.moveaxis(np.abs(coefs) ** 2, 0, 1)
    return out[0] if single else out


def band_power_cwt(rates: np.ndarray, bin_ms: float, bands: dict[str, list[float]], wavelet: str = "cmor1.5-1.0",
                   n_freqs_per_band: int = 5, chunk_rows: int = 2048) -> tuple[np.ndarray, list[str]]:
    """Mean CWT power inside each band, averaged over time (cone of influence trimmed).

    ``rates``: (N, T) -> (N, n_bands) float32. Works on arbitrarily many rows thanks to chunking.
    """
    names, freqs, band_idx = [], [], []
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        f = np.geomspace(max(lo, 0.5), hi, n_freqs_per_band)
        freqs.append(f)
        band_idx += [i] * len(f)
        names.append(name)
    freqs = np.concatenate(freqs)
    band_idx = np.asarray(band_idx)
    rates = np.asarray(rates, dtype=np.float32)
    N, T = rates.shape
    lo, hi = int(0.1 * T), int(0.9 * T)   # trim the cone of influence (edge effects)
    out = np.empty((N, len(names)), dtype=np.float32)
    for i in range(0, N, max(int(chunk_rows), 1)):
        power = cwt_scalogram(rates[i:i + chunk_rows], bin_ms, freqs, wavelet, chunk_rows=chunk_rows)  # (n, F, T)
        mean_p = power[..., lo:hi].mean(axis=-1)  # (n, F)
        out[i:i + chunk_rows] = np.stack([mean_p[:, band_idx == b].mean(axis=1) for b in range(len(names))], axis=1)
    return out, names
