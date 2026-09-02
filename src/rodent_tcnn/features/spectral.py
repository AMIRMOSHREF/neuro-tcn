"""Wavelet (CWT) and STFT time-frequency features on delay-binned rasters."""

from __future__ import annotations

import numpy as np
from scipy import signal

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


def _smooth_rate(raster: np.ndarray, win: int = 5) -> np.ndarray:
    if raster.size == 0:
        return raster
    kernel = np.ones(win, dtype=np.float64) / win
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), -1, raster)


def wavelet_power(raster: np.ndarray, bin_size: float, n_freq: int = 16) -> np.ndarray:
    """Morlet CWT power. raster: (units, time) → (units, freq, time)."""
    rate = _smooth_rate(raster.astype(np.float64))
    n_units, n_t = rate.shape
    freqs = np.geomspace(4.0, 80.0, n_freq)
    if pywt is None or n_t < 8:
        return stft_power(raster, bin_size, n_freq)
    scales = pywt.frequency2scale("cmor1.5-1.0", freqs * bin_size)
    out = np.zeros((n_units, n_freq, n_t), dtype=np.float32)
    for i in range(n_units):
        coef, _ = pywt.cwt(rate[i], scales, "cmor1.5-1.0")
        out[i] = np.abs(coef).astype(np.float32)
    return out


def stft_power(raster: np.ndarray, bin_size: float, n_freq: int = 16) -> np.ndarray:
    """STFT magnitude resampled onto n_freq × T. raster: (units, time)."""
    rate = _smooth_rate(raster.astype(np.float64))
    n_units, n_t = rate.shape
    nperseg = min(32, max(8, n_t // 4))
    noverlap = nperseg // 2
    out = np.zeros((n_units, n_freq, n_t), dtype=np.float32)
    if n_t < nperseg:
        return out
    for i in range(n_units):
        f, t, z = signal.stft(rate[i], fs=1.0 / bin_size, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(z)
        # resample frequency then time to (n_freq, n_t)
        if mag.shape[0] == 1:
            freq_rs = np.repeat(mag, n_freq, axis=0)
        else:
            src = np.linspace(0, 1, mag.shape[0])
            dst = np.linspace(0, 1, n_freq)
            freq_rs = np.vstack([np.interp(dst, src, mag[:, j]) for j in range(mag.shape[1])]).T
        src_t = np.linspace(0, 1, freq_rs.shape[1])
        dst_t = np.linspace(0, 1, n_t)
        time_rs = np.vstack([np.interp(dst_t, src_t, freq_rs[k]) for k in range(n_freq)])
        out[i] = time_rs.astype(np.float32)
    return out


def trial_tf_maps(delay_stack: np.ndarray, bin_size: float, n_freq: int = 16) -> np.ndarray:
    """delay_stack: (4, N, T) → (4, N, F, T) average of CWT and STFT."""
    maps = []
    for r in range(delay_stack.shape[0]):
        w = wavelet_power(delay_stack[r], bin_size, n_freq)
        s = stft_power(delay_stack[r], bin_size, n_freq)
        maps.append(0.5 * (w + s))
    return np.stack(maps, axis=0)


def band_energy(tf: np.ndarray, bin_size: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Mean power in [f_lo, f_hi]. tf: (units, freq, time) → (units,)."""
    n_freq = tf.shape[1]
    freqs = np.geomspace(4.0, 80.0, n_freq)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return tf.mean(axis=(1, 2))
    return tf[:, mask, :].mean(axis=(1, 2))
