"""Time-frequency features of spike-count rasters (STFT band power and Morlet CWT).

The continuous wavelet transform is *linear* and identical for every row, so for a fixed trace length ``T``
it is a fixed ``(T, F*T)`` complex matrix: the response of ``pywt.cwt`` to every unit impulse ``e_t``
(``pywt.cwt(np.eye(T))``).  Instead of running pywt's per-scale FFT convolution (complex128 buffers of
``n_scales x n_rows x fft_len``) on every chunk of single-trial rate traces, the impulse-response matrix is
computed once per ``(T, bin_ms, freqs, wavelet)`` (``functools.lru_cache``) and applied to the rates with two
real BLAS matmuls (real and imaginary parts).  For 350 trials x hundreds of units per session this is ~10x
faster on a CPU workstation and needs no complex intermediates; the numerical result is the same up to
float32 round-off (checked against pywt in ``tests/test_data_layer.py``).
"""
from __future__ import annotations

import functools

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft


def _as_float(a: np.ndarray) -> np.ndarray:
    """Integer (uint8 cache) input -> float32; float input is passed through unchanged.

    Every consumer divides or filters, which would silently promote uint8 counts to float64 and double the
    memory of a (n_trials*n_units, T) matrix; float32 is precise enough for rates and spectral power.
    """
    a = np.asarray(a)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float32)


def smooth_rates(counts: np.ndarray, bin_ms: float, sigma_ms: float) -> np.ndarray:
    """Gaussian-smoothed firing rate (Hz) along the last axis of a count array."""
    counts = _as_float(counts)
    rate = counts / np.asarray(bin_ms / 1000.0, dtype=counts.dtype)
    if sigma_ms <= 0:
        return rate
    return gaussian_filter1d(rate, sigma=sigma_ms / bin_ms, axis=-1, mode="nearest")


def band_power_stft(rates: np.ndarray, bin_ms: float, bands: dict[str, list[float]],
                    win_ms: float = 300.0, hop_bins: int = 1) -> np.ndarray:
    """STFT band power along time. ``rates``: (..., T) -> (..., n_bands, T) (interpolated to T bins).

    A 300 ms Hann window at 100 Hz gives ~3.3 Hz resolution, which is sufficient to separate the
    slow (1-4 Hz ramp/offset), theta (4-12 Hz) and beta (12-30 Hz) bands used here.
    """
    rates = _as_float(rates)
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


def cwt_scales(freqs_hz: np.ndarray, bin_ms: float, wavelet: str) -> np.ndarray:
    """pywt scales that place the wavelet's centre frequency at ``freqs_hz`` for ``bin_ms`` sampling."""
    dt = bin_ms / 1000.0
    fc = pywt.central_frequency(wavelet)
    return fc / (np.asarray(freqs_hz, dtype=float) * dt)


@functools.lru_cache(maxsize=32)
def _cwt_kernel(T: int, bin_ms: float, freqs_hz: tuple[float, ...], wavelet: str) -> tuple[np.ndarray, np.ndarray]:
    """Impulse-response matrices of the CWT: ``(K_real, K_imag)``, each ``(T, F*T)`` float64.

    ``K[t_in, f*T + t_out]`` is the coefficient at frequency ``f`` and time ``t_out`` produced by a unit
    impulse at ``t_in``; because the transform is linear, ``cwt(x) = x @ K`` for any row ``x`` of length
    ``T``.  Computed with ``pywt.cwt`` itself (``method="fft"``, identity input) so that the boundary
    handling and normalisation are exactly pywt's.  Cached per ``(T, bin_ms, freqs, wavelet)``; the
    120 x (15*120) matrices of the default settings take ~3.5 MB.
    """
    dt = bin_ms / 1000.0
    scales = cwt_scales(np.asarray(freqs_hz), bin_ms, wavelet)
    coefs, _ = pywt.cwt(np.eye(int(T)), scales, wavelet, sampling_period=dt, axis=-1, method="fft")  # (F, T_in, T_out)
    K = np.moveaxis(coefs, 0, 1).reshape(int(T), -1)  # (T_in, F*T_out)
    return np.ascontiguousarray(K.real), np.ascontiguousarray(K.imag)


def _cwt_power(X: np.ndarray, bin_ms: float, freqs_hz: np.ndarray, wavelet: str, chunk_rows: int = 2048,
               t_slice: slice | None = None) -> np.ndarray:
    """CWT power ``|cwt(X)|^2`` of the rows of ``X`` (N, T) -> (N, F, T_out) in the dtype of ``X``.

    ``t_slice`` restricts the output times (e.g. the cone-of-influence trim) *before* the matmul, so the
    kernel columns outside the window are never multiplied.  Rows are processed in chunks of ``chunk_rows``
    so the (n, F*T_out) products stay in cache-friendly blocks and the peak memory is bounded.
    """
    X = np.asarray(X)
    N, T = X.shape
    F = len(freqs_hz)
    Kr, Ki = _cwt_kernel(int(T), float(bin_ms), tuple(float(f) for f in np.asarray(freqs_hz)), str(wavelet))
    if t_slice is not None:
        Kr = Kr.reshape(T, F, T)[:, :, t_slice].reshape(T, -1)
        Ki = Ki.reshape(T, F, T)[:, :, t_slice].reshape(T, -1)
    Kr = np.ascontiguousarray(Kr, dtype=X.dtype)
    Ki = np.ascontiguousarray(Ki, dtype=X.dtype)
    T_out = Kr.shape[1] // F
    out = np.empty((N, F, T_out), dtype=X.dtype)
    step = max(int(chunk_rows), 1)
    for i in range(0, N, step):
        Xc = X[i:i + step]
        Cr = Xc @ Kr
        Ci = Xc @ Ki
        Cr *= Cr
        Ci *= Ci
        Cr += Ci
        out[i:i + step] = Cr.reshape(len(Xc), F, T_out)
    return out


def cwt_scalogram(rates: np.ndarray, bin_ms: float, freqs_hz: np.ndarray, wavelet: str = "cmor1.5-1.0",
                  chunk_rows: int = 2048) -> np.ndarray:
    """Complex Morlet CWT power. ``rates``: (T,) or (N, T) -> (N, F, T) (or (F, T)), float32.

    Each row is mean-centred first (the wavelet has zero mean, but centring removes the DC leakage of the
    truncated low-frequency scales at the edges).  Uses the cached impulse-response matrix of ``pywt.cwt``
    (see module docstring) applied chunk-wise, so hundreds of thousands of single-trial rate traces
    (2000 units x 350 trials) never materialise an (N, F, T) complex array.
    """
    rates = _as_float(rates)
    single = rates.ndim == 1
    X = rates[None] if single else rates
    X = (X - X.mean(axis=-1, keepdims=True)).astype(np.float32, copy=False)
    out = _cwt_power(X, bin_ms, np.asarray(freqs_hz, dtype=float), wavelet, chunk_rows=chunk_rows)
    return out[0] if single else out


def band_frequencies(bands: dict[str, list[float]], n_freqs_per_band: int = 5, min_freq_hz: float = 2.0
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Log-spaced CWT frequencies per band: ``(freqs, band_index, names)``.

    Frequencies run from ``max(lo, min_freq_hz)`` to ``hi``: at 100 Hz sampling and a 1.2 s delay a
    ``cmor1.5-1.0`` wavelet below ~2 Hz spans more than the whole window, so its "power" would be pure
    cone-of-influence leakage of the ramp rather than an oscillation - the slow band therefore starts at 2 Hz.
    """
    names, freqs, band_idx = [], [], []
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        lo_eff = max(float(lo), float(min_freq_hz))
        f = np.geomspace(lo_eff, max(float(hi), lo_eff), int(n_freqs_per_band))
        freqs.append(f)
        band_idx += [i] * len(f)
        names.append(name)
    return np.concatenate(freqs), np.asarray(band_idx), names


def band_power_cwt(rates: np.ndarray, bin_ms: float, bands: dict[str, list[float]], wavelet: str = "cmor1.5-1.0",
                   n_freqs_per_band: int = 5, chunk_rows: int = 2048, min_freq_hz: float = 2.0
                   ) -> tuple[np.ndarray, list[str]]:
    """Mean CWT power inside each band, averaged over time (cone of influence trimmed).

    ``rates``: (N, T) -> (N, n_bands) float32.  The central 80 % of the window (``[0.1 T, 0.9 T)``) is
    averaged, which discards the edge bins where the wavelet overlaps the window boundaries.  Only the
    kernel columns inside that window are multiplied, so the trim also saves 20 % of the work.
    ``min_freq_hz`` is the lower limit of the wavelet frequencies (see ``band_frequencies``).
    """
    freqs, band_idx, names = band_frequencies(bands, n_freqs_per_band, min_freq_hz)
    rates = np.asarray(rates, dtype=np.float32)
    N, T = rates.shape
    lo, hi = int(0.1 * T), int(0.9 * T)   # trim the cone of influence (edge effects)
    out = np.empty((N, len(names)), dtype=np.float32)
    step = max(int(chunk_rows), 1)
    for i in range(0, N, step):
        X = rates[i:i + step]
        X = X - X.mean(axis=-1, keepdims=True)
        power = _cwt_power(X, bin_ms, freqs, wavelet, chunk_rows=step, t_slice=slice(lo, hi))  # (n, F, hi-lo)
        mean_p = power.mean(axis=-1)  # (n, F)
        out[i:i + step] = np.stack([mean_p[:, band_idx == b].mean(axis=1) for b in range(len(names))], axis=1)
    return out, names
