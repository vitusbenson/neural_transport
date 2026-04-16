"""Spectral analysis utilities for comparing spatial fields.

Provides 2D power spectrum computation and spectral divergence metrics
for diagnosing conditioning artifacts (e.g., orbit-track stripes).
"""

from __future__ import annotations

import numpy as np


def power_spectrum_2d(
    field_2d: np.ndarray,
    cos_lat_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2D power spectrum of a lat-lon field.

    Parameters
    ----------
    field_2d : np.ndarray, shape [nlat, nlon]
        Input field (e.g., column XCO2 map).
    cos_lat_weights : np.ndarray, shape [nlat], optional
        Cosine latitude weights. If provided, field is pre-weighted
        by sqrt(cos_lat) to approximate area-weighting in spectral space.

    Returns
    -------
    wavenumbers : np.ndarray, shape [n_bins]
        Radial wavenumber bins.
    power : np.ndarray, shape [n_bins]
        Radially-averaged power at each wavenumber.
    """
    nlat, nlon = field_2d.shape

    # Apply latitude weighting if provided
    if cos_lat_weights is not None:
        weights = np.sqrt(np.maximum(cos_lat_weights, 0.0))[:, None]
        field = field_2d * weights
    else:
        field = field_2d

    # Remove mean
    field = field - field.mean()

    # 2D FFT
    fft2 = np.fft.fft2(field)
    power_2d = np.abs(fft2) ** 2 / (nlat * nlon)

    # Shift so DC is at center
    power_2d = np.fft.fftshift(power_2d)

    # Build radial wavenumber grid
    ky = np.fft.fftshift(np.fft.fftfreq(nlat, d=1.0 / nlat))
    kx = np.fft.fftshift(np.fft.fftfreq(nlon, d=1.0 / nlon))
    KX, KY = np.meshgrid(kx, ky)
    k_rad = np.sqrt(KX**2 + KY**2)

    # Radial binning
    k_max = min(nlat, nlon) // 2
    bins = np.arange(0.5, k_max + 0.5, 1.0)
    wavenumbers = np.arange(1, k_max + 1)
    power = np.zeros(len(wavenumbers))

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (k_rad >= lo) & (k_rad < hi)
        if mask.any():
            power[i] = power_2d[mask].mean()

    return wavenumbers, power


def spectral_divergence(
    power_pred: np.ndarray,
    power_gt: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Log-spectral distance between two power spectra.

    D = sqrt(mean((log(P_pred) - log(P_gt))^2))

    Lower is better. Zero means identical spectra.

    Parameters
    ----------
    power_pred, power_gt : np.ndarray
        Power spectra (same length). Must be non-negative.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float — Log-spectral distance.
    """
    log_pred = np.log(np.maximum(power_pred, eps))
    log_gt = np.log(np.maximum(power_gt, eps))
    return float(np.sqrt(np.mean((log_pred - log_gt) ** 2)))


def gradient_magnitude(field_2d: np.ndarray) -> float:
    """Mean absolute gradient magnitude — measures spatial sharpness.

    Higher values indicate sharper, more detailed fields.
    Lower values indicate smoother, blurrier fields.

    Parameters
    ----------
    field_2d : np.ndarray, shape [nlat, nlon]

    Returns
    -------
    float — Mean gradient magnitude.
    """
    gy, gx = np.gradient(field_2d)
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def high_freq_power_ratio(
    power_pred: np.ndarray,
    power_gt: np.ndarray,
    frac: float = 1 / 3,
) -> float:
    """Ratio of high-frequency power (pred / gt).

    Values < 1 indicate the prediction lacks fine-scale detail.
    Values > 1 indicate excess high-frequency energy (artifacts).

    Parameters
    ----------
    power_pred, power_gt : np.ndarray
        Power spectra (same length).
    frac : float
        Fraction of wavenumber range considered "high frequency" (from the top).

    Returns
    -------
    float — High-frequency power ratio.
    """
    n = len(power_gt)
    start = int(n * (1 - frac))
    hf_gt = power_gt[start:].sum()
    hf_pred = power_pred[start:].sum()
    return float(hf_pred / max(hf_gt, 1e-12))


def detail_metrics(
    pred_2d: np.ndarray,
    gt_2d: np.ndarray,
    cos_lat_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute fine-scale detail metrics comparing a prediction field to GT.

    Returns dict with:
        spectral_div: log-spectral distance
        spectral_slope_gt: GT spectral slope
        spectral_slope_pred: prediction spectral slope
        grad_mag_gt: GT gradient magnitude
        grad_mag_pred: prediction gradient magnitude
        grad_ratio: pred/gt gradient ratio (<1 = smoother than GT)
        high_freq_power_ratio: high-freq power ratio (<1 = less detail)
    """
    wn_gt, ps_gt = power_spectrum_2d(gt_2d, cos_lat_weights)
    wn_pred, ps_pred = power_spectrum_2d(pred_2d, cos_lat_weights)

    return {
        "spectral_div": spectral_divergence(ps_pred, ps_gt),
        "spectral_slope_gt": spectral_slope(wn_gt, ps_gt),
        "spectral_slope_pred": spectral_slope(wn_pred, ps_pred),
        "grad_mag_gt": gradient_magnitude(gt_2d),
        "grad_mag_pred": gradient_magnitude(pred_2d),
        "grad_ratio": gradient_magnitude(pred_2d) / max(gradient_magnitude(gt_2d), 1e-12),
        "high_freq_power_ratio": high_freq_power_ratio(ps_pred, ps_gt),
    }


def spectral_slope(wavenumbers: np.ndarray, power: np.ndarray) -> float:
    """Fit spectral slope in log-log space (linear regression).

    For natural fields, slope is typically -2 to -3 (Kolmogorov-like).
    Conditioning artifacts may flatten the spectrum at certain wavenumbers.

    Parameters
    ----------
    wavenumbers : np.ndarray
    power : np.ndarray

    Returns
    -------
    float — slope of log(power) vs log(wavenumber).
    """
    valid = power > 0
    if valid.sum() < 3:
        return np.nan
    log_k = np.log(wavenumbers[valid])
    log_p = np.log(power[valid])
    # Linear regression
    coeffs = np.polyfit(log_k, log_p, 1)
    return float(coeffs[0])
