"""Spatial utility functions for geophysical fields.

Extracted from flowmatching.py — provides geo-aware 2D Gaussian smoothing
with periodic longitude wrapping and reflective latitude padding.
"""

import torch
import torch.nn.functional as F


def gaussian_smooth_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian smoothing with geo-aware padding.

    Args:
        field: [B, C, Nlat, Nlon] tensor to smooth.
        sigma: Gaussian kernel standard deviation in grid cells. If <= 0, returns input unchanged.

    Returns:
        Smoothed tensor of same shape.

    Padding:
        - Longitude: circular (periodic).
        - Latitude: reflect (no periodicity at poles).
    """
    if sigma <= 0:
        return field

    Nlat, Nlon = field.shape[2], field.shape[3]

    ks = int(6 * sigma + 1)
    if ks % 2 == 0:
        ks += 1
    # Clamp kernel size so padding doesn't exceed spatial dimensions
    ks = min(ks, 2 * min(Nlat, Nlon) - 1)
    if ks < 3:
        return field
    half = ks // 2

    # 1D Gaussian kernel
    coords = torch.arange(ks, dtype=field.dtype, device=field.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    C = field.shape[1]

    # --- Longitude (dim=-1): circular padding ---
    half_lon = min(half, Nlon - 1)
    kernel_lon_1d = kernel_1d[half - half_lon : half + half_lon + 1]
    kernel_lon_1d = kernel_lon_1d / kernel_lon_1d.sum()
    ks_lon = 2 * half_lon + 1
    kernel_lon = kernel_lon_1d.view(1, 1, 1, ks_lon).expand(C, 1, 1, ks_lon)
    padded = F.pad(field, (half_lon, half_lon, 0, 0), mode='circular')
    out = F.conv2d(padded, kernel_lon, groups=C)

    # --- Latitude (dim=-2): reflect padding ---
    half_lat = min(half, Nlat - 1)
    kernel_lat_1d = kernel_1d[half - half_lat : half + half_lat + 1]
    kernel_lat_1d = kernel_lat_1d / kernel_lat_1d.sum()
    ks_lat = 2 * half_lat + 1
    kernel_lat = kernel_lat_1d.view(1, 1, ks_lat, 1).expand(C, 1, ks_lat, 1)
    padded = F.pad(out, (0, 0, half_lat, half_lat), mode='reflect')
    out = F.conv2d(padded, kernel_lat, groups=C)

    return out
