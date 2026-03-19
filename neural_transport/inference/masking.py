"""Masking logic for observation conditioning in flow-matching inference.

Section A: Mask creation functions (from generative.py)
Section B: Masking application functions (from MaskedVelocityWrapper)
"""

import numpy as np
import torch

from neural_transport.configs import DT_FALLBACK, SATELLITE_TILT_RAD, SWATH_SPACING_FACTOR
from neural_transport.tools.conversion import molemix_to_massmix

# ── Section A: Mask creation ─────────────────────────────────────────────


def create_oco2_mask(batch, target_var="xco2_2019_scale"):
    """
    Create OCO-2 observation mask and values from sparse target_var field.
    Args:
      batch: dict of tensors, each of shape [B T N C]
      target_var: the variable to create the mask for

    Returns:
      obs_mask:  bool[B,T,N,C] with True where OCO-2 has data
      obs_values: float[B,T,N,C] containing observed values at mask locations,
                  NaN elsewhere
    """
    obs_mask = ~torch.isnan(batch[target_var])
    obs_values = batch[target_var].clone()
    obs_values = molemix_to_massmix(obs_values)
    batch[f"{target_var}_offset"] = molemix_to_massmix(batch[f"{target_var}_offset"])
    batch[f"{target_var}_scale"] = molemix_to_massmix(batch[f"{target_var}_scale"])

    # Handle xco2_averaging_kernel
    ak = batch["xco2_averaging_kernel"].clone()  # [B, T, N, C=10]
    n_levels = ak.shape[-1]
    ak_mask = obs_mask.expand(-1, -1, -1, n_levels)  # [B, T, N, C=10]
    valid_ak = ak[ak_mask]
    mean_ak_per_level = valid_ak.reshape(-1, n_levels).mean(dim=0)  # [C=10]
    mean_ak_full = mean_ak_per_level.view(1, 1, 1, -1).expand_as(ak)
    ak_cleaned = torch.where(ak_mask, ak, mean_ak_full)
    batch["xco2_averaging_kernel"] = ak_cleaned

    return obs_mask, obs_values  # [B T N C] each


def create_oco2_mask_test(
    batch,
    target_var="xco2_2019_scale",
    mask_pattern="diagonal",
    nlat=32,
    nlon=64,
):
    """
    Create synthetic OCO-2-like observation masks for testing.

    Args:
        batch: dict of tensors [B, T, N, C]
        target_var: variable used to infer shape
        mask_pattern:
            - "diagonal"   : diagonal stripe (lat = lon)
            - "leftright"  : left half observed
            - "topbottom"  : top half observed
            - "checkerboard"
            - "center_box" : central rectangle
        nlat, nlon: grid dimensions (must satisfy nlat * nlon == N)

    Returns:
        obs_mask   : bool [B, T, N, C]
        obs_values : float [B, T, N, C] (NaN outside mask)
    """

    B, T, N, C = batch[target_var].shape
    device = batch[target_var].device

    assert nlat * nlon == N, "nlat * nlon must equal N"

    grid = torch.arange(N, device=device).reshape(nlat, nlon)

    mask2d = torch.zeros((nlat, nlon), dtype=torch.bool, device=device)

    if mask_pattern == "diagonal":
        for i in range(min(nlat, nlon)):
            mask2d[i, i] = True

    elif mask_pattern == "leftright":
        mask2d[:, : nlon // 2] = True

    elif mask_pattern == "topbottom":
        mask2d[: nlat // 2, :] = True

    elif mask_pattern == "checkerboard":
        mask2d = (torch.arange(nlat, device=device)[:, None] + torch.arange(nlon, device=device)[None, :]) % 2 == 0

    elif mask_pattern == "center_box":
        lat0, lat1 = nlat // 4, 3 * nlat // 4
        lon0, lon1 = nlon // 4, 3 * nlon // 4
        mask2d[lat0:lat1, lon0:lon1] = True

    else:
        raise ValueError(f"Unknown test pattern: {mask_pattern}")

    obs_indices = grid[mask2d].reshape(-1)

    obs_mask = torch.zeros((B, T, N, C), dtype=torch.bool, device=device)
    obs_values = torch.full((B, T, N, C), float("nan"), device=device)

    obs_mask[:, :, obs_indices, :] = True
    gt_values = molemix_to_massmix(batch[target_var].clone())
    obs_values[:, :, obs_indices, :] = gt_values[:, :, obs_indices, :]

    # Handle xco2_averaging_kernel
    ak = batch["xco2_averaging_kernel"].clone()  # [B, T, N, C=10]
    n_levels = ak.shape[-1]
    true_mask = ~torch.isnan(batch[target_var])
    ak_mask = true_mask.expand(-1, -1, -1, n_levels)  # [B, T, N, C=10]
    valid_ak = ak[ak_mask]
    mean_ak_per_level = valid_ak.reshape(-1, n_levels).mean(dim=0)  # [C=10]
    mean_ak_full = mean_ak_per_level.view(1, 1, 1, -1).expand_as(ak)
    batch["xco2_averaging_kernel"] = mean_ak_full

    return obs_mask, obs_values  # [B T N C] each


def create_column_mask(
    batch,
    target_var="co2massmix",
    obs_fraction=0.1,
    mask_pattern="random",
    nlat=32,
    nlon=64,
    ak_10=None,
):
    """
    Create synthetic XCO2 column observations from a 3D CO2 field using
    average OCO-2 averaging kernel and CarbonTracker pressure weights.

    Computes XCO2 = sum(h_k * a_k * x_k) where:
        h_k = (p_bottom_k - p_top_k) / p_surface  (pressure weight)
        a_k = mean OCO-2 averaging kernel at level k
        x_k = CO2 mass mixing ratio at level k

    Args:
        batch: dict of tensors [B, T, N, C]
        target_var: 3D CO2 variable name
        obs_fraction: fraction of spatial points to observe
        mask_pattern: spatial pattern for observations
        nlat, nlon: grid dimensions
        ak_10: averaging kernel on 10 aggregated levels [C=10], numpy array.
               If None, uses uniform AK (all ones).

    Returns:
        obs_mask: bool [B, T, N, 1] spatial mask
        obs_values: float [B, T, N, 1] synthetic XCO2 at observed locations
        Also modifies batch in-place to add:
            xco2_averaging_kernel [B, T, N, C]
            xco2_apriori [B, T, N, 1]  (set to zero - no prior correction needed for OSSE)
            co2_profile_apriori [B, T, N, C]  (set to zero)
            pressure_weight [B, T, N, C]
    """
    device = batch[target_var].device
    B, T, N, C = batch[target_var].shape

    # Compute pressure weights h_k = (p_bottom_k - p_top_k) / p_surface
    p_bottom = batch["p_bottom"]  # [B, T, N, C]
    p_top = batch["p_top"]  # [B, T, N, C]
    dp = p_bottom - p_top  # [B, T, N, C]
    p_surface = p_bottom[:, :, :, 0:1]  # surface level (level 0)
    h_k = dp / p_surface.clamp(min=1e-6)  # [B, T, N, C]

    # Averaging kernel
    if ak_10 is not None:
        ak = torch.tensor(ak_10, dtype=torch.float32, device=device)
        ak = ak.view(1, 1, 1, C).expand(B, T, N, C)
    else:
        ak = torch.ones(B, T, N, C, device=device)

    # Compute synthetic XCO2 = sum(h_k * a_k * x_k)
    co2 = batch[target_var]  # [B, T, N, C]
    xco2 = (h_k * ak * co2).sum(dim=-1, keepdim=True)  # [B, T, N, 1]

    # Create spatial mask
    obs_mask = torch.zeros((B, T, N, 1), dtype=torch.bool, device=device)
    obs_values = torch.full((B, T, N, 1), float('nan'), device=device)

    for t_idx in range(T):
        if mask_pattern == "random":
            num_obs = int(obs_fraction * N)
            obs_indices = torch.randperm(N, device=device)[:num_obs]
        elif mask_pattern == "checkerboard":
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            mask2d = (torch.arange(nlat, device=device)[:, None] + torch.arange(nlon, device=device)[None, :]) % 2 == 0
            obs_indices = grid[mask2d].reshape(-1)
        elif mask_pattern == "satellite":
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            swath_width = max(1, int(obs_fraction * nlon / SWATH_SPACING_FACTOR))
            tilt = SATELLITE_TILT_RAD
            cols = []
            for i in range(0, nlon, swath_width * SWATH_SPACING_FACTOR):
                for w in range(swath_width):
                    col_idx = i + w
                    if col_idx < nlon:
                        lat_offsets = ((torch.arange(nlat, device=device) * np.tan(tilt)).long()) % nlon
                        col_with_tilt = (col_idx + lat_offsets) % nlon
                        cols.append(col_with_tilt.unsqueeze(0))
            cols = torch.cat(cols, dim=0)
            obs_indices = grid[torch.arange(nlat).unsqueeze(0), cols].reshape(-1)
        elif mask_pattern == "vertical":
            num_cols = max(1, int(obs_fraction * nlon))
            cols_sel = torch.arange(0, nlon, nlon // num_cols, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[:, cols_sel].reshape(-1)
        elif mask_pattern == "horizontal":
            num_rows = max(1, int(obs_fraction * nlat))
            rows = torch.arange(0, nlat, nlat // num_rows, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[rows, :].reshape(-1)
        else:
            raise ValueError(f"Unknown mask pattern: {mask_pattern}")

        obs_mask[:, t_idx, obs_indices, :] = True
        obs_values[:, t_idx, obs_indices, :] = xco2[:, t_idx, obs_indices, :]

    # Add AK, prior, and pressure_weight to batch for prepare_masking_config
    batch["xco2_averaging_kernel"] = ak
    # For OSSE with CT as truth, prior is zero (XCO2 = sum(h*a*x), no prior correction)
    batch["xco2_apriori"] = torch.zeros(B, T, N, 1, device=device)
    batch["co2_profile_apriori"] = torch.zeros(B, T, N, C, device=device)
    batch["pressure_weight"] = h_k

    return obs_mask, obs_values  # [B T N 1], [B T N 1]


def create_mask(batch, target_var="co2massmix", obs_fraction=0.1, mask_pattern="random", nlat=32, nlon=64):
    """
    Create a random observation mask for the input batch.
    Args:
        batch: dict of tensors, each of shape [B T N C]
        target_var: the variable to create the mask for
        obs_fraction: fraction of points to keep as observations
        mask_pattern: "random", "vertical", "horizontal", "checkerboard", "satellite"
    """
    device = batch[target_var].device
    B, T, N, C = batch[target_var].shape

    obs_mask = torch.zeros((B, T, N, C), dtype=torch.bool, device=device)
    obs_values = torch.full_like(batch[target_var], float('nan'), device=device)

    for t in range(T):
        if mask_pattern == "random":
            num_obs = int(obs_fraction * N)
            obs_indices = torch.randperm(N, device=device)[:num_obs]

        elif mask_pattern == "vertical":
            # keep a fixed fraction of longitude columns
            num_cols = max(1, int(obs_fraction * nlon))
            cols = torch.arange(0, nlon, nlon // num_cols, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[:, cols].reshape(-1)

        elif mask_pattern == "horizontal":
            # keep a fixed fraction of latitude rows
            num_rows = max(1, int(obs_fraction * nlat))
            rows = torch.arange(0, nlat, nlat // num_rows, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[rows, :].reshape(-1)

        elif mask_pattern == "checkerboard":
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            mask2d = (torch.arange(nlat, device=device)[:, None] + torch.arange(nlon, device=device)[None, :]) % 2 == 0
            obs_indices = grid[mask2d].reshape(-1)

        elif mask_pattern == "satellite":
            # grid = [nlat, nlon]
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            # choose swath width (fraction of nlon)
            swath_width = max(1, int(obs_fraction * nlon / SWATH_SPACING_FACTOR))
            # tilt angle in radians (small tilt)
            tilt = SATELLITE_TILT_RAD
            cols = []
            for i in range(0, nlon, swath_width * SWATH_SPACING_FACTOR):  # spacing between swaths
                for w in range(swath_width):
                    col_idx = i + w
                    if col_idx < nlon:
                        # apply tilt shift proportional to latitude
                        lat_offsets = ((torch.arange(nlat, device=device) * np.tan(tilt)).long()) % nlon
                        col_with_tilt = (col_idx + lat_offsets) % nlon
                        cols.append(col_with_tilt.unsqueeze(0))
            # stack and mask
            cols = torch.cat(cols, dim=0)  # shape [n_swath, nlat]
            obs_indices = grid[torch.arange(nlat).unsqueeze(0), cols].reshape(-1)

        else:
            raise ValueError(f"Unknown mask pattern: {mask_pattern}")

        # fill mask + values
        obs_mask[:, t, obs_indices, :] = True
        obs_values[:, t, obs_indices, :] = batch[target_var][:, t, obs_indices, :]

    return obs_mask, obs_values  # [B T N C] each


# ── Section B: Masking application ───────────────────────────────────────


def get_temporal_weight(t, masking_time, t_threshold=0.9):
    """Compute scalar temporal weight for conditioning strength.

    Args:
        t: current timestep tensor
        masking_time: one of "smooth_late_masking", "step_late_masking",
                      "smooth_early_masking", "step_early_masking", or None
        t_threshold: threshold for step/smooth transitions

    Returns:
        Scalar or tensor weight in [0, 1].
    """
    if masking_time == "smooth_late_masking":
        return torch.sigmoid((t - t_threshold) * 20.0).view(-1, 1, 1, 1)
    elif masking_time == "step_late_masking":
        return (t >= t_threshold).float().view(-1, 1, 1, 1)
    elif masking_time == "smooth_early_masking":
        return torch.sigmoid((t_threshold - t) * 20.0).view(-1, 1, 1, 1)
    elif masking_time == "step_early_masking":
        return (t < t_threshold).float().view(-1, 1, 1, 1)
    else:
        return 1.0


def masking_simple(x, obs_mask, obs_values):
    """Replace x at observed locations with obs_values."""
    return torch.where(obs_mask.bool(), obs_values.detach(), x)


def masking_interpolate(x, t, obs_mask, obs_values):
    """Time-weighted linear blend at observed locations."""
    blended = t * obs_values.detach() + (1.0 - t) * x
    return torch.where(obs_mask.bool(), blended, x)


def masking_total_column_average_simple(x, obs_mask, obs_values, forward_model, ak=None, pressure_weights=None):
    """Uniform additive column correction at observed locations."""
    if ak is None:
        ak = torch.ones(x.shape, device=x.device)

    h = pressure_weights if pressure_weights is not None else 1.0 / x.shape[1]
    h_ak = h * ak
    h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)

    xco2 = forward_model.forward(x)  # [B 1 Nlat Nlon]
    column_error = obs_values.detach() - xco2  # [B 1 Nlat Nlon]

    distributed_correction = column_error / h_ak_sum  # [B C Nlat Nlon] - uniform per level
    return torch.where(obs_mask.bool(), x + distributed_correction, x)


def masking_total_column_average_mult(
    x,
    obs_mask,
    obs_values,
    forward_model,
    ak=None,
    pressure_weights=None,
    target_mean=None,
    target_std=None,
    obs_mean=None,
    obs_std=None,
    xco2_prior=None,
    co2_profile_prior=None,
):
    """Multiplicative column correction at observed locations.

    Args:
        x: [B, C, Nlat, Nlon] - the C-level CO2 field
        obs_mask: observation mask
        obs_values: observed XCO2 values (normalized)
        forward_model: XCO2ForwardModel instance
        ak: averaging kernel
        pressure_weights: pressure layer weights
        target_mean, target_std: target normalization parameters
        obs_mean, obs_std: observation normalization parameters
        xco2_prior: prior column-averaged XCO2
        co2_profile_prior: prior CO2 profile
    """
    if ak is None:
        ak = torch.ones(x.shape, device=x.device)
    h = pressure_weights if pressure_weights is not None else 1.0 / x.shape[1]
    x_physical = x * target_std + target_mean
    obs_physical = obs_values * obs_std + obs_mean

    xco2_physical = xco2_prior + (h * ak * (x_physical - co2_profile_prior)).sum(dim=1, keepdim=True)  # [B 1 Nlat Nlon]

    h_ak = h * ak
    correction = (xco2_prior * h / h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12) + x_physical - co2_profile_prior) * (
        obs_physical / xco2_physical - 1
    )  # [B C Nlat Nlon]

    x_scaled_physical = x_physical + correction  # [B C Nlat Nlon]
    x_scaled = (x_scaled_physical - target_mean) / target_std

    return torch.where(obs_mask.bool(), x_scaled, x)


# Dispatch table for masking methods
_MASKING_METHODS = {
    "simple": masking_simple,
    "interpolate": masking_interpolate,
    "total_column_average_simple": masking_total_column_average_simple,
    "total_column_average_mult": masking_total_column_average_mult,
}

# Methods that require the `t` argument
_METHODS_WITH_T = {"interpolate"}


_COLUMN_SIMPLE_KWARGS = {"ak", "pressure_weights"}
_COLUMN_MULT_KWARGS = {
    "ak",
    "pressure_weights",
    "target_mean",
    "target_std",
    "obs_mean",
    "obs_std",
    "xco2_prior",
    "co2_profile_prior",
}


def apply_masking(method, x, t, obs_mask, obs_values, forward_model=None, **kwargs):
    """Dispatch to appropriate masking method by name.

    Args:
        method: masking method name (e.g. "simple", "interpolate",
                "total_column_average_simple", "total_column_average_mult")
        x: current state tensor
        t: current timestep
        obs_mask: observation mask
        obs_values: observed values
        forward_model: XCO2ForwardModel (needed for column methods)
        **kwargs: additional keyword arguments passed to the masking function

    Returns:
        Masked state tensor.

    Raises:
        ValueError: if method name is unknown.
    """
    fn = _MASKING_METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown masking method: {method}")

    if method in _METHODS_WITH_T:
        return fn(x, t, obs_mask, obs_values)
    elif method == "total_column_average_simple":
        filtered = {k: v for k, v in kwargs.items() if k in _COLUMN_SIMPLE_KWARGS}
        return fn(x, obs_mask, obs_values, forward_model, **filtered)
    elif method == "total_column_average_mult":
        filtered = {k: v for k, v in kwargs.items() if k in _COLUMN_MULT_KWARGS}
        return fn(x, obs_mask, obs_values, forward_model, **filtered)
    else:
        return fn(x, obs_mask, obs_values)


def compute_dt(t, time_grid, fallback=DT_FALLBACK):
    """Compute dt from time_grid at position t, with fallback."""
    if time_grid is not None:
        idx = torch.searchsorted(time_grid, t.item())
        if idx == 0:
            dt = time_grid[1] - time_grid[0]
        elif idx >= len(time_grid):
            dt = time_grid[-1] - time_grid[-2]
        else:
            dt = time_grid[idx] - time_grid[idx - 1]
    else:
        dt = fallback
    return dt


def apply_temporal_weighting(x, x_masked, t, masking_time, t_threshold=0.9):
    """Blend masked and unmasked states using temporal weight.

    Args:
        x: original state
        x_masked: masked state
        t: current timestep
        masking_time: temporal weighting strategy
        t_threshold: threshold for step/smooth transitions

    Returns:
        Weighted blend of x and x_masked.
    """
    mask_weight = get_temporal_weight(t, masking_time, t_threshold)
    return mask_weight * x_masked + (1.0 - mask_weight) * x
