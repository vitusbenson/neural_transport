"""Tests for neural_transport.inference.masking module."""

import pytest
import torch

from neural_transport.inference.masking import (
    apply_masking,
    apply_temporal_weighting,
    create_column_mask,
    create_mask,
    get_temporal_weight,
    masking_interpolate,
    masking_simple,
    masking_total_column_average_simple,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def batch_3d():
    """Batch with 3D CO2 field [B=2, T=1, N=128, C=5] and pressure data."""
    torch.manual_seed(42)
    B, T, nlat, nlon, C = 2, 1, 8, 16, 5
    N = nlat * nlon

    # Pressure decreases with altitude; surface is level 0
    p_surface = 101325.0
    p_levels = torch.linspace(p_surface, 10000.0, C + 1)
    p_bottom = p_levels[:-1].view(1, 1, 1, C).expand(B, T, N, C)
    p_top = p_levels[1:].view(1, 1, 1, C).expand(B, T, N, C)

    return {
        "co2massmix": 400.0 + 2.0 * torch.randn(B, T, N, C),
        "p_bottom": p_bottom.clone(),
        "p_top": p_top.clone(),
    }


@pytest.fixture
def spatial_dims():
    return {"nlat": 8, "nlon": 16}


# ── Mask creation tests ──────────────────────────────────────────────────


@pytest.mark.parametrize("pattern", ["random", "vertical", "horizontal", "checkerboard", "satellite"])
def test_create_mask_shapes_and_patterns(batch_3d, spatial_dims, pattern):
    """Verify output shapes and boolean dtype for all mask patterns."""
    obs_mask, obs_values = create_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=0.2,
        mask_pattern=pattern,
        **spatial_dims,
    )
    B, T, N, C = batch_3d["co2massmix"].shape
    assert obs_mask.shape == (B, T, N, C)
    assert obs_mask.dtype == torch.bool
    assert obs_values.shape == (B, T, N, C)
    # At least some observations
    assert obs_mask.any()


def test_create_mask_satellite_uses_config_constants(batch_3d, spatial_dims):
    """Satellite pattern should produce non-trivial mask using config constants."""
    from neural_transport.configs import SATELLITE_TILT_RAD, SWATH_SPACING_FACTOR

    obs_mask, _ = create_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=0.2,
        mask_pattern="satellite",
        **spatial_dims,
    )
    # The satellite pattern should not be all-True or all-False
    assert obs_mask.any() and not obs_mask.all()
    # Verify constants are accessible (sanity)
    assert isinstance(SATELLITE_TILT_RAD, float)
    assert isinstance(SWATH_SPACING_FACTOR, int)


def test_create_column_mask_xco2_computation(batch_3d, spatial_dims):
    """Verify obs_values = sum(h_k * a_k * x_k) at observed locations."""
    obs_mask, obs_values = create_column_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=0.3,
        mask_pattern="random",
        **spatial_dims,
    )
    # obs_values should be [B, T, N, 1]
    assert obs_values.shape[-1] == 1
    assert obs_mask.shape[-1] == 1

    # Recompute XCO2 manually
    p_bottom = batch_3d["p_bottom"]
    p_top = batch_3d["p_top"]
    dp = p_bottom - p_top
    p_surface = p_bottom[:, :, :, 0:1]
    h_k = dp / p_surface.clamp(min=1e-6)
    # ak_10=None -> ak=1
    xco2_manual = (h_k * batch_3d["co2massmix"]).sum(dim=-1, keepdim=True)

    # At observed locations, obs_values should match manual computation
    torch.testing.assert_close(
        obs_values[obs_mask].float(),
        xco2_manual[obs_mask.expand_as(xco2_manual)].float(),
        atol=1e-4,
        rtol=1e-4,
    )


def test_create_column_mask_adds_batch_keys(batch_3d, spatial_dims):
    """Verify batch gets xco2_averaging_kernel, xco2_apriori, co2_profile_apriori, pressure_weight."""
    create_column_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=0.2,
        mask_pattern="random",
        **spatial_dims,
    )
    assert "xco2_averaging_kernel" in batch_3d
    assert "xco2_apriori" in batch_3d
    assert "co2_profile_apriori" in batch_3d
    assert "pressure_weight" in batch_3d


def test_create_mask_obs_fraction(batch_3d, spatial_dims):
    """Random pattern coverage should be approximately obs_fraction."""
    torch.manual_seed(0)
    obs_fraction = 0.3
    obs_mask, _ = create_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=obs_fraction,
        mask_pattern="random",
        **spatial_dims,
    )
    # Coverage per spatial point (averaged over batch and channels)
    N = batch_3d["co2massmix"].shape[2]
    actual_fraction = obs_mask[:, 0, :, 0].float().sum(dim=-1) / N
    for frac in actual_fraction:
        assert abs(frac.item() - obs_fraction) < 0.05


def test_create_mask_edge_cases(batch_3d, spatial_dims):
    """obs_fraction=0 gives empty mask, obs_fraction=1 gives full mask."""
    obs_mask_empty, _ = create_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=0.0,
        mask_pattern="random",
        **spatial_dims,
    )
    assert not obs_mask_empty.any()

    obs_mask_full, _ = create_mask(
        batch_3d,
        target_var="co2massmix",
        obs_fraction=1.0,
        mask_pattern="random",
        **spatial_dims,
    )
    assert obs_mask_full.all()


# ── Masking application tests ────────────────────────────────────────────


def test_masking_simple():
    """Obs locations replaced, non-obs unchanged."""
    x = torch.randn(2, 5, 4, 8)
    obs_mask = torch.zeros(2, 1, 4, 8, dtype=torch.bool)
    obs_mask[:, :, :2, :] = True
    obs_values = torch.ones(2, 1, 4, 8) * 42.0

    result = masking_simple(x, obs_mask, obs_values)

    # Observed locations should be 42.0 (broadcast across channels)
    assert (result[:, :, :2, :] == 42.0).all()
    # Non-observed should be unchanged
    torch.testing.assert_close(result[:, :, 2:, :], x[:, :, 2:, :])


def test_masking_interpolate_endpoints():
    """At t=0 result=x at obs, at t=1 result=obs_values at obs."""
    x = torch.randn(2, 5, 4, 8)
    obs_mask = torch.ones(2, 1, 4, 8, dtype=torch.bool)
    obs_values = torch.ones(2, 1, 4, 8) * 10.0

    # t=0: should return x at observed locations
    result_t0 = masking_interpolate(x, 0.0, obs_mask, obs_values)
    torch.testing.assert_close(result_t0, x)

    # t=1: should return obs_values
    result_t1 = masking_interpolate(x, 1.0, obs_mask, obs_values)
    # obs_values broadcasts to match x channels via torch.where
    assert (result_t1 == 10.0).all()


def test_masking_interpolate_midpoint():
    """At t=0.5, result is average of x and obs_values at obs locations."""
    x = torch.zeros(1, 1, 2, 2)
    obs_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    obs_values = torch.ones(1, 1, 2, 2) * 10.0

    result = masking_interpolate(x, 0.5, obs_mask, obs_values)
    torch.testing.assert_close(result, torch.ones(1, 1, 2, 2) * 5.0)


def test_masking_total_column_average_simple_constraint(sample_forward_model, sample_masking_config):
    """After masking, forward_model(x_masked) should approximately equal obs_values at obs locations."""
    B, nlev, nlat, nlon = 2, 5, 4, 8
    torch.manual_seed(42)
    x = torch.randn(B, nlev, nlat, nlon)

    obs_mask = sample_masking_config["obs_mask"]
    obs_values = sample_masking_config["obs_values"]
    ak = sample_masking_config["ak"]
    pw = sample_masking_config["pressure_weights"]

    x_masked = masking_total_column_average_simple(
        x,
        obs_mask,
        obs_values,
        sample_forward_model,
        ak=ak,
        pressure_weights=pw,
    )

    # At observed locations, forward_model(x_masked) should be close to obs_values
    xco2_after = sample_forward_model.forward(x_masked)
    obs_bool = obs_mask.bool()
    if obs_bool.any():
        torch.testing.assert_close(
            xco2_after[obs_bool.expand_as(xco2_after)],
            obs_values[obs_bool.expand_as(obs_values)],
            atol=1e-3,
            rtol=1e-3,
        )


@pytest.mark.parametrize(
    "mode,t_val,expected_high",
    [
        ("smooth_late_masking", 0.95, True),
        ("smooth_late_masking", 0.1, False),
        ("step_late_masking", 0.95, True),
        ("step_late_masking", 0.1, False),
        ("smooth_early_masking", 0.1, True),
        ("smooth_early_masking", 0.95, False),
        ("step_early_masking", 0.1, True),
        ("step_early_masking", 0.95, False),
        (None, 0.5, True),  # None -> always 1.0
    ],
)
def test_get_temporal_weight_modes(mode, t_val, expected_high):
    """Test all temporal weight modes produce correct high/low weights."""
    t = torch.tensor([t_val])
    w = get_temporal_weight(t, mode, t_threshold=0.9)
    if isinstance(w, float):
        val = w
    else:
        val = w.item()

    if expected_high:
        assert val > 0.5
    else:
        assert val < 0.5


def test_apply_masking_dispatch():
    """Correct routing for each method name."""
    x = torch.randn(1, 3, 4, 4)
    obs_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    obs_values = torch.ones(1, 1, 4, 4) * 5.0
    t = torch.tensor([0.5])

    # simple
    result = apply_masking("simple", x, t, obs_mask, obs_values)
    assert (result == 5.0).all()

    # interpolate
    result = apply_masking("interpolate", x, t, obs_mask, obs_values)
    expected = 0.5 * 5.0 + 0.5 * x
    torch.testing.assert_close(result, expected)


def test_apply_masking_unknown_raises():
    """ValueError for unknown method."""
    x = torch.randn(1, 3, 4, 4)
    obs_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    obs_values = torch.ones(1, 1, 4, 4)
    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="Unknown masking method"):
        apply_masking("nonexistent", x, t, obs_mask, obs_values)


def test_apply_temporal_weighting():
    """weight=0 gives x, weight=1 gives x_masked."""
    x = torch.zeros(1, 3, 4, 4)
    x_masked = torch.ones(1, 3, 4, 4)

    # None masking_time -> weight=1.0 -> should return x_masked
    result = apply_temporal_weighting(x, x_masked, torch.tensor([0.5]), None)
    torch.testing.assert_close(result, x_masked)

    # step_late at t=0.1 (below threshold) -> weight=0 -> should return x
    result = apply_temporal_weighting(x, x_masked, torch.tensor([0.1]), "step_late_masking", t_threshold=0.9)
    torch.testing.assert_close(result, x)

    # step_late at t=0.95 (above threshold) -> weight=1 -> should return x_masked
    result = apply_temporal_weighting(x, x_masked, torch.tensor([0.95]), "step_late_masking", t_threshold=0.9)
    torch.testing.assert_close(result, x_masked)
