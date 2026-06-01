"""Unit tests for MaskedVelocityWrapper and XCO2ForwardModel."""

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import MockSubmodel

from neural_transport.forward_model import XCO2ForwardModel, build_interp_matrix
from neural_transport.models.flowmatching import MaskedVelocityWrapper
from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d


def _make_wrapper(
    nlev=5,
    nlat=4,
    nlon=4,
    batch_size=2,
    target_mean=400.0,
    target_std=5.0,
    obs_mean=400.0,
    obs_std=5.0,
    targshift_mean=None,
    use_prior=True,
    masking_method="total_column_average_simple",
    conditioning_mode="correction",
    guidance_scale=1.0,
    sigma_obs=1.0,
    spatial_smoothing_sigma=0.0,
):
    """Build a MaskedVelocityWrapper with synthetic data for testing."""
    B, C, Nlat, Nlon = batch_size, nlev, nlat, nlon

    # Pressure weights: surface-heavy (decreasing with altitude)
    pw = torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08]).view(1, C, 1, 1).expand(B, C, Nlat, Nlon)
    # Averaging kernel: near 1 at surface, decreasing aloft
    ak = torch.tensor([0.95, 0.85, 0.60, 0.30, 0.10]).view(1, C, 1, 1).expand(B, C, Nlat, Nlon)

    # Physical CO2 profile ~400 ppm + small variation
    torch.manual_seed(42)
    x_phys = 400.0 + 2.0 * torch.randn(B, C, Nlat, Nlon)
    x_norm = (x_phys - target_mean) / target_std

    # Build masking_config
    target_mean_t = torch.tensor(target_mean).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    target_std_t = torch.tensor(target_std).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    obs_mean_t = torch.tensor(obs_mean).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    obs_std_t = torch.tensor(obs_std).view(1, 1, 1, 1).expand(B, 1, 1, 1)

    # XCO2 ground truth from physical profile
    xco2_phys = (pw * ak * x_phys).sum(dim=1, keepdim=True)
    obs_values_norm = (xco2_phys - obs_mean) / obs_std  # [B, 1, Nlat, Nlon]
    obs_mask = torch.ones(B, 1, Nlat, Nlon, dtype=torch.bool)

    masking_config = {
        "obs_mask": obs_mask,
        "obs_values": obs_values_norm,
        "obs_mean": obs_mean_t,
        "obs_std": obs_std_t,
        "target_mean": target_mean_t,
        "target_std": target_std_t,
        "ak": ak,
        "pressure_weights": pw,
        "targshift_mean": targshift_mean,
        "time_grid": torch.linspace(0, 1, 11),
    }

    if use_prior:
        co2_profile_prior = 400.0 * torch.ones(B, C, Nlat, Nlon)
        xco2_prior = (pw * ak * co2_profile_prior).sum(dim=1, keepdim=True)
        masking_config["xco2_prior"] = xco2_prior
        masking_config["co2_profile_prior"] = co2_profile_prior
    else:
        masking_config["xco2_prior"] = None
        masking_config["co2_profile_prior"] = None

    submodel = MockSubmodel(nlev=nlev)
    wrapper = MaskedVelocityWrapper(
        submodel=submodel,
        masking_config=masking_config,
        nlev=nlev,
        masking_method=masking_method,
        conditioning_mode=conditioning_mode,
        guidance_scale=guidance_scale,
        sigma_obs=sigma_obs,
        spatial_smoothing_sigma=spatial_smoothing_sigma,
    )

    return wrapper, x_norm, x_phys, masking_config


class TestComputeXCO2WithPrior:
    def test_roundtrip(self):
        """XCO2 computed from x should match the obs_values we derived from x."""
        wrapper, x_norm, x_phys, masking_config = _make_wrapper(use_prior=True)
        xco2_norm = wrapper.forward_model.forward(x_norm)
        # obs_values were computed as (sum(pw*ak*x_phys) - obs_mean) / obs_std
        assert torch.allclose(xco2_norm, wrapper.obs_values, atol=1e-5), (
            f"Max error: {(xco2_norm - wrapper.obs_values).abs().max().item()}"
        )

    def test_with_targshift(self):
        """Roundtrip should also work when targshift_mean is active."""
        targshift_mean = torch.tensor(0.3).view(1, 1, 1, 1)
        wrapper, x_norm, x_phys, masking_config = _make_wrapper(use_prior=True, targshift_mean=targshift_mean)
        # x_norm was computed WITHOUT targshift. If targshift is active, the wrapper
        # adds it back, so we need to shift x_norm to simulate the targshift subtraction.
        x_shifted = x_norm - targshift_mean
        xco2_norm = wrapper.forward_model.forward(x_shifted)
        assert torch.allclose(xco2_norm, wrapper.obs_values, atol=1e-5), (
            f"Max error: {(xco2_norm - wrapper.obs_values).abs().max().item()}"
        )


class TestComputeXCO2Fallback:
    def test_fallback_formula(self):
        """Fallback (no prior) should match sum(h_ak*x) + (mean/std)*(h_ak_sum-1)."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=False)
        xco2 = wrapper.forward_model.forward(x_norm)
        # Manual computation
        h_ak = wrapper.pressure_weights * wrapper.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        expected = (h_ak * x_norm).sum(dim=1, keepdim=True) + (
            masking_config["target_mean"] / masking_config["target_std"]
        ) * (h_ak_sum - 1.0)
        assert torch.allclose(xco2, expected, atol=1e-5)

    def test_fallback_with_targshift(self):
        """Fallback with targshift should add targshift_mean * h_ak_sum."""
        targshift_mean = torch.tensor(0.5).view(1, 1, 1, 1)
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=False, targshift_mean=targshift_mean)
        xco2 = wrapper.forward_model.forward(x_norm)
        # Manual computation
        h_ak = wrapper.pressure_weights * wrapper.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        expected = (
            (h_ak * x_norm).sum(dim=1, keepdim=True)
            + (masking_config["target_mean"] / masking_config["target_std"]) * (h_ak_sum - 1.0)
            + targshift_mean * h_ak_sum
        )
        assert torch.allclose(xco2, expected, atol=1e-5)


class TestGuidanceLevelWeighting:
    def test_surface_stronger_than_upper(self):
        """Guidance at surface level should be ~proportional to h_k*a_k (much stronger than upper)."""
        wrapper, x_norm, _, masking_config = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
        )
        # Make obs_values differ from compute_xco2(x) to create nonzero guidance
        wrapper.obs_values = wrapper.obs_values + 1.0

        t = torch.tensor(0.5)
        v = wrapper.forward(x_norm, t)
        # With MockSubmodel returning 0, v = -guidance_scale * mask_weight * guidance
        # guidance = h_ak * column_error (same column_error for all levels)
        # So |v| at level k should be proportional to h_k * a_k

        # h*a values: [0.4*0.95, 0.25*0.85, 0.15*0.60, 0.12*0.30, 0.08*0.10]
        #           = [0.38,     0.2125,    0.09,      0.036,     0.008]
        h_ak = wrapper.pressure_weights * wrapper.ak  # [B, C, Nlat, Nlon]

        # Check that guidance magnitude at each level is proportional to h_ak
        v_abs = v.abs().mean(dim=(0, 2, 3))  # [C] - average over batch, lat, lon
        h_ak_avg = h_ak.mean(dim=(0, 2, 3))  # [C]

        # Ratio should be approximately constant across levels
        ratios = v_abs / h_ak_avg
        ratio_spread = ratios.max() / ratios.min()
        assert ratio_spread < 1.5, f"Guidance not proportional to h_ak. Ratios: {ratios}"

        # Surface (level 0) should be much stronger than top (level 4)
        surface_to_top = v_abs[0] / v_abs[4]
        h_ak_surface_to_top = h_ak_avg[0] / h_ak_avg[4]
        assert surface_to_top > 5.0, f"Surface/top ratio only {surface_to_top:.1f}, expected >5"
        assert abs(surface_to_top - h_ak_surface_to_top) / h_ak_surface_to_top < 0.1, (
            f"Surface/top ratio {surface_to_top:.1f} doesn't match h_ak ratio {h_ak_surface_to_top:.1f}"
        )


class TestMaskingMethodsKept:
    @pytest.mark.parametrize(
        "method",
        ["simple", "interpolate", "total_column_average_simple", "total_column_average_mult"],
    )
    def test_method_runs(self, method):
        """Kept masking methods should execute without error."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=True, masking_method=method)
        t = torch.tensor(0.5)
        result = wrapper.apply_masking(x_norm, t)
        assert result.shape == x_norm.shape
        assert not torch.isnan(result).any()


class TestMaskingMethodsRemoved:
    @pytest.mark.parametrize(
        "method",
        [
            "preserve_global_mean",
            "preserve_global_mean_and_var",
            "total_column_average_test",
            "total_column_average_test_basic",
            "total_column_average_add",
            "total_column_average_simple_unitary",
            "total_column_average_simple_diag",
            "total_column_average_simple_invdiag",
        ],
    )
    def test_method_raises(self, method):
        """Removed masking methods should raise ValueError via apply_masking."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=True, masking_method=method)
        t = torch.tensor(0.5)
        with pytest.raises(ValueError, match="Unknown masking method"):
            wrapper.apply_masking(x_norm, t)


class TestGaussianSmooth2D:
    def test_constant_field_unchanged(self):
        """Constant field should be unchanged by smoothing."""
        field = torch.ones(2, 3, 16, 32) * 5.0
        result = _gaussian_smooth_2d(field, sigma=2.0)
        assert torch.allclose(result, field, atol=1e-5)

    def test_delta_produces_bump(self):
        """Delta function should produce a Gaussian-like bump."""
        field = torch.zeros(1, 1, 32, 64)
        field[0, 0, 16, 32] = 1.0
        result = _gaussian_smooth_2d(field, sigma=2.0)
        # Peak should be at original location but smaller
        assert result[0, 0, 16, 32] > 0
        assert result[0, 0, 16, 32] < 1.0
        # Neighbors should be positive
        assert result[0, 0, 15, 32] > 0
        assert result[0, 0, 16, 31] > 0

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape."""
        for shape in [(1, 1, 8, 16), (2, 5, 32, 64), (1, 3, 4, 4)]:
            field = torch.randn(*shape)
            result = _gaussian_smooth_2d(field, sigma=1.5)
            assert result.shape == field.shape

    def test_sigma_zero_returns_input(self):
        """sigma=0 should return the input unchanged."""
        field = torch.randn(2, 3, 16, 32)
        result = _gaussian_smooth_2d(field, sigma=0.0)
        assert torch.equal(result, field)

    def test_periodic_longitude_wrapping(self):
        """A spike at the right edge should wrap around to the left edge."""
        field = torch.zeros(1, 1, 16, 32)
        field[0, 0, 8, 31] = 1.0  # rightmost column
        result = _gaussian_smooth_2d(field, sigma=2.0)
        # Left edge should receive some weight from periodic wrapping
        assert result[0, 0, 8, 0] > 1e-6, "Periodic wrapping failed: left edge should see the right-edge spike"


class TestSigmaObs:
    def test_sigma_obs_scaling(self):
        """sigma_obs=2.0 should produce 4x weaker guidance than sigma_obs=1.0."""
        wrapper_s1, x_norm, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            sigma_obs=1.0,
        )
        wrapper_s1.obs_values = wrapper_s1.obs_values + 1.0

        wrapper_s2, _, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            sigma_obs=2.0,
        )
        wrapper_s2.obs_values = wrapper_s2.obs_values + 1.0

        t = torch.tensor(0.5)
        v1 = wrapper_s1.forward(x_norm, t)
        v2 = wrapper_s2.forward(x_norm, t)

        # v = -guidance_scale * mask_weight * guidance; guidance scales as 1/sigma_obs^2
        ratio = v1.abs().mean() / v2.abs().mean()
        assert abs(ratio - 4.0) < 0.5, f"Expected ~4x ratio, got {ratio:.2f}"

    def test_sigma_obs_1_backward_compat(self):
        """sigma_obs=1.0 should match behavior without sigma_obs parameter."""
        wrapper, x_norm, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            sigma_obs=1.0,
        )
        wrapper.obs_values = wrapper.obs_values + 1.0
        t = torch.tensor(0.5)
        v = wrapper.forward(x_norm, t)
        assert not torch.isnan(v).any()
        assert v.abs().mean() > 0  # non-trivial guidance


class TestSpatialSmoothingSigma:
    def test_smoothed_guidance_is_smoother(self):
        """Guidance with smoothing should have lower Laplacian norm (smoother)."""
        wrapper_no, x_norm, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            spatial_smoothing_sigma=0.0,
        )
        wrapper_no.obs_values = wrapper_no.obs_values + 1.0

        wrapper_sm, _, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            spatial_smoothing_sigma=2.0,
        )
        wrapper_sm.obs_values = wrapper_sm.obs_values + 1.0

        t = torch.tensor(0.5)
        v_no = wrapper_no.forward(x_norm, t)
        v_sm = wrapper_sm.forward(x_norm, t)

        # Compute Laplacian norm (sum of second derivatives) as proxy for roughness
        def laplacian_norm(v):
            d2_lat = v[:, :, 2:, :] - 2 * v[:, :, 1:-1, :] + v[:, :, :-2, :]
            d2_lon = v[:, :, :, 2:] - 2 * v[:, :, :, 1:-1] + v[:, :, :, :-2]
            return d2_lat.pow(2).mean() + d2_lon.pow(2).mean()

        lap_no = laplacian_norm(v_no)
        lap_sm = laplacian_norm(v_sm)
        assert lap_sm < lap_no, f"Smoothed Laplacian {lap_sm:.6f} should be < unsmoothed {lap_no:.6f}"

    def test_no_nan_with_both_params(self):
        """No NaN when both sigma_obs and smoothing are enabled."""
        wrapper, x_norm, _, _ = _make_wrapper(
            use_prior=False,
            conditioning_mode="guidance",
            guidance_scale=1.0,
            sigma_obs=0.5,
            spatial_smoothing_sigma=2.0,
        )
        wrapper.obs_values = wrapper.obs_values + 1.0
        t = torch.tensor(0.5)
        v = wrapper.forward(x_norm, t)
        assert not torch.isnan(v).any()


# ── XCO2ForwardModel tests ──────────────────────────────────────────────


class TestXCO2ForwardModelConstruction:
    def test_from_masking_config(self, sample_masking_config):
        """from_masking_config() creates a valid model."""
        fm = XCO2ForwardModel.from_masking_config(sample_masking_config)
        assert fm.has_priors
        assert fm.h_ak is not None
        assert fm.pressure_weights is not None
        assert fm.ak is not None

    def test_minimal_construction(self):
        """Minimal construction with just pw + ak."""
        pw = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1)
        ak = torch.tensor([1.0, 1.0]).view(1, 2, 1, 1)
        fm = XCO2ForwardModel(pressure_weights=pw, ak=ak)
        assert not fm.has_priors
        assert fm.h_ak is not None

    def test_uniform_weights_with_nlev(self):
        """When pressure_weights is None, uses 1/nlev if nlev provided."""
        ak = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
        fm = XCO2ForwardModel(pressure_weights=None, ak=ak, nlev=3)
        expected_h_ak = (1.0 / 3.0) * ak
        assert torch.allclose(fm.h_ak, expected_h_ak)


class TestXCO2ForwardModelParityWithWrapper:
    """forward() matches wrapper.forward_model.forward() exactly."""

    def test_parity_with_prior(self):
        """With-prior path matches wrapper."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=True)
        fm = XCO2ForwardModel.from_masking_config(masking_config)
        expected = wrapper.forward_model.forward(x_norm)
        actual = fm.forward(x_norm)
        assert torch.allclose(actual, expected, atol=1e-7), f"Max error: {(actual - expected).abs().max().item()}"

    def test_parity_fallback(self):
        """Fallback (no prior) path matches wrapper."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=False)
        fm = XCO2ForwardModel.from_masking_config(masking_config)
        expected = wrapper.forward_model.forward(x_norm)
        actual = fm.forward(x_norm)
        assert torch.allclose(actual, expected, atol=1e-7), f"Max error: {(actual - expected).abs().max().item()}"

    def test_parity_with_targshift(self):
        """With-prior + targshift path matches wrapper."""
        targshift_mean = torch.tensor(0.3).view(1, 1, 1, 1)
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=True, targshift_mean=targshift_mean)
        fm = XCO2ForwardModel.from_masking_config(masking_config)
        x_shifted = x_norm - targshift_mean
        expected = wrapper.forward_model.forward(x_shifted)
        actual = fm.forward(x_shifted)
        assert torch.allclose(actual, expected, atol=1e-7)


class TestXCO2ForwardModelParityWithFlowDPS:
    """FlowDPSSampler.forward_model.forward() matches standalone XCO2ForwardModel."""

    def test_parity_with_prior(self, sample_masking_config):
        from neural_transport.inference.samplers import FlowDPSSampler

        velocity_model = MockSubmodel(nlev=5)
        sampler = FlowDPSSampler(velocity_model, sample_masking_config)
        fm = XCO2ForwardModel.from_masking_config(sample_masking_config)

        torch.manual_seed(99)
        x = torch.randn(2, 5, 4, 8)
        expected = sampler.forward_model.forward(x)
        actual = fm.forward(x)
        assert torch.allclose(actual, expected, atol=1e-7), f"Max error: {(actual - expected).abs().max().item()}"

    def test_parity_fallback(self):
        from neural_transport.inference.samplers import FlowDPSSampler

        B, nlev, nlat, nlon = 2, 5, 4, 8
        pw = torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08]).view(1, nlev, 1, 1).expand(B, nlev, nlat, nlon)
        ak = torch.tensor([0.95, 0.85, 0.60, 0.30, 0.10]).view(1, nlev, 1, 1).expand(B, nlev, nlat, nlon)
        config = {
            "obs_mask": torch.ones(B, 1, nlat, nlon, dtype=torch.bool),
            "obs_values": torch.zeros(B, 1, nlat, nlon),
            "obs_mean": None,
            "obs_std": None,
            "target_mean": torch.tensor(400.0).view(1, 1, 1, 1),
            "target_std": torch.tensor(5.0).view(1, 1, 1, 1),
            "ak": ak,
            "pressure_weights": pw,
            "xco2_prior": None,
            "co2_profile_prior": None,
            "targshift_mean": None,
        }
        velocity_model = MockSubmodel(nlev=5)
        sampler = FlowDPSSampler(velocity_model, config)
        fm = XCO2ForwardModel.from_masking_config(config)

        torch.manual_seed(99)
        x = torch.randn(B, nlev, nlat, nlon)
        expected = sampler.forward_model.forward(x)
        actual = fm.forward(x)
        assert torch.allclose(actual, expected, atol=1e-7)


class TestXCO2ForwardModelLinearity:
    def test_linearity_fallback(self):
        """H(a*x1 + b*x2) == a*H(x1) + b*H(x2) for fallback path (no priors, no targshift)."""
        pw = torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08]).view(1, 5, 1, 1)
        ak = torch.tensor([0.95, 0.85, 0.60, 0.30, 0.10]).view(1, 5, 1, 1)
        fm = XCO2ForwardModel(
            pressure_weights=pw,
            ak=ak,
            target_mean=torch.zeros(1, 1, 1, 1),
            target_std=torch.ones(1, 1, 1, 1),
        )

        torch.manual_seed(42)
        x1 = torch.randn(2, 5, 4, 8)
        x2 = torch.randn(2, 5, 4, 8)
        a, b = 0.7, 0.3

        lhs = fm.forward(a * x1 + b * x2)
        rhs = a * fm.forward(x1) + b * fm.forward(x2)
        assert torch.allclose(lhs, rhs, atol=1e-6), f"Max error: {(lhs - rhs).abs().max().item()}"

    def test_affine_linearity_with_prior(self):
        """With-prior path is affine: H(a*x1+(1-a)*x2) == a*H(x1)+(1-a)*H(x2)."""
        wrapper, _, _, masking_config = _make_wrapper(use_prior=True)
        fm = XCO2ForwardModel.from_masking_config(masking_config)

        torch.manual_seed(42)
        x1 = torch.randn(2, 5, 4, 4)
        x2 = torch.randn(2, 5, 4, 4)
        a = 0.6

        lhs = fm.forward(a * x1 + (1 - a) * x2)
        rhs = a * fm.forward(x1) + (1 - a) * fm.forward(x2)
        assert torch.allclose(lhs, rhs, atol=1e-5), f"Max error: {(lhs - rhs).abs().max().item()}"


class TestXCO2ForwardModelProjection:
    def test_projection_reduces_error(self):
        """|y - H(project(x))| < |y - H(x)| at observed locations."""
        wrapper, x_norm, x_phys, masking_config = _make_wrapper(use_prior=True)
        fm = XCO2ForwardModel.from_masking_config(masking_config)

        # Add noise to create a gap between H(x) and obs
        x_noisy = x_norm + 0.5 * torch.randn_like(x_norm)
        y = wrapper.obs_values
        mask = wrapper.obs_mask

        error_before = (y - fm.forward(x_noisy)).abs()
        x_proj = fm.project(x_noisy, y, mask, sigma=0.1)
        error_after = (y - fm.forward(x_proj)).abs()

        # Error should decrease at observed locations
        obs_error_before = error_before[mask].mean()
        obs_error_after = error_after[mask].mean()
        assert obs_error_after < obs_error_before, (
            f"Projection didn't reduce error: {obs_error_after:.6f} >= {obs_error_before:.6f}"
        )

    def test_projection_idempotence(self):
        """project(project(x)) ≈ project(x)."""
        wrapper, x_norm, _, masking_config = _make_wrapper(use_prior=True)
        fm = XCO2ForwardModel.from_masking_config(masking_config)

        y = wrapper.obs_values
        mask = wrapper.obs_mask

        x_proj1 = fm.project(x_norm, y, mask, sigma=0.01)
        x_proj2 = fm.project(x_proj1, y, mask, sigma=0.01)
        assert torch.allclose(x_proj1, x_proj2, atol=1e-4), f"Max diff: {(x_proj1 - x_proj2).abs().max().item()}"


class TestXCO2ForwardModelNumPyParity:
    def test_numpy_pytorch_parity(self):
        """forward_numpy(x_np) ≈ forward(x_torch).numpy() for simple case."""
        B, nlev, nlat, nlon = 1, 5, 4, 8
        pw = torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08]).view(1, nlev, 1, 1).expand(B, nlev, nlat, nlon)
        ak = torch.tensor([0.95, 0.85, 0.60, 0.30, 0.10]).view(1, nlev, 1, 1).expand(B, nlev, nlat, nlon)

        fm = XCO2ForwardModel(pressure_weights=pw, ak=ak)

        torch.manual_seed(42)
        x_torch = 400.0 + 2.0 * torch.randn(B, nlev, nlat, nlon)
        x_np = x_torch.numpy()

        # PyTorch forward (simple sum, no normalization — use forward_numpy equivalent)
        result_np = fm.forward_numpy(x_np, levels_axis=1)

        # Manual PyTorch computation for comparison
        result_torch = (pw * ak * x_torch).sum(dim=1).numpy()

        np.testing.assert_allclose(result_np, result_torch, atol=1e-5)

    def test_forward_numpy_matches_metrics(self):
        """forward_numpy matches metrics.compute_xco2_column for levels-last data."""
        from neural_transport.inference.metrics import compute_xco2_column

        nlat, nlon, nlev = 4, 8, 5
        pw_np = np.array([0.4, 0.25, 0.15, 0.12, 0.08]).reshape(1, 1, nlev)
        ak_np = np.array([0.95, 0.85, 0.60, 0.30, 0.10]).reshape(1, 1, nlev)

        np.random.seed(42)
        x_np = 400.0 + 2.0 * np.random.randn(nlat, nlon, nlev)

        expected = compute_xco2_column(x_np, pw_np, ak_np)

        # forward_numpy with levels_axis=-1
        pw_torch = torch.tensor(pw_np.reshape(1, 1, nlev))
        ak_torch = torch.tensor(ak_np.reshape(1, 1, nlev))
        fm = XCO2ForwardModel(pressure_weights=pw_torch, ak=ak_torch)
        actual = fm.forward_numpy(x_np, levels_axis=-1)

        np.testing.assert_allclose(actual, expected, atol=1e-10)


# ── P1: interpolate-then-apply (MIP-correct) forward operator ────────────

# CarbonTracker l10 model pressure levels [hPa] (surface -> top of atmosphere).
P_MODEL_L10 = torch.tensor([1013.0, 1005.0, 995.0, 971.0, 943.0, 843.0, 642.0, 441.0, 243.0, 73.0])

OCO2_ASSIM_ZARR = Path("/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/oco2_assimilate.zarr")


class TestBuildInterpMatrix:
    def test_rows_sum_to_one(self):
        """Every interpolation row is a convex combination (sums to 1)."""
        p_src = torch.tensor([1000.0, 800.0, 500.0, 200.0, 50.0])
        p_dst = torch.linspace(1050.0, 30.0, 20)  # includes out-of-range ends
        W = build_interp_matrix(p_src, p_dst)
        assert torch.allclose(W.sum(-1), torch.ones(20), atol=1e-6)
        assert (W >= -1e-7).all(), "interpolation weights must be non-negative"

    def test_reproduces_linear_in_logp(self):
        """A profile linear in log-pressure is interpolated exactly."""
        p_src = torch.tensor([1000.0, 700.0, 400.0, 100.0])
        p_dst = torch.tensor([900.0, 550.0, 250.0])
        # y = c0 + c1 * log(p)
        y_src = 3.0 + 2.0 * torch.log(p_src)
        y_dst_true = 3.0 + 2.0 * torch.log(p_dst)
        W = build_interp_matrix(p_src, p_dst, log_pressure=True)
        y_dst = W @ y_src
        assert torch.allclose(y_dst, y_dst_true, atol=1e-5), f"{y_dst} vs {y_dst_true}"

    def test_constant_extrapolation(self):
        """Destination levels outside the source range clamp to the nearest level."""
        p_src = torch.tensor([900.0, 500.0, 100.0])
        p_dst = torch.tensor([1050.0, 30.0])  # below-surface and above-top
        y_src = torch.tensor([10.0, 5.0, 1.0])
        W = build_interp_matrix(p_src, p_dst)
        y_dst = W @ y_src
        assert torch.allclose(y_dst, torch.tensor([10.0, 1.0]), atol=1e-5)

    def test_identity_when_grids_match(self):
        """dst == src yields the identity matrix."""
        p = torch.tensor([1000.0, 600.0, 300.0, 80.0])
        W = build_interp_matrix(p, p)
        assert torch.allclose(W, torch.eye(4), atol=1e-5)

    def test_batched_shapes(self):
        """Batched per-gridcell destination pressures produce [...,D,S]."""
        B, Nlat, Nlon, S, D = 2, 3, 4, 10, 20
        p_src = P_MODEL_L10.view(1, 1, 1, S).expand(B, Nlat, Nlon, S)
        p_dst = torch.linspace(1010.0, 40.0, D).view(1, 1, 1, D).expand(B, Nlat, Nlon, D).contiguous()
        W = build_interp_matrix(p_src, p_dst)
        assert W.shape == (B, Nlat, Nlon, D, S)
        assert torch.allclose(W.sum(-1), torch.ones(B, Nlat, Nlon, D), atol=1e-5)


def _make_retrieval_operator(B=2, Cret=20, Nlat=3, Nlon=4, seed=0):
    """Build a from_retrieval_levels operator + the raw retrieval fields."""
    torch.manual_seed(seed)
    sig = torch.linspace(1e-4, 1.0, Cret)
    psurf = 980.0 + 60.0 * torch.rand(B, Nlat, Nlon)
    p_ret = sig.view(1, Cret, 1, 1) * psurf.unsqueeze(1)  # [B, Cret, Nlat, Nlon], top->surf
    h = torch.softmax(torch.randn(B, Cret, Nlat, Nlon), dim=1)  # sums to 1 over levels
    a = 0.2 + 0.85 * torch.rand(B, Cret, Nlat, Nlon)
    xa = 395.0 + 10.0 * torch.rand(B, Cret, Nlat, Nlon)
    xco2_prior = (h * xa).sum(1, keepdim=True)  # consistent prior
    tm = torch.tensor(400.0).view(1, 1, 1, 1)
    ts = torch.tensor(5.0).view(1, 1, 1, 1)
    fm = XCO2ForwardModel.from_retrieval_levels(
        p_model=P_MODEL_L10,
        p_ret=p_ret,
        pressure_weights=h,
        ak=a,
        co2_profile_prior=xa,
        xco2_prior=xco2_prior,
        obs_mean=tm,
        obs_std=ts,
        target_mean=tm,
        target_std=ts,
    )
    return fm, dict(p_ret=p_ret, h=h, a=a, xa=xa, xco2_prior=xco2_prior, tm=tm, ts=ts)


class TestInterpolateThenApplyOperator:
    def test_affine_matches_explicit_formula(self):
        """Effective-kernel forward == explicit interpolate-then-apply formula."""
        fm, r = _make_retrieval_operator()
        B, Cmod, Nlat, Nlon = fm.h_ak.shape
        torch.manual_seed(1)
        x_phys = 398.0 + 6.0 * torch.rand(B, Cmod, Nlat, Nlon)
        x = (x_phys - r["tm"]) / r["ts"]
        out_phys = fm.forward(x) * r["ts"] + r["tm"]

        W = build_interp_matrix(
            P_MODEL_L10.view(1, 1, 1, Cmod).expand(B, Nlat, Nlon, Cmod),
            r["p_ret"].permute(0, 2, 3, 1),
        )
        x_interp = (W * x_phys.permute(0, 2, 3, 1).unsqueeze(-2)).sum(-1).permute(0, 3, 1, 2)
        ref = r["xco2_prior"] + (r["h"] * r["a"] * (x_interp - r["xa"])).sum(1, keepdim=True)
        assert torch.allclose(out_phys, ref, atol=1e-3), f"max|diff|={(out_phys - ref).abs().max().item()}"

    def test_apriori_reproduction(self):
        """REFERENCE GATE: feeding the a-priori profile reproduces xco2_apriori.

        When the model grid equals the retrieval grid (W = identity) and the
        model profile equals the a-priori, H(x) must equal xco2_prior exactly.
        """
        torch.manual_seed(2)
        B, C, Nlat, Nlon = 2, 20, 3, 4
        p_ret = torch.linspace(1e-4, 1.0, C).view(1, C, 1, 1) * (980.0 + 40.0 * torch.rand(B, Nlat, Nlon)).unsqueeze(1)
        h = torch.softmax(torch.randn(B, C, Nlat, Nlon), dim=1)
        a = 0.2 + 0.85 * torch.rand(B, C, Nlat, Nlon)
        xa = 395.0 + 10.0 * torch.rand(B, C, Nlat, Nlon)
        xco2_prior = (h * xa).sum(1, keepdim=True)
        tm = torch.tensor(400.0).view(1, 1, 1, 1)
        ts = torch.tensor(5.0).view(1, 1, 1, 1)
        # p_model == p_ret per gridcell -> identity interpolation.
        fm = XCO2ForwardModel.from_retrieval_levels(
            p_model=p_ret,
            p_ret=p_ret,
            pressure_weights=h,
            ak=a,
            co2_profile_prior=xa,
            xco2_prior=xco2_prior,
            obs_mean=tm,
            obs_std=ts,
            target_mean=tm,
            target_std=ts,
        )
        x = (xa - tm) / ts  # a-priori profile in normalized space
        out_phys = fm.forward(x) * ts + tm
        assert torch.allclose(out_phys, xco2_prior, atol=1e-3), (
            f"a-priori not reproduced: max|diff|={(out_phys - xco2_prior).abs().max().item()}"
        )

    def test_linearity(self):
        """The operator is affine: H(a*x1+(1-a)*x2) == a*H(x1)+(1-a)*H(x2)."""
        fm, r = _make_retrieval_operator()
        B, Cmod, Nlat, Nlon = fm.h_ak.shape
        torch.manual_seed(3)
        x1 = torch.randn(B, Cmod, Nlat, Nlon)
        x2 = torch.randn(B, Cmod, Nlat, Nlon)
        al = 0.6
        lhs = fm.forward(al * x1 + (1 - al) * x2)
        rhs = al * fm.forward(x1) + (1 - al) * fm.forward(x2)
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_adjoint_matches_autograd(self):
        """jacobian_transpose (phys-space) is consistent with autograd of forward."""
        fm, r = _make_retrieval_operator()
        B, Cmod, Nlat, Nlon = fm.h_ak.shape
        torch.manual_seed(4)
        x = torch.randn(B, Cmod, Nlat, Nlon, requires_grad=True)
        y = fm.forward(x)
        err = torch.randn_like(y)
        (y * err).sum().backward()
        # d forward/dx = h_ak * (target_std / obs_std); jacobian_transpose returns h_ak*err (phys).
        expected_grad = fm.h_ak * err * r["ts"] / r["ts"]
        assert torch.allclose(x.grad, expected_grad, atol=1e-5)
        # jacobian_transpose itself is the phys-space adjoint h_ak * err.
        assert torch.allclose(fm.jacobian_transpose(err), fm.h_ak * err, atol=1e-6)

    def test_project_reduces_error(self):
        """project() still reduces the observation residual with the new operator."""
        fm, r = _make_retrieval_operator()
        B, Cmod, Nlat, Nlon = fm.h_ak.shape
        torch.manual_seed(5)
        x = torch.randn(B, Cmod, Nlat, Nlon)
        y = fm.forward(x + 0.3 * torch.randn_like(x))  # a feasible target
        mask = torch.ones(B, 1, Nlat, Nlon, dtype=torch.bool)
        err_before = (y - fm.forward(x)).abs()[mask].mean()
        x_proj = fm.project(x, y, mask, sigma=0.05)
        err_after = (y - fm.forward(x_proj)).abs()[mask].mean()
        assert err_after < err_before


@pytest.mark.skipif(not OCO2_ASSIM_ZARR.exists(), reason="OCO-2 assimilate zarr not staged")
class TestOperatorDiscrepancyRealSoundings:
    """Quantify the down-aggregated vs interpolate-then-apply discrepancy on real soundings.

    Deliverable for P1: a number proving the operator choice matters.
    """

    @staticmethod
    def _load(n=2000, seed=0):
        import xarray as xr

        ds = xr.open_zarr(OCO2_ASSIM_ZARR)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(ds.sizes["sounding_id"], n, replace=False))
        sub = ds.isel(sounding_id=idx).compute()
        h = sub["pressure_weight"].values
        a = sub["xco2_averaging_kernel"].values
        xa = sub["co2_profile_apriori"].values
        xco2_ap = sub["xco2_apriori"].values
        p_ret = sub["sigma_levels"].values.T * sub["psurf"].values[:, None]
        return h, a, xa, xco2_ap, p_ret

    def test_apriori_reproduction_real(self):
        """H(a-priori) == xco2_apriori to round-off on real soundings."""
        h, a, xa, xco2_ap, _ = self._load(n=1000)
        recon = (h * xa).sum(1)
        # stored xco2_apriori is rounded to 2 decimals; agreement to that precision.
        assert np.nanmax(np.abs(recon - xco2_ap)) < 0.02

    def test_down_agg_vs_interp_discrepancy(self):
        """The two operators disagree by a decision-relevant amount (>0.5 ppm RMSE)."""
        from neural_transport.datasets.mip_oco2 import MIP_OCO2_LEVEL_AGG

        h, a, xa, xco2_ap, p_ret = self._load(n=3000)
        N = h.shape[0]
        p_model = P_MODEL_L10.numpy()
        rng = np.random.default_rng(7)
        # Realistic structured model profiles on l10 (boundary-layer enhancement + curvature).
        pn = p_model / p_model.max()
        amp = rng.uniform(2, 8, (N, 1))
        curv = rng.uniform(-4, 4, (N, 1))
        base = rng.uniform(395, 405, (N, 1))
        x_l10 = base + amp * pn[None] + curv * pn[None] ** 2  # [N, 10]

        # Down-aggregated l10 operator (legacy path).
        groups = MIP_OCO2_LEVEL_AGG["l10"]
        h_agg = np.zeros((N, 10))
        a_agg = np.zeros((N, 10))
        xa_agg = np.zeros((N, 10))
        for g, grp in enumerate(groups):
            hg = h[:, grp]
            hs = hg.sum(1)
            w = hg / hs[:, None]
            h_agg[:, g] = hs
            a_agg[:, g] = (a[:, grp] * w).sum(1)
            xa_agg[:, g] = (xa[:, grp] * w).sum(1)
        H_agg = xco2_ap + (h_agg * a_agg * (x_l10 - xa_agg)).sum(1)

        # Interpolate-then-apply operator via the production builder.
        fm = XCO2ForwardModel.from_retrieval_levels(
            p_model=torch.tensor(p_model, dtype=torch.float32),
            p_ret=torch.tensor(p_ret[:, :, None, None], dtype=torch.float32),
            pressure_weights=torch.tensor(h[:, :, None, None], dtype=torch.float32),
            ak=torch.tensor(a[:, :, None, None], dtype=torch.float32),
            co2_profile_prior=torch.tensor(xa[:, :, None, None], dtype=torch.float32),
            xco2_prior=torch.tensor(xco2_ap[:, None, None, None], dtype=torch.float32),
            obs_mean=torch.zeros(1, 1, 1, 1),
            obs_std=torch.ones(1, 1, 1, 1),
            target_mean=torch.zeros(1, 1, 1, 1),
            target_std=torch.ones(1, 1, 1, 1),
        )
        x_t = torch.tensor(x_l10[:, :, None, None], dtype=torch.float32)
        H_int = fm.forward(x_t).squeeze().numpy()

        d = H_agg - H_int
        rmse = float(np.sqrt(np.mean(d**2)))
        bias = float(np.mean(d))
        print(f"\n[P1] down-agg vs interp: bias={bias:.3f} ppm  RMSE={rmse:.3f} ppm  max|d|={np.abs(d).max():.3f} ppm")
        assert rmse > 0.5, f"Expected a decision-relevant discrepancy; got RMSE={rmse:.3f} ppm"
