"""Unit tests for MaskedVelocityWrapper: compute_xco2 and guidance fixes."""

import pytest
import torch
import torch.nn as nn

from neural_transport.models.flowmatching import MaskedVelocityWrapper


class MockSubmodel(nn.Module):
    """Submodel that returns zero velocity."""

    def __init__(self, nlev):
        super().__init__()
        self.nlev = nlev
        self.model = (
            self  # forward_guidance calls super().forward -> VelocityWrapper.forward -> self.submodel.model(x_in)
        )

    def forward(self, x):
        # Input: [B, C+1, Nlat, Nlon] (channels + time). Return zero velocity.
        return torch.zeros(x.shape[0], self.nlev, x.shape[2], x.shape[3], device=x.device)


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
    )

    return wrapper, x_norm, x_phys


class TestComputeXCO2WithPrior:
    def test_roundtrip(self):
        """XCO2 computed from x should match the obs_values we derived from x."""
        wrapper, x_norm, x_phys = _make_wrapper(use_prior=True)
        xco2_norm = wrapper.compute_xco2(x_norm)
        # obs_values were computed as (sum(pw*ak*x_phys) - obs_mean) / obs_std
        assert torch.allclose(xco2_norm, wrapper.obs_values, atol=1e-5), (
            f"Max error: {(xco2_norm - wrapper.obs_values).abs().max().item()}"
        )

    def test_with_targshift(self):
        """Roundtrip should also work when targshift_mean is active."""
        targshift_mean = torch.tensor(0.3).view(1, 1, 1, 1)
        wrapper, x_norm, x_phys = _make_wrapper(use_prior=True, targshift_mean=targshift_mean)
        # x_norm was computed WITHOUT targshift. If targshift is active, the wrapper
        # adds it back, so we need to shift x_norm to simulate the targshift subtraction.
        x_shifted = x_norm - targshift_mean
        xco2_norm = wrapper.compute_xco2(x_shifted)
        assert torch.allclose(xco2_norm, wrapper.obs_values, atol=1e-5), (
            f"Max error: {(xco2_norm - wrapper.obs_values).abs().max().item()}"
        )


class TestComputeXCO2Fallback:
    def test_fallback_formula(self):
        """Fallback (no prior) should match sum(h_ak*x) + (mean/std)*(h_ak_sum-1)."""
        wrapper, x_norm, _ = _make_wrapper(use_prior=False)
        xco2 = wrapper.compute_xco2(x_norm)
        # Manual computation
        h_ak = wrapper.pressure_weights * wrapper.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        expected = (h_ak * x_norm).sum(dim=1, keepdim=True) + (wrapper.target_mean / wrapper.target_std) * (
            h_ak_sum - 1.0
        )
        assert torch.allclose(xco2, expected, atol=1e-5)

    def test_fallback_with_targshift(self):
        """Fallback with targshift should add targshift_mean * h_ak_sum."""
        targshift_mean = torch.tensor(0.5).view(1, 1, 1, 1)
        wrapper, x_norm, _ = _make_wrapper(use_prior=False, targshift_mean=targshift_mean)
        xco2 = wrapper.compute_xco2(x_norm)
        # Manual computation
        h_ak = wrapper.pressure_weights * wrapper.ak
        h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
        expected = (
            (h_ak * x_norm).sum(dim=1, keepdim=True)
            + (wrapper.target_mean / wrapper.target_std) * (h_ak_sum - 1.0)
            + targshift_mean * h_ak_sum
        )
        assert torch.allclose(xco2, expected, atol=1e-5)


class TestGuidanceLevelWeighting:
    def test_surface_stronger_than_upper(self):
        """Guidance at surface level should be ~proportional to h_k*a_k (much stronger than upper)."""
        wrapper, x_norm, _ = _make_wrapper(
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
        wrapper, x_norm, _ = _make_wrapper(use_prior=True, masking_method=method)
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
        wrapper, x_norm, _ = _make_wrapper(use_prior=True, masking_method=method)
        t = torch.tensor(0.5)
        with pytest.raises(ValueError, match="Unknown masking method"):
            wrapper.apply_masking(x_norm, t)
