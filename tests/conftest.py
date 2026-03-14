"""Shared test fixtures and helpers for neural_transport tests."""

import pytest
import torch
import torch.nn as nn


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


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_velocity_model():
    """Zero-velocity MockSubmodel for testing samplers/wrappers."""
    return MockSubmodel(nlev=5)


@pytest.fixture(params=[(4, 8, 10), (32, 64, 10)], ids=["tiny-4x8", "realistic-32x64"])
def synthetic_co2_field(request):
    """Random CO2 field [B=2, nlev, nlat, nlon] with physically plausible range (~400 ppm)."""
    nlat, nlon, nlev = request.param
    torch.manual_seed(42)
    return 400.0 + 2.0 * torch.randn(2, nlev, nlat, nlon)


@pytest.fixture
def synthetic_pressure_weights():
    """Pressure layer weights for XCO2 computation. Shape [1, 5, 1, 1]."""
    return torch.tensor([0.4, 0.25, 0.15, 0.12, 0.08]).view(1, 5, 1, 1)


@pytest.fixture
def synthetic_averaging_kernel():
    """Averaging kernel (ak) for column observation forward model. Shape [1, 5, 1, 1]."""
    return torch.tensor([0.95, 0.85, 0.60, 0.30, 0.10]).view(1, 5, 1, 1)


@pytest.fixture
def sample_obs_mask():
    """Binary observation mask [B=2, 1, 4, 8] with ~30% coverage."""
    torch.manual_seed(123)
    return (torch.rand(2, 1, 4, 8) < 0.3).float()


@pytest.fixture
def sample_masking_config(synthetic_pressure_weights, synthetic_averaging_kernel, sample_obs_mask):
    """Full masking_config dict matching MaskedVelocityWrapper API."""
    B, nlev, nlat, nlon = 2, 5, 4, 8
    pw = synthetic_pressure_weights.expand(B, nlev, nlat, nlon)
    ak = synthetic_averaging_kernel.expand(B, nlev, nlat, nlon)

    torch.manual_seed(42)
    x_phys = 400.0 + 2.0 * torch.randn(B, nlev, nlat, nlon)
    xco2_phys = (pw * ak * x_phys).sum(dim=1, keepdim=True)

    target_mean = torch.tensor(400.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    target_std = torch.tensor(5.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    obs_mean = torch.tensor(400.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)
    obs_std = torch.tensor(5.0).view(1, 1, 1, 1).expand(B, 1, 1, 1)

    obs_values_norm = (xco2_phys - 400.0) / 5.0

    co2_profile_prior = 400.0 * torch.ones(B, nlev, nlat, nlon)
    xco2_prior = (pw * ak * co2_profile_prior).sum(dim=1, keepdim=True)

    return {
        "obs_mask": sample_obs_mask,
        "obs_values": obs_values_norm,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "ak": ak,
        "pressure_weights": pw,
        "targshift_mean": None,
        "time_grid": torch.linspace(0, 1, 11),
        "xco2_prior": xco2_prior,
        "co2_profile_prior": co2_profile_prior,
    }


@pytest.fixture
def sample_forward_model(sample_masking_config):
    """XCO2ForwardModel built from sample_masking_config."""
    from neural_transport.forward_model import XCO2ForwardModel

    return XCO2ForwardModel.from_masking_config(sample_masking_config)
