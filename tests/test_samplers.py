"""Tests for sampler abstract base classes, ODE sampler, and all posterior samplers."""

import math

import pytest
import torch

from neural_transport.configs import KNOWN_SAMPLERS
from neural_transport.forward_model import XCO2ForwardModel
from neural_transport.inference.samplers import (
    SAMPLER_REGISTRY,
    BaseSampler,
    FIGSampler,
    FlowDPSSampler,
    ICTMSampler,
    ODESampler,
    PosteriorSampler,
    StochasticPosteriorSampler,
    create_sampler,
)

# ── Minimal concrete subclass for testing PosteriorSampler ──────────────


class MockPosteriorSampler(PosteriorSampler):
    """Concrete subclass using simple Euler integration for testing."""

    @torch.no_grad()
    def sample(self, x_init, time_grid, return_intermediates=False):
        x_t = x_init
        z = x_init.clone()
        intermediates = [x_t] if return_intermediates else None

        for i in range(len(time_grid) - 1):
            t_n = time_grid[i]
            t_next = time_grid[i + 1]

            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)
            # VelocityWrapper expects scalar t for view(1,1,1,1)
            v_theta = self.velocity_model(x_t, t_n)

            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)
            x_hat_proj = self._project_column(x_hat)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat_proj, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t


# ── ABC enforcement ─────────────────────────────────────────────────────


class TestABCEnforcement:
    @pytest.mark.quick
    def test_base_sampler_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseSampler()

    @pytest.mark.quick
    def test_posterior_sampler_cannot_instantiate(self):
        with pytest.raises(TypeError):
            PosteriorSampler(None, {})


# ── PosteriorSampler shared methods ────────────────────────────────────


class TestPosteriorSamplerSharedMethods:
    @pytest.fixture
    def sampler(self, mock_velocity_wrapper, sample_masking_config):
        return MockPosteriorSampler(
            velocity_model=mock_velocity_wrapper,
            masking_config=sample_masking_config,
        )

    @pytest.mark.quick
    def test_has_forward_model(self, sampler):
        assert isinstance(sampler.forward_model, XCO2ForwardModel)

    @pytest.mark.quick
    def test_has_velocity_model(self, sampler, mock_velocity_wrapper):
        assert sampler.velocity_model is mock_velocity_wrapper

    @pytest.mark.quick
    def test_obs_mask_is_bool(self, sampler):
        """PosteriorSampler.__init__ should cast obs_mask to bool."""
        assert sampler.obs_mask.dtype == torch.bool

    @pytest.mark.quick
    def test_tweedie_estimate_formula(self, sampler):
        B, C, H, W = 2, 5, 4, 8
        torch.manual_seed(0)
        x_t = torch.randn(B, C, H, W)
        v = torch.randn(B, C, H, W)
        t = 0.3 * torch.ones(B)

        x_hat = sampler._tweedie_estimate(x_t, t, v)
        expected = x_t + (1.0 - 0.3) * v
        assert torch.allclose(x_hat, expected, atol=1e-6)

    @pytest.mark.quick
    def test_renoise_formula(self, sampler):
        B, C, H, W = 2, 5, 4, 8
        torch.manual_seed(0)
        x_hat = torch.randn(B, C, H, W)
        z = torch.randn(B, C, H, W)
        t_next = 0.6 * torch.ones(B)

        result = sampler._renoise(x_hat, t_next, z)
        expected = (1.0 - 0.6) * z + 0.6 * x_hat
        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.quick
    def test_renoise_at_t1_gives_signal(self, sampler):
        B, C, H, W = 2, 5, 4, 8
        torch.manual_seed(0)
        x_hat = torch.randn(B, C, H, W)
        z = torch.randn(B, C, H, W)
        t_next = torch.ones(B)

        result = sampler._renoise(x_hat, t_next, z)
        assert torch.allclose(result, x_hat, atol=1e-6)

    @pytest.mark.quick
    def test_renoise_at_t0_gives_noise(self, sampler):
        B, C, H, W = 2, 5, 4, 8
        torch.manual_seed(0)
        x_hat = torch.randn(B, C, H, W)
        z = torch.randn(B, C, H, W)
        t_next = torch.zeros(B)

        result = sampler._renoise(x_hat, t_next, z)
        assert torch.allclose(result, z, atol=1e-6)

    @pytest.mark.quick
    def test_project_column_delegates(self, sampler):
        B, C, H, W = 2, 5, 4, 8
        torch.manual_seed(0)
        x_hat = torch.randn(B, C, H, W)

        result = sampler._project_column(x_hat)
        expected = sampler.forward_model.project(
            x_hat,
            sampler.obs_values,
            sampler.obs_mask,
            sampler.sigma_obs,
            sampler.spatial_smoothing_sigma,
        )
        assert torch.allclose(result, expected, atol=1e-6)


# ── ODESampler ──────────────────────────────────────────────────────────


class TestODESampler:
    @pytest.fixture
    def ode_sampler(self, mock_velocity_wrapper):
        return ODESampler(velocity_model=mock_velocity_wrapper, method="euler")

    @pytest.mark.quick
    def test_sample_returns_correct_shape(self, ode_sampler):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = ode_sampler.sample(x_init, time_grid)
        assert result.shape == (B, C, H, W)

    @pytest.mark.quick
    def test_sample_no_nan(self, ode_sampler):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = ode_sampler.sample(x_init, time_grid)
        assert torch.isfinite(result).all()

    @pytest.mark.quick
    def test_sample_with_intermediates_shape(self, ode_sampler):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = ode_sampler.sample(x_init, time_grid, return_intermediates=True)
        # T time points -> T entries in trajectory
        assert result.shape == (len(time_grid), B, C, H, W)


# ── Sampler interface (parametrized over all real samplers) ─────────────


class TestSamplerInterface:
    """Interface tests parametrized over all sampler types."""

    @pytest.fixture(params=["ode", "flowdps", "sde", "fig", "ictm"])
    def sampler_instance(self, request, mock_velocity_wrapper, sample_masking_config):
        if request.param == "ode":
            return ODESampler(velocity_model=mock_velocity_wrapper, method="euler")
        elif request.param == "flowdps":
            return FlowDPSSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
            )
        elif request.param == "sde":
            return StochasticPosteriorSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
            )
        elif request.param == "fig":
            return FIGSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
            )
        elif request.param == "ictm":
            return ICTMSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
            )

    @pytest.mark.quick
    def test_is_base_sampler(self, sampler_instance):
        assert isinstance(sampler_instance, BaseSampler)

    @pytest.mark.quick
    def test_sample_returns_tensor(self, sampler_instance):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = sampler_instance.sample(x_init, time_grid)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.quick
    def test_sample_correct_spatial_shape(self, sampler_instance):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = sampler_instance.sample(x_init, time_grid)
        assert result.shape == (B, C, H, W)

    @pytest.mark.quick
    def test_no_nan_in_output(self, sampler_instance):
        B, C, H, W = 2, 5, 4, 8
        x_init = torch.randn(B, C, H, W)
        time_grid = torch.linspace(0, 1, 6)

        result = sampler_instance.sample(x_init, time_grid)
        assert torch.isfinite(result).all()


# ── create_sampler factory ──────────────────────────────────────────────


class TestCreateSampler:
    @pytest.mark.quick
    @pytest.mark.parametrize(
        "name,expected_cls",
        [
            ("ode", ODESampler),
            ("flowdps", FlowDPSSampler),
            ("sde", StochasticPosteriorSampler),
            ("fig", FIGSampler),
            ("ictm", ICTMSampler),
        ],
    )
    def test_creates_each_type(self, name, expected_cls, mock_velocity_wrapper, sample_masking_config):
        if name == "ode":
            sampler = create_sampler(name, mock_velocity_wrapper)
        else:
            sampler = create_sampler(name, mock_velocity_wrapper, sample_masking_config)
        assert isinstance(sampler, expected_cls)

    @pytest.mark.quick
    def test_unknown_name_raises(self, mock_velocity_wrapper):
        with pytest.raises(KeyError, match="Unknown sampler"):
            create_sampler("nonexistent", mock_velocity_wrapper)

    @pytest.mark.quick
    def test_posterior_without_masking_config_raises(self, mock_velocity_wrapper):
        with pytest.raises(TypeError, match="requires masking_config"):
            create_sampler("flowdps", mock_velocity_wrapper)

    @pytest.mark.quick
    def test_registry_contains_known_samplers(self):
        """Registry keys should be a superset of KNOWN_SAMPLERS from configs."""
        assert KNOWN_SAMPLERS <= set(SAMPLER_REGISTRY.keys())


# ── ICTM schedule tests ────────────────────────────────────────────────


class TestICTMSchedule:
    @pytest.fixture
    def ictm(self, mock_velocity_wrapper, sample_masking_config):
        def _make(r_schedule="decreasing", r_max=1.0):
            return ICTMSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
                r_schedule=r_schedule,
                r_max=r_max,
            )

        return _make

    @pytest.mark.quick
    def test_decreasing(self, ictm):
        s = ictm("decreasing", r_max=2.0)
        assert s.r(0.0) == pytest.approx(2.0)
        assert s.r(0.5) == pytest.approx(1.0)
        assert s.r(1.0) == pytest.approx(1e-6)  # clamped

    @pytest.mark.quick
    def test_constant(self, ictm):
        s = ictm("constant", r_max=0.5)
        assert s.r(0.0) == pytest.approx(0.5)
        assert s.r(0.5) == pytest.approx(0.5)
        assert s.r(1.0) == pytest.approx(0.5)

    @pytest.mark.quick
    def test_increasing(self, ictm):
        s = ictm("increasing", r_max=2.0)
        assert s.r(0.0) == pytest.approx(1e-6)  # clamped
        assert s.r(0.5) == pytest.approx(1.0)
        assert s.r(1.0) == pytest.approx(2.0)

    @pytest.mark.quick
    def test_cosine(self, ictm):
        s = ictm("cosine", r_max=1.0)
        assert s.r(0.0) == pytest.approx(1.0)
        assert s.r(0.5) == pytest.approx(math.cos(math.pi * 0.5 / 2.0))
        assert s.r(1.0) == pytest.approx(1e-6)  # clamped


# ── SDE noise schedule tests ───────────────────────────────────────────


class TestSDENoiseSchedule:
    @pytest.fixture
    def sde(self, mock_velocity_wrapper, sample_masking_config):
        def _make(noise_schedule="annealed", sigma_max=1.0):
            return StochasticPosteriorSampler(
                velocity_model=mock_velocity_wrapper,
                masking_config=sample_masking_config,
                noise_schedule=noise_schedule,
                sigma_max=sigma_max,
            )

        return _make

    @pytest.mark.quick
    def test_annealed(self, sde):
        s = sde("annealed", sigma_max=2.0)
        assert s.sigma(0.0) == pytest.approx(2.0)
        assert s.sigma(0.5) == pytest.approx(1.0)
        assert s.sigma(1.0) == pytest.approx(0.0)

    @pytest.mark.quick
    def test_constant(self, sde):
        s = sde("constant", sigma_max=0.5)
        assert s.sigma(0.0) == pytest.approx(0.5)
        assert s.sigma(0.5) == pytest.approx(0.5)
        assert s.sigma(1.0) == pytest.approx(0.5)

    @pytest.mark.quick
    def test_cosine(self, sde):
        s = sde("cosine", sigma_max=1.0)
        assert s.sigma(0.0) == pytest.approx(1.0)
        assert s.sigma(0.5) == pytest.approx(math.cos(math.pi * 0.5 / 2.0))
        assert s.sigma(1.0) == pytest.approx(0.0, abs=1e-7)
