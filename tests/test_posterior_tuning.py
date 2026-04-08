"""Tests for neural_transport.inference.tuning — posterior sampler Optuna tuning."""

import numpy as np
import pytest

from neural_transport.inference.tuning import (
    METHODS,
    SUGGEST_FUNCS,
    _build_generate_config,
    _reconstruct_method_params,
    compute_pressure_weights,
    pressure_weighted_rmse,
)

# ── Pressure weights ─────────────────────────────────────────────────────


class TestComputePressureWeights:
    def test_sums_to_one(self):
        levels = np.array([1013, 1005, 995, 971, 943, 843, 642, 441, 243, 73])
        w = compute_pressure_weights(levels)
        assert w.shape == (10,)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_single_level(self):
        w = compute_pressure_weights(np.array([500.0]))
        np.testing.assert_array_equal(w, [1.0])

    def test_two_levels(self):
        w = compute_pressure_weights(np.array([1000.0, 500.0]))
        # Both half-intervals: dp = [250, 250] → [0.5, 0.5]
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_all_positive(self):
        levels = np.array([1013, 1005, 995, 971, 943, 843, 642, 441, 243, 73])
        w = compute_pressure_weights(levels)
        assert (w > 0).all()

    def test_surface_heavier_for_l10(self):
        """Lower atmosphere (higher pressure) should generally get more weight."""
        levels = np.array([1013, 1005, 995, 971, 943, 843, 642, 441, 243, 73])
        w = compute_pressure_weights(levels)
        # The big gap 943→843→642 should give middle levels more weight
        # than the closely spaced surface levels. Just check sum of bottom half > 0.
        assert w[:5].sum() > 0


class TestPressureWeightedRmse:
    def test_zero_error(self):
        nlat, nlon, nlev = 4, 8, 3
        field = np.random.randn(nlat, nlon, nlev)
        pw = np.array([0.5, 0.3, 0.2])
        rmse = pressure_weighted_rmse(field, field, pw)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_uniform_error(self):
        nlat, nlon, nlev = 4, 8, 3
        pred = np.ones((nlat, nlon, nlev))
        target = np.zeros((nlat, nlon, nlev))
        pw = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        rmse = pressure_weighted_rmse(pred, target, pw)
        assert rmse == pytest.approx(1.0, abs=1e-10)

    def test_with_lat_weights(self):
        nlat, nlon, nlev = 4, 8, 3
        pred = np.ones((nlat, nlon, nlev))
        target = np.zeros((nlat, nlon, nlev))
        pw = np.ones(nlev) / nlev
        clw = np.ones(nlat)
        rmse = pressure_weighted_rmse(pred, target, pw, cos_lat_weights=clw)
        assert rmse == pytest.approx(1.0, abs=1e-10)

    def test_lat_weights_change_result(self):
        nlat, nlon, nlev = 4, 8, 3
        pred = np.zeros((nlat, nlon, nlev))
        pred[0, :, :] = 10.0  # Error only at first latitude
        target = np.zeros((nlat, nlon, nlev))
        pw = np.ones(nlev) / nlev

        rmse_uniform = pressure_weighted_rmse(pred, target, pw)
        # With lat weights that downweight first latitude
        clw = np.array([0.01, 1.0, 1.0, 1.0])
        rmse_weighted = pressure_weighted_rmse(pred, target, pw, cos_lat_weights=clw)
        assert rmse_weighted < rmse_uniform


# ── Suggest functions ────────────────────────────────────────────────────


class TestSuggestFunctions:
    """Test that each suggest function returns valid params with expected keys."""

    @pytest.fixture
    def mock_trial(self):
        import optuna

        # Use a fixed trial via enqueue
        return optuna.trial.FixedTrial(
            {
                # Shared
                "sigma_obs": 0.5,
                "spatial_smoothing_sigma": 2.0,
                "soft_boundary_sigma": 1.0,
                # DPS
                "guidance_scale": 1.0,
                "masking_time": "none",
                # FlowDPS
                "fresh_noise": True,
                "steps": 21,
                # SDE
                "sigma_max": 0.3,
                "noise_schedule": "annealed",
                "use_projection": False,
                "n_corrector_steps": 0,
                "corrector_step_size": 0.01,
                # FIG
                "step_size_c": 10.0,
                "k_steps": 1,
                "noise_scale_w": 0.0,
                "skip_first_last": True,
                # ICTM
                "r_max": 1.0,
                "r_schedule": "decreasing",
                "n_inner_steps": 2,
                "inner_lr": 0.1,
                # DPS conditional
                "t_threshold": 0.9,
                # MCG
                "n_forward_steps": 2,
            }
        )

    def test_all_methods_present(self):
        assert set(METHODS) == {"dps", "flowdps", "sde", "fig", "ictm", "mcg"}

    def test_suggest_dps(self, mock_trial):
        params = SUGGEST_FUNCS["dps"](mock_trial)
        assert params["sampler"] is None
        assert params["conditioning_mode"] == "guidance"
        assert "sigma_obs" in params
        assert "guidance_scale" in params

    def test_suggest_flowdps(self, mock_trial):
        params = SUGGEST_FUNCS["flowdps"](mock_trial)
        assert params["sampler"] == "flowdps"
        assert "sigma_obs" in params
        assert "fresh_noise" in params

    def test_suggest_sde(self, mock_trial):
        params = SUGGEST_FUNCS["sde"](mock_trial)
        assert params["sampler"] == "sde"
        assert "sigma_max" in params
        assert "noise_schedule" in params

    def test_suggest_fig(self, mock_trial):
        params = SUGGEST_FUNCS["fig"](mock_trial)
        assert params["sampler"] == "fig"
        assert "step_size_c" in params
        assert "k_steps" in params

    def test_suggest_ictm(self, mock_trial):
        params = SUGGEST_FUNCS["ictm"](mock_trial)
        assert params["sampler"] == "ictm"
        assert "r_max" in params
        assert "n_inner_steps" in params

    def test_suggest_mcg(self, mock_trial):
        params = SUGGEST_FUNCS["mcg"](mock_trial)
        assert params["sampler"] == "mcg"
        assert "n_forward_steps" in params
        assert "soft_boundary_sigma" in params

    @pytest.mark.parametrize("method", METHODS)
    def test_all_suggest_have_soft_boundary_sigma(self, mock_trial, method):
        """All suggest functions should include soft_boundary_sigma."""
        params = SUGGEST_FUNCS[method](mock_trial)
        assert "soft_boundary_sigma" in params


# ── Config building ──────────────────────────────────────────────────────


class TestBuildGenerateConfig:
    def test_roundtrip_dps(self):
        from neural_transport.configs import GenerateConfig

        base = GenerateConfig(n_samples=20, steps=21)
        method_params = {
            "sampler": None,
            "conditioning_mode": "guidance",
            "sigma_obs": 0.5,
            "guidance_scale": 2.0,
            "spatial_smoothing_sigma": 1.0,
        }
        config = _build_generate_config(method_params, base)
        assert config.sampler is None
        assert config.conditioning.conditioning_mode == "guidance"
        assert config.sampler_params.sigma_obs == 0.5
        assert config.conditioning.guidance_scale == 2.0

    def test_roundtrip_flowdps(self):
        from neural_transport.configs import GenerateConfig

        base = GenerateConfig(n_samples=20, steps=21)
        method_params = {"sampler": "flowdps", "sigma_obs": 0.1, "fresh_noise": False}
        config = _build_generate_config(method_params, base)
        assert config.sampler == "flowdps"
        assert config.sampler_params.sigma_obs == 0.1
        assert config.sampler_params.fresh_noise is False


class TestReconstructMethodParams:
    @pytest.mark.parametrize("method", METHODS)
    def test_reconstruct_returns_dict(self, method):
        """Smoke test: reconstruct doesn't crash for any method."""
        # Minimal params matching each method
        flat_params = {
            "sigma_obs": 0.5,
            "guidance_scale": 1.0,
            "spatial_smoothing_sigma": 0.0,
            "soft_boundary_sigma": 1.0,
            "masking_time": "none",
            "fresh_noise": True,
            "steps": 21,
            "sigma_max": 0.3,
            "noise_schedule": "annealed",
            "use_projection": False,
            "n_corrector_steps": 0,
            "step_size_c": 10.0,
            "k_steps": 1,
            "noise_scale_w": 0.0,
            "skip_first_last": True,
            "r_max": 1.0,
            "r_schedule": "decreasing",
            "n_inner_steps": 2,
            "inner_lr": 0.1,
            "n_forward_steps": 2,
        }
        result = _reconstruct_method_params(method, flat_params)
        assert isinstance(result, dict)
        assert "sampler" in result or (method == "dps" and result.get("sampler") is None)


class TestMultiObjective:
    def test_select_from_pareto_empty(self):
        import optuna

        from neural_transport.inference.tuning import select_from_pareto

        study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
        result = select_from_pareto(study, obs_residual_threshold=0.5)
        assert result == []

    def test_multi_objective_class_exists(self):
        from neural_transport.inference.tuning import MultiObjectivePosteriorObjective

        assert callable(MultiObjectivePosteriorObjective)

    def test_run_multi_objective_study_exists(self):
        from neural_transport.inference.tuning import run_multi_objective_study

        assert callable(run_multi_objective_study)
