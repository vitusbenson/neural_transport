"""Tests for the evaluation metrics package.

Covers pointwise (NumPy + xarray), ensemble, and backward-compatibility.
"""

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Pointwise NumPy metrics
# ---------------------------------------------------------------------------


class TestPointwiseNumPy:
    """Tests for evaluation.pointwise NumPy functions."""

    def test_rmse_zeros(self):
        from neural_transport.evaluation.pointwise import rmse_np

        pred = np.zeros((4, 4))
        targ = np.zeros((4, 4))
        assert rmse_np(pred, targ) == 0.0

    def test_rmse_known_value(self):
        from neural_transport.evaluation.pointwise import rmse_np

        pred = np.array([1.0, 2.0, 3.0])
        targ = np.array([1.5, 2.5, 3.5])
        expected = np.sqrt(np.mean((pred - targ) ** 2))
        assert rmse_np(pred, targ) == pytest.approx(expected)

    def test_rmse_weighted_differs(self):
        from neural_transport.evaluation.pointwise import rmse_np

        pred = np.array([1.0, 2.0, 3.0, 4.0])
        targ = np.array([0.0, 0.0, 0.0, 0.0])
        weights = np.array([10.0, 1.0, 1.0, 1.0])
        unweighted = rmse_np(pred, targ)
        weighted = rmse_np(pred, targ, weights=weights)
        assert unweighted != pytest.approx(weighted)

    def test_mae_known_value(self):
        from neural_transport.evaluation.pointwise import mae_np

        pred = np.array([1.0, 3.0, 5.0])
        targ = np.array([2.0, 2.0, 2.0])
        # |1-2| + |3-2| + |5-2| = 1+1+3 = 5, mean = 5/3
        assert mae_np(pred, targ) == pytest.approx(5.0 / 3.0)

    def test_bias_positive(self):
        from neural_transport.evaluation.pointwise import bias_np

        pred = np.array([3.0, 3.0, 3.0])
        targ = np.array([1.0, 1.0, 1.0])
        assert bias_np(pred, targ) > 0

    def test_bias_negative(self):
        from neural_transport.evaluation.pointwise import bias_np

        pred = np.array([1.0, 1.0, 1.0])
        targ = np.array([3.0, 3.0, 3.0])
        assert bias_np(pred, targ) < 0

    def test_r2_perfect(self):
        from neural_transport.evaluation.pointwise import r2_np

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_np(data, data) == pytest.approx(1.0)

    def test_r2_random_low(self):
        from neural_transport.evaluation.pointwise import r2_np

        rng = np.random.default_rng(42)
        targ = rng.normal(0, 1, 10000)
        pred = rng.normal(0, 1, 10000)
        # Independent random → R^2 should be well below 1
        assert r2_np(pred, targ) < 0.5

    def test_nse_equals_r2(self):
        from neural_transport.evaluation.pointwise import nse_np, r2_np

        rng = np.random.default_rng(0)
        pred = rng.normal(0, 1, 100)
        targ = pred + rng.normal(0, 0.1, 100)
        assert nse_np(pred, targ) == pytest.approx(r2_np(pred, targ))

    def test_rel_mean(self):
        from neural_transport.evaluation.pointwise import rel_mean_np

        pred = np.array([2.0, 4.0, 6.0])
        targ = np.array([1.0, 2.0, 3.0])
        assert rel_mean_np(pred, targ) == pytest.approx(2.0)

    def test_rel_std(self):
        from neural_transport.evaluation.pointwise import rel_std_np

        pred = np.array([2.0, 4.0, 6.0])
        targ = np.array([1.0, 2.0, 3.0])
        # std(pred)/std(targ) = std([2,4,6])/std([1,2,3]) = 2.0
        assert rel_std_np(pred, targ) == pytest.approx(2.0)

    def test_metrics_np_dict(self):
        from neural_transport.evaluation.pointwise import METRICS_NP

        assert set(METRICS_NP.keys()) == {"rmse", "mae", "bias", "r2", "nse", "rel_mean", "rel_std"}


# ---------------------------------------------------------------------------
# Pointwise xarray metrics
# ---------------------------------------------------------------------------


class TestPointwiseXarray:
    """Tests for evaluation.pointwise xarray functions."""

    @pytest.fixture
    def xr_data(self):
        rng = np.random.default_rng(123)
        lat = np.linspace(-90, 90, 8)
        lon = np.linspace(0, 360, 16)
        pred_vals = rng.normal(400, 5, (8, 16))
        targ_vals = pred_vals + rng.normal(0, 1, (8, 16))
        pred = xr.DataArray(pred_vals, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        targ = xr.DataArray(targ_vals, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        weights = xr.DataArray(np.cos(np.deg2rad(lat)), dims=["lat"], coords={"lat": lat})
        _, weights = xr.broadcast(pred, weights)
        return pred, targ, weights

    def test_rmse_xr_scalar(self, xr_data):
        from neural_transport.evaluation.pointwise import rmse_xr

        pred, targ, weights = xr_data
        result = rmse_xr(pred, targ, weights)
        assert float(result) > 0

    def test_bias_xr_sign(self, xr_data):
        from neural_transport.evaluation.pointwise import bias_xr

        pred, targ, weights = xr_data
        result = bias_xr(pred, targ, weights)
        # Just check it returns a valid float
        assert np.isfinite(float(result))

    def test_r2_xr_returns_xarray(self, xr_data):
        from neural_transport.evaluation.pointwise import r2_xr

        pred, targ, weights = xr_data
        result = r2_xr(pred, targ, weights)
        assert isinstance(result, xr.DataArray)

    def test_nse_xr_returns_xarray(self, xr_data):
        from neural_transport.evaluation.pointwise import nse_xr

        pred, targ, weights = xr_data
        result = nse_xr(pred, targ, weights)
        assert isinstance(result, xr.DataArray)

    def test_metrics_xr_dict(self):
        from neural_transport.evaluation.pointwise import METRICS_XR

        assert set(METRICS_XR.keys()) == {"rmse", "mae", "bias", "r2", "nse", "rel_mean", "rel_std"}


# ---------------------------------------------------------------------------
# Error maps & scalars
# ---------------------------------------------------------------------------


class TestErrorMapsScalars:
    """Tests for compute_error_maps and compute_error_scalars."""

    def test_perfect_ensemble(self):
        from neural_transport.evaluation.pointwise import compute_error_maps

        gt = xr.DataArray(np.ones((4, 8)), dims=["lat", "lon"])
        # Perfect ensemble: all samples equal gt
        samples = xr.DataArray(np.ones((5, 4, 8)), dims=["sample", "lat", "lon"])
        bias_map, rmse_map, mean_map, spread_map = compute_error_maps(samples, gt)
        np.testing.assert_allclose(bias_map, 0, atol=1e-12)
        np.testing.assert_allclose(rmse_map, 0, atol=1e-12)

    def test_biased_ensemble(self):
        from neural_transport.evaluation.pointwise import compute_error_maps

        gt = xr.DataArray(np.zeros((4, 8)), dims=["lat", "lon"])
        samples = xr.DataArray(np.ones((5, 4, 8)), dims=["sample", "lat", "lon"])
        bias_map, rmse_map, mean_map, spread_map = compute_error_maps(samples, gt)
        np.testing.assert_allclose(bias_map, 1.0)
        np.testing.assert_allclose(rmse_map, 1.0)

    def test_scalars_weighted_vs_unweighted(self):
        from neural_transport.evaluation.pointwise import compute_error_scalars

        bias_map = np.ones((4, 8))
        rmse_map = np.ones((4, 8)) * 2
        mean_map = np.ones((4, 8)) * 3
        spread_map = np.ones((4, 8)) * 0.5
        b1, r1, m1, s1 = compute_error_scalars(bias_map, rmse_map, mean_map, spread_map)
        weights = np.random.default_rng(0).random((4, 8))
        b2, r2, m2, s2 = compute_error_scalars(bias_map, rmse_map, mean_map, spread_map, weights=weights)
        # Uniform maps → weighted and unweighted should give same result
        assert b1 == pytest.approx(b2)


# ---------------------------------------------------------------------------
# Ensemble metrics
# ---------------------------------------------------------------------------


class TestEnsembleMetrics:
    """Tests for ensemble metrics (CRPS, spread-skill, calibration, rank histogram)."""

    def test_crps_known_value(self):
        from neural_transport.evaluation.ensemble import crps

        rng = np.random.default_rng(42)
        gt = np.zeros((4, 8))
        # Ensemble samples close to ground truth
        samples = rng.normal(0, 0.01, (10, 4, 8))
        crps_map, crps_mean = crps(samples, gt)
        assert crps_map.shape == (4, 8)
        assert crps_mean >= 0
        assert crps_mean < 0.1  # should be small for near-perfect ensemble

    def test_crps_ensemble_3d(self):
        from neural_transport.evaluation.ensemble import crps_ensemble

        rng = np.random.default_rng(42)
        gt = np.zeros((4, 8, 3))
        samples = rng.normal(0, 0.01, (5, 4, 8, 3))
        crps_map, crps_mean = crps_ensemble(samples, gt)
        assert crps_map.shape == (4, 8, 3)
        assert crps_mean >= 0

    def test_crps_ensemble_2d(self):
        from neural_transport.evaluation.ensemble import crps_ensemble

        rng = np.random.default_rng(42)
        gt = np.zeros((4, 8))
        samples = rng.normal(0, 0.01, (5, 4, 8))
        crps_map, crps_mean = crps_ensemble(samples, gt)
        assert crps_map.shape == (4, 8)

    def test_spread_skill_ratio_basic(self):
        from neural_transport.evaluation.ensemble import spread_skill_ratio

        rng = np.random.default_rng(42)
        gt = rng.normal(100, 5, (4, 8, 3))
        # Ensemble centered on gt with known spread
        noise = rng.normal(0, 1.0, (200, 4, 8, 3))
        samples = gt[None, ...] + noise
        ratio = spread_skill_ratio(samples, gt)
        # With centered noise, spread ~ mean(|error of ensemble mean|)
        # should be a finite positive number
        assert ratio > 0
        assert np.isfinite(ratio)

    def test_spread_skill_overconfident(self):
        from neural_transport.evaluation.ensemble import spread_skill_ratio

        gt = np.ones((4, 8, 3)) * 10
        # All samples very tight but wrong → overconfident
        samples = np.zeros((20, 4, 8, 3)) + np.random.default_rng(0).normal(0, 0.001, (20, 4, 8, 3))
        ratio = spread_skill_ratio(samples, gt)
        assert ratio < 1.0

    def test_calibration_score_keys(self):
        from neural_transport.evaluation.ensemble import calibration_score

        rng = np.random.default_rng(42)
        gt = rng.normal(0, 1, (4, 8))
        samples = rng.normal(0, 1, (20, 4, 8))
        result = calibration_score(samples, gt)
        assert "nominal" in result
        assert "observed" in result
        assert "calibration_error" in result
        assert len(result["nominal"]) == 19
        assert len(result["observed"]) == 19

    def test_rank_histogram_shape(self):
        from neural_transport.evaluation.ensemble import rank_histogram

        rng = np.random.default_rng(42)
        n_samples = 10
        gt = rng.normal(0, 1, (4, 8))
        samples = rng.normal(0, 1, (n_samples, 4, 8))
        hist = rank_histogram(samples, gt)
        assert hist.shape == (n_samples + 1,)
        assert hist.sum() == pytest.approx(1.0)

    def test_rank_histogram_uniform_calibrated(self):
        from neural_transport.evaluation.ensemble import rank_histogram

        rng = np.random.default_rng(42)
        n_samples = 10
        n_points = 10000
        # All drawn from same distribution → uniform rank histogram
        all_data = rng.normal(0, 1, (n_samples + 1, n_points))
        gt = all_data[0]
        samples = all_data[1:]
        hist = rank_histogram(samples, gt)
        expected = 1.0 / (n_samples + 1)
        # Each bin should be close to expected
        assert np.all(np.abs(hist - expected) < 0.05)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify that old import paths still work via shims."""

    def test_inference_metrics_crps_ensemble(self):
        from neural_transport.inference.metrics import crps_ensemble

        assert callable(crps_ensemble)

    def test_inference_metrics_spread_skill(self):
        from neural_transport.inference.metrics import spread_skill_ratio

        assert callable(spread_skill_ratio)

    def test_inference_metrics_calibration_score(self):
        from neural_transport.inference.metrics import calibration_score

        assert callable(calibration_score)

    def test_inference_metrics_rank_histogram(self):
        from neural_transport.inference.metrics import rank_histogram

        assert callable(rank_histogram)

    def test_inference_distributional_energy_distance(self):
        from neural_transport.inference.distributional_metrics import energy_distance

        assert callable(energy_distance)

    def test_inference_distributional_compute(self):
        from neural_transport.inference.distributional_metrics import compute_distributional_metrics

        assert callable(compute_distributional_metrics)

    def test_tools_metrics_crps(self):
        from neural_transport.tools.metrics import crps

        assert callable(crps)

    def test_tools_metrics_compute_error_maps(self):
        from neural_transport.tools.metrics import compute_error_maps

        assert callable(compute_error_maps)

    def test_tools_metrics_compute_error_scalars(self):
        from neural_transport.tools.metrics import compute_error_scalars

        assert callable(compute_error_scalars)


# ---------------------------------------------------------------------------
# Conditioning diagnostic metrics (Phase 23c)
# ---------------------------------------------------------------------------


class TestGradientAtBoundary:
    """Tests for gradient_at_boundary metric."""

    def test_smooth_field_low_ratio(self):
        from neural_transport.inference.metrics import gradient_at_boundary

        # Smooth field + arbitrary mask → gradient ratio near 1
        nlat, nlon = 32, 64
        y, x = np.mgrid[0:nlat, 0:nlon]
        field = np.sin(2 * np.pi * x / nlon)  # smooth sine wave
        mask = np.zeros((nlat, nlon), dtype=bool)
        mask[8:24, 16:48] = True
        result = gradient_at_boundary(field, mask)
        assert np.isfinite(result["gradient_ratio"])
        # Smooth field should have ratio near 1 (no sharp boundary)
        assert result["gradient_ratio"] < 3.0

    def test_step_at_boundary_high_ratio(self):
        from neural_transport.inference.metrics import gradient_at_boundary

        # Field with a step exactly at the mask edge
        nlat, nlon = 32, 64
        mask = np.zeros((nlat, nlon), dtype=bool)
        mask[8:24, 16:48] = True
        field = mask.astype(float) * 10.0  # step function at mask edge
        result = gradient_at_boundary(field, mask)
        assert result["gradient_ratio"] > 2.0

    def test_empty_mask_returns_nan(self):
        from neural_transport.inference.metrics import gradient_at_boundary

        field = np.random.randn(16, 32)
        mask = np.zeros((16, 32), dtype=bool)
        result = gradient_at_boundary(field, mask)
        assert np.isnan(result["gradient_ratio"])


class TestXco2ObsResidual:
    """Tests for xco2_obs_residual metric."""

    def test_perfect_match_zero_residual(self):
        from neural_transport.inference.metrics import xco2_obs_residual

        nlat, nlon, nlev = 8, 16, 10
        field = np.random.randn(nlat, nlon, nlev)
        pw = np.ones(nlev) / nlev
        ak = np.ones(nlev)
        mask = np.ones((nlat, nlon), dtype=bool)
        # Ensemble mean == GT → zero residual
        assert xco2_obs_residual(field, field, mask, pw, ak) == pytest.approx(0.0)

    def test_different_fields_positive_residual(self):
        from neural_transport.inference.metrics import xco2_obs_residual

        nlat, nlon, nlev = 8, 16, 10
        pred = np.random.randn(nlat, nlon, nlev)
        gt = pred + 1.0  # offset
        pw = np.ones(nlev) / nlev
        ak = np.ones(nlev)
        mask = np.ones((nlat, nlon), dtype=bool)
        residual = xco2_obs_residual(pred, gt, mask, pw, ak)
        assert residual > 0

    def test_no_mask_returns_nan(self):
        from neural_transport.inference.metrics import xco2_obs_residual

        nlat, nlon, nlev = 8, 16, 10
        field = np.random.randn(nlat, nlon, nlev)
        pw = np.ones(nlev) / nlev
        ak = np.ones(nlev)
        assert np.isnan(xco2_obs_residual(field, field, None, pw, ak))
