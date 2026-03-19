"""Tests for EvaluationSuite orchestrator and EvalResult container."""

import json

import numpy as np
import pandas as pd
import pytest

from neural_transport.configs import EvalConfig
from neural_transport.evaluation.suite import EvalResult, EvaluationSuite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(nlat=8, nlon=16, seed=42):
    """Return lat, lon, and cos-lat weights for a small grid."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    weights = np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones((1, nlon))
    return lat, lon, weights, rng


# ===========================================================================
# TestEvalResult
# ===========================================================================


class TestEvalResult:
    def test_to_flat_dict_pointwise_only(self):
        result = EvalResult(pointwise={"rmse": 1.0, "bias": 0.5})
        flat = result.to_flat_dict()
        assert flat == {"pointwise/rmse": 1.0, "pointwise/bias": 0.5}

    def test_to_flat_dict_all_sections(self):
        result = EvalResult(
            pointwise={"rmse": 1.0},
            ensemble={"crps_mean": 0.3},
            distributional={"energy_distance": 0.1},
        )
        flat = result.to_flat_dict()
        assert "pointwise/rmse" in flat
        assert "ensemble/crps_mean" in flat
        assert "distributional/energy_distance" in flat

    def test_to_json_roundtrip(self, tmp_path):
        result = EvalResult(
            pointwise={"rmse": 1.23, "bias": -0.5},
            ensemble={"crps_mean": 0.45},
            metadata={"experiment": "test"},
        )
        path = tmp_path / "result.json"
        result.to_json(path)
        loaded = EvalResult.from_json(path)
        assert loaded.pointwise["rmse"] == pytest.approx(1.23)
        assert loaded.pointwise["bias"] == pytest.approx(-0.5)
        assert loaded.ensemble["crps_mean"] == pytest.approx(0.45)
        assert loaded.metadata["experiment"] == "test"

    def test_from_json_missing_sections(self, tmp_path):
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"pointwise": {"rmse": 1.0}}))
        loaded = EvalResult.from_json(path)
        assert loaded.pointwise == {"rmse": 1.0}
        assert loaded.ensemble is None
        assert loaded.distributional is None

    def test_maps_excluded_from_json(self, tmp_path):
        result = EvalResult(
            pointwise={"rmse": 1.0},
            maps={"bias_map": np.zeros((4, 8))},
        )
        path = tmp_path / "result.json"
        result.to_json(path)
        data = json.loads(path.read_text())
        assert "maps" not in data


# ===========================================================================
# TestEvaluateDeterministic
# ===========================================================================


class TestEvaluateDeterministic:
    def setup_method(self):
        self.suite = EvaluationSuite(EvalConfig())

    def test_perfect_prediction(self):
        rng = np.random.default_rng(0)
        gt = rng.standard_normal((8, 16))
        result = self.suite.evaluate_deterministic(gt, gt)
        assert result.pointwise["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result.pointwise["r2"] == pytest.approx(1.0, abs=1e-10)
        assert result.pointwise["bias"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_eval_result(self):
        gt = np.ones((4, 8))
        result = self.suite.evaluate_deterministic(gt, gt)
        assert isinstance(result, EvalResult)

    def test_pointwise_keys(self):
        rng = np.random.default_rng(1)
        pred = rng.standard_normal((4, 8))
        gt = rng.standard_normal((4, 8))
        result = self.suite.evaluate_deterministic(pred, gt)
        expected_keys = {"rmse", "mae", "bias", "r2", "nse", "rel_mean", "rel_std"}
        assert set(result.pointwise.keys()) == expected_keys

    def test_ensemble_is_none(self):
        gt = np.ones((4, 8))
        result = self.suite.evaluate_deterministic(gt, gt)
        assert result.ensemble is None

    def test_weighted_differs_from_unweighted(self):
        rng = np.random.default_rng(2)
        pred = rng.standard_normal((8, 16))
        gt = rng.standard_normal((8, 16))
        lat = np.linspace(-90, 90, 8)
        weights = np.cos(np.deg2rad(lat))[:, np.newaxis] * np.ones((1, 16))

        unweighted = self.suite.evaluate_deterministic(pred, gt)
        weighted = self.suite.evaluate_deterministic(pred, gt, lat_weights=weights)
        assert unweighted.pointwise["rmse"] != pytest.approx(weighted.pointwise["rmse"], abs=1e-6)


# ===========================================================================
# TestEvaluateEnsemble
# ===========================================================================


class TestEvaluateEnsemble:
    def setup_method(self):
        self.suite = EvaluationSuite(EvalConfig())
        self.rng = np.random.default_rng(42)
        self.nlat, self.nlon = 8, 16
        self.n_samples = 10
        self.gt = self.rng.standard_normal((self.nlat, self.nlon))
        self.samples = self.gt[np.newaxis, :, :] + 0.1 * self.rng.standard_normal(
            (self.n_samples, self.nlat, self.nlon)
        )

    def test_returns_eval_result(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert isinstance(result, EvalResult)

    def test_pointwise_populated(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "rmse" in result.pointwise
        assert "bias" in result.pointwise

    def test_ensemble_populated(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert result.ensemble is not None
        assert "crps_mean" in result.ensemble
        assert "spread_skill_ratio" in result.ensemble

    def test_maps_populated(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "bias_map" in result.maps
        assert "rmse_map" in result.maps
        assert "mean_map" in result.maps
        assert "spread_map" in result.maps

    def test_diagnostics_has_rank_histogram(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "rank_histogram" in result.diagnostics
        assert isinstance(result.diagnostics["rank_histogram"], np.ndarray)

    def test_diagnostics_has_calibration(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "calibration" in result.diagnostics
        cal = result.diagnostics["calibration"]
        assert "nominal" in cal
        assert "observed" in cal
        assert "calibration_error" in cal

    def test_crps_in_ensemble(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "crps_mean" in result.ensemble

    def test_spread_skill_in_ensemble(self):
        result = self.suite.evaluate_ensemble(self.samples, self.gt)
        assert "spread_skill_ratio" in result.ensemble


# ===========================================================================
# TestEvaluateDistributional
# ===========================================================================


class TestEvaluateDistributional:
    def setup_method(self):
        self.suite = EvaluationSuite(EvalConfig())
        self.rng = np.random.default_rng(123)
        self.nlat, self.nlon = 8, 16
        self.lat = np.linspace(-90, 90, self.nlat)
        self.lon = np.linspace(0, 360, self.nlon, endpoint=False)
        self.gt_pool = self.rng.standard_normal((5, self.nlat, self.nlon))
        self.gen_pool = self.rng.standard_normal((5, self.nlat, self.nlon))

    def test_returns_eval_result(self):
        result = self.suite.evaluate_distributional(self.gt_pool, self.gen_pool, self.lat, self.lon)
        assert isinstance(result, EvalResult)

    def test_distributional_populated(self):
        result = self.suite.evaluate_distributional(self.gt_pool, self.gen_pool, self.lat, self.lon)
        assert result.distributional is not None
        assert "energy_distance" in result.distributional
        assert "mmd_rbf" in result.distributional

    def test_pointwise_is_empty(self):
        result = self.suite.evaluate_distributional(self.gt_pool, self.gen_pool, self.lat, self.lon)
        assert result.pointwise == {}

    def test_energy_distance_key(self):
        result = self.suite.evaluate_distributional(self.gt_pool, self.gen_pool, self.lat, self.lon)
        assert isinstance(result.distributional["energy_distance"], float)
        assert result.distributional["energy_distance"] >= 0


# ===========================================================================
# TestToDataframe
# ===========================================================================


class TestToDataframe:
    def setup_method(self):
        self.suite = EvaluationSuite(EvalConfig())

    def test_single_row(self):
        result = EvalResult(pointwise={"rmse": 1.0, "bias": 0.5})
        df = self.suite.to_dataframe(result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_columns_match_flat_dict(self):
        result = EvalResult(
            pointwise={"rmse": 1.0},
            ensemble={"crps_mean": 0.3},
        )
        df = self.suite.to_dataframe(result)
        flat = result.to_flat_dict()
        assert set(df.columns) == set(flat.keys())
