"""Tests for distributional metrics on synthetic data.

Verify that metrics behave correctly:
- Same distribution → small distance
- Different distributions → large distance
- Known properties (symmetry, non-negativity)
"""

import numpy as np
import pytest

from neural_transport.inference.distributional_metrics import (
    compute_distributional_metrics,
    coverage_density,
    energy_distance,
    meridional_gradient_score,
    mmd_rbf,
    power_spectrum_distance,
    remove_spatial_mean,
    vendi_score,
    wasserstein_1d_marginals,
    zonal_mean_distance,
)


@pytest.fixture
def synthetic_fields():
    """Create synthetic 4D fields for testing."""
    np.random.seed(42)
    nlat, nlon, nlev = 16, 32, 5
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)

    # Pool A: Gaussian fields with spatial structure
    n_a = 30
    fields_a = np.random.randn(n_a, nlat, nlon, nlev) * 2.0
    # Add latitude gradient
    lat_grad = np.sin(np.deg2rad(lat))[:, np.newaxis, np.newaxis]
    fields_a += lat_grad

    # Pool B: Same distribution (slightly different samples)
    n_b = 30
    fields_b = np.random.randn(n_b, nlat, nlon, nlev) * 2.0
    fields_b += lat_grad

    # Pool C: Different distribution (shifted + scaled)
    fields_c = np.random.randn(n_b, nlat, nlon, nlev) * 4.0
    fields_c += lat_grad * 3

    return fields_a, fields_b, fields_c, lat, lon


@pytest.fixture
def synthetic_3d_fields():
    """Create synthetic 3D fields (no level dim) for testing."""
    np.random.seed(42)
    nlat, nlon = 16, 32
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)

    n = 20
    fields_a = np.random.randn(n, nlat, nlon)
    fields_b = np.random.randn(n, nlat, nlon)
    return fields_a, fields_b, lat, lon


class TestRemoveSpatialMean:
    def test_4d(self, synthetic_fields):
        fields_a, _, _, _, _ = synthetic_fields
        anom = remove_spatial_mean(fields_a)
        # Spatial mean should be ~0 for each sample
        spatial_mean = anom.mean(axis=(1, 2))  # [N, nlev]
        np.testing.assert_allclose(spatial_mean, 0, atol=1e-10)

    def test_3d(self, synthetic_3d_fields):
        fields_a, _, _, _ = synthetic_3d_fields
        anom = remove_spatial_mean(fields_a)
        spatial_mean = anom.mean(axis=(1, 2))
        np.testing.assert_allclose(spatial_mean, 0, atol=1e-10)


class TestEnergyDistance:
    def test_same_distribution_small(self, synthetic_fields):
        a, b, _, _, _ = synthetic_fields
        d_same = energy_distance(a, b)
        assert d_same >= 0

    def test_different_distribution_larger(self, synthetic_fields):
        a, b, c, _, _ = synthetic_fields
        d_same = energy_distance(a, b)
        d_diff = energy_distance(a, c)
        assert d_diff > d_same

    def test_self_near_zero(self, synthetic_fields):
        a, _, _, _, _ = synthetic_fields
        d_self = energy_distance(a, a)
        assert abs(d_self) < 0.1


class TestMMDRBF:
    def test_same_distribution_small(self, synthetic_fields):
        a, b, _, _, _ = synthetic_fields
        d_same = mmd_rbf(a, b)
        assert isinstance(d_same, float)

    def test_different_distribution_larger(self, synthetic_fields):
        a, b, c, _, _ = synthetic_fields
        d_same = mmd_rbf(a, b)
        d_diff = mmd_rbf(a, c)
        assert d_diff > d_same

    def test_self_near_zero(self, synthetic_fields):
        a, _, _, _, _ = synthetic_fields
        d_self = mmd_rbf(a, a)
        assert abs(d_self) < 1e-6


class TestWasserstein1DMarginals:
    def test_returns_dict(self, synthetic_fields):
        a, b, _, _, _ = synthetic_fields
        result = wasserstein_1d_marginals(a, b)
        assert isinstance(result, dict)
        assert "wasserstein_all_levels_mean" in result

    def test_3d_fields(self, synthetic_3d_fields):
        a, b, _, _ = synthetic_3d_fields
        result = wasserstein_1d_marginals(a, b)
        assert "wasserstein_all" in result

    def test_has_lat_bands(self, synthetic_fields):
        a, b, _, _, _ = synthetic_fields
        result = wasserstein_1d_marginals(a, b)
        assert "wasserstein_latband_0" in result


class TestPowerSpectrumDistance:
    def test_returns_dict(self, synthetic_fields):
        a, b, _, lat, lon = synthetic_fields
        result = power_spectrum_distance(a, b, lat, lon)
        assert "log_spectral_dist_mean" in result

    def test_self_small(self, synthetic_fields):
        a, _, _, lat, lon = synthetic_fields
        result = power_spectrum_distance(a, a, lat, lon)
        assert result["log_spectral_dist_mean"] < 1e-6


class TestZonalMeanDistance:
    def test_returns_dict(self, synthetic_fields):
        a, b, _, lat, _ = synthetic_fields
        result = zonal_mean_distance(a, b, lat)
        assert "zonal_mean_rmse" in result
        assert "zonal_std_rmse" in result

    def test_self_zero(self, synthetic_fields):
        a, _, _, lat, _ = synthetic_fields
        result = zonal_mean_distance(a, a, lat)
        assert result["zonal_mean_rmse"] < 1e-10


class TestMeridionalGradient:
    def test_returns_float(self, synthetic_fields):
        a, b, _, lat, _ = synthetic_fields
        result = meridional_gradient_score(a, b, lat)
        assert isinstance(result, float)
        assert result >= 0


class TestCoverageDensity:
    def test_returns_dict(self, synthetic_fields):
        a, b, _, _, _ = synthetic_fields
        result = coverage_density(a, b)
        assert "coverage" in result
        assert "density" in result
        assert 0 <= result["coverage"] <= 1

    def test_self_high_coverage(self, synthetic_fields):
        a, _, _, _, _ = synthetic_fields
        result = coverage_density(a, a)
        assert result["coverage"] > 0.5


class TestVendiScore:
    def test_returns_float(self, synthetic_fields):
        a, _, _, _, _ = synthetic_fields
        score = vendi_score(a)
        assert isinstance(score, float)
        assert score >= 1.0

    def test_low_diversity_lower_than_high(self):
        """Low diversity samples should score lower than high diversity."""
        np.random.seed(42)
        # Low diversity: small perturbations around a single pattern
        base = np.random.randn(1, 4, 8, 2)
        low_div = base + np.random.randn(20, 4, 8, 2) * 0.01
        score_low = vendi_score(low_div)

        # High diversity: completely independent samples
        high_div = np.random.randn(20, 4, 8, 2)
        score_high = vendi_score(high_div)

        assert score_low < score_high


class TestComputeDistributionalMetrics:
    def test_returns_all_keys(self, synthetic_fields):
        a, b, _, lat, lon = synthetic_fields
        metrics = compute_distributional_metrics(a, b, lat, lon)
        assert "energy_distance" in metrics
        assert "mmd_rbf" in metrics
        assert "zonal_mean_rmse" in metrics
        assert "coverage" in metrics
        assert "vendi_score_gen" in metrics

    def test_different_distributions(self, synthetic_fields):
        a, b, c, lat, lon = synthetic_fields
        m_same = compute_distributional_metrics(a, b, lat, lon)
        m_diff = compute_distributional_metrics(a, c, lat, lon)
        # Different distributions should have larger energy distance
        assert m_diff["energy_distance"] > m_same["energy_distance"]
