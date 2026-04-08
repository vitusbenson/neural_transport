"""Tests for evaluation.spectral — power spectrum and spectral metrics."""

import numpy as np
import pytest


class TestPowerSpectrum2D:
    def test_shape(self):
        from neural_transport.evaluation.spectral import power_spectrum_2d

        field = np.random.randn(32, 64)
        wn, power = power_spectrum_2d(field)
        assert wn.shape == power.shape
        assert len(wn) == 16  # min(32,64)//2 = 16, bins 1..16

    def test_constant_field_zero_power(self):
        from neural_transport.evaluation.spectral import power_spectrum_2d

        field = np.ones((16, 32)) * 42.0
        wn, power = power_spectrum_2d(field)
        # Constant field has zero power (mean removed)
        assert np.allclose(power, 0.0, atol=1e-10)

    def test_single_frequency(self):
        from neural_transport.evaluation.spectral import power_spectrum_2d

        nlat, nlon = 32, 64
        x = np.arange(nlon) / nlon * 2 * np.pi
        field = np.sin(3 * x)[None, :] * np.ones((nlat, 1))  # wavenumber 3 in lon
        wn, power = power_spectrum_2d(field)
        # Peak should be near wavenumber 3
        peak_wn = wn[np.argmax(power)]
        assert abs(peak_wn - 3) <= 1

    def test_with_lat_weights(self):
        from neural_transport.evaluation.spectral import power_spectrum_2d

        field = np.random.randn(16, 32)
        lat = np.linspace(-90, 90, 16)
        cos_lat = np.cos(np.deg2rad(lat))
        wn1, p1 = power_spectrum_2d(field)
        wn2, p2 = power_spectrum_2d(field, cos_lat_weights=cos_lat)
        # Same shape, different values due to weighting
        assert wn1.shape == wn2.shape
        assert not np.allclose(p1, p2)


class TestSpectralDivergence:
    def test_identical_spectra(self):
        from neural_transport.evaluation.spectral import spectral_divergence

        p = np.array([100.0, 50.0, 10.0, 1.0])
        assert spectral_divergence(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_different_spectra_positive(self):
        from neural_transport.evaluation.spectral import spectral_divergence

        p1 = np.array([100.0, 50.0, 10.0])
        p2 = np.array([100.0, 50.0, 1.0])
        d = spectral_divergence(p1, p2)
        assert d > 0

    def test_symmetry(self):
        from neural_transport.evaluation.spectral import spectral_divergence

        p1 = np.array([100.0, 50.0, 10.0])
        p2 = np.array([90.0, 40.0, 8.0])
        assert spectral_divergence(p1, p2) == pytest.approx(spectral_divergence(p2, p1))


class TestSpectralSlope:
    def test_known_slope(self):
        from neural_transport.evaluation.spectral import spectral_slope

        # Power law: P(k) = k^(-2)
        wn = np.arange(1, 20, dtype=float)
        power = wn ** (-2.0)
        slope = spectral_slope(wn, power)
        assert slope == pytest.approx(-2.0, abs=0.01)

    def test_flat_spectrum(self):
        from neural_transport.evaluation.spectral import spectral_slope

        wn = np.arange(1, 10, dtype=float)
        power = np.ones_like(wn) * 5.0
        slope = spectral_slope(wn, power)
        assert slope == pytest.approx(0.0, abs=0.01)
