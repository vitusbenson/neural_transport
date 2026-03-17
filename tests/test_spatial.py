"""Unit tests for spatial utility functions."""

import pytest
import torch

from neural_transport.tools.spatial import gaussian_smooth_2d


class TestGaussianSmooth2D:
    def test_sigma_zero_returns_input(self):
        """sigma=0 returns input unchanged (identity)."""
        field = torch.randn(2, 3, 16, 32)
        result = gaussian_smooth_2d(field, sigma=0.0)
        assert torch.equal(result, field)

    def test_sigma_negative_returns_input(self):
        """sigma < 0 returns input unchanged."""
        field = torch.randn(2, 3, 16, 32)
        result = gaussian_smooth_2d(field, sigma=-1.0)
        assert torch.equal(result, field)

    @pytest.mark.parametrize(
        "shape",
        [(1, 1, 8, 16), (2, 5, 32, 64), (1, 3, 4, 4), (3, 2, 16, 16)],
    )
    def test_output_shape_matches_input(self, shape):
        """Output shape matches input."""
        field = torch.randn(*shape)
        result = gaussian_smooth_2d(field, sigma=1.5)
        assert result.shape == field.shape

    def test_constant_field_unchanged(self):
        """Constant field unchanged after smoothing."""
        field = torch.ones(2, 3, 16, 32) * 5.0
        result = gaussian_smooth_2d(field, sigma=2.0)
        assert torch.allclose(result, field, atol=1e-5)

    def test_periodic_longitude_wrapping(self):
        """Spike at rightmost col visible at leftmost col via periodic wrapping."""
        field = torch.zeros(1, 1, 16, 32)
        field[0, 0, 8, 31] = 1.0  # rightmost column
        result = gaussian_smooth_2d(field, sigma=2.0)
        # Left edge should receive some weight from periodic wrapping
        assert result[0, 0, 8, 0] > 1e-6, "Periodic wrapping failed: left edge should see the right-edge spike"

    def test_energy_conservation(self):
        """Sum of output approximately equals sum of input."""
        field = torch.randn(1, 1, 32, 64)
        result = gaussian_smooth_2d(field, sigma=2.0)
        assert torch.allclose(result.sum(), field.sum(), rtol=0.05), (
            f"Energy not conserved: input sum={field.sum().item():.4f}, output sum={result.sum().item():.4f}"
        )

    def test_kernel_size_via_spread(self):
        """Kernel size = int(6*sigma + 1) verified via spread of smoothed delta."""
        sigma = 2.0
        expected_ks = int(6 * sigma + 1)
        half = expected_ks // 2

        field = torch.zeros(1, 1, 64, 128)
        field[0, 0, 32, 64] = 1.0
        result = gaussian_smooth_2d(field, sigma=sigma)

        # Values beyond the kernel half-width should be zero (or very near zero)
        # Check latitude direction (reflect padded, so strictly bounded)
        far_lat = 32 + half + 2  # well beyond kernel reach
        if far_lat < 64:
            assert result[0, 0, far_lat, 64].abs() < 1e-6, (
                f"Value at distance {half + 2} from delta is {result[0, 0, far_lat, 64].item()}, expected ~0"
            )
