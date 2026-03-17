"""Unit tests for noise generation utilities."""

import pytest
import torch

from neural_transport.inference.noise import generate_noise, noise

T, N, C = 2, 32, 5


@pytest.fixture
def batch():
    """Batch dict with co2massmix tensor of shape [T, N, C]."""
    return {"co2massmix": torch.randn(T, N, C)}


ALL_PATTERNS = [
    None,
    "spiral_outward_noise",
    "spiral_noise",
    "geodesic_noise",
    "linear_noise",
    "antipodal_orthogonal_noise",
]


class TestGenerateNoise:
    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_output_length_and_shape(self, batch, pattern):
        """Each pattern returns list of length n_samples, each tensor [1, T, N, C]."""
        n_samples = 6
        result = generate_noise(batch, n_samples=n_samples, noise_pattern=pattern)
        assert len(result) == n_samples
        for tensor in result:
            assert tensor.shape == (1, T, N, C)

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_no_nan_inf(self, batch, pattern):
        """No NaN or Inf in any output."""
        result = generate_noise(batch, n_samples=4, noise_pattern=pattern)
        for tensor in result:
            assert not torch.isnan(tensor).any(), f"NaN found for pattern {pattern}"
            assert not torch.isinf(tensor).any(), f"Inf found for pattern {pattern}"

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_deterministic_with_seed(self, batch, pattern):
        """Deterministic given same torch.manual_seed."""
        torch.manual_seed(42)
        result1 = generate_noise(batch, n_samples=4, noise_pattern=pattern)
        torch.manual_seed(42)
        result2 = generate_noise(batch, n_samples=4, noise_pattern=pattern)
        for t1, t2 in zip(result1, result2):
            assert torch.equal(t1, t2), f"Not deterministic for pattern {pattern}"

    @pytest.mark.parametrize("pattern", ALL_PATTERNS)
    def test_n_samples_1(self, batch, pattern):
        """n_samples=1 works for all patterns."""
        result = generate_noise(batch, n_samples=1, noise_pattern=pattern)
        assert len(result) == 1
        assert result[0].shape == (1, T, N, C)

    def test_unknown_pattern_raises(self, batch):
        """Unknown pattern raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise type"):
            generate_noise(batch, noise_pattern="nonexistent_pattern")

    def test_antipodal_pairs_sum_to_zero(self, batch):
        """For antipodal_orthogonal_noise with n_samples=4, pairs 0+1 and 2+3 sum to near-zero."""
        torch.manual_seed(123)
        result = generate_noise(batch, n_samples=4, noise_pattern="antipodal_orthogonal_noise")
        assert len(result) == 4
        # Pair 0+1 should sum to near-zero
        pair1_sum = result[0] + result[1]
        assert torch.allclose(pair1_sum, torch.zeros_like(pair1_sum), atol=1e-5), (
            f"Pair 0+1 sum max: {pair1_sum.abs().max().item()}"
        )
        # Pair 2+3 should sum to near-zero
        pair2_sum = result[2] + result[3]
        assert torch.allclose(pair2_sum, torch.zeros_like(pair2_sum), atol=1e-5), (
            f"Pair 2+3 sum max: {pair2_sum.abs().max().item()}"
        )


class TestNoiseWrapper:
    def test_returns_correct_length(self, batch):
        """noise() with analyze_noise=False returns list of correct length."""
        n_samples = 5
        result = noise(batch, n_samples=n_samples, analyze_noise=False)
        assert len(result) == n_samples

    def test_delegates_to_generate_noise(self, batch):
        """noise() delegates correctly to generate_noise."""
        torch.manual_seed(42)
        result_wrapper = noise(batch, n_samples=3, noise_pattern="spiral_noise", analyze_noise=False)
        torch.manual_seed(42)
        result_direct = generate_noise(batch, n_samples=3, noise_pattern="spiral_noise")
        for t1, t2 in zip(result_wrapper, result_direct):
            assert torch.equal(t1, t2)
