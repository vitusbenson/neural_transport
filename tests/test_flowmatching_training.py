"""Tests for OT coupling and time grid utilities.

Usage:
    pytest tests/test_flowmatching_training.py -v
    pytest tests/test_flowmatching_training.py -m quick -v
"""

import pytest
import torch

from neural_transport.models.flowmatching import FlowMatching, compute_ot_coupling

# ── OT coupling tests ────────────────────────────────────────────────────


@pytest.mark.quick
def test_ot_coupling_correct_shape():
    """Output has same shape as input."""
    x_0 = torch.randn(8, 4, 16, 32)
    x_1 = torch.randn(8, 4, 16, 32)
    result = compute_ot_coupling(x_0, x_1)
    assert result.shape == x_0.shape


@pytest.mark.quick
def test_ot_coupling_is_reordering():
    """Output rows are drawn from input rows (each output row matches some input row)."""
    B = 16
    x_0 = torch.randn(B, 4, 16, 32)
    x_1 = torch.randn(B, 4, 16, 32)
    result = compute_ot_coupling(x_0, x_1)

    # Each row in result should match at least one row in x_0
    x0_flat = x_0.reshape(B, -1)
    res_flat = result.reshape(B, -1)

    matched_outputs = 0
    for i in range(B):
        for j in range(B):
            if torch.allclose(res_flat[i], x0_flat[j], atol=1e-6):
                matched_outputs += 1
                break

    assert matched_outputs == B, f"Only {matched_outputs}/{B} output rows match input rows"


@pytest.mark.quick
def test_ot_coupling_reduces_cost():
    """OT coupling produces lower total L2 cost than random pairing."""
    torch.manual_seed(42)
    B = 32
    x_0 = torch.randn(B, 4, 8, 16)
    x_1 = torch.randn(B, 4, 8, 16)

    # Random pairing cost
    random_cost = (x_0 - x_1).reshape(B, -1).pow(2).sum(dim=1).mean().item()

    # OT pairing cost
    x_0_ot = compute_ot_coupling(x_0, x_1)
    ot_cost = (x_0_ot - x_1).reshape(B, -1).pow(2).sum(dim=1).mean().item()

    assert ot_cost <= random_cost, f"OT cost {ot_cost:.4f} > random cost {random_cost:.4f}"


@pytest.mark.quick
def test_ot_coupling_small_batch():
    """Works correctly with B=2."""
    x_0 = torch.randn(2, 4, 8, 16)
    x_1 = torch.randn(2, 4, 8, 16)
    result = compute_ot_coupling(x_0, x_1)
    assert result.shape == x_0.shape
    assert not torch.isnan(result).any()


@pytest.mark.quick
def test_ot_coupling_no_grad():
    """OT coupling should not track gradients."""
    x_0 = torch.randn(4, 2, 8, 8, requires_grad=True)
    x_1 = torch.randn(4, 2, 8, 8, requires_grad=True)
    result = compute_ot_coupling(x_0, x_1)
    assert not result.requires_grad


# ── Time grid tests ──────────────────────────────────────────────────────


@pytest.mark.quick
def test_time_grid_uniform():
    """Uniform grid has equal spacing."""
    grid = FlowMatching._build_time_grid(11, torch.device('cpu'), 'uniform')
    assert grid.shape == (11,)
    diffs = grid[1:] - grid[:-1]
    assert torch.allclose(diffs, diffs[0], atol=1e-6)


@pytest.mark.quick
def test_time_grid_cosine():
    """Cosine grid is denser at endpoints (larger steps in the middle)."""
    grid = FlowMatching._build_time_grid(21, torch.device('cpu'), 'cosine')
    diffs = grid[1:] - grid[:-1]
    # Middle step should be larger than first step
    assert diffs[len(diffs) // 2] > diffs[0]
    # Middle step should be larger than last step
    assert diffs[len(diffs) // 2] > diffs[-1]


@pytest.mark.quick
def test_time_grid_front_loaded():
    """Front-loaded grid is denser near t=0 (first step < last step)."""
    grid = FlowMatching._build_time_grid(21, torch.device('cpu'), 'front_loaded')
    diffs = grid[1:] - grid[:-1]
    assert diffs[0] < diffs[-1], "Front-loaded grid should have smaller steps near t=0"


@pytest.mark.quick
def test_all_grids_monotonic_0_to_1():
    """All grid types are monotonically increasing from 0 to 1."""
    for spacing in ['uniform', 'cosine', 'front_loaded']:
        grid = FlowMatching._build_time_grid(11, torch.device('cpu'), spacing)
        assert grid[0] == 0.0, f"{spacing}: grid doesn't start at 0"
        assert grid[-1] == 1.0, f"{spacing}: grid doesn't end at 1"
        diffs = grid[1:] - grid[:-1]
        assert (diffs > 0).all(), f"{spacing}: grid is not monotonically increasing"
