"""Tests for `_next` suffix in input_vars (Phase 24 transport prior setup).

Usage:
    pytest tests/test_regulargrid_next_inputs.py -v
"""

import pytest
import torch

from neural_transport.models.regulargrid import RegularGridModel


class _BareRegularGrid(RegularGridModel):
    """Bypass init_model so we can exercise normalize_batch / preprocess_inputs."""

    def init_model(self, **model_kwargs):
        self._submodel = None

    def model(self, x):
        return x


def _build_model(input_vars, target_vars, targshift=True, nlev=10, nlat=4, nlon=8):
    return _BareRegularGrid(
        input_vars=input_vars,
        target_vars=target_vars,
        nlat=nlat,
        nlon=nlon,
        nlev=nlev,
        targshift=targshift,
        horizontal_interpolation=None,
    )


def _make_batch(vars_and_next, B=2, N=32, C=10):
    """Create a minimal batch dict with {var}, {var}_next, {var}_offset, {var}_scale."""
    batch = {}
    for v, has_next in vars_and_next.items():
        batch[v] = torch.randn(B, N, C)
        batch[f"{v}_offset"] = torch.zeros(B, 1, 1)
        batch[f"{v}_scale"] = torch.ones(B, 1, 1)
        if has_next:
            batch[f"{v}_next"] = torch.randn(B, N, C)
    return batch


@pytest.mark.quick
def test_phase11_unconditional_channel_count():
    """Phase 11 layout (input_vars=[co2massmix]) yields nlev input channels."""
    m = _build_model(input_vars=["co2massmix"], target_vars=["co2massmix"])
    batch = _make_batch({"co2massmix": True}, C=10)
    x_in = m.preprocess_inputs(batch)
    assert x_in.shape == (2, 10, 4, 8)  # [B, nlev, nlat, nlon]


@pytest.mark.quick
def test_phase24_transport_prior_channel_count():
    """Phase 24 layout yields 4*nlev input channels (before time is appended)."""
    m = _build_model(
        input_vars=["co2massmix_next", "co2massmix", "u", "v"],
        target_vars=["co2massmix"],
    )
    batch = _make_batch({"co2massmix": True, "u": False, "v": False}, C=10)
    x_in = m.preprocess_inputs(batch)
    assert x_in.shape == (2, 40, 4, 8)


@pytest.mark.quick
def test_next_slot_uses_base_offset_scale():
    """co2massmix_next must use co2massmix_{offset,scale} (no _next_offset key exists)."""
    m = _build_model(
        input_vars=["co2massmix_next", "co2massmix"],
        target_vars=["co2massmix"],
    )
    batch = _make_batch({"co2massmix": True}, C=10)
    # Force distinguishable offset/scale, mutate batch stats:
    batch["co2massmix_offset"] = torch.full((2, 1, 1), 3.0)
    batch["co2massmix_scale"] = torch.full((2, 1, 1), 2.0)
    # No co2massmix_next_offset/scale in batch — must not raise.
    x_in = m.preprocess_inputs(batch)
    assert x_in.shape == (2, 20, 4, 8)


@pytest.mark.quick
def test_targshift_applied_to_next_slot():
    """Targshift must treat co2massmix_next as a target (base var in target_vars)."""
    m = _build_model(
        input_vars=["co2massmix_next", "co2massmix", "u"],
        target_vars=["co2massmix"],
        targshift=True,
    )
    batch = _make_batch({"co2massmix": True, "u": False}, C=10)
    normalized = m.normalize_batch(batch)

    # Mean over (N, C) should be ~0 for both co2 slots (targshift applied)
    assert torch.allclose(
        normalized["co2massmix_next"].mean(dim=(1, 2)),
        torch.zeros(2),
        atol=1e-5,
    )
    assert torch.allclose(
        normalized["co2massmix"].mean(dim=(1, 2)),
        torch.zeros(2),
        atol=1e-5,
    )
    # u is NOT in target_vars → no targshift, nonzero mean expected in general
    u_mean = normalized["u"].mean(dim=(1, 2)).abs().sum().item()
    # (not strictly guaranteed to be nonzero, but with random data essentially is)
    assert u_mean > 1e-8


@pytest.mark.quick
def test_input_vars_ordering_preserved():
    """Concat order follows input_vars list order (dict insertion order = input_vars)."""
    m = _build_model(
        input_vars=["co2massmix_next", "u", "v", "co2massmix"],
        target_vars=["co2massmix"],
    )
    batch = _make_batch({"co2massmix": True, "u": False, "v": False}, C=10)
    # Zero everything except u, which is set to constant 5.
    for v in ("co2massmix", "co2massmix_next", "v"):
        batch[v] = torch.zeros_like(batch[v])
    batch["u"] = torch.full_like(batch["u"], 5.0)
    x_in = m.preprocess_inputs(batch)  # [B, 40, H, W]
    # With input_vars order [co2_next, u, v, co2_t]:
    # channels 0-9 = co2_next (zero, minus targshift mean = 0), ≈0
    # channels 10-19 = u = 5
    # channels 20-29 = v = 0
    # channels 30-39 = co2_t (zero minus targshift) = 0
    assert torch.allclose(x_in[:, 0:10], torch.zeros_like(x_in[:, 0:10]))
    assert torch.allclose(x_in[:, 10:20], torch.full_like(x_in[:, 10:20], 5.0))
    assert torch.allclose(x_in[:, 20:30], torch.zeros_like(x_in[:, 20:30]))
    assert torch.allclose(x_in[:, 30:40], torch.zeros_like(x_in[:, 30:40]))
