"""Tests for GridInfo and InferenceDataLoader (Phase 14)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from neural_transport.configs import DataConfig
from neural_transport.data import GridInfo, InferenceDataLoader
from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS, VERTICAL_LAYERS_PROTOTYPE_COORDS

# ---------------------------------------------------------------------------
# GridInfo tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_grid_info_from_config():
    """Default DataConfig (latlon5.625/l10) → nlat=32, nlon=64, nlev=10."""
    cfg = DataConfig()
    gi = GridInfo.from_config(cfg)
    assert gi.nlat == 32
    assert gi.nlon == 64
    assert gi.nlev == 10
    np.testing.assert_array_equal(gi.lat, LATLON_PROTOTYPE_COORDS["latlon5.625"]["lat"])
    np.testing.assert_array_equal(gi.lon, LATLON_PROTOTYPE_COORDS["latlon5.625"]["lon"])
    np.testing.assert_array_equal(gi.levels, VERTICAL_LAYERS_PROTOTYPE_COORDS["l10"]["level"])


@pytest.mark.quick
def test_cos_lat_weights_2d_shape_and_normalization():
    """cos_lat_weights_2d has shape (nlat, 1) and mean ≈ 1.0."""
    gi = GridInfo.from_config(DataConfig())
    w = gi.cos_lat_weights_2d
    assert w.shape == (32, 1)
    np.testing.assert_allclose(w.mean(), 1.0, atol=1e-10)


@pytest.mark.quick
def test_cos_lat_weights_flat_shape():
    """cos_lat_weights_flat has shape (nlat*nlon, 1) and mean ≈ 1.0."""
    gi = GridInfo.from_config(DataConfig())
    w = gi.cos_lat_weights_flat
    assert w.shape == (32 * 64, 1)
    np.testing.assert_allclose(w.mean(), 1.0, atol=1e-10)


@pytest.mark.quick
def test_cos_lat_weights_flat_matches_manual():
    """cos_lat_weights_flat matches the carbonbench manual computation."""
    cfg = DataConfig()
    gi = GridInfo.from_config(cfg)

    # Manual computation (the carbonbench pattern)
    lat = LATLON_PROTOTYPE_COORDS["latlon5.625"]["lat"]
    nlon = 64
    cos_lat = np.cos(np.radians(lat))[:, None, None].repeat(nlon, axis=1).reshape(-1, 1)
    cos_lat = cos_lat / cos_lat.mean()

    np.testing.assert_allclose(gi.cos_lat_weights_flat, cos_lat, atol=1e-12)


@pytest.mark.quick
def test_cos_lat_weights_alias():
    """cos_lat_weights is an alias for cos_lat_weights_flat."""
    gi = GridInfo.from_config(DataConfig())
    np.testing.assert_array_equal(gi.cos_lat_weights, gi.cos_lat_weights_flat)


@pytest.mark.quick
def test_grid_info_non_latlon_raises():
    """Non-latlon grids raise ValueError."""
    cfg = DataConfig(grid="icon")
    with pytest.raises(ValueError, match="non-latlon"):
        GridInfo.from_config(cfg)


@pytest.mark.quick
@pytest.mark.parametrize(
    "grid, vlev, expected_nlat, expected_nlon, expected_nlev",
    [
        ("latlon5.625", "l10", 32, 64, 10),
        ("latlon2.8125", "l19", 64, 128, 19),
        ("latlon2x3", "l34", 90, 120, 34),
        ("latlon1.5", "l20", 120, 240, 20),
    ],
)
def test_grid_info_multiple_grids(grid, vlev, expected_nlat, expected_nlon, expected_nlev):
    """GridInfo works for several grid/level combos."""
    cfg = DataConfig(grid=grid, vertical_levels=vlev)
    gi = GridInfo.from_config(cfg)
    assert gi.nlat == expected_nlat
    assert gi.nlon == expected_nlon
    assert gi.nlev == expected_nlev


# ---------------------------------------------------------------------------
# InferenceDataLoader tests
# ---------------------------------------------------------------------------


def _make_mock_dataset(nlat=32, nlon=64, nlev=10, n_samples=5):
    """Create a mock CarbonDataset with the right interface."""
    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n_samples)

    n_cells = nlat * nlon
    sample = {
        "co2massmix": torch.randn(n_cells, nlev),
        "co2massmix_offset": torch.tensor(400.0),
        "co2massmix_scale": torch.tensor(10.0),
        "forcing_a": torch.randn(n_cells, 1),
        "forcing_a_offset": torch.tensor(0.0),
        "forcing_a_scale": torch.tensor(1.0),
    }
    ds.__getitem__ = MagicMock(return_value=sample)
    return ds


@pytest.mark.quick
def test_inference_data_loader_init():
    """Constructs without loading dataset; grid_info is correct."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    assert loader.grid_info.nlat == 32
    assert loader.grid_info.nlon == 64
    assert loader._dataset is None  # Not loaded yet


@pytest.mark.quick
def test_inference_data_loader_get_batch():
    """get_batch returns dict with batch dim 0 added."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    batch = loader.get_batch(0, device="cpu")
    assert isinstance(batch, dict)
    assert batch["co2massmix"].shape == (1, 32 * 64, 10)
    mock_ds.__getitem__.assert_called_with(0)


@pytest.mark.quick
def test_inference_data_loader_get_gt_field():
    """get_gt_field returns numpy array with shape [cell, level]."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    field = loader.get_gt_field(2)
    assert isinstance(field, np.ndarray)
    assert field.shape == (32 * 64, 10)
    mock_ds.__getitem__.assert_called_with(2)


@pytest.mark.quick
def test_inference_data_loader_get_normalization_stats():
    """get_normalization_stats extracts _offset and _scale keys."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    stats = loader.get_normalization_stats()
    assert "co2massmix_offset" in stats
    assert "co2massmix_scale" in stats
    assert "forcing_a_offset" in stats
    assert "forcing_a_scale" in stats
    # Non-stat keys should not be present
    assert "co2massmix" not in stats
    assert "forcing_a" not in stats


@pytest.mark.quick
def test_inference_data_loader_len():
    """__len__ delegates to dataset."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset(n_samples=42)
    loader._dataset = mock_ds

    assert len(loader) == 42


# ---------------------------------------------------------------------------
# InferenceDataLoader.get_window_batch tests
# ---------------------------------------------------------------------------


@pytest.mark.quick
def test_get_window_batch_single_step():
    """Single-step window returns same shape as get_batch."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    batch = loader.get_window_batch(0, window_steps=1, device="cpu")
    assert isinstance(batch, dict)
    assert batch["co2massmix"].shape == (1, 32 * 64, 10)


@pytest.mark.quick
def test_get_window_batch_multi_step():
    """Multi-step window aggregates correctly."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    batch = loader.get_window_batch(0, window_steps=3, device="cpu")
    assert isinstance(batch, dict)
    assert batch["co2massmix"].shape == (1, 32 * 64, 10)


@pytest.mark.quick
def test_get_window_batch_clamp_to_length():
    """Window extending past dataset length is clamped."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset(n_samples=3)
    loader._dataset = mock_ds

    # Request 10 steps but only 3 available
    batch = loader.get_window_batch(1, window_steps=10, device="cpu")
    assert "co2massmix" in batch


@pytest.mark.quick
def test_get_window_batch_nanmean_agg():
    """nanmean aggregation works."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    batch = loader.get_window_batch(0, window_steps=2, device="cpu", agg="nanmean")
    assert "co2massmix" in batch


@pytest.mark.quick
def test_get_window_batch_preserves_normalization():
    """Normalization stat keys are preserved."""
    cfg = DataConfig()
    loader = InferenceDataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_mock_dataset()
    loader._dataset = mock_ds

    batch = loader.get_window_batch(0, window_steps=1, device="cpu")
    assert "co2massmix_offset" in batch
    assert "co2massmix_scale" in batch
