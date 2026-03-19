"""Tests for OCO2DataLoader and ObservationBatch (Phase 15)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from neural_transport.data.oco2_loader import ObservationBatch, OCO2DataLoader

# ── Mock helpers ──────────────────────────────────────────────────────────


def _make_oco2_mock_dataset(n_samples=10, nlat=4, nlon=8, nlev=10, sparse_fraction=0.3, seed=42):
    """Create a mock CarbonDataset with OCO-2 fields (xco2 with NaNs, AK, priors)."""
    N = nlat * nlon
    torch.manual_seed(seed)

    # XCO2 field with NaN sparsity
    xco2_template = torch.randn(1, N, 1) + 400.0
    nan_mask = torch.rand(1, N, 1) > sparse_fraction
    xco2_template[nan_mask] = float("nan")

    template = {
        "xco2_2019_scale": xco2_template.clone(),
        "xco2_2019_scale_offset": torch.tensor(400.0),
        "xco2_2019_scale_scale": torch.tensor(10.0),
        "xco2_averaging_kernel": torch.randn(1, N, nlev).abs(),
        "xco2_apriori": torch.randn(1, N, 1) + 400.0,
        "co2_profile_apriori": torch.randn(1, N, nlev) + 400.0,
        "co2massmix": torch.randn(1, N, nlev),
        "gph_bottom": torch.randn(1, N, nlev),
        "gph_top": torch.randn(1, N, nlev),
        "co2flux_anthro": torch.randn(1, N, 1),
        "co2flux_land": torch.randn(1, N, 1),
        "co2flux_ocean": torch.randn(1, N, 1),
    }

    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=n_samples)

    times = pd.date_range("2019-01-01", periods=n_samples, freq="6h")
    ds.ds = xr.Dataset(coords={"time": times})

    def getitem(idx):
        result = {k: v.clone() for k, v in template.items()}
        # Vary xco2 slightly per sample
        result["xco2_2019_scale"] = xco2_template.clone() + idx * 0.01
        return result

    ds.__getitem__ = MagicMock(side_effect=getitem)

    # tensor_to_xarray stub
    def tensor_to_xarray(tensor):
        data = tensor.detach().cpu().numpy()
        if data.ndim == 4:
            return xr.DataArray(data, dims=["batch", "time", "cell", "level"])
        elif data.ndim == 3:
            return xr.DataArray(data, dims=["batch", "cell", "level"])
        return xr.DataArray(data)

    ds.tensor_to_xarray = tensor_to_xarray

    return ds


def _make_oco2_loader(n_samples=10, nlat=4, nlon=8, nlev=10, sparse_fraction=0.3):
    """Create an OCO2DataLoader with a mock dataset injected."""
    from neural_transport.configs import DataConfig

    cfg = DataConfig()
    loader = OCO2DataLoader(cfg, data_path="/fake/path")
    mock_ds = _make_oco2_mock_dataset(
        n_samples=n_samples, nlat=nlat, nlon=nlon, nlev=nlev, sparse_fraction=sparse_fraction
    )
    loader._dataset = mock_ds
    # Override grid_info for test dimensions
    from neural_transport.data.inference_loader import GridInfo

    loader._grid_info = GridInfo(
        nlat=nlat,
        nlon=nlon,
        nlev=nlev,
        lat=np.linspace(-90, 90, nlat),
        lon=np.linspace(0, 360, nlon, endpoint=False),
        levels=np.arange(nlev),
    )
    return loader


# ── ObservationBatch tests ────────────────────────────────────────────────


@pytest.mark.quick
class TestObservationBatch:
    def test_creation(self):
        B, T, N, C = 1, 1, 32, 1
        C_lev = 10
        obs = ObservationBatch(
            obs_values=torch.randn(B, T, N, C),
            obs_mask=torch.ones(B, T, N, C, dtype=torch.bool),
            averaging_kernel=torch.randn(B, T, N, C_lev),
            xco2_prior=torch.randn(B, T, N, 1),
            co2_profile_prior=torch.randn(B, T, N, C_lev),
            pressure_weights=torch.randn(B, T, N, C_lev),
            obs_mean=torch.tensor(400.0),
            obs_std=torch.tensor(10.0),
            raw_batch={"xco2_2019_scale": torch.randn(B, T, N, C)},
        )
        assert obs.obs_values.shape == (B, T, N, C)
        assert obs.obs_mask.shape == (B, T, N, C)
        assert obs.averaging_kernel.shape == (B, T, N, C_lev)

    def test_inject_into_batch(self):
        B, T, N, C = 1, 1, 32, 1
        obs = ObservationBatch(
            obs_values=torch.ones(B, T, N, C),
            obs_mask=torch.ones(B, T, N, C, dtype=torch.bool),
            averaging_kernel=None,
            xco2_prior=None,
            co2_profile_prior=None,
            pressure_weights=None,
            obs_mean=torch.tensor(0.0),
            obs_std=torch.tensor(1.0),
            raw_batch={
                "xco2_2019_scale": torch.randn(B, T, N, C),
                "xco2_averaging_kernel": torch.randn(B, T, N, 10),
            },
        )
        batch = {"co2massmix": torch.randn(B, T, N, 10)}
        obs.inject_into_batch(
            batch,
            target_var="xco2_2019_scale",
            forcing_vars=["xco2_averaging_kernel"],
        )
        assert "obs_mask" in batch
        assert "obs_mask_original" in batch
        assert "obs_values" in batch
        assert "xco2_2019_scale" in batch
        assert "xco2_averaging_kernel" in batch
        # Original key preserved
        assert "co2massmix" in batch

    def test_inject_preserves_shape(self):
        B, T, N, C = 2, 1, 16, 1
        obs = ObservationBatch(
            obs_values=torch.ones(B, T, N, C),
            obs_mask=torch.zeros(B, T, N, C, dtype=torch.bool),
            averaging_kernel=None,
            xco2_prior=None,
            co2_profile_prior=None,
            pressure_weights=None,
            obs_mean=torch.tensor(0.0),
            obs_std=torch.tensor(1.0),
            raw_batch={"xco2_2019_scale": torch.randn(B, T, N, C)},
        )
        batch = {}
        obs.inject_into_batch(batch, target_var="xco2_2019_scale", forcing_vars=[])
        assert batch["obs_mask"].shape == (B, T, N, C)
        assert batch["obs_values"].shape == (B, T, N, C)


# ── OCO2DataLoader.align_time tests ───────────────────────────────────────


@pytest.mark.quick
class TestOCO2DataLoaderAlignTime:
    def test_same_start(self):
        t = np.array([1, 2, 3])
        assert OCO2DataLoader.align_time(t, t) == 0

    def test_oco2_later(self):
        time = np.array([3, 4, 5])
        time_gen = np.array([1, 2, 3, 4, 5])
        assert OCO2DataLoader.align_time(time, time_gen) == 2

    def test_oco2_earlier(self):
        time = np.array([1, 2, 3, 4, 5])
        time_gen = np.array([3, 4, 5])
        assert OCO2DataLoader.align_time(time, time_gen) == -2

    def test_no_overlap_before(self):
        with pytest.raises(ValueError, match="No overlap"):
            OCO2DataLoader.align_time(np.array([1, 2, 3]), np.array([5, 6, 7]))

    def test_no_overlap_after(self):
        with pytest.raises(ValueError, match="No overlap"):
            OCO2DataLoader.align_time(np.array([5, 6, 7]), np.array([1, 2, 3]))


# ── OCO2DataLoader.get_window_batch tests ─────────────────────────────────


@pytest.mark.quick
class TestOCO2DataLoaderGetWindowBatch:
    def test_single_step(self):
        loader = _make_oco2_loader()
        batch = loader.get_window_batch(0, window_steps=1, device="cpu")
        assert "xco2_2019_scale" in batch
        # Mock dataset returns [1, N, C], unsqueeze adds batch dim → [1, 1, N, C]
        assert batch["xco2_2019_scale"].ndim == 4

    def test_multi_step_nanmean(self):
        loader = _make_oco2_loader()
        batch = loader.get_window_batch(0, window_steps=3, device="cpu", agg="nanmean")
        assert "xco2_2019_scale" in batch
        assert batch["xco2_2019_scale"].shape[0] == 1  # B=1

    def test_multi_step_mean(self):
        loader = _make_oco2_loader()
        batch = loader.get_window_batch(0, window_steps=3, device="cpu", agg="mean")
        assert "xco2_2019_scale" in batch

    def test_clamp_to_length(self):
        loader = _make_oco2_loader(n_samples=5)
        # Request more steps than available
        batch = loader.get_window_batch(3, window_steps=10, device="cpu")
        # Should not crash, just uses available samples
        assert "xco2_2019_scale" in batch

    def test_normalization_stats_preserved(self):
        loader = _make_oco2_loader()
        batch = loader.get_window_batch(0, window_steps=1, device="cpu")
        assert "xco2_2019_scale_offset" in batch
        assert "xco2_2019_scale_scale" in batch


# ── OCO2DataLoader.get_observations tests ─────────────────────────────────


@pytest.mark.quick
class TestOCO2DataLoaderGetObservations:
    def test_returns_observation_batch(self):
        loader = _make_oco2_loader()
        obs = loader.get_observations(0, device="cpu")
        assert isinstance(obs, ObservationBatch)

    def test_mask_from_sparsity(self):
        loader = _make_oco2_loader(sparse_fraction=0.5)
        obs = loader.get_observations(0, device="cpu")
        # Mask should have both True and False values (sparse data)
        assert obs.obs_mask.any()
        # With 50% sparse, should not be all True
        assert not obs.obs_mask.all()

    def test_mask_from_test_pattern(self):
        loader = _make_oco2_loader(nlat=4, nlon=8)
        obs = loader.get_observations(0, device="cpu", mask_pattern="checkerboard", nlat=4, nlon=8)
        assert obs.obs_mask.any()
        # Checkerboard should have ~50% observed
        frac = obs.obs_mask.float().mean().item()
        assert 0.3 < frac < 0.7

    def test_shapes(self):
        nlat, nlon = 4, 8
        loader = _make_oco2_loader(nlat=nlat, nlon=nlon)
        obs = loader.get_observations(0, device="cpu")
        N = nlat * nlon
        assert obs.obs_values.shape[-2] == N
        assert obs.obs_mask.shape[-2] == N

    def test_normalization_stats(self):
        loader = _make_oco2_loader()
        obs = loader.get_observations(0, device="cpu")
        assert obs.obs_mean is not None
        assert obs.obs_std is not None

    def test_ak_present(self):
        loader = _make_oco2_loader()
        obs = loader.get_observations(0, device="cpu")
        assert obs.averaging_kernel is not None

    def test_raw_batch_available(self):
        loader = _make_oco2_loader()
        obs = loader.get_observations(0, device="cpu")
        assert isinstance(obs.raw_batch, dict)
        assert "xco2_2019_scale" in obs.raw_batch

    def test_window_steps(self):
        loader = _make_oco2_loader(n_samples=10)
        obs = loader.get_observations(0, window_steps=3, device="cpu")
        assert isinstance(obs, ObservationBatch)

    def test_with_offset(self):
        loader = _make_oco2_loader(n_samples=10)
        obs = loader.get_observations(0, offset=2, device="cpu")
        assert isinstance(obs, ObservationBatch)


# ── Backward compatibility tests ──────────────────────────────────────────


@pytest.mark.quick
class TestBackwardCompat:
    def test_align_time_importable_from_generation(self):
        """align_time should still be importable from generation module."""
        from neural_transport.inference.generation import align_time

        t = np.array([1, 2, 3])
        assert align_time(t, t) == 0

    def test_align_time_delegates_to_oco2(self):
        """generation.align_time should produce same results as OCO2DataLoader.align_time."""
        from neural_transport.inference.generation import align_time

        time = np.array([3, 4, 5])
        time_gen = np.array([1, 2, 3, 4, 5])
        assert align_time(time, time_gen) == OCO2DataLoader.align_time(time, time_gen)

    def test_get_batches_still_works(self):
        """get_batches should still be importable and functional."""
        # Inline minimal mock dataset
        import pandas as pd
        import torch
        import xarray as xr

        from neural_transport.inference.generation import get_batches

        class _SimpleDataset:
            def __init__(self):
                self._n = 5
                times = pd.date_range("2019-01-01", periods=5, freq="6h")
                self.ds = xr.Dataset(coords={"time": times})
                self._t = {"co2massmix": torch.randn(1, 32, 10)}

            def __len__(self):
                return self._n

            def __getitem__(self, idx):
                return {k: v.clone() for k, v in self._t.items()}

        ds = _SimpleDataset()
        ds_gen = _SimpleDataset()
        batch, batch_gen = get_batches(0, 0, ds, ds_gen, window_steps=1, device="cpu")
        assert "co2massmix" in batch
        assert "co2massmix" in batch_gen

    def test_oco2_loader_exports(self):
        """OCO2DataLoader and ObservationBatch should be importable from data package."""
        from neural_transport.data import ObservationBatch, OCO2DataLoader

        assert OCO2DataLoader is not None
        assert ObservationBatch is not None
