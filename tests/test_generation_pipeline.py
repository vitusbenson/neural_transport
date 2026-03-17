"""Tests for the unified GenerationPipeline.

Phase 8: TDD tests written first, then implementation to make them pass.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from neural_transport.inference.generation import (
    GenerationPipeline,
    align_time,
    get_batches,
    get_zarrpath_obspath,
    is_bad_sample,
    parse_freq,
)

# ── Mock objects for pipeline tests ──────────────────────────────────────


class _MockFlowMatchingModel:
    """Lightweight mock of the FlowMatching inner model."""

    def __init__(self, nlat=4, nlon=8, nlev=5):
        self.in_nlat = nlat
        self.in_nlon = nlon
        self.nlev = nlev
        self.generate_kwargs = {}
        self.generating = False
        self.return_intermediates = True

    def normalize_observations(self, obs_values, batch, target_var=None, targshift=False):
        return obs_values


class _MockModel:
    """Lightweight mock of NeuralTransport LightningModule."""

    def __init__(self, nlat=4, nlon=8, nlev=5):
        self.model = _MockFlowMatchingModel(nlat, nlon, nlev)
        self.return_intermediates = True
        self._nlat = nlat
        self._nlon = nlon
        self._nlev = nlev

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch):
        first_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
        B = first_tensor.shape[0]
        N = self._nlat * self._nlon
        C = self._nlev
        T_steps = 2
        torch.manual_seed(0)
        traj = torch.randn(B, T_steps, N, C)
        return {"trajectory": traj, "co2massmix": traj.clone()}


class _MockDataset:
    """Lightweight mock of CarbonDataset."""

    def __init__(self, n_samples=10, nlat=4, nlon=8, nlev=5):
        self.nlat = nlat
        self.nlon = nlon
        self.nlev = nlev
        self._n = n_samples
        N = nlat * nlon

        times = pd.date_range("2019-01-01", periods=n_samples, freq="6h")
        self.ds = xr.Dataset(coords={"time": times})

        torch.manual_seed(42)
        self._template = {
            "co2massmix": torch.randn(1, N, nlev),
            "gph_bottom": torch.randn(1, N, nlev),
            "gph_top": torch.randn(1, N, nlev),
            "co2flux_anthro": torch.randn(1, N, 1),
            "co2flux_land": torch.randn(1, N, 1),
            "co2flux_ocean": torch.randn(1, N, 1),
        }

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {k: v.clone() for k, v in self._template.items()}

    def create_prototype_zarr(self, path, target_vars_3d=None, target_vars_2d=None, grid=None):
        return xr.Dataset(coords={"time": self.ds.time.values[: self._n]})

    def tensor_to_xarray(self, tensor):
        data = tensor.detach().cpu().numpy()
        if data.ndim == 4:
            return xr.DataArray(data, dims=["batch", "time", "cell", "level"])
        elif data.ndim == 3:
            return xr.DataArray(data, dims=["batch", "cell", "level"])
        return xr.DataArray(data)

    def readout_stations(self, ds, grid=None):
        return xr.Dataset()


@pytest.fixture
def mock_model():
    return _MockModel(nlat=4, nlon=8, nlev=5)


@pytest.fixture
def mock_dataset():
    return _MockDataset(n_samples=10, nlat=4, nlon=8, nlev=5)


@pytest.fixture
def mock_dataset_gen():
    """Second dataset for timeseries/OCO-2 mode."""
    return _MockDataset(n_samples=10, nlat=4, nlon=8, nlev=5)


# ── Utility function tests ──────────────────────────────────────────────


class TestParseFreq:
    def test_hours(self):
        assert parse_freq("6h") == 6
        assert parse_freq("3h") == 3
        assert parse_freq("1h") == 1

    def test_days(self):
        assert parse_freq("1D") == 24
        assert parse_freq("2D") == 48

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_freq("3m")


class TestIsBadSample:
    def test_good_sample(self):
        assert not is_bad_sample(np.array([1.0, 2.0, 3.0]))

    def test_nan(self):
        assert is_bad_sample(np.array([1.0, np.nan, 3.0]))

    def test_inf(self):
        assert is_bad_sample(np.array([1.0, np.inf, 3.0]))

    def test_large_values(self):
        assert is_bad_sample(np.array([1e7]))

    def test_custom_threshold(self):
        assert not is_bad_sample(np.array([1e7]), thresh=1e8)


class TestAlignTime:
    def test_same_start(self):
        t = np.array([1, 2, 3])
        assert align_time(t, t) == 0

    def test_time_starts_after_gen(self):
        time = np.array([3, 4, 5])
        time_gen = np.array([1, 2, 3, 4, 5])
        assert align_time(time, time_gen) == 2

    def test_time_starts_before_gen(self):
        time = np.array([1, 2, 3, 4, 5])
        time_gen = np.array([3, 4, 5])
        assert align_time(time, time_gen) == -2

    def test_no_overlap_before(self):
        with pytest.raises(ValueError, match="No overlap"):
            align_time(np.array([1, 2, 3]), np.array([5, 6, 7]))

    def test_no_overlap_after(self):
        with pytest.raises(ValueError, match="No overlap"):
            align_time(np.array([5, 6, 7]), np.array([1, 2, 3]))


class TestGetZarrpathObspath:
    def test_default_rollout(self, tmp_path):
        zarr, obs = get_zarrpath_obspath(tmp_path, rollout=True, freq="6h")
        assert zarr.name == "co2_pred_rollout_6h.zarr"
        assert obs.name == "obs_co2_pred_rollout_6h.zarr"

    def test_default_singlestep(self, tmp_path):
        zarr, obs = get_zarrpath_obspath(tmp_path, rollout=False, freq="6h")
        assert zarr.name == "co2_pred_singlestep.zarr"

    def test_custom_filename(self, tmp_path):
        zarr, _ = get_zarrpath_obspath(tmp_path, rollout=True, freq="6h", zarr_filename="custom.zarr")
        assert zarr.name == "custom.zarr"

    def test_zero_surfflux(self, tmp_path):
        zarr, _ = get_zarrpath_obspath(tmp_path, rollout=True, freq="6h", zero_surfflux=True)
        assert "zeroflux" in zarr.name

    def test_creates_dir(self, tmp_path):
        sub = tmp_path / "sub" / "dir"
        get_zarrpath_obspath(sub, rollout=True, freq="6h")
        assert sub.exists()


class TestGetBatches:
    def test_basic(self, mock_dataset, mock_dataset_gen):
        batch, batch_gen = get_batches(0, 0, mock_dataset, mock_dataset_gen, window_steps=1, device="cpu")
        assert "co2massmix" in batch
        assert "co2massmix" in batch_gen
        # Tensors should have batch dim
        assert batch["co2massmix"].ndim == 4  # [1, T, N, C]

    def test_window_aggregation(self, mock_dataset, mock_dataset_gen):
        batch, batch_gen = get_batches(0, 0, mock_dataset, mock_dataset_gen, window_steps=3, device="cpu")
        assert "co2massmix" in batch
        assert batch["co2massmix"].shape[0] == 1  # B=1 after mean

    def test_positive_offset(self, mock_dataset, mock_dataset_gen):
        batch, batch_gen = get_batches(0, 2, mock_dataset, mock_dataset_gen, window_steps=1, device="cpu")
        assert "co2massmix" in batch
        assert "co2massmix" in batch_gen


# ── GenerationPipeline tests ────────────────────────────────────────────


class TestGenerationPipelineInit:
    def test_basic_construction(self, mock_model, mock_dataset):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        assert pipeline.nlat == 4
        assert pipeline.nlon == 8
        assert pipeline.mode == "sample"

    def test_timeseries_mode(self, mock_model, mock_dataset, mock_dataset_gen):
        pipeline = GenerationPipeline(mock_model, mock_dataset, dataset_gen=mock_dataset_gen, device="cpu")
        assert pipeline.mode == "timeseries"

    def test_default_target_vars(self, mock_model, mock_dataset):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        assert pipeline.target_vars_3d == ["co2massmix"]
        assert pipeline.target_vars_2d == []

    def test_custom_target_vars(self, mock_model, mock_dataset):
        pipeline = GenerationPipeline(
            mock_model,
            mock_dataset,
            target_vars_3d=["co2massmix", "co2density"],
            target_vars_2d=["xco2_obs"],
            device="cpu",
        )
        assert pipeline.target_vars_3d == ["co2massmix", "co2density"]
        assert pipeline.target_vars_2d == ["xco2_obs"]


@pytest.mark.quick
class TestGenerationPipelineRunSample:
    """Test sample mode (no dataset_gen → iterates over samples)."""

    def test_produces_output(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=3,
            masking=False,
        )
        assert isinstance(ds, xr.Dataset)
        assert "co2massmix" in ds
        assert "sample" in ds.dims

    def test_sample_count(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=5,
            masking=False,
        )
        assert ds.sizes["sample"] == 5

    def test_zarr_saved(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
        )
        zarr_files = list(tmp_path.glob("*.zarr"))
        assert len(zarr_files) >= 1

    def test_unconditional_no_obs_mask(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
        )
        assert "obs_mask" not in ds
        assert "obs_values" not in ds

    def test_zero_surfflux(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
            zero_surfflux=True,
        )
        assert isinstance(ds, xr.Dataset)

    def test_no_nan_in_output(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
        )
        assert not np.isnan(ds["co2massmix"].values).any()


@pytest.mark.quick
class TestGenerationPipelineRunTimeseries:
    """Test timeseries mode (dataset_gen provided → iterates over timesteps)."""

    def test_produces_output(self, mock_model, mock_dataset, mock_dataset_gen, tmp_path):
        pipeline = GenerationPipeline(
            mock_model,
            mock_dataset,
            dataset_gen=mock_dataset_gen,
            device="cpu",
        )
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
            mask_pattern="vertical",
            generate_data_kwargs={"freq": "6h", "forcing_vars": ["co2massmix"]},
        )
        assert isinstance(ds, xr.Dataset)
        assert "co2massmix" in ds
        assert "time" in ds.dims

    def test_sample_dim_present(self, mock_model, mock_dataset, mock_dataset_gen, tmp_path):
        pipeline = GenerationPipeline(
            mock_model,
            mock_dataset,
            dataset_gen=mock_dataset_gen,
            device="cpu",
        )
        ds = pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=3,
            masking=False,
            mask_pattern="vertical",
            generate_data_kwargs={"freq": "6h", "forcing_vars": ["co2massmix"]},
        )
        assert "sample" in ds.dims
        assert ds.sizes["sample"] == 3

    def test_zarr_saved(self, mock_model, mock_dataset, mock_dataset_gen, tmp_path):
        pipeline = GenerationPipeline(
            mock_model,
            mock_dataset,
            dataset_gen=mock_dataset_gen,
            device="cpu",
        )
        pipeline.run(
            tmp_path,
            freq="6h",
            rollout=True,
            save_obs=False,
            n_samples=2,
            masking=False,
            mask_pattern="vertical",
            generate_data_kwargs={"freq": "6h", "forcing_vars": ["co2massmix"]},
        )
        zarr_files = list(tmp_path.glob("*.zarr"))
        assert len(zarr_files) >= 1


@pytest.mark.quick
class TestGenerationPipelineRunDistributional:
    """Test distributional evaluation mode."""

    def test_returns_two_datasets(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt_ds, gen_ds = pipeline.run_distributional(
            tmp_path,
            n_gt_samples=5,
            n_gen_samples=8,
            batch_size=4,
        )
        assert isinstance(gt_ds, xr.Dataset)
        assert isinstance(gen_ds, xr.Dataset)

    def test_gt_sample_count(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt_ds, gen_ds = pipeline.run_distributional(
            tmp_path,
            n_gt_samples=5,
            n_gen_samples=8,
            batch_size=4,
        )
        assert gt_ds.sizes["sample"] == 5

    def test_gen_sample_count(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt_ds, gen_ds = pipeline.run_distributional(
            tmp_path,
            n_gt_samples=5,
            n_gen_samples=8,
            batch_size=4,
        )
        assert gen_ds.sizes["sample"] == 8

    def test_has_target_var(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt_ds, gen_ds = pipeline.run_distributional(
            tmp_path,
            n_gt_samples=3,
            n_gen_samples=4,
            batch_size=4,
        )
        assert "co2massmix" in gt_ds
        assert "co2massmix" in gen_ds

    def test_has_spatial_dims(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt_ds, gen_ds = pipeline.run_distributional(
            tmp_path,
            n_gt_samples=3,
            n_gen_samples=4,
            batch_size=4,
        )
        assert "lat" in gt_ds.dims
        assert "lon" in gt_ds.dims
        assert "level" in gt_ds.dims

    def test_files_saved(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        pipeline.run_distributional(
            tmp_path,
            n_gt_samples=3,
            n_gen_samples=4,
            batch_size=4,
        )
        assert (tmp_path / "gt_pool.nc").exists()
        assert (tmp_path / "gen_pool.nc").exists()

    def test_reproducible_with_seed(self, mock_model, mock_dataset, tmp_path):
        pipeline = GenerationPipeline(mock_model, mock_dataset, device="cpu")
        gt1, gen1 = pipeline.run_distributional(
            tmp_path / "a",
            n_gt_samples=3,
            n_gen_samples=4,
            batch_size=4,
            seed=42,
        )
        gt2, gen2 = pipeline.run_distributional(
            tmp_path / "b",
            n_gt_samples=3,
            n_gen_samples=4,
            batch_size=4,
            seed=42,
        )
        np.testing.assert_array_equal(gt1["co2massmix"].values, gt2["co2massmix"].values)


# ── Backward-compatibility wrapper tests ─────────────────────────────────


class TestBackwardCompatWrappers:
    """Test that legacy function API still works."""

    def test_iterative_generate_wrapper(self, mock_model, mock_dataset, tmp_path):
        from neural_transport.inference.generation import iterative_generate

        ds = iterative_generate(
            mock_model,
            mock_dataset,
            tmp_path,
            rollout=True,
            device="cpu",
            freq="6h",
            save_obs=False,
            n_samples=2,
            masking=False,
        )
        assert isinstance(ds, xr.Dataset)
        assert "sample" in ds.dims

    def test_iterative_generate_oco2_wrapper(self, mock_model, mock_dataset, mock_dataset_gen, tmp_path):
        from neural_transport.inference.generation import iterative_generate_oco2

        ds = iterative_generate_oco2(
            mock_model,
            mock_dataset,
            mock_dataset_gen,
            tmp_path,
            rollout=True,
            device="cpu",
            freq="6h",
            save_obs=False,
            n_samples=2,
            masking=False,
            mask_pattern="vertical",
            generate_data_kwargs={"freq": "6h", "forcing_vars": ["co2massmix"]},
        )
        assert isinstance(ds, xr.Dataset)
        assert "time" in ds.dims

    def test_generate_for_distributional_eval_wrapper(self, mock_model, mock_dataset, tmp_path):
        from neural_transport.inference.generation import generate_for_distributional_eval

        gt_ds, gen_ds = generate_for_distributional_eval(
            mock_model,
            mock_dataset,
            tmp_path,
            n_gt_samples=3,
            n_gen_samples=4,
            device="cpu",
            batch_size=4,
        )
        assert isinstance(gt_ds, xr.Dataset)
        assert isinstance(gen_ds, xr.Dataset)
