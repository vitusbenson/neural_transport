"""Tests for AblationRunner — Phase 16 experiment runner deduplication.

TDD: tests written first, implementation follows.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from neural_transport.configs import DataConfig, EvalConfig, GenerateConfig, PlotConfig

# ── Mock objects ──────────────────────────────────────────────────────────


class _MockFlowMatchingModel:
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


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    return _MockModel(nlat=4, nlon=8, nlev=5)


@pytest.fixture
def mock_dataset():
    return _MockDataset(n_samples=10, nlat=4, nlon=8, nlev=5)


@pytest.fixture
def data_config():
    return DataConfig(
        dataset="carbontracker",
        grid="latlon5.625",
        vertical_levels="l10",
        freq="6h",
        target_vars=["co2massmix"],
        forcing_vars=[],
    )


@pytest.fixture
def runner(tmp_path, data_config):
    from neural_transport.experiments.ablation_runner import AblationRunner

    return AblationRunner(
        experiment_dir=tmp_path / "experiment",
        data_config=data_config,
        model_dirs=[tmp_path / "model_dir1", tmp_path / "model_dir2"],
        device="cpu",
        data_path=tmp_path / "data",
    )


# ── TestAblationRunnerInit ────────────────────────────────────────────────


class TestAblationRunnerInit:
    def test_stores_experiment_dir_and_model_dirs(self, runner, tmp_path):
        assert runner.experiment_dir == tmp_path / "experiment"
        assert len(runner.model_dirs) == 2

    def test_creates_default_configs(self, runner):
        assert isinstance(runner.eval_config, EvalConfig)
        assert isinstance(runner.plot_config, PlotConfig)

    def test_accepts_custom_configs(self, tmp_path, data_config):
        from neural_transport.experiments.ablation_runner import AblationRunner

        eval_cfg = EvalConfig(n_gt_samples=10, n_gen_samples=20)
        plot_cfg = PlotConfig(dpi=300, imgformats=["png"])
        r = AblationRunner(
            experiment_dir=tmp_path,
            data_config=data_config,
            model_dirs=[tmp_path],
            eval_config=eval_cfg,
            plot_config=plot_cfg,
            device="cpu",
        )
        assert r.eval_config.n_gt_samples == 10
        assert r.plot_config.dpi == 300


# ── TestLoadModel ─────────────────────────────────────────────────────────


class TestLoadModel:
    def test_first_dir_succeeds(self, runner, mock_model):
        with patch(
            "neural_transport.experiments.ablation_runner.train_load_model",
            return_value=mock_model,
        ):
            model = runner.load_model()
            assert model is mock_model

    def test_first_fails_second_succeeds(self, runner, mock_model):
        call_count = 0

        def _side_effect(exp_dir, ckpt="best", device="cpu"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FileNotFoundError("No checkpoint")
            return mock_model

        with patch(
            "neural_transport.experiments.ablation_runner.train_load_model",
            side_effect=_side_effect,
        ):
            model = runner.load_model()
            assert model is mock_model
            assert call_count == 2

    def test_all_fail_raises(self, runner):
        with patch(
            "neural_transport.experiments.ablation_runner.train_load_model",
            side_effect=FileNotFoundError("No checkpoint"),
        ):
            with pytest.raises(RuntimeError, match="No model checkpoint found"):
                runner.load_model()


# ── TestRunSingleEval ─────────────────────────────────────────────────────


class TestRunSingleEval:
    def test_returns_eval_result_with_pointwise(self, runner, mock_model, mock_dataset, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        out_dir = tmp_path / "eval_out"

        # Mock the pipeline run and scoring
        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(5, 32, 5))},
            coords={"sample": np.arange(5)},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5, "R2_3D_co2molemix": 0.9, "Mass_RMSE": 10.0}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            config = GenerateConfig(n_samples=5, steps=11)
            result = runner.run_single_eval(
                config, "test_config", model=mock_model, dataset=mock_dataset, out_dir=out_dir
            )

        assert isinstance(result, EvalResult)
        assert "config_name" in result.metadata

    def test_returns_eval_result_with_ensemble(self, runner, mock_model, mock_dataset, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        out_dir = tmp_path / "eval_out"

        # Create mock prediction dataset with proper shape for _extract_samples
        n_samples, nlat, nlon, nlev = 5, 4, 8, 5
        mock_ds = xr.Dataset(
            {
                "co2massmix": (
                    ("sample", "cell", "level"),
                    np.random.randn(n_samples, nlat * nlon, nlev),
                )
            },
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5, "R2_3D_co2molemix": 0.9}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            config = GenerateConfig(n_samples=5, steps=11)
            result = runner.run_single_eval(
                config, "test_config", model=mock_model, dataset=mock_dataset, out_dir=out_dir
            )

        assert isinstance(result, EvalResult)

    def test_saves_eval_result_json(self, runner, mock_model, mock_dataset, tmp_path):
        out_dir = tmp_path / "eval_out"

        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(3, 32, 5))},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            config = GenerateConfig(n_samples=3, steps=11)
            runner.run_single_eval(config, "test_config", model=mock_model, dataset=mock_dataset, out_dir=out_dir)

        assert (out_dir / "test_config" / "metrics_summary.json").exists()

    def test_stores_config_name_in_metadata(self, runner, mock_model, mock_dataset, tmp_path):
        out_dir = tmp_path / "eval_out"

        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(3, 32, 5))},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            config = GenerateConfig(n_samples=3, steps=11)
            result = runner.run_single_eval(
                config, "my_config", model=mock_model, dataset=mock_dataset, out_dir=out_dir
            )

        assert result.metadata["config_name"] == "my_config"


# ── TestRunAblation ───────────────────────────────────────────────────────


class TestRunAblation:
    def _make_runner_with_mocks(self, runner, mock_model, mock_dataset):
        """Patch load_model and _load_dataset to use mocks."""
        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset
        return runner

    def test_iterates_all_configs(self, runner, mock_model, mock_dataset, tmp_path):
        runner = self._make_runner_with_mocks(runner, mock_model, mock_dataset)
        base_config = GenerateConfig(n_samples=3, steps=11)
        ablation_configs = {
            "cfg_a": {"steps": 21},
            "cfg_b": {"steps": 51},
            "cfg_c": {},
        }

        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(3, 32, 5))},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            results = runner.run_ablation(base_config, ablation_configs)

        assert set(results.keys()) == {"cfg_a", "cfg_b", "cfg_c"}

    def test_filter_str_selects_matching(self, runner, mock_model, mock_dataset):
        runner = self._make_runner_with_mocks(runner, mock_model, mock_dataset)
        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {
            "sigma_0.1": {"steps": 11},
            "sigma_0.5": {"steps": 11},
            "steps_21": {"steps": 21},
        }

        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(3, 32, 5))},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            results = runner.run_ablation(base_config, ablation_configs, filter_str="sigma")

        assert set(results.keys()) == {"sigma_0.1", "sigma_0.5"}

    def test_filter_no_match_returns_empty(self, runner, mock_model, mock_dataset):
        runner = self._make_runner_with_mocks(runner, mock_model, mock_dataset)
        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {"cfg_a": {}, "cfg_b": {}}

        results = runner.run_ablation(base_config, ablation_configs, filter_str="nonexistent")
        assert results == {}

    def test_config_merge_applied(self, runner, mock_model, mock_dataset):
        runner = self._make_runner_with_mocks(runner, mock_model, mock_dataset)
        base_config = GenerateConfig(n_samples=3, steps=11)
        ablation_configs = {"fast": {"steps": 51}}

        merged_configs = []

        def _capture_config(config, name, **kwargs):
            merged_configs.append(config)
            return MagicMock()

        with (
            patch.object(runner, "run_single_eval", side_effect=_capture_config),
        ):
            runner.run_ablation(base_config, ablation_configs)

        assert len(merged_configs) == 1
        assert merged_configs[0].steps == 51

    def test_saves_ablation_summary_json(self, runner, mock_model, mock_dataset, tmp_path):
        runner = self._make_runner_with_mocks(runner, mock_model, mock_dataset)
        runner.experiment_dir = tmp_path
        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {"cfg_a": {}}

        mock_ds = xr.Dataset(
            {"co2massmix": (("sample", "cell", "level"), np.random.randn(3, 32, 5))},
        )
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.compute_score_df_generate") as mock_score,
        ):
            MockPipeline.return_value.run.return_value = mock_ds
            mock_df = pd.DataFrame([{"RMSE_3D_co2molemix": 0.5}])
            mock_df.loc["mean"] = mock_df.mean()
            mock_score.return_value = (mock_df, pd.DataFrame(), xr.Dataset())

            runner.run_ablation(base_config, ablation_configs)

        summary_path = runner.experiment_dir / "results" / "ablation_summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert "cfg_a" in data


# ── TestRunDistributionalEval ─────────────────────────────────────────────


class TestRunDistributionalEval:
    def test_returns_eval_result_with_distributional(self, runner, mock_model, mock_dataset, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset

        nlat, nlon, nlev = 4, 8, 5
        gt_ds = xr.Dataset(
            {"co2massmix": (("sample", "lat", "lon", "level"), np.random.randn(5, nlat, nlon, nlev))},
            coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(0, 360, nlon), "level": np.arange(nlev)},
        )
        gen_ds = xr.Dataset(
            {"co2massmix": (("sample", "lat", "lon", "level"), np.random.randn(10, nlat, nlon, nlev))},
            coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(0, 360, nlon), "level": np.arange(nlev)},
        )

        config = GenerateConfig(n_samples=3, steps=11)
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.EvaluationSuite") as MockSuite,
        ):
            MockPipeline.return_value.run_distributional.return_value = (gt_ds, gen_ds)
            mock_result = EvalResult(
                distributional={"energy_distance": 0.1, "mmd": 0.05},
            )
            MockSuite.return_value.evaluate_distributional.return_value = mock_result

            result = runner.run_distributional_eval(config)

        assert isinstance(result, EvalResult)
        assert result.distributional is not None

    def test_saves_distributional_json(self, runner, mock_model, mock_dataset, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset
        runner.experiment_dir = tmp_path

        nlat, nlon, nlev = 4, 8, 5
        gt_ds = xr.Dataset(
            {"co2massmix": (("sample", "lat", "lon", "level"), np.random.randn(5, nlat, nlon, nlev))},
            coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(0, 360, nlon), "level": np.arange(nlev)},
        )
        gen_ds = xr.Dataset(
            {"co2massmix": (("sample", "lat", "lon", "level"), np.random.randn(10, nlat, nlon, nlev))},
            coords={"lat": np.linspace(-90, 90, nlat), "lon": np.linspace(0, 360, nlon), "level": np.arange(nlev)},
        )

        config = GenerateConfig(n_samples=3, steps=11)
        with (
            patch("neural_transport.experiments.ablation_runner.GenerationPipeline") as MockPipeline,
            patch("neural_transport.experiments.ablation_runner.EvaluationSuite") as MockSuite,
        ):
            MockPipeline.return_value.run_distributional.return_value = (gt_ds, gen_ds)
            mock_result = EvalResult(
                distributional={"energy_distance": 0.1},
            )
            MockSuite.return_value.evaluate_distributional.return_value = mock_result

            runner.run_distributional_eval(config)

        dist_dir = tmp_path / "results" / "distributional_eval"
        json_files = list(dist_dir.glob("*.json"))
        assert len(json_files) >= 1


# ── TestSaveResults ───────────────────────────────────────────────────────


class TestSaveResults:
    def test_ablation_summary_is_valid_json(self, runner, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        runner.experiment_dir = tmp_path
        results = {
            "cfg_a": EvalResult(pointwise={"rmse": 0.5}, metadata={"compat_metrics": {"RMSE_3D_co2molemix": 0.5}}),
            "cfg_b": EvalResult(pointwise={"rmse": 0.3}, metadata={"compat_metrics": {"RMSE_3D_co2molemix": 0.3}}),
        }
        out_dir = runner.save_results(results)
        summary = json.loads((out_dir / "ablation_summary.json").read_text())
        assert "cfg_a" in summary
        assert "cfg_b" in summary

    def test_per_config_json_written(self, runner, tmp_path):
        from neural_transport.evaluation.suite import EvalResult

        runner.experiment_dir = tmp_path
        results = {
            "cfg_a": EvalResult(pointwise={"rmse": 0.5}, metadata={"compat_metrics": {"RMSE_3D_co2molemix": 0.5}}),
        }
        out_dir = runner.save_results(results)
        assert (out_dir / "cfg_a" / "eval_result.json").exists()


# ── TestPrintSummaryTable ─────────────────────────────────────────────────


class TestPrintSummaryTable:
    def test_default_columns_appear(self, caplog):
        import logging

        from neural_transport.evaluation.suite import EvalResult
        from neural_transport.experiments.ablation_runner import AblationRunner

        results = {
            "cfg_a": EvalResult(
                metadata={
                    "compat_metrics": {
                        "RMSE_3D_co2molemix": 0.5,
                        "R2_3D_co2molemix": 0.9,
                        "RelRMSE_3D_co2molemix": 0.01,
                        "Mass_RMSE": 10.0,
                    }
                }
            ),
        }
        with caplog.at_level(logging.INFO, logger="neural_transport.experiments.ablation_runner"):
            AblationRunner.print_summary_table(results)
        assert "RMSE_3D" in caplog.text
        assert "R2" in caplog.text

    def test_custom_columns(self, caplog):
        import logging

        from neural_transport.evaluation.suite import EvalResult
        from neural_transport.experiments.ablation_runner import AblationRunner

        results = {
            "cfg_a": EvalResult(metadata={"compat_metrics": {"custom_metric": 42.0}}),
        }
        columns = [("Custom", "custom_metric")]
        with caplog.at_level(logging.INFO, logger="neural_transport.experiments.ablation_runner"):
            AblationRunner.print_summary_table(results, columns=columns)
        assert "Custom" in caplog.text

    def test_missing_metrics_show_na(self, caplog):
        import logging

        from neural_transport.evaluation.suite import EvalResult
        from neural_transport.experiments.ablation_runner import AblationRunner

        results = {
            "cfg_a": EvalResult(metadata={"compat_metrics": {}}),
        }
        with caplog.at_level(logging.INFO, logger="neural_transport.experiments.ablation_runner"):
            AblationRunner.print_summary_table(results)
        assert "N/A" in caplog.text


# ── TestMainCli ───────────────────────────────────────────────────────────


class TestMainCli:
    def test_filter_arg_dispatches(self, runner, mock_model, mock_dataset):
        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset

        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {"cfg_a": {}, "cfg_b": {}}

        with (
            patch.object(runner, "run_ablation", return_value={}) as mock_run,
            patch.object(runner, "run_distributional_eval", return_value=MagicMock()),
        ):
            runner.main_cli(
                base_config,
                ablation_configs,
                args=["--filter", "cfg_a", "--device", "cpu"],
            )
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs[1].get("filter_str") == "cfg_a" or call_kwargs.kwargs.get("filter_str") == "cfg_a"

    def test_distributional_flag_dispatches(self, runner, mock_model, mock_dataset):
        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset

        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {"cfg_a": {}}

        with patch.object(runner, "run_distributional_eval", return_value=MagicMock()) as mock_dist:
            runner.main_cli(
                base_config,
                ablation_configs,
                args=["--dist-eval-only", "--device", "cpu"],
            )
            mock_dist.assert_called_once()

    def test_plot_only_loads_from_json(self, runner, tmp_path, mock_model, mock_dataset):
        runner._cached_model = mock_model
        runner._cached_dataset = mock_dataset
        runner.experiment_dir = tmp_path

        # Create a fake ablation_summary.json
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True)
        summary = {"cfg_a": {"RMSE_3D_co2molemix": 0.5}}
        (results_dir / "ablation_summary.json").write_text(json.dumps(summary))

        base_config = GenerateConfig(n_samples=3)
        ablation_configs = {"cfg_a": {}}

        with patch.object(runner, "plot_results") as mock_plot:
            runner.main_cli(
                base_config,
                ablation_configs,
                args=["--plot-only", "--device", "cpu"],
            )
            mock_plot.assert_called_once()
