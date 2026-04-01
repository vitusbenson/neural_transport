"""AblationRunner — shared experiment runner for carbonbench ablation studies.

Absorbs ~85% of the boilerplate duplicated across 5 run_ablation.py files.
Each experiment reduces to ~50 lines: sampler defaults + ablation config dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_transport.data.inference_loader import GridInfo

logger = logging.getLogger(__name__)

from neural_transport.configs import (
    DataConfig,
    EvalConfig,
    GenerateConfig,
    PlotConfig,
    compat_to_generate_kwargs,
)
from neural_transport.evaluation.suite import EvalResult, EvaluationSuite

# Lazy import for backward-compatible scoring
from neural_transport.inference.analyse import compute_score_df_generate
from neural_transport.inference.generation import GenerationPipeline
from neural_transport.training.train import load_model as train_load_model

DEFAULT_TRAINING_DATA_ROOT = "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/Carbontracker"
DEFAULT_FORECAST_DATA_ROOT = DEFAULT_TRAINING_DATA_ROOT + "/test"

DEFAULT_SUMMARY_COLUMNS = [
    ("RMSE_3D", "RMSE_3D_co2molemix"),
    ("RelRMSE", "RelRMSE_3D_co2molemix"),
    ("Mass_RMSE", "Mass_RMSE"),
    ("R2", "R2_3D_co2molemix"),
]


class AblationRunner:
    """Shared experiment runner for carbonbench ablation studies.

    Absorbs model loading, data setup, evaluation loops, distributional eval,
    and summary table printing. Each experiment only defines sampler defaults
    and an ablation config dict.
    """

    def __init__(
        self,
        experiment_dir: Path,
        data_config: DataConfig,
        model_dirs: list[Path],
        *,
        eval_config: EvalConfig | None = None,
        plot_config: PlotConfig | None = None,
        device: str = "cuda",
        target_vars: list[str] | None = None,
        data_path: str | Path | None = None,
        summary_columns: list[tuple[str, str]] | None = None,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.data_config = data_config
        self.model_dirs = [Path(d) for d in model_dirs]
        self.eval_config = eval_config or EvalConfig()
        self.plot_config = plot_config or PlotConfig()
        self.device = device
        self.target_vars = target_vars or ["co2massmix"]
        self.data_path = Path(data_path) if data_path else Path(DEFAULT_FORECAST_DATA_ROOT)
        self.summary_columns = summary_columns or DEFAULT_SUMMARY_COLUMNS

        # Cached model/dataset (set by run_ablation or externally for testing)
        self._cached_model = None
        self._cached_dataset = None

    # ── Model loading ─────────────────────────────────────────────────────

    def load_model(self) -> Any:
        """Load best model, iterating through model_dirs as fallbacks."""
        if self._cached_model is not None:
            return self._cached_model

        for exp_dir in self.model_dirs:
            try:
                model = train_load_model(exp_dir, ckpt="best", device=self.device)
                logger.info("Loaded model from %s", exp_dir)
                self._cached_model = model
                return model
            except Exception as e:
                logger.warning("Could not load from %s: %s", exp_dir, e)

        raise RuntimeError(f"No model checkpoint found in any of: {[str(d) for d in self.model_dirs]}")

    # ── Dataset loading ───────────────────────────────────────────────────

    def _load_dataset(self) -> Any:
        """Load CarbonDataset for inference."""
        if self._cached_dataset is not None:
            return self._cached_dataset

        from neural_transport.training.train import load_dataset

        data_kwargs = self.data_config.to_dict()
        dataset = load_dataset(self.data_path, data_kwargs, load_obspack=False)
        self._cached_dataset = dataset
        return dataset

    def _get_grid_info(self) -> GridInfo:
        """Get GridInfo from data config."""
        from neural_transport.data.inference_loader import GridInfo

        return GridInfo.from_config(self.data_config)

    # ── Single evaluation ─────────────────────────────────────────────────

    def run_single_eval(
        self,
        config: GenerateConfig,
        name: str,
        *,
        model=None,
        dataset=None,
        out_dir: Path | None = None,
    ) -> EvalResult:
        """Run a single ablation config: generate samples, compute metrics.

        Parameters
        ----------
        config : GenerateConfig
            Full generation config (after merge with overrides).
        name : str
            Config name for output directory and metadata.
        model : optional
            Pre-loaded model (avoids reloading per config).
        dataset : optional
            Pre-loaded dataset.
        out_dir : Path, optional
            Output directory. Defaults to experiment_dir/results.

        Returns
        -------
        EvalResult with backward-compatible metrics in metadata["compat_metrics"].
        """
        if model is None:
            model = self.load_model()
        if dataset is None:
            dataset = self._load_dataset()
        if out_dir is None:
            out_dir = self.experiment_dir / "results"

        eval_dir = Path(out_dir) / name
        eval_dir.mkdir(parents=True, exist_ok=True)

        logger.info("--- Evaluating: %s ---", name)
        generate_kwargs = compat_to_generate_kwargs(config)

        import time as _time

        _t0 = _time.perf_counter()
        try:
            # 1. Generate samples via GenerationPipeline
            pipeline = GenerationPipeline(model, dataset, target_vars_3d=self.target_vars, device=self.device)
            ds_pred = pipeline.run(
                eval_dir,
                freq="QS",
                rollout=False,
                save_obs=False,
                **generate_kwargs,
            )

            # 2. Backward-compatible scoring via compute_score_df_generate
            import xarray as xr

            dc = self.data_config
            target_path = self.data_path / f"carbontracker_{dc.grid}_{dc.vertical_levels}_{dc.freq}.zarr"

            if target_path.exists():
                _raw = xr.open_zarr(str(target_path))
                co2targ = xr.merge(
                    [
                        _raw.variables_2d.to_dataset("vari_2d"),
                        _raw.variables_3d.to_dataset("vari_3d"),
                    ]
                ).compute()

                pred_zarrs = sorted(eval_dir.glob("co2_pred_*.zarr"))
                if pred_zarrs:
                    co2pred = xr.open_zarr(str(pred_zarrs[0])).compute()
                    df_full, _, _ = compute_score_df_generate(
                        co2targ, co2pred, target_var=self.target_vars[0], **generate_kwargs
                    )
                    compat_metrics = df_full.loc["mean"].to_dict()
                else:
                    compat_metrics = {"error": f"No prediction zarr in {eval_dir}"}
            else:
                # Target zarr not available (e.g., in tests); skip compat scoring
                compat_metrics = {}

            compat_metrics["wall_time_sec"] = _time.perf_counter() - _t0

            # 3. Save metrics JSON
            metrics_path = eval_dir / "metrics_summary.json"
            with open(metrics_path, "w") as f:
                json.dump(compat_metrics, f, indent=2, default=str)

            # 4. Try ensemble eval via EvaluationSuite if we have numpy arrays
            from neural_transport.inference.osse_runner import _extract_samples

            grid_info = self._get_grid_info()
            eval_result = EvalResult(
                metadata={
                    "config_name": name,
                    "generate_config": config.to_dict(),
                    "compat_metrics": compat_metrics,
                },
            )

            try:
                samples = _extract_samples(ds_pred, grid_info.nlat, grid_info.nlon)
                gt_field = dataset[0][self.target_vars[0]].numpy()
                if gt_field.ndim == 3:
                    gt_field = gt_field[0]
                gt_2d = gt_field.reshape(grid_info.nlat, grid_info.nlon, -1)

                suite = EvaluationSuite(self.eval_config)
                # Reshape lat weights to broadcast with [nlat, nlon, nlev] data
                lat_w = grid_info.cos_lat_weights_2d
                if gt_2d.ndim == 3 and lat_w.ndim == 2:
                    lat_w = lat_w[:, :, None]
                eval_result = suite.evaluate_ensemble(
                    samples,
                    gt_2d,
                    lat_weights=lat_w,
                    metadata={
                        "config_name": name,
                        "generate_config": config.to_dict(),
                        "compat_metrics": compat_metrics,
                    },
                )
            except Exception as e:
                logger.warning("EvaluationSuite failed (%s), using compat metrics only", e)

            logger.info("Metrics saved to %s", metrics_path)
            for k, v in compat_metrics.items():
                if isinstance(v, int | float):
                    logger.info("  %s: %.4f", k, v)

            return eval_result

        except Exception as e:
            logger.error("Error in %s: %s", name, e, exc_info=True)
            return EvalResult(
                metadata={
                    "config_name": name,
                    "compat_metrics": {"error": str(e)},
                },
            )

    # ── Ablation sweep ────────────────────────────────────────────────────

    def run_ablation(
        self,
        base_config: GenerateConfig,
        ablation_configs: dict[str, dict],
        *,
        filter_str: str | None = None,
    ) -> dict[str, EvalResult]:
        """Run ablation sweep over all configs.

        Parameters
        ----------
        base_config : GenerateConfig
            Base configuration. Each ablation overrides specific fields.
        ablation_configs : dict[str, dict]
            Maps config name to override dict.
        filter_str : str, optional
            If given, only run configs whose name contains this substring.

        Returns
        -------
        dict mapping config name to EvalResult.
        """
        model = self.load_model()
        dataset = self._load_dataset()

        # Filter configs
        if filter_str is not None:
            configs = {k: v for k, v in ablation_configs.items() if filter_str in k}
            if not configs:
                warnings.warn(f"No configs matching '{filter_str}'. Available: {list(ablation_configs.keys())}")
                return {}
        else:
            configs = ablation_configs

        out_dir = self.experiment_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, EvalResult] = {}
        for name, overrides in configs.items():
            merged = base_config.merge(**overrides)
            result = self.run_single_eval(merged, name, model=model, dataset=dataset, out_dir=out_dir)
            results[name] = result

        # Save summary and print table
        self.save_results(results)
        self.print_summary_table(results, columns=self.summary_columns)

        return results

    # ── Distributional evaluation ─────────────────────────────────────────

    def run_distributional_eval(
        self,
        config: GenerateConfig,
        *,
        n_gt_samples: int = 50,
        n_gen_samples: int = 200,
    ) -> EvalResult:
        """Run distributional evaluation (unconditional generation).

        Returns EvalResult with distributional metrics populated.
        """
        model = self.load_model()
        dataset = self._load_dataset()

        out_dir = self.experiment_dir / "results" / "distributional_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_kwargs = compat_to_generate_kwargs(config)

        logger.info("--- Running Distributional Evaluation ---")
        pipeline = GenerationPipeline(model, dataset, target_vars_3d=self.target_vars, device=self.device)
        gt_ds, gen_ds = pipeline.run_distributional(
            out_dir,
            n_gt_samples=n_gt_samples,
            n_gen_samples=n_gen_samples,
            generate_kwargs=generate_kwargs,
            batch_size=20,
        )

        # Compute distributional metrics
        target_var = self.target_vars[0]
        gt_fields = gt_ds[target_var].values
        gen_fields = gen_ds[target_var].values
        lat_vals = gt_ds.lat.values
        lon_vals = gt_ds.lon.values

        suite = EvaluationSuite(self.eval_config)
        result = suite.evaluate_distributional(
            gt_fields,
            gen_fields,
            lat_vals,
            lon_vals,
        )

        # Save metrics
        result.to_json(out_dir / "distributional_eval_result.json")

        # Also save CSV for backward compatibility
        try:
            from neural_transport.inference.analyse import compute_distributional_score_df

            df = compute_distributional_score_df(gt_ds, gen_ds, target_var=target_var)
            df.to_csv(out_dir / "distributional_metrics.csv", index=False)
        except Exception as e:
            logger.warning("CSV export failed: %s", e)

        logger.info("Distributional eval saved to %s", out_dir)
        return result

    # ── Save results ──────────────────────────────────────────────────────

    def save_results(self, results: dict[str, EvalResult]) -> Path:
        """Save ablation_summary.json and per-config eval_result.json."""
        out_dir = self.experiment_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build summary from compat_metrics
        summary: dict[str, Any] = {}
        for name, result in results.items():
            compat = result.metadata.get("compat_metrics", {})
            if compat:
                summary[name] = compat
            else:
                summary[name] = result.to_flat_dict()

        summary_path = out_dir / "ablation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Per-config eval_result.json
        for name, result in results.items():
            config_dir = out_dir / name
            config_dir.mkdir(parents=True, exist_ok=True)
            result.to_json(config_dir / "eval_result.json")

        return out_dir

    # ── Plot results ──────────────────────────────────────────────────────

    def plot_results(
        self,
        results: dict[str, EvalResult | dict],
        *,
        custom_plots: list[Any] | None = None,
    ) -> None:
        """Generate standard plots for ablation results.

        Parameters
        ----------
        results : dict mapping config name to EvalResult or metrics dict.
        custom_plots : list of callables(results, plot_dir), optional.
        """
        from neural_transport.plots.base import PlotContext, run_plots
        from neural_transport.plots.metrics_plots import plot_summary_bars

        plot_dir = self.experiment_dir / "results" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Standard per-config plots via run_plots
        ctx = PlotContext(PlotConfig(save_dir=str(plot_dir)))
        for name, result in results.items():
            if isinstance(result, EvalResult):
                try:
                    run_plots(result, ctx)
                except Exception as e:
                    logger.warning("Plots for %s failed: %s", name, e)

        # Summary bar charts across configs
        metrics_dict = {}
        for name, result in results.items():
            if isinstance(result, EvalResult):
                compat = result.metadata.get("compat_metrics", {})
                if compat and "error" not in compat:
                    metrics_dict[name] = compat
            elif isinstance(result, dict) and "error" not in result:
                metrics_dict[name] = result

        if metrics_dict:
            for col_label, col_key in self.summary_columns:
                try:
                    plot_summary_bars(
                        metrics_dict,
                        col_key,
                        plot_dir,
                        title=f"Ablation: {col_label}",
                    )
                except Exception as e:
                    logger.warning("Summary bar plot for %s failed: %s", col_key, e)

        # Custom plots
        if custom_plots:
            for plot_fn in custom_plots:
                try:
                    plot_fn(results, plot_dir)
                except Exception as e:
                    logger.warning("Custom plot failed: %s", e)

        logger.info("Plots saved to %s", plot_dir)

    # ── Print summary table ───────────────────────────────────────────────

    @staticmethod
    def print_summary_table(
        results: dict[str, EvalResult],
        columns: list[tuple[str, str]] | None = None,
    ) -> None:
        """Print a formatted summary table of ablation results.

        Parameters
        ----------
        results : dict mapping config name to EvalResult.
        columns : list of (display_name, metric_key) tuples.
        """
        if columns is None:
            columns = DEFAULT_SUMMARY_COLUMNS

        # Build full table as single string to avoid timestamp-per-line disruption
        col_width = 12
        header = f"{'Config':<30}"
        for label, _ in columns:
            header += f" {label:>{col_width}}"
        sep = "=" * len(header)

        lines = [sep, header, "-" * len(header)]

        for name, result in results.items():
            compat = result.metadata.get("compat_metrics", {})
            if isinstance(compat, dict) and "error" not in compat:
                row = f"  {name:<28}"
                for _, key in columns:
                    val = compat.get(key)
                    if val is not None and isinstance(val, int | float):
                        row += f" {val:>{col_width}.4f}"
                    else:
                        row += f" {'N/A':>{col_width}}"
                lines.append(row)
            else:
                error_msg = compat.get("error", "unknown error") if isinstance(compat, dict) else str(compat)
                lines.append(f"  {name:<28} ERROR: {error_msg}")

        lines.append(sep)
        logger.info("\n%s", "\n".join(lines))

    # ── CLI entry point ───────────────────────────────────────────────────

    def main_cli(
        self,
        base_config: GenerateConfig,
        ablation_configs: dict[str, dict],
        *,
        args: list[str] | None = None,
    ) -> None:
        """Parse CLI args and dispatch to run_ablation / run_distributional_eval.

        Parameters
        ----------
        base_config : GenerateConfig
            Base generation config for this experiment.
        ablation_configs : dict[str, dict]
            Maps config name to override dict.
        args : list[str], optional
            CLI args (for testing). If None, uses sys.argv.
        """
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
        parser = argparse.ArgumentParser(description="Ablation experiment runner")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument(
            "--filter",
            type=str,
            default=None,
            help="Filter configs by substring",
        )
        parser.add_argument("--out-dir", type=str, default=None)
        parser.add_argument(
            "--dist-eval-only",
            action="store_true",
            help="Only run distributional evaluation",
        )
        parser.add_argument(
            "--plot-only",
            action="store_true",
            help="Load saved results and plot only",
        )
        parsed = parser.parse_args(args)

        self.device = parsed.device
        if parsed.out_dir:
            self.experiment_dir = Path(parsed.out_dir)

        if parsed.plot_only:
            # Load from saved ablation_summary.json
            from neural_transport.plots.metrics_plots import load_ablation_results

            results_dir = self.experiment_dir / "results"
            raw = load_ablation_results(results_dir)
            # Wrap in EvalResult for plot_results
            results = {name: EvalResult(metadata={"compat_metrics": metrics}) for name, metrics in raw.items()}
            self.plot_results(results)
        elif parsed.dist_eval_only:
            dist_config = base_config.merge(**{"conditioning.masking": False})
            self.run_distributional_eval(dist_config)
        else:
            self.run_ablation(
                base_config,
                ablation_configs,
                filter_str=parsed.filter,
            )
            # Also run distributional eval
            dist_config = base_config.merge(**{"conditioning.masking": False})
            self.run_distributional_eval(dist_config)
