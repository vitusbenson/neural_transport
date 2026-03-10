"""Orchestrates OSSE experiments: run inference, compute metrics, produce plots, save results.

Usage:
    results = run_osse_comparison(model, dataset, experiments, base_generate_kwargs, ...)
    # Automatically produces metrics_summary.json + all diagnostic plots.
"""

import copy
import json
from pathlib import Path

import numpy as np

from neural_transport.inference.generative import iterative_generate
from neural_transport.inference.metrics import (
    MetricsResult,
    OSSEResult,
    compute_all_metrics,
    _to_numpy,
)
from neural_transport.plots.conditioning_diagnostics import (
    plot_conditioning_comparison,
    plot_metrics_summary,
    plot_ensemble_diagnostics,
    plot_xco2_maps,
    plot_zonal_mean,
)


def _extract_mask_2d(ds_pred, nlat, nlon):
    """Extract 2D spatial mask from xarray Dataset.

    Handles both column masks (no level dim) and 3D masks (with level dim).

    Parameters
    ----------
    ds_pred : xarray.Dataset with optional 'obs_mask' variable
    nlat, nlon : int

    Returns
    -------
    np.ndarray [nlat, nlon] bool, or None if no mask present.
    """
    if "obs_mask" not in ds_pred:
        return None
    mask = ds_pred["obs_mask"]
    if "time" in mask.dims:
        mask = mask.isel(time=0)
    if "level" in mask.dims:
        mask_np = mask.isel(level=0).values.astype(bool)
    else:
        mask_np = mask.values.astype(bool).squeeze()
    if mask_np.ndim > 1:
        mask_np = mask_np.flatten()
    n_cells = nlat * nlon
    if mask_np.size > n_cells:
        mask_np = mask_np[:n_cells]
    return mask_np.reshape(nlat, nlon)


def _extract_samples(ds_pred, nlat, nlon):
    """Extract ensemble samples from xarray Dataset.

    Parameters
    ----------
    ds_pred : xarray.Dataset with 'co2massmix' variable
    nlat, nlon : int

    Returns
    -------
    np.ndarray [n_samples, nlat, nlon, nlev]
    """
    pred = ds_pred["co2massmix"]
    if "trajectory_steps" in pred.dims:
        pred = pred.isel(trajectory_steps=-1)
    if "time" in pred.dims:
        pred = pred.isel(time=0)

    pred_np = pred.values  # [sample, cell, level] or [sample, nlat, nlon, level]
    n_samples = pred_np.shape[0]
    if pred_np.ndim == 3:
        pred_np = pred_np.reshape(n_samples, nlat, nlon, -1)
    return pred_np


def run_single_osse(model, dataset, generate_kwargs, device="cuda",
                    nlat=32, nlon=64, gt_field=None,
                    pressure_weights=None, ak=None, lat=None,
                    name="unnamed", outpath=None, seed=42):
    """Run a single OSSE experiment.

    1. Deep-copy model
    2. Call iterative_generate() with generate_kwargs
    3. Extract samples, mask, GT via helper functions
    4. Call compute_all_metrics()
    5. Return OSSEResult

    Parameters
    ----------
    model : NeuralTransport lightning module
    dataset : dataset object compatible with iterative_generate
    generate_kwargs : dict, passed as **kwargs to iterative_generate
    device : str
    nlat, nlon : int, grid dimensions
    gt_field : np.ndarray [nlat, nlon, nlev], ground truth
    pressure_weights : np.ndarray [nlat, nlon, nlev], optional
    ak : np.ndarray [nlat, nlon, nlev], optional
    lat : np.ndarray [nlat], optional (for cos-lat weighting)
    name : str, experiment name
    outpath : str or Path, output directory for zarr files
    seed : int

    Returns
    -------
    OSSEResult
    """
    import pytorch_lightning as pl
    pl.seed_everything(seed)

    if outpath is None:
        outpath = Path(f"/tmp/osse_{name}")
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    model_copy = copy.deepcopy(model)

    print(f"\n{'=' * 60}")
    print(f"  Running OSSE: {name}")
    config_display = {k: v for k, v in generate_kwargs.items()
                      if k not in ("generate_data_kwargs", "data_path_generate")}
    print(f"  Config: {json.dumps(config_display, default=str, indent=4)}")
    print(f"{'=' * 60}")

    ds_all = iterative_generate(
        model_copy,
        dataset,
        outpath,
        rollout=False,
        device=device,
        verbose=True,
        freq="QS",
        zero_surfflux=False,
        remap=False,
        target_vars_3d=["co2massmix"],
        target_vars_2d=[],
        save_obs=False,
        **generate_kwargs,
    )

    # Extract arrays
    samples = _extract_samples(ds_all, nlat, nlon)
    mask_2d = _extract_mask_2d(ds_all, nlat, nlon)

    if gt_field is None:
        raise ValueError("gt_field must be provided for OSSE evaluation")

    # Compute all metrics
    metrics, extra_data = compute_all_metrics(
        samples, gt_field,
        mask_2d=mask_2d,
        pressure_weights=pressure_weights,
        ak=ak,
        lat=lat,
    )

    result = OSSEResult(
        name=name,
        config=config_display,
        metrics=metrics,
        samples=samples,
        ensemble_mean=samples.mean(axis=0),
        gt=gt_field,
        mask_2d=mask_2d,
        pressure_weights=pressure_weights,
        ak=ak,
        rank_hist=extra_data.get("rank_hist"),
        calibration_data=extra_data.get("calibration_data"),
        crps_map=extra_data.get("crps_map"),
        lat=lat,
    )

    print(f"  -> Metrics: {metrics.to_dict()}")
    return result


def run_osse_comparison(model, dataset, experiments, base_generate_kwargs,
                        device="cuda", out_dir="osse_results",
                        nlat=32, nlon=64, lat=None, lon=None,
                        pressure_weights=None, ak=None,
                        gt_field=None, seed=42):
    """Run multiple OSSE experiments, save results, and produce all plots.

    Parameters
    ----------
    model : NeuralTransport lightning module
    dataset : dataset object
    experiments : dict[str, dict], maps experiment name -> override kwargs
    base_generate_kwargs : dict, base kwargs for iterative_generate
    device : str
    out_dir : str or Path
    nlat, nlon : int
    lat, lon : np.ndarray, optional
    pressure_weights, ak : np.ndarray, optional
    gt_field : np.ndarray [nlat, nlon, nlev], ground truth
    seed : int

    Returns
    -------
    dict[str, OSSEResult]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if gt_field is None:
        raise ValueError("gt_field must be provided for OSSE comparison")

    results = {}

    for exp_name, overrides in experiments.items():
        merged_kwargs = copy.deepcopy(base_generate_kwargs)
        merged_kwargs.update(overrides)

        exp_outpath = out_dir / "predictions" / exp_name

        try:
            result = run_single_osse(
                model=model,
                dataset=dataset,
                generate_kwargs=merged_kwargs,
                device=device,
                nlat=nlat,
                nlon=nlon,
                gt_field=gt_field,
                pressure_weights=pressure_weights,
                ak=ak,
                lat=lat,
                name=exp_name,
                outpath=exp_outpath,
                seed=seed,
            )
            if lon is not None:
                result.lon = lon
            results[exp_name] = result
        except Exception as e:
            import traceback
            print(f"  !! FAILED: {exp_name}: {e}")
            traceback.print_exc()
            continue

    if not results:
        print("No experiments succeeded!")
        return results

    # Save metrics and produce plots
    save_osse_results(results, out_dir)

    plot_dir = out_dir / "plots"
    try:
        plot_conditioning_comparison(results, plot_dir)
    except Exception as e:
        print(f"  Warning: plot_conditioning_comparison failed: {e}")
    try:
        plot_metrics_summary(results, plot_dir)
    except Exception as e:
        print(f"  Warning: plot_metrics_summary failed: {e}")
    try:
        plot_ensemble_diagnostics(results, plot_dir)
    except Exception as e:
        print(f"  Warning: plot_ensemble_diagnostics failed: {e}")
    try:
        plot_xco2_maps(results, plot_dir)
    except Exception as e:
        print(f"  Warning: plot_xco2_maps failed: {e}")
    try:
        plot_zonal_mean(results, plot_dir)
    except Exception as e:
        print(f"  Warning: plot_zonal_mean failed: {e}")

    return results


def save_osse_results(results, out_dir):
    """Save metrics summary JSON and print comparison table.

    Parameters
    ----------
    results : dict[str, OSSEResult]
    out_dir : str or Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics JSON
    summary = {name: res.metrics.to_dict() for name, res in results.items()}
    json_path = out_dir / "metrics_summary.json"

    # Convert NaN to null for JSON serialization
    def _clean_for_json(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        return obj

    with open(json_path, "w") as f:
        json.dump(_clean_for_json(summary), f, indent=2)
    print(f"\nMetrics summary saved to {json_path}")

    # Print comparison table
    _print_comparison_table(results)


def _print_comparison_table(results):
    """Print a formatted comparison table of metrics."""
    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return "N/A"
        return f"{v:.4f}"

    header = (
        f"{'Experiment':<25} {'RMSE_3d':>10} {'RMSE_3d_o':>10} "
        f"{'RMSE_3d_a':>10} {'RMSE_xco2':>10} {'R2':>8} "
        f"{'Spr/Skl':>8} {'CRPS':>8} {'CalErr':>8} {'Spread':>10}"
    )
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print(f"{'-' * len(header)}")

    for name, res in results.items():
        m = res.metrics
        print(
            f"{name:<25} {_fmt(m.rmse_3d_full):>10} {_fmt(m.rmse_3d_obs):>10} "
            f"{_fmt(m.rmse_3d_away):>10} {_fmt(m.rmse_xco2_full):>10} "
            f"{_fmt(m.r2):>8} {_fmt(m.spread_skill):>8} "
            f"{_fmt(m.crps_mean):>8} {_fmt(m.calibration_error):>8} "
            f"{_fmt(m.sample_spread):>10}"
        )
    print(sep)
