"""Plotting script to call all available plotting functions."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import xarray as xr

from neural_transport.plots.utilities.plot_utils import (
    save_figure,
    load_carbontracker_tests,
    parse_projections
)
from neural_transport.plots.utilities.cmaps import get_cmap_list
from neural_transport.plots.plotting.plot_samples_comparison import (
    plot_samples_with_comparison
)
from neural_transport.plots.plotting.plot_trajectory import plot_trajectory_timeseries
from neural_transport.plots.plotting.plot_samples_and_ground_truth import plot_samples_and_ground_truth
from neural_transport.plots.plotting.plot_trajectories import plot_trajectories
from neural_transport.plots.plotting.plot_cmaps import plot_cmaps
from neural_transport.plots.plotting.plot_samples import plot_samples
from neural_transport.plots.plotting.plot_projections import (
    plot_samples_projection
)


def one_to_rule_them_all(args):
    # --- Load predictions and ground truth ---
    path = Path(args.samples_path)
    samples = xr.open_zarr(path) if path.suffix == ".zarr" else xr.open_dataset(path)
    co2tests = load_carbontracker_tests(data_path=args.data_path)

    # --- Colormaps ---
    cmaps = get_cmap_list(args.use_ipcc, args.use_ipcc_one, args.use_selected, n_samples=args.n_samples)

    # --- Parse projections ---
    projections = parse_projections(args.projections)

    # --- Call all plotting functions ---

    # 1. plot_samples_with_comparison
    for proj in projections:
        fig = plot_samples_with_comparison(
            traj=samples.trajectory,
            tests=co2tests,
            projection=proj,
            n_samples=1,
            level_idx=args.level_idx,
            cmap=cmaps[0],
            bias_hidden=args.bias_hidden,
            title="Which one is a generated sample, which one is ground truth?",
        )
        save_figure(fig, args.out_dir,
                    f"samples_comparison_{'minmax_' if args.bias_hidden else ''}{proj.__class__.__name__}",
                    imgformats=["pdf"], dpi=300)

    # 2. plot_trajectory_timeseries
    for sample_idx in range(min(args.n_samples, samples.sizes["sample"])):
        fig = plot_trajectory_timeseries(
            trajectory=samples.trajectory,
            sample_idx=sample_idx,
            level_idx=args.level_idx,
            cmap=cmaps[0],
            bias_hidden=args.bias_hidden,
            title="Trajectory Time Series",
        )
        save_figure(fig, args.out_dir, f"trajectory_timeseries_{sample_idx}", imgformats=["pdf"], dpi=300)

    fig = plot_trajectory_timeseries(
            trajectory=samples.trajectory,
            sample_idx=sample_idx,
            level_idx=args.level_idx,
            projection=projections[0],
            cmap=cmaps[0],
            bias_hidden=args.bias_hidden,
            title="Trajectory Time Series",
        )
    save_figure(fig, args.out_dir,
                f"trajectory_timeseries_{projections[0].__class__.__name__}", imgformats=["pdf"], dpi=300)

    # 3. plot_samples_and_ground_truth
    for proj in projections:
        fig = plot_samples_and_ground_truth(
            traj=samples.trajectory,
            tests=co2tests,
            projection=proj,
            n_samples=args.n_samples,
            level_idx=args.level_idx,
            cmap=cmaps[0],
            bias_hidden=args.bias_hidden,
            title="Generated Samples vs. Ground Truth",
        )
        proj_name = proj.__class__.__name__
        suffix = "minmax_" if args.bias_hidden else ""
        save_figure(fig, args.out_dir, f"samples_comparison_gt_{suffix}{proj_name}", imgformats=["pdf"], dpi=300)

    # 4. plot_trajectories
    fig = plot_trajectories(
        traj=samples.trajectory,
        n_samples=args.n_samples,
        sample_indices=None,
        level_idx=args.level_idx,
        time_indices=args.time_indices,
        cmaps=cmaps,
        title="Sample Trajectories",
    )
    save_figure(fig, args.out_dir, "trajectories", imgformats=["pdf"])

    # 5. plot_cmaps
    fig = plot_cmaps(samples, cmap_list=cmaps)
    save_figure(fig, args.out_dir, "colormap_comparison", imgformats=["pdf"])

    # 6. plot_samples_projection
    fig = plot_samples_projection(
        samples.trajectory,
        projection=[ccrs.Robinson(), ccrs.Aitoff(), ccrs.InterruptedGoodeHomolosine(), ccrs.Mollweide()],
        terrain=False,
        grid=True,
        land=False, ocean=False, borders=False, lakes=False, rivers=False,
        n_samples=4,
        ncol=args.ncol,
        level_idx=args.level_idx,
        cmaps=cmaps,
        title="CO₂ Projection Samples"
    )
    save_figure(fig, args.out_dir, "samples_projection", imgformats=["pdf"])

    # 7. plot_samples
    fig, _ = plot_samples(
        traj=samples.trajectory,
        n_samples=args.n_samples,
        ncol=args.ncol,
        sample_indices=list(range(args.n_samples)),
        level_idx=args.level_idx,
        cmaps=cmaps,
        title="Samples",
    )
    save_figure(fig, args.out_dir, "samples", imgformats=["pdf"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master plotting script for CO₂ samples and trajectories.")
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to CarbonTracker data directory.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--level_idx", type=int, default=0)
    parser.add_argument("--projections", nargs="*", default=["Robinson"])
    parser.add_argument("--use_ipcc", action="store_true")
    parser.add_argument("--use_ipcc_one", action="store_true")
    parser.add_argument("--use_selected", action="store_true")
    parser.add_argument("--bias_hidden", action="store_true")
    parser.add_argument("--time_indices", type=int, nargs="+", default=[0, 5, 7, 9])
    parser.add_argument("--ncol", type=int, default=2)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    one_to_rule_them_all(args)

# python /Net/Groups/BGI/work_5/CO2_diffusion/neural_transport/neural_transport/plots/one_to_rule_them_all.py --samples_path /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/transport_models/carbontracker_lowres/flowmatching_dev/flowmatching_20251030_1_unet_baseline_dev/singlestep/preds/ckpt=best_massfixer=default/co2_pred_rollout_QS.zarr --out_dir /Net/Groups/BGI/work_5/CO2_diffusion/carbonbench/plotting/flowmatching/plots/20251030_1_unet_baseline_dev/ --use_ipcc_one --n_samples 100
