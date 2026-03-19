"""Unified generation pipeline for all use cases.

Replaces three separate functions from generative.py:
- iterative_generate → GenerationPipeline.run() in sample mode
- iterative_generate_oco2 → GenerationPipeline.run() in timeseries mode
- generate_for_distributional_eval → GenerationPipeline.run_distributional()
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from neural_transport.configs import DEFAULT_T
from neural_transport.inference.masking import (
    create_column_mask,
    create_mask,
    create_oco2_mask,
    create_oco2_mask_test,
)
from neural_transport.inference.noise import noise

# ── Utility functions ────────────────────────────────────────────────────


def get_zarrpath_obspath(out_path, rollout, freq, zarr_filename=None, zero_surfflux=False):
    """Construct output zarr and obs file paths."""
    if zarr_filename is None:
        zarr_filename = f"co2_pred_rollout_{freq}.zarr" if rollout else "co2_pred_singlestep.zarr"
    if zero_surfflux:
        zarr_filename = zarr_filename.replace("co2_pred", "co2_pred_zeroflux")

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    zarrpath = out_path / zarr_filename
    obspath = out_path / f"obs_{zarr_filename}"

    return zarrpath, obspath


def remap_with_cdo(dataset, prototype_zarr, ds):
    """CDO-based conservative remapping to a lat/lon grid."""
    from cdo import Cdo

    cdo = Cdo()
    prototype_zarr.lat.attrs = {
        "long_name": "latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    }
    prototype_zarr.lon.attrs = {
        "long_name": "longitude",
        "units": "degrees_east",
        "standard_name": "longitude",
    }

    ds["clon_vertices"] = dataset.grid_ds.clon_vertices
    ds["clat_vertices"] = dataset.grid_ds.clat_vertices
    ds["clon"] = dataset.grid_ds.clon
    ds["clat"] = dataset.grid_ds.clat
    ds["co2massmix"].attrs = {"CDI_grid_type": "unstructured"}
    ds["co2density"].attrs = {"CDI_grid_type": "unstructured"}
    ds["time"].attrs = {"standard_name": "time"}
    ds["height"].attrs = {"standard_name": "air_pressure"}

    grid_temp_file = tempfile.NamedTemporaryFile(delete=True, prefix="grid_temp_file_", dir=tempfile.gettempdir())
    prototype_zarr.to_netcdf(grid_temp_file.name)

    ds_temp_file = tempfile.NamedTemporaryFile(delete=True, prefix="ds_temp_file_", dir=tempfile.gettempdir())
    ds.transpose("time", "height", "cell", "nv").to_netcdf(ds_temp_file.name)

    ds_remap = cdo.remapcon(grid_temp_file.name, input=ds_temp_file.name, returnXDataset=True)

    grid_temp_file.close()
    ds_temp_file.close()

    cdo.cleanTempDir()
    return ds_remap


def get_batches(t, offset, dataset, dataset_gen, window_steps, device):
    """Collect batches over a time window.

    For sparse observation variables (target_vars_2d, xco2*): union all non-NaN observations.
    For other variables: take mean over window.
    """
    if offset > 0:
        t_start_gen = t + offset
        t_start_gt = t
    else:
        t_start_gen = t
        t_start_gt = t - offset
    t_end_gen = min(t_start_gen + window_steps, len(dataset_gen))
    t_end_gt = min(t_start_gt + window_steps, len(dataset))

    batch_list = []
    for t_win in range(t_start_gt, t_end_gt):
        batch_list.append({k: v.unsqueeze(0).to(device) for k, v in dataset[t_win].items()})
    batch = {}
    for k in batch_list[0].keys():
        batch[k] = torch.stack([b[k] for b in batch_list], dim=0).mean(dim=0)

    batch_gen_list = []
    for t_win in range(t_start_gen, t_end_gen):
        batch_gen_list.append({k: v.unsqueeze(0).to(device) for k, v in dataset_gen[t_win].items()})
    batch_gen = {}
    for k in batch_gen_list[0].keys():
        batch_gen[k] = torch.nanmean(torch.stack([b[k] for b in batch_gen_list], dim=0), dim=0)

    return batch, batch_gen


def is_bad_sample(arr, thresh=1e6):
    """Check if array contains NaN, Inf, or values exceeding threshold."""
    return np.isnan(arr).any() or np.isinf(arr).any() or np.nanmax(np.abs(arr)) > thresh


def parse_freq(freq: str) -> int:
    """Convert frequency string (e.g. '6h', '1D') to integer hours."""
    num = int("".join(filter(str.isdigit, freq)))
    unit = "".join(filter(str.isalpha, freq))

    valid_units = {"h", "D"}
    if unit not in valid_units:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    if unit == "h":
        hours = num
    elif unit == "D":
        hours = num * 24
    return hours


def align_time(time, time_gen):
    """Find index offset to align two time axes.

    Returns:
      offset: int, such that time_gen[i + offset] == time[i]
    """
    if time[-1] < time_gen[0]:
        raise ValueError("No overlap: time from trained dataset is before masking dataset.")
    elif time[0] > time_gen[-1]:
        raise ValueError("No overlap: time from trained dataset is after masking dataset.")
    elif time[0] == time_gen[0]:
        return 0
    elif time[0] > time_gen[0]:
        offset = (time_gen == time[0]).argmax().item()
        return offset
    else:  # time[0] < time_gen[0]
        offset = -(time == time_gen[0]).argmax().item()
        return offset


# ── GenerationPipeline ──────────────────────────────────────────────────


class GenerationPipeline:
    """Unified generation for all use cases.

    Modes:
    - Sample mode (dataset_gen=None): per-sample generation loop for OSSE/synthetic.
    - Timeseries mode (dataset_gen provided): per-timestep loop for real OCO-2.
    """

    def __init__(
        self,
        model,
        dataset,
        *,
        dataset_gen=None,
        target_vars_3d=None,
        target_vars_2d=None,
        device="cuda",
        verbose=True,
    ):
        from neural_transport.data import InferenceDataLoader

        self.model = model
        self.target_vars_3d = target_vars_3d or ["co2massmix"]
        self.target_vars_2d = target_vars_2d or []
        self.device = device
        self.verbose = verbose

        if isinstance(dataset, InferenceDataLoader):
            self.dataset = dataset.dataset
            self.nlat = dataset.grid_info.nlat
            self.nlon = dataset.grid_info.nlon
        else:
            self.dataset = dataset
            self.nlat = model.model.in_nlat
            self.nlon = model.model.in_nlon

        self.dataset_gen = dataset_gen

    @property
    def mode(self):
        """'timeseries' if dataset_gen provided, else 'sample'."""
        return "timeseries" if self.dataset_gen is not None else "sample"

    def run(
        self,
        out_dir,
        *,
        zarr_filename=None,
        freq=None,
        rollout=False,
        zero_surfflux=False,
        remap=False,
        save_obs=True,
        **generate_kwargs,
    ) -> xr.Dataset:
        """Run generation. Dispatches to sample or timeseries mode."""
        if self.mode == "timeseries":
            return self._run_timeseries(
                out_dir,
                zarr_filename=zarr_filename,
                freq=freq,
                rollout=rollout,
                zero_surfflux=zero_surfflux,
                remap=remap,
                save_obs=save_obs,
                **generate_kwargs,
            )
        else:
            return self._run_sample(
                out_dir,
                zarr_filename=zarr_filename,
                freq=freq,
                rollout=rollout,
                zero_surfflux=zero_surfflux,
                remap=remap,
                save_obs=save_obs,
                **generate_kwargs,
            )

    def run_distributional(
        self,
        out_dir,
        *,
        n_gt_samples=50,
        n_gen_samples=200,
        seed=42,
        batch_size=20,
        generate_kwargs=None,
    ) -> tuple:
        """Build GT and generated pools for distributional comparison.

        Returns:
            gt_ds: xr.Dataset [sample, lat, lon, level] -- GT CO2 fields
            gen_ds: xr.Dataset [sample, lat, lon, level] -- generated CO2 fields
        """
        if generate_kwargs is None:
            generate_kwargs = {}

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(seed)
        target_var = self.target_vars_3d[0]

        # --- GT pool ---
        n_gt = min(n_gt_samples, len(self.dataset))
        gt_indices = rng.choice(len(self.dataset), n_gt, replace=False)
        gt_fields = []
        for idx in gt_indices:
            batch = self.dataset[idx]
            field = batch[target_var]  # [T, N, C] or [N, C]
            if field.ndim == 3:
                field = field[0]  # first timestep: [N, C]
            gt_fields.append(field.numpy())
        gt_fields = np.stack(gt_fields)  # [n_gt, N, C]

        nlev = gt_fields.shape[-1]
        gt_fields = gt_fields.reshape(n_gt, self.nlat, self.nlon, nlev)

        # --- Gen pool ---
        model_eval = self.model.eval().to(self.device)
        model_eval.model.generating = True
        model_eval.model.generate_kwargs = generate_kwargs
        model_eval.return_intermediates = False
        model_eval.model.return_intermediates = False

        gen_fields = []
        n_remaining = n_gen_samples

        template = {
            k: v.unsqueeze(0).to(self.device) for k, v in self.dataset[0].items() if isinstance(v, torch.Tensor)
        }

        while n_remaining > 0:
            this_batch_size = min(batch_size, n_remaining)
            batch_in = {k: v.expand(this_batch_size, *v.shape[1:]).clone() for k, v in template.items()}

            with torch.no_grad():
                preds = model_eval(batch_in)

            if target_var in preds:
                field = preds[target_var].cpu().numpy()  # [B, T, N, C] or [B, N, C]
                if field.ndim == 4:
                    field = field[:, -1]  # Take last timestep: [B, N, C]
                for i in range(this_batch_size):
                    if not is_bad_sample(field[i]):
                        gen_fields.append(field[i])

            n_remaining -= this_batch_size

        gen_fields = np.stack(gen_fields[:n_gen_samples])  # [n_gen, N, C]
        gen_fields = gen_fields.reshape(len(gen_fields), self.nlat, self.nlon, nlev)

        # Build xarray datasets with proper 1D lat/lon coordinate arrays
        lat_vals, lon_vals = self._get_grid_coords()

        if hasattr(self.dataset, "ds") and "level" in self.dataset.ds:
            level_vals = self.dataset.ds.level.values[:nlev]
        else:
            level_vals = np.arange(nlev)

        gt_ds = xr.Dataset(
            {target_var: (("sample", "lat", "lon", "level"), gt_fields)},
            coords={
                "sample": np.arange(len(gt_fields)),
                "lat": lat_vals,
                "lon": lon_vals,
                "level": level_vals,
            },
        )

        gen_ds = xr.Dataset(
            {target_var: (("sample", "lat", "lon", "level"), gen_fields)},
            coords={
                "sample": np.arange(len(gen_fields)),
                "lat": lat_vals,
                "lon": lon_vals,
                "level": level_vals,
            },
        )

        gt_ds.to_netcdf(out_dir / "gt_pool.nc")
        gen_ds.to_netcdf(out_dir / "gen_pool.nc")

        return gt_ds, gen_ds

    # ── Shared internal helpers ──

    def _get_grid_coords(self):
        """Look up lat/lon coordinates from grid prototypes."""
        try:
            from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS

            for grid_name, coords in LATLON_PROTOTYPE_COORDS.items():
                if len(coords["lat"]) == self.nlat and len(coords["lon"]) == self.nlon:
                    return coords["lat"], coords["lon"]
        except ImportError:
            pass

        return (
            np.linspace(-90, 90, self.nlat),
            np.linspace(0, 360, self.nlon, endpoint=False),
        )

    def _setup(self, out_dir, generate_kwargs, rollout, freq, zarr_filename, zero_surfflux, remap):
        """Common setup: output paths, prototype zarr, model config."""
        zarrpath, obspath = get_zarrpath_obspath(out_dir, rollout, freq, zarr_filename, zero_surfflux)
        prototype_zarr = self.dataset.create_prototype_zarr(
            zarrpath,
            target_vars_3d=self.target_vars_3d,
            target_vars_2d=self.target_vars_2d,
            grid="default" if remap else None,
        )
        model = self.model.eval().to(self.device)
        model.return_intermediates = True
        model.model.generate_kwargs = generate_kwargs
        return zarrpath, obspath, prototype_zarr, model

    def _run_inference(self, model, batch):
        """Run model forward pass and fix list-valued predictions."""
        with torch.no_grad():
            preds = model(batch)
            traj = preds["trajectory"]

        preds_fixed = {}
        for k, v in preds.items():
            if isinstance(v, list):
                v = v[0]
            preds_fixed[k] = v
        return preds_fixed, traj

    def _finalize(self, dss, obss, zarrpath, obspath, concat_dim, save_obs, outpath, generate_kwargs):
        """Concatenate, save, and optionally plot masking diagnostics."""
        ds_all = xr.concat(dss, dim=concat_dim)

        if save_obs:
            obs_all = xr.concat(obss, dim=concat_dim).fillna({"obs_filename": ""})
            obs_all.to_zarr(obspath, mode="w")
            ds_all = ds_all.fillna({"obs_filename": ""})

        ds_all.to_zarr(zarrpath, mode="w")

        masking = generate_kwargs.get("masking", False)
        analyze_masking = generate_kwargs.get("analyze_masking", False)
        if analyze_masking and masking:
            from neural_transport.plots.plot_results import plot_masking_diagnostics

            plot_masking_diagnostics(
                ds_all,
                str(outpath).replace("preds", "plots"),
                varnames=self.target_vars_3d,
                nlat=self.nlat,
                nlon=self.nlon,
                imgformats=["png"],
            )

        return ds_all

    # ── Sample mode ──

    def _apply_sample_masking(self, batch, model, mask_source, mask_pattern, obs_fraction, ak_10, target_var):
        """Apply masking for sample mode. Modifies batch in-place."""
        if mask_source in ("column", "xco2"):
            obs_mask, obs_values = create_column_mask(
                batch,
                target_var=target_var,
                obs_fraction=obs_fraction,
                mask_pattern=mask_pattern,
                nlat=self.nlat,
                nlon=self.nlon,
                ak_10=ak_10,
            )
            batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(
                obs_values, batch, target_var=target_var, targshift=False
            )
            batch["obs_values"] = obs_values_normed
        elif mask_pattern is None:
            obs_mask, obs_values = create_oco2_mask(batch, target_var=target_var)
            batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(obs_values, batch, target_var=target_var)
            batch["obs_values"] = obs_values_normed
        else:
            obs_mask, obs_values = create_mask(
                batch,
                target_var=target_var,
                obs_fraction=obs_fraction,
                mask_pattern=mask_pattern,
                nlat=self.nlat,
                nlon=self.nlon,
            )
            batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(obs_values, batch, target_var=target_var)
            batch["obs_values"] = obs_values_normed

    def _run_sample(self, out_dir, *, zarr_filename, freq, rollout, zero_surfflux, remap, save_obs, **generate_kwargs):
        """Per-sample generation loop (OSSE/synthetic observations)."""
        condition_one_timestep = generate_kwargs.get("condition_one_timestep", True)
        n_samples = generate_kwargs.get("n_samples", 10)
        masking = generate_kwargs.get("masking", False)
        mask_source = generate_kwargs.get("mask_source", "3d")
        mask_pattern = generate_kwargs.get("mask_pattern", "vertical")
        obs_fraction = generate_kwargs.get("obs_fraction", 0.2)
        noise_pattern = generate_kwargs.get("noise_pattern", None)
        analyze_noise = generate_kwargs.get("analyze_noise", False)
        ak_10 = generate_kwargs.get("ak_10", None)

        zarrpath, obspath, prototype_zarr, model = self._setup(
            out_dir, generate_kwargs, rollout, freq, zarr_filename, zero_surfflux, remap
        )

        dss = []
        obss = []

        # Noise
        noise_list = noise(
            self.dataset[0],
            target_var=self.target_vars_3d[0],
            n_samples=n_samples,
            noise_pattern=noise_pattern,
            analyze_noise=analyze_noise,
            outpath=out_dir,
        )

        if condition_one_timestep:
            base_batch = {k: v.unsqueeze(0).to(self.device) for k, v in self.dataset[0].items()}
            if masking:
                self._apply_sample_masking(
                    base_batch, model, mask_source, mask_pattern, obs_fraction, ak_10, self.target_vars_3d[0]
                )

        iterator = range(n_samples)
        if self.verbose:
            iterator = tqdm(iterator, desc="Generating samples")

        for i in iterator:
            if not condition_one_timestep:
                base_batch = {k: v.unsqueeze(0).to(self.device) for k, v in self.dataset[i].items()}
                if masking:
                    self._apply_sample_masking(
                        base_batch, model, mask_source, mask_pattern, obs_fraction, ak_10, self.target_vars_3d[0]
                    )

            batch = {k: v.clone() for k, v in base_batch.items()}
            batch["noise"] = noise_list[i].to(self.device)

            if zero_surfflux:
                for var in ["co2flux_anthro", "co2flux_land", "co2flux_ocean"]:
                    batch[var] = torch.zeros_like(batch[var])

            preds_fixed, traj = self._run_inference(model, batch)

            if save_obs:
                preds_fixed["gph_bottom"] = batch["gph_bottom"]
                preds_fixed["gph_top"] = batch["gph_top"]

            ds = xr.Dataset({k: self.dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()})

            ds = ds.assign_coords(
                time=("time", prototype_zarr.time[: traj.shape[1]].values if traj.ndim > 0 else [0]),
                sample=("sample", [i]),
            )

            if masking:
                obs_mask_xr = self.dataset.tensor_to_xarray(batch["obs_mask"][:1])
                obs_values_xr = self.dataset.tensor_to_xarray(batch["obs_values"][:1])
                ds["obs_mask"] = obs_mask_xr
                ds["obs_values"] = obs_values_xr

            if remap:
                ds = remap_with_cdo(self.dataset, prototype_zarr.isel(time=0), ds)

            if save_obs:
                obs = self.dataset.readout_stations(ds, grid="default" if remap else None)
                ds = ds.drop_vars(["gph_bottom", "gph_top"])
                obss.append(obs)

            dss.append(ds)

        return self._finalize(dss, obss, zarrpath, obspath, "sample", save_obs, out_dir, generate_kwargs)

    # ── Timeseries mode ──

    def _run_timeseries(
        self, out_dir, *, zarr_filename, freq, rollout, zero_surfflux, remap, save_obs, **generate_kwargs
    ):
        """Per-timestep generation loop (real OCO-2 assimilation)."""
        n_samples = generate_kwargs.get("n_samples", 10)
        masking = generate_kwargs.get("masking", True)
        mask_pattern = generate_kwargs.get("mask_pattern", None)
        noise_pattern = generate_kwargs.get("noise_pattern", None)
        analyze_noise = generate_kwargs.get("analyze_noise", False)
        freq_int = parse_freq(generate_kwargs.get("generate_data_kwargs", {}).get("freq", "6h"))
        window_hours = generate_kwargs.get("window_hours", freq_int)

        zarrpath, obspath, prototype_zarr, model = self._setup(
            out_dir, generate_kwargs, rollout, freq, zarr_filename, zero_surfflux, remap
        )

        dss = []
        obss = []

        if mask_pattern is None:
            T = DEFAULT_T
        else:
            T = min(len(self.dataset), len(self.dataset_gen))

        offset = align_time(self.dataset.ds.time.values, self.dataset_gen.ds.time.values)
        print(f"Time alignment offset: {offset} timesteps")
        window_steps = max(1, window_hours // freq_int)
        print(f"Using observation window: {window_hours} hours = {window_steps} timesteps")

        iterator = range(T)
        if self.verbose:
            iterator = tqdm(iterator, desc="Generating")

        for t in iterator:
            batch, batch_gen = get_batches(t, offset, self.dataset, self.dataset_gen, window_steps, self.device)

            # Noise
            noise_list = noise(
                self.dataset_gen[0],
                target_var=generate_kwargs["generate_data_kwargs"]["forcing_vars"][0],
                n_samples=n_samples,
                noise_pattern=noise_pattern,
                analyze_noise=analyze_noise,
                outpath=out_dir,
            )

            # Masking
            if masking:
                target_var = self.target_vars_2d[0]
                if mask_pattern is None:
                    obs_mask, obs_values = create_oco2_mask(batch_gen, target_var=target_var)
                else:
                    obs_mask, obs_values = create_oco2_mask_test(
                        batch_gen, target_var=target_var, mask_pattern=mask_pattern, nlat=self.nlat, nlon=self.nlon
                    )
                batch_gen["obs_mask_original"] = obs_mask.clone()
                batch_gen["obs_mask"] = obs_mask
                obs_values_normed = model.model.normalize_observations(
                    obs_values, batch_gen, target_var=target_var, targshift=False
                )
                for k in (
                    self.target_vars_2d
                    + generate_kwargs["generate_data_kwargs"]["forcing_vars"]
                    + ["obs_mask", "obs_mask_original"]
                ):
                    batch[k] = batch_gen[k]
                batch["obs_values"] = obs_values_normed

            for k in batch.keys():
                batch[k] = batch[k].expand(n_samples, -1, -1, -1)
            batch["noise"] = torch.cat(noise_list, dim=0).to(self.device)

            if zero_surfflux:
                for var in ["co2flux_anthro", "co2flux_land", "co2flux_ocean"]:
                    batch[var] = torch.zeros_like(batch[var])

            preds_fixed, traj = self._run_inference(model, batch)

            if save_obs:
                preds_fixed["gph_bottom"] = batch["gph_bottom"]
                preds_fixed["gph_top"] = batch["gph_top"]

            ds = xr.Dataset({k: self.dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()})

            ds = ds.rename({"batch": "sample", "time": "trajectory_steps"})
            ds = ds.assign_coords(
                trajectory_steps=("trajectory_steps", np.arange(traj.shape[1])),
                sample=("sample", np.arange(n_samples)),
            )
            ds = ds.expand_dims(time=[prototype_zarr.isel(time=t).time.values])

            if masking:
                obs_mask_xr = self.dataset_gen.tensor_to_xarray(batch_gen["obs_mask"][:1])
                obs_values_xr = self.dataset_gen.tensor_to_xarray(obs_values[:1])
                ds["obs_mask"] = obs_mask_xr
                ds["obs_values"] = obs_values_xr

            if remap:
                ds = remap_with_cdo(self.dataset, prototype_zarr.isel(time=0), ds)

            if save_obs:
                obs = self.dataset.readout_stations(ds, grid="default" if remap else None)
                ds = ds.drop_vars(["gph_bottom", "gph_top"])
                obss.append(obs)

            dss.append(ds)

        return self._finalize(dss, obss, zarrpath, obspath, "time", save_obs, out_dir, generate_kwargs)


# ── Backward-compatible function API ─────────────────────────────────────


def iterative_generate(
    model,
    dataset,
    outpath,
    rollout=False,
    device="cuda",
    verbose=False,
    zarr_filename=None,
    freq=None,
    zero_surfflux=False,
    remap=False,
    target_vars_3d=[],
    target_vars_2d=[],
    save_obs=True,
    **generate_kwargs,
):
    """Backward-compatible wrapper around GenerationPipeline.run() in sample mode."""
    pipeline = GenerationPipeline(
        model,
        dataset,
        target_vars_3d=target_vars_3d,
        target_vars_2d=target_vars_2d,
        device=device,
        verbose=verbose,
    )
    return pipeline.run(
        outpath,
        zarr_filename=zarr_filename,
        freq=freq,
        rollout=rollout,
        zero_surfflux=zero_surfflux,
        remap=remap,
        save_obs=save_obs,
        **generate_kwargs,
    )


def iterative_generate_oco2(
    model,
    dataset,
    dataset_gen,
    outpath,
    rollout=False,
    device="cuda",
    verbose=False,
    zarr_filename=None,
    freq=None,
    zero_surfflux=False,
    remap=False,
    target_vars_3d=[],
    target_vars_2d=[],
    save_obs=True,
    **generate_kwargs,
):
    """Backward-compatible wrapper around GenerationPipeline.run() in timeseries mode."""
    pipeline = GenerationPipeline(
        model,
        dataset,
        dataset_gen=dataset_gen,
        target_vars_3d=target_vars_3d,
        target_vars_2d=target_vars_2d,
        device=device,
        verbose=verbose,
    )
    return pipeline.run(
        outpath,
        zarr_filename=zarr_filename,
        freq=freq,
        rollout=rollout,
        zero_surfflux=zero_surfflux,
        remap=remap,
        save_obs=save_obs,
        **generate_kwargs,
    )


def generate_for_distributional_eval(
    model,
    dataset,
    outpath,
    n_gt_samples=50,
    n_gen_samples=200,
    device="cuda",
    target_vars_3d=None,
    generate_kwargs=None,
    seed=42,
    batch_size=20,
):
    """Backward-compatible wrapper around GenerationPipeline.run_distributional()."""
    pipeline = GenerationPipeline(
        model,
        dataset,
        target_vars_3d=target_vars_3d,
        device=device,
    )
    return pipeline.run_distributional(
        outpath,
        n_gt_samples=n_gt_samples,
        n_gen_samples=n_gen_samples,
        seed=seed,
        batch_size=batch_size,
        generate_kwargs=generate_kwargs,
    )
