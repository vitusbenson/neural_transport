"""Unified generation pipeline for all use cases.

Replaces three separate functions from generative.py:
- iterative_generate → GenerationPipeline.run() in sample mode
- iterative_generate_oco2 → GenerationPipeline.run() in timeseries mode
- generate_for_distributional_eval → GenerationPipeline.run_distributional()
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)

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

    Delegates to :meth:`OCO2DataLoader.align_time` (canonical implementation).

    Returns:
      offset: int, such that time_gen[i + offset] == time[i]
    """
    from neural_transport.data.oco2_loader import OCO2DataLoader

    return OCO2DataLoader.align_time(time, time_gen)


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
        oco2_loader=None,
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
        self.oco2_loader = oco2_loader

    @property
    def mode(self):
        """'timeseries' if dataset_gen or oco2_loader provided, else 'sample'."""
        return "timeseries" if (self.dataset_gen is not None or self.oco2_loader is not None) else "sample"

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

    def _apply_sample_masking(
        self, batch, model, mask_source, mask_pattern, obs_fraction, ak_10, target_var, soft_boundary_sigma=0.0
    ):
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
                soft_boundary_sigma=soft_boundary_sigma,
            )
            batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(
                obs_values, batch, target_var=target_var, targshift=False
            )
            # Replace NaN at unobserved locations with 0. This is safe because:
            # - With binary mask (no obs_weight): torch.where(obs_mask, ...) skips NaN locations
            # - With soft mask (obs_weight): obs_weight * (... - obs_values) needs finite obs_values
            #   to avoid 0 * NaN = NaN at boundary pixels
            import torch

            obs_values_normed = torch.nan_to_num(obs_values_normed, nan=0.0)
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
        soft_boundary_sigma = generate_kwargs.get("soft_boundary_sigma", 0.0)

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
                    base_batch,
                    model,
                    mask_source,
                    mask_pattern,
                    obs_fraction,
                    ak_10,
                    self.target_vars_3d[0],
                    soft_boundary_sigma=soft_boundary_sigma,
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

        # Determine dataset_gen source: oco2_loader or raw dataset_gen
        use_oco2_loader = self.oco2_loader is not None
        if use_oco2_loader:
            dataset_gen_raw = self.oco2_loader.dataset
        else:
            dataset_gen_raw = self.dataset_gen

        if mask_pattern is None:
            T = DEFAULT_T
        else:
            T = min(len(self.dataset), len(dataset_gen_raw))

        if use_oco2_loader:
            offset = self.oco2_loader.compute_offset(self)
        else:
            offset = align_time(self.dataset.ds.time.values, dataset_gen_raw.ds.time.values)
        logger.info("Time alignment offset: %d timesteps", offset)
        window_steps = max(1, window_hours // freq_int)
        logger.info("Using observation window: %d hours = %d timesteps", window_hours, window_steps)

        iterator = range(T)
        if self.verbose:
            iterator = tqdm(iterator, desc="Generating")

        for t in iterator:
            if use_oco2_loader:
                # New path: delegate to OCO2DataLoader
                from neural_transport.data import InferenceDataLoader

                gt_loader = InferenceDataLoader.__new__(InferenceDataLoader)
                gt_loader._dataset = self.dataset
                gt_loader._grid_info = self.oco2_loader.grid_info
                batch = gt_loader.get_window_batch(
                    t if offset <= 0 else t,
                    window_steps=window_steps,
                    device=self.device,
                    agg="mean",
                )
                # Adjust GT start for negative offset
                if offset < 0:
                    gt_start = t - offset
                    batch = gt_loader.get_window_batch(
                        gt_start,
                        window_steps=window_steps,
                        device=self.device,
                        agg="mean",
                    )

                obs_batch = self.oco2_loader.get_observations(
                    t,
                    offset=offset,
                    window_steps=window_steps,
                    device=self.device,
                    mask_pattern=mask_pattern,
                    nlat=self.nlat,
                    nlon=self.nlon,
                )
                batch_gen = obs_batch.raw_batch

                # Noise (use oco2_loader's dataset for noise template)
                noise_list = noise(
                    self.oco2_loader.dataset[0],
                    target_var=generate_kwargs["generate_data_kwargs"]["forcing_vars"][0],
                    n_samples=n_samples,
                    noise_pattern=noise_pattern,
                    analyze_noise=analyze_noise,
                    outpath=out_dir,
                )

                # Masking via OCO2DataLoader
                if masking:
                    target_var = self.target_vars_2d[0]
                    obs_values = obs_batch.obs_values
                    obs_values_normed = model.model.normalize_observations(
                        obs_values, batch_gen, target_var=target_var, targshift=False
                    )
                    obs_batch_injected = obs_batch
                    obs_batch_injected.obs_values = obs_values_normed
                    obs_batch_injected.inject_into_batch(
                        batch,
                        target_var=target_var,
                        forcing_vars=generate_kwargs["generate_data_kwargs"]["forcing_vars"],
                    )
                    # Also copy target_vars_2d
                    for k in self.target_vars_2d:
                        if k in batch_gen and k not in batch:
                            batch[k] = batch_gen[k]
            else:
                # Legacy path: raw dataset_gen
                batch, batch_gen = get_batches(t, offset, self.dataset, dataset_gen_raw, window_steps, self.device)

                # Noise
                noise_list = noise(
                    dataset_gen_raw[0],
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
                if use_oco2_loader:
                    obs_mask_xr = self.oco2_loader.dataset.tensor_to_xarray(batch_gen["obs_mask"][:1])
                    obs_values_xr = self.oco2_loader.dataset.tensor_to_xarray(obs_batch.obs_values[:1])
                else:
                    obs_mask_xr = dataset_gen_raw.tensor_to_xarray(batch_gen["obs_mask"][:1])
                    obs_values_xr = dataset_gen_raw.tensor_to_xarray(obs_values[:1])
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


def generate_autoregressive(
    model,
    data_loader,
    n_steps,
    target_var="co2massmix",
    init_idx=0,
    reinit_every=None,
    device="cuda",
    verbose=False,
):
    """Auto-regressive rollout for Phase 24 transport-prior FM models.

    At each step k, the model is fed:
      - previous CO2 (fed back from the last prediction, or from ground truth
        at the k-th step when `reinit_every` triggers a re-initialisation)
      - GT wind forcings (u, v) from the dataset at step k
    and it predicts CO2 at step k+1.

    Parameters
    ----------
    model : nn.Module
        A Phase 24 FlowMatching model wrapper (expects
        input_vars=["co2massmix_next", "co2massmix", "u", "v"]).
    data_loader : InferenceDataLoader
        Loader over the test period (provides per-timestep batches with GT CO2
        and wind fields).
    n_steps : int
        Number of forward steps to generate.
    target_var : str
        Target variable name (default "co2massmix").
    init_idx : int
        Dataset index used as the initial state and as the starting point of
        the trajectory.
    reinit_every : int | None
        If set, every ``reinit_every`` steps the rolled CO2 state is replaced
        by GT (sliding-window mode). ``None`` → pure auto-regressive rollout.
    device : str
    verbose : bool

    Returns
    -------
    xr.Dataset
        Trajectory with dims ``[time, sample, lat, lon, level]`` and key
        ``target_var``. Time coord is the dataset's original time axis,
        aligned to timesteps ``init_idx + 1 ... init_idx + n_steps``.
    """
    model.eval()
    model.to(device)

    init_batch = data_loader.get_batch(init_idx, device=device)
    current_co2 = init_batch[target_var].clone()  # [1, N, C]

    predictions = []
    times = []

    iterator = range(n_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Auto-regressive rollout")

    with torch.no_grad():
        for k in iterator:
            idx = init_idx + k
            if idx + 1 >= len(data_loader):
                break

            batch = data_loader.get_batch(idx, device=device)

            # Feed back the rolled CO2 state (or re-init from GT at window edges).
            if reinit_every is not None and k > 0 and k % reinit_every == 0:
                current_co2 = batch[target_var].clone()

            batch[target_var] = current_co2

            # The _next slot is overwritten by x_init=noise inside
            # inference_forward, but its value still seeds the denormalization
            # targshift_mean in postprocess. Using the current (rolled) state is
            # a close proxy for the unknown next mean (CO2 evolves slowly).
            batch[f"{target_var}_next"] = current_co2.clone()

            preds = model(batch, _mode="generate")
            pred_co2 = preds[target_var]  # [1, N, C]
            current_co2 = pred_co2.detach()

            predictions.append(pred_co2.cpu())
            times.append(data_loader.dataset.ds.time.values[idx + 1])

    # Assemble xarray Dataset using the dataset's helper.
    pred_tensor = torch.cat(predictions, dim=0)  # [T, N, C] (B collapses across steps)
    ds = xr.Dataset({target_var: data_loader.dataset.tensor_to_xarray(pred_tensor)})
    ds = ds.rename({"batch": "time"}).assign_coords(time=("time", np.array(times)))
    return ds


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


# ── Multi-target batched generation ─────────────────────────────────────


def generate_multi_target(
    model,
    dataset,
    target_indices: list[int],
    n_samples_per_target: int = 10,
    batch_size: int = 20,
    generate_kwargs: dict | None = None,
    out_dir: str | Path = ".",
    device: str = "cuda",
    target_var: str = "co2massmix",
    verbose: bool = True,
    seed: int = 42,
) -> Path:
    """Generate conditioned ensembles for multiple targets in GPU-efficient batches.

    Pre-allocates a zarr store with ``(target, sample)`` as separate dimensions,
    processes work items in batches of ``batch_size`` on GPU, and flushes each
    target's results to zarr as soon as all its samples are collected.

    Parameters
    ----------
    model : NeuralTransport
        Pre-trained model (will be set to eval/generate mode).
    dataset : CarbonDataset
        Dataset providing target fields and normalization stats.
    target_indices : list[int]
        Which dataset time indices to use as conditioning targets.
    n_samples_per_target : int
        Ensemble members to generate per target.
    batch_size : int
        GPU batch size (items processed in parallel).
    generate_kwargs : dict, optional
        Generation config (masking, sampler, steps, etc.).
    out_dir : str or Path
        Output directory for zarr store.
    device : str
        Torch device.
    target_var : str
        Target variable name (e.g. "co2massmix").
    verbose : bool
        Show progress bar.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Path
        Path to the written zarr store.
    """
    import zarr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / "multitarget_predictions.zarr"

    if generate_kwargs is None:
        generate_kwargs = {}

    n_targets = len(target_indices)

    # ── Setup model ──
    model = model.eval().to(device)
    model.return_intermediates = False
    model.model.generate_kwargs = generate_kwargs

    # Determine grid shape from first dataset item
    sample_data = dataset[target_indices[0]][target_var]
    if sample_data.ndim == 3:  # [T, N, C]
        _, N, nlev = sample_data.shape
    else:  # [N, C]
        N, nlev = sample_data.shape

    # Get nlat, nlon from dataset
    grid = dataset.grid
    if grid.startswith("latlon"):
        res = float(grid.replace("latlon", ""))
        nlat = int(180 / res)
        nlon = int(360 / res)
    else:
        nlat = nlon = int(np.sqrt(N))

    # ── Pre-allocate zarr ──
    store = zarr.open(str(zarr_path), mode="w")
    store.create_array(
        "predictions",
        shape=(n_targets, n_samples_per_target, nlat, nlon, nlev),
        chunks=(1, n_samples_per_target, nlat, nlon, nlev),
        dtype="float32",
    )
    store.create_array("gt", shape=(n_targets, nlat, nlon, nlev), chunks=(1, nlat, nlon, nlev), dtype="float32")
    store.create_array("obs_mask", shape=(n_targets, nlat, nlon), chunks=(1, nlat, nlon), dtype="bool")
    store.create_array("obs_values", shape=(n_targets, nlat, nlon), chunks=(1, nlat, nlon), dtype="float32")
    store.create_array("target_time_idx", shape=(n_targets,), dtype="int64")
    # Optional: pressure weights and AK (same shape per target if column obs)
    store.create_array(
        "pressure_weights",
        shape=(n_targets, nlat, nlon, nlev),
        chunks=(1, nlat, nlon, nlev),
        dtype="float32",
    )
    store.create_array("ak", shape=(n_targets, nlat, nlon, nlev), chunks=(1, nlat, nlon, nlev), dtype="float32")

    # ── Prepare masked batches per target (computed once, cached) ──
    mask_source = generate_kwargs.get("mask_source", "column")
    mask_pattern = generate_kwargs.get("mask_pattern", "satellite")
    obs_fraction = generate_kwargs.get("obs_fraction", 0.2)
    ak_10 = generate_kwargs.get("ak_10", None)
    masking = generate_kwargs.get("masking", True)
    soft_boundary_sigma = generate_kwargs.get("soft_boundary_sigma", 0.0)

    # Temporary pipeline for masking utility (reuses _apply_sample_masking)
    pipeline = GenerationPipeline(model, dataset, target_vars_3d=[target_var], device=device, verbose=False)

    cached_batches = {}
    cached_gt = {}
    cached_obs_info = {}

    logger.info("Preparing masked batches for %d targets...", n_targets)
    for t_pos, t_idx in enumerate(target_indices):
        base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[t_idx].items()}

        # Extract ground truth BEFORE masking
        gt_tensor = base_batch[target_var].clone()  # [1, T, N, C] or [1, N, C]
        if gt_tensor.ndim == 4:
            gt_np = gt_tensor[0, 0].cpu().numpy().reshape(nlat, nlon, nlev)
        else:
            gt_np = gt_tensor[0].cpu().numpy().reshape(nlat, nlon, nlev)

        if masking:
            pipeline._apply_sample_masking(
                base_batch,
                model,
                mask_source,
                mask_pattern,
                obs_fraction,
                ak_10,
                target_var,
                soft_boundary_sigma=soft_boundary_sigma,
            )

            # Extract obs info for zarr
            obs_mask_tensor = base_batch.get("obs_mask")
            obs_values_tensor = base_batch.get("obs_values")
            if obs_mask_tensor is not None:
                # Column obs: obs_mask is [B, T, N, 1] → [nlat, nlon]
                om = obs_mask_tensor[0, 0, :, 0].cpu().numpy().reshape(nlat, nlon)
                ov = obs_values_tensor[0, 0, :, 0].cpu().numpy().reshape(nlat, nlon)
            else:
                om = np.zeros((nlat, nlon), dtype=bool)
                ov = np.full((nlat, nlon), np.nan, dtype=np.float32)

            # Extract pressure weights and AK if available
            pw_tensor = base_batch.get("pressure_weight")
            ak_tensor = base_batch.get("xco2_averaging_kernel")
            if pw_tensor is not None:
                pw_np = pw_tensor[0, 0].cpu().numpy().reshape(nlat, nlon, nlev)
                ak_np = ak_tensor[0, 0].cpu().numpy().reshape(nlat, nlon, nlev)
            else:
                pw_np = np.zeros((nlat, nlon, nlev), dtype=np.float32)
                ak_np = np.zeros((nlat, nlon, nlev), dtype=np.float32)
        else:
            om = np.zeros((nlat, nlon), dtype=bool)
            ov = np.full((nlat, nlon), np.nan, dtype=np.float32)
            pw_np = np.zeros((nlat, nlon, nlev), dtype=np.float32)
            ak_np = np.zeros((nlat, nlon, nlev), dtype=np.float32)

        cached_batches[t_pos] = base_batch
        cached_gt[t_pos] = gt_np
        cached_obs_info[t_pos] = (om, ov, pw_np, ak_np)

        # Write GT and obs info immediately (these don't depend on generation)
        store["gt"][t_pos] = gt_np
        store["obs_mask"][t_pos] = om
        store["obs_values"][t_pos] = ov
        store["pressure_weights"][t_pos] = pw_np
        store["ak"][t_pos] = ak_np
        store["target_time_idx"][t_pos] = t_idx

    # ── Build work items ordered by target ──
    work_items = []
    for t_pos in range(n_targets):
        for s_idx in range(n_samples_per_target):
            work_items.append((t_pos, s_idx))

    # ── Process in GPU batches ──
    # Buffer: accumulate predictions per target
    target_buffers: dict[int, list[np.ndarray]] = {t: [] for t in range(n_targets)}

    n_batches = (len(work_items) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Generating (batched)", total=n_batches)

    for batch_idx in iterator:
        start = batch_idx * batch_size
        end = min(start + batch_size, len(work_items))
        batch_items = work_items[start:end]

        # Stack batches: cat along batch dimension
        stacked = {}
        for key in cached_batches[0]:
            tensors = [cached_batches[t_pos][key] for (t_pos, _) in batch_items]
            stacked[key] = torch.cat(tensors, dim=0)  # [B, ...]

        # Generate noise
        noise_shape = stacked[target_var].shape  # [B, T, N, C]
        stacked["noise"] = torch.randn(
            noise_shape, device=device, generator=torch.Generator(device=device).manual_seed(seed + start)
        )

        # Forward pass
        with torch.no_grad():
            preds = model(stacked)

        # Extract predictions
        pred_tensor = preds[target_var]  # [B, N, C]
        pred_np = pred_tensor.cpu().numpy()  # [B, N, C]

        # Distribute to per-target buffers
        for i, (t_pos, s_idx) in enumerate(batch_items):
            sample = pred_np[i].reshape(nlat, nlon, nlev)
            target_buffers[t_pos].append(sample)

            # Flush when target complete
            if len(target_buffers[t_pos]) == n_samples_per_target:
                samples_array = np.stack(target_buffers[t_pos])  # [n_samples, nlat, nlon, nlev]
                store["predictions"][t_pos] = samples_array.astype(np.float32)
                target_buffers[t_pos] = []  # Free memory
                logger.debug("Flushed target %d (dataset idx %d) to zarr", t_pos, target_indices[t_pos])

    # Flush any remaining buffers (shouldn't happen if work items are ordered correctly)
    for t_pos, buf in target_buffers.items():
        if buf:
            logger.warning(
                "Target %d has %d/%d samples (incomplete), flushing partial", t_pos, len(buf), n_samples_per_target
            )
            samples_array = np.stack(buf)
            store["predictions"][t_pos, : len(buf)] = samples_array.astype(np.float32)

    logger.info(
        "Multi-target generation complete: %d targets × %d samples → %s", n_targets, n_samples_per_target, zarr_path
    )
    return zarr_path
