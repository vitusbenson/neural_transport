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
from neural_transport.tools.spatial import gaussian_smooth_2d

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
        n_ref_timesteps=20,
        n_gen_per_ref=10,
        seed=42,
        generate_kwargs=None,
        # Legacy aliases (ignored except to map old→new names).
        n_gt_samples=None,
        n_gen_samples=None,
        batch_size=None,
    ) -> tuple:
        """Build GT and generated pools for distributional comparison.

        Ensemble-based protocol: picks ``n_ref_timesteps`` random test-period
        init points and draws ``n_gen_per_ref`` independent one-step samples
        per init. GT pool = GT co2_{t+1} at those init points (size
        ``n_ref_timesteps``); gen pool = all samples flattened (size
        ``n_ref_timesteps * n_gen_per_ref``).

        Returns:
            gt_ds, gen_ds: xr.Dataset [sample, lat, lon, level]
        """
        from neural_transport.data import InferenceDataLoader

        if generate_kwargs is None:
            generate_kwargs = {}
        if n_gt_samples is not None:
            n_ref_timesteps = n_gt_samples
        if n_gen_samples is not None:
            # Distribute legacy total gen pool size across refs.
            n_gen_per_ref = max(1, n_gen_samples // n_ref_timesteps)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(seed)
        target_var = self.target_vars_3d[0]

        # Need a proper InferenceDataLoader for generate_ensemble. If one was
        # not provided at construction, build one from the dataset.
        if isinstance(self.dataset, InferenceDataLoader):
            loader = self.dataset
        else:
            loader = InferenceDataLoader.__new__(InferenceDataLoader)
            loader._dataset = self.dataset
            from neural_transport.data.inference_loader import GridInfo

            loader._grid_info = GridInfo(
                nlat=self.nlat,
                nlon=self.nlon,
                nlev=self.dataset[0][target_var].shape[-1],
                lat=np.asarray(self._get_grid_coords()[0]),
                lon=np.asarray(self._get_grid_coords()[1]),
                levels=np.arange(self.dataset[0][target_var].shape[-1]),
            )

        # Restrict to indices that have a valid t+1 target.
        valid_range = len(loader) - 1
        n_ref = min(n_ref_timesteps, valid_range)
        init_indices = sorted(rng.choice(valid_range, n_ref, replace=False).tolist())

        # --- GT pool: co2_{t+1} at each init index ---
        gt_fields = []
        for idx in init_indices:
            sample = loader.dataset[idx]
            next_key = f"{target_var}_next"
            if next_key in sample:
                field = sample[next_key]
            else:
                nxt = loader.dataset[idx + 1]
                field = nxt[target_var]
            if isinstance(field, torch.Tensor):
                field = field.numpy()
            if field.ndim == 3:
                field = field[0]
            gt_fields.append(field)
        gt_fields = np.stack(gt_fields)
        nlev = gt_fields.shape[-1]
        gt_fields = gt_fields.reshape(len(gt_fields), self.nlat, self.nlon, nlev)

        # --- Gen pool: generate_ensemble with n_steps=1 ---
        self.model.model.generate_kwargs = generate_kwargs
        ens = generate_ensemble(
            self.model,
            loader,
            init_indices=init_indices,
            n_samples=n_gen_per_ref,
            n_steps=1,
            target_var=target_var,
            device=self.device,
            seed=seed,
            verbose=self.verbose,
        )
        # [init, sample, 1, lat, lon, level] → flatten to [init*sample, lat, lon, level]
        gen_np = ens[target_var].values[:, :, 0]  # [init, sample, lat, lon, level]
        gen_fields = gen_np.reshape(-1, self.nlat, self.nlon, nlev)
        # Filter NaN/Inf bad samples.
        good = np.array([not is_bad_sample(s) for s in gen_fields])
        gen_fields = gen_fields[good]

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


def generate_ensemble(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps=1,
    reinit_every=None,
    target_var="co2massmix",
    device="cuda",
    seed=42,
    verbose=False,
):
    """Unified ensemble generator for Phase 24 FM transport-prior models.

    For each init index, draws ``n_samples`` independent samples (independent
    noise, same conditioning) and rolls each sample forward ``n_steps`` steps.
    Wind forcings come from GT at every step. With ``n_steps=1`` this reduces
    to the one-step probabilistic eval used by ``run_distributional``.

    Parameters
    ----------
    model : LightningModule
        A Phase 24 FlowMatching model. Set to eval+generating here.
    data_loader : InferenceDataLoader
    init_indices : list[int]
        Test-period indices used as starting points.
    n_samples : int
        Number of independent samples per init.
    n_steps : int
        Trajectory length (>=1).
    reinit_every : int | None
        If set, every ``reinit_every`` steps the rolled CO2 state is replaced
        by GT (sliding-window mode).
    target_var : str
    device : str
    seed : int
        Global seed, applied once so the full ensemble is reproducible.
    verbose : bool

    Returns
    -------
    xr.Dataset
        Dims ``[init, sample, lead, lat, lon, level]`` with variable
        ``target_var``. ``time`` is a 2-D coord ``[init, lead]`` giving the
        timestamp of each prediction (``init_indices[i] + lead + 1``).
    """
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    # Ensure FlowMatching dispatches to _mode="generate" inside forward.
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    inner.generating = True

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)

    # Lazy-determine grid shape from a dummy pass of the first batch.
    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon
    assert N == nlat * nlon, f"grid mismatch: N={N}, nlat*nlon={nlat * nlon}"

    all_preds = np.full((n_inits, n_samples, n_steps, nlat, nlon, C), np.nan, dtype=np.float32)
    time_coord = np.empty((n_inits, n_steps), dtype=times_axis.dtype)

    outer = enumerate(init_indices)
    if verbose:
        outer = tqdm(list(outer), desc="Ensemble rollout")

    with torch.no_grad():
        for i_pos, init_idx in outer:
            # Initial rolled state: GT at init_idx, broadcast to n_samples.
            init_batch = data_loader.get_batch(init_idx, device=device)
            current_co2 = init_batch[target_var].expand(n_samples, *init_batch[target_var].shape[1:]).clone()

            for k in range(n_steps):
                idx = init_idx + k
                if idx + 1 >= T_total:
                    logger.warning("Init %d hit end of dataset at lead %d; remaining leads left NaN.", init_idx, k)
                    break

                batch_single = data_loader.get_batch(idx, device=device)
                # Expand forcings to n_samples (wind is shared across samples).
                batch = {
                    k_: v.expand(n_samples, *v.shape[1:]).clone() if isinstance(v, torch.Tensor) else v
                    for k_, v in batch_single.items()
                }

                if reinit_every is not None and k > 0 and k % reinit_every == 0:
                    current_co2 = batch[target_var].clone()

                batch[target_var] = current_co2
                # Placeholder for _next: same as current rolled state. Its value
                # is only used for targshift mean in postprocess (close proxy).
                if f"{target_var}_next" in batch:
                    batch[f"{target_var}_next"] = current_co2.clone()

                preds = model(batch, mode="generate")
                pred_co2 = preds[target_var]  # [n_samples, N, C]
                current_co2 = pred_co2.detach()

                pred_np = pred_co2.cpu().numpy().reshape(n_samples, nlat, nlon, C)
                all_preds[i_pos, :, k] = pred_np
                time_coord[i_pos, k] = times_axis[idx + 1]

    inner.generating = prev_generating

    # Build xarray dataset.
    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels

    ds = xr.Dataset(
        {target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds)},
        coords={
            "init": np.array(init_indices),
            "sample": np.arange(n_samples),
            "lead": np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def generate_trajectory_ensemble_with_obs(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps,
    sampler_generate_kwargs,
    free_generate_kwargs=None,
    obs_every=1,
    obs_offset=0,
    reinit_every=None,
    target_var="co2massmix",
    device="cuda",
    seed=42,
    verbose=False,
):
    """Auto-regressive ensemble rollout with per-step posterior conditioning.

    Extends ``generate_ensemble`` to apply a posterior sampler (e.g. FMPS or
    D-Flow) at observation steps, and free unconditional ODE sampling elsewhere.

    At every lead ``k``:
      1. Assemble AR batch with the rolled CO2 state and GT winds.
      2. If ``k % obs_every == obs_offset``: build a synthetic XCO2 column
         observation via ``create_column_mask`` and run the velocity model
         with ``sampler_generate_kwargs`` (must contain a posterior ``sampler``
         key). The rolled state is fed as ``static_inputs`` — the sampler
         only differentiates through one ODE solve, so the trajectory length
         does not grow the autograd graph.
      3. Else: run a standard unconditional ODE solve with ``free_generate_kwargs``.

    Parameters
    ----------
    model : LightningModule
        Phase 24 transport-prior flow matching model.
    data_loader : InferenceDataLoader
        Must have ``target_vars`` including ``p_bottom`` and ``p_top`` so the
        column observation operator can compute XCO2 at each step.
    init_indices : list[int]
    n_samples : int
        Ensemble members per init.
    n_steps : int
        Trajectory length.
    sampler_generate_kwargs : dict
        ``generate_kwargs`` applied at observation steps — must include a
        ``sampler`` key (e.g. ``"fmps"`` or ``"dflow"``) and the masking config
        (``obs_fraction``, ``mask_pattern`` etc).
    free_generate_kwargs : dict | None
        ``generate_kwargs`` for unconditional steps. Defaults to sampler kwargs
        with masking disabled and ``sampler`` removed.
    obs_every : int
        Observation cadence in steps (6h per step at freq=6h).
    obs_offset : int
        Apply observations at ``k`` such that ``(k - obs_offset) % obs_every == 0``.
    reinit_every : int | None
        Sliding window re-init to GT.
    target_var : str
    device : str
    seed : int
    verbose : bool

    Returns
    -------
    xr.Dataset
        Dims ``[init, sample, lead, lat, lon, level]`` with ``target_var`` and
        an ``obs_present`` bool array ``[lead]``.
    """
    import numpy as np

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    inner.generating = True

    if free_generate_kwargs is None:
        free_generate_kwargs = {
            k: v for k, v in sampler_generate_kwargs.items() if k not in ("sampler", "sampler_params", "masking")
        }
        # drop conditioning sub-dict entries that imply masking
        free_generate_kwargs["masking"] = False

    # Full generate_kwargs always need the non-sampler plumbing keys present.
    for k, v in sampler_generate_kwargs.items():
        if k not in free_generate_kwargs and k not in (
            "sampler",
            "masking",
            "mask_source",
            "mask_pattern",
            "obs_fraction",
        ):
            free_generate_kwargs.setdefault(k, v)

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)

    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon
    assert N == nlat * nlon, f"grid mismatch: N={N}, nlat*nlon={nlat * nlon}"

    all_preds = np.full((n_inits, n_samples, n_steps, nlat, nlon, C), np.nan, dtype=np.float32)
    time_coord = np.empty((n_inits, n_steps), dtype=times_axis.dtype)
    obs_present = np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        obs_present[k] = ((k - obs_offset) >= 0) and ((k - obs_offset) % obs_every == 0)

    mask_pattern = sampler_generate_kwargs.get("mask_pattern", "satellite")
    obs_fraction = sampler_generate_kwargs.get("obs_fraction", 0.3)
    ak_10 = sampler_generate_kwargs.get("ak_10", None)
    soft_boundary_sigma = sampler_generate_kwargs.get("soft_boundary_sigma", 0.0)

    outer = enumerate(init_indices)
    if verbose:
        outer = tqdm(list(outer), desc="Trajectory+obs rollout")

    for i_pos, init_idx in outer:
        init_batch = data_loader.get_batch(init_idx, device=device)
        current_co2 = init_batch[target_var].expand(n_samples, *init_batch[target_var].shape[1:]).clone()

        for k in range(n_steps):
            idx = init_idx + k
            if idx + 1 >= T_total:
                logger.warning("Init %d hit end of dataset at lead %d; remaining leads left NaN.", init_idx, k)
                break

            batch_single = data_loader.get_batch(idx, device=device)
            batch = {
                k_: v.expand(n_samples, *v.shape[1:]).clone() if isinstance(v, torch.Tensor) else v
                for k_, v in batch_single.items()
            }

            if reinit_every is not None and k > 0 and k % reinit_every == 0:
                current_co2 = batch[target_var].clone()

            batch[target_var] = current_co2
            if f"{target_var}_next" in batch:
                batch[f"{target_var}_next"] = current_co2.clone()

            use_obs = bool(obs_present[k])

            if use_obs:
                # Build synthetic XCO2 column mask from GT next field. For OSSE
                # we use the GT target (dataset[idx+1]) to generate observations.
                gt_next_batch = data_loader.get_batch(idx + 1, device=device)
                obs_batch = {k_: v for k_, v in batch.items()}
                obs_batch[target_var] = (
                    gt_next_batch[target_var].expand(n_samples, *gt_next_batch[target_var].shape[1:]).clone()
                )
                # p_bottom / p_top come from obs_batch (current step; approx OK).
                om, ov = create_column_mask(
                    obs_batch,
                    target_var=target_var,
                    obs_fraction=obs_fraction,
                    mask_pattern=mask_pattern,
                    nlat=nlat,
                    nlon=nlon,
                    ak_10=ak_10,
                    soft_boundary_sigma=soft_boundary_sigma,
                )
                obs_batch["obs_mask"] = om
                ov_normed = inner.normalize_observations(ov, obs_batch, target_var=target_var, targshift=False)
                obs_batch["obs_values"] = torch.nan_to_num(ov_normed, nan=0.0)
                # Put conditioning keys onto the AR batch (not obs_batch, which
                # had GT swapped in).
                for key in (
                    "obs_mask",
                    "obs_values",
                    "obs_weight",
                    "xco2_averaging_kernel",
                    "xco2_apriori",
                    "co2_profile_apriori",
                    "pressure_weight",
                ):
                    if key in obs_batch:
                        batch[key] = obs_batch[key]

                inner.generate_kwargs = sampler_generate_kwargs
                # Sampler needs grads (D-Flow). FMPS is wrapped in no_grad inside
                # its own sample(). Either way, let autograd decide.
                preds = model(batch, mode="generate")
            else:
                inner.generate_kwargs = free_generate_kwargs
                with torch.no_grad():
                    preds = model(batch, mode="generate")

            pred_co2 = preds[target_var]
            current_co2 = pred_co2.detach()
            pred_np = current_co2.cpu().numpy().reshape(n_samples, nlat, nlon, C)
            all_preds[i_pos, :, k] = pred_np
            time_coord[i_pos, k] = times_axis[idx + 1]

    inner.generating = prev_generating

    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels

    ds = xr.Dataset(
        {
            target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
            "obs_present": (("lead",), obs_present),
        },
        coords={
            "init": np.array(init_indices),
            "sample": np.arange(n_samples),
            "lead": np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def _model_call_chunked(model, batch, chunk_size, total, no_grad=True):
    """Call ``model(batch, mode="generate")`` in batch-dim chunks of size
    ``chunk_size``, concatenating predictions back to the original batch dim.

    For D-Flow this caps peak VRAM at chunk_size × ODE-graph (instead of
    BATCH × graph). When chunk_size >= total, falls back to a single call.
    """
    if chunk_size >= total:
        if no_grad:
            with torch.no_grad():
                return model(batch, mode="generate")
        return model(batch, mode="generate")

    chunks_out = []
    for s in range(0, total, chunk_size):
        e = min(s + chunk_size, total)
        sub = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == total:
                sub[k] = v[s:e]
            else:
                sub[k] = v
        if no_grad:
            with torch.no_grad():
                sub_out = model(sub, mode="generate")
        else:
            sub_out = model(sub, mode="generate")
        chunks_out.append({k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in sub_out.items()})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # concat tensors along batch dim
    out = {}
    keys = chunks_out[0].keys()
    for k in keys:
        v0 = chunks_out[0][k]
        if isinstance(v0, torch.Tensor) and v0.shape[0] == (
            chunks_out[0][k].shape[0] if hasattr(chunks_out[0][k], 'shape') else None
        ):
            out[k] = torch.cat([c[k] for c in chunks_out], dim=0)
        else:
            out[k] = v0
    return out


def generate_trajectory_ensemble_batched(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps,
    sampler_generate_kwargs=None,
    free_generate_kwargs=None,
    obs_every=1,
    obs_offset=0,
    reinit_every=None,
    target_var="co2massmix",
    device="cuda",
    seed=42,
    chunk_size=None,
    verbose=False,
):
    """Batched (init × sample) auto-regressive ensemble rollout with optional
    per-step posterior conditioning.

    This is the canonical Phase 25b generator. Improvements over
    ``generate_trajectory_ensemble_with_obs``:
      * **Flat batching**: all `n_inits × n_samples` trajectories are
        propagated in a single forward call per step (or `chunk_size`
        sub-batches if VRAM-limited). Wallclock scales sub-linearly in
        `n_inits` rather than linearly.
      * **Per-trajectory storage**: individual sample trajectories are kept;
        ensemble mean / spread are computed posthoc.
      * **No early collapse** to ensemble mean: the rolled state stays
        per-sample throughout.

    `sampler_generate_kwargs` is None ⟺ unconditional AR baseline.
    """
    import numpy as _np

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    inner.generating = True

    has_obs = sampler_generate_kwargs is not None

    if has_obs and free_generate_kwargs is None:
        free_generate_kwargs = {
            k: v for k, v in sampler_generate_kwargs.items() if k not in ("sampler", "sampler_params", "masking")
        }
        free_generate_kwargs["masking"] = False

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon

    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    assert N == nlat * nlon

    BATCH = n_inits * n_samples
    if chunk_size is None or chunk_size >= BATCH:
        chunk_size = BATCH

    # Build initial rolled state [BATCH, ...] from per-init init batch
    init_batches = [data_loader.get_batch(i, device=device) for i in init_indices]
    co2_shape = init_batches[0][target_var].shape[1:]  # without batch dim
    current_co2 = torch.empty(BATCH, *co2_shape, device=device, dtype=init_batches[0][target_var].dtype)
    for i, ib in enumerate(init_batches):
        current_co2[i * n_samples : (i + 1) * n_samples] = ib[target_var]

    obs_present = _np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        obs_present[k] = has_obs and ((k - obs_offset) >= 0) and ((k - obs_offset) % obs_every == 0)

    if has_obs:
        mask_pattern = sampler_generate_kwargs.get("mask_pattern", "satellite")
        obs_fraction = sampler_generate_kwargs.get("obs_fraction", 0.3)
        ak_10 = sampler_generate_kwargs.get("ak_10", None)
        soft_boundary_sigma = sampler_generate_kwargs.get("soft_boundary_sigma", 0.0)

    all_preds = _np.full((n_inits, n_samples, n_steps, nlat, nlon, C), _np.nan, dtype=_np.float32)
    time_coord = _np.empty((n_inits, n_steps), dtype=times_axis.dtype)

    iters = range(n_steps)
    if verbose:
        iters = tqdm(iters, desc="Batched AR rollout", total=n_steps)

    for k in iters:
        # Build per-init batches at lead k, then expand to BATCH dim
        batches_per_init = []
        for i_pos, init_idx in enumerate(init_indices):
            idx = init_idx + k
            if idx + 1 >= T_total:
                logger.warning("Init %d hit end at lead %d", init_idx, k)
                batches_per_init.append(None)
                continue
            b = data_loader.get_batch(idx, device=device)
            batches_per_init.append(b)
            time_coord[i_pos, k] = times_axis[idx + 1]

        if any(b is None for b in batches_per_init):
            break

        # Stack & expand each tensor field across n_samples
        batch = {}
        for key, v in batches_per_init[0].items():
            if isinstance(v, torch.Tensor):
                stacked = torch.cat(
                    [bb[key].expand(n_samples, *bb[key].shape[1:]) for bb in batches_per_init],
                    dim=0,
                )
                batch[key] = stacked
            else:
                batch[key] = v

        if reinit_every is not None and k > 0 and k % reinit_every == 0:
            current_co2 = batch[target_var].clone()

        batch[target_var] = current_co2
        if f"{target_var}_next" in batch:
            batch[f"{target_var}_next"] = current_co2.clone()

        use_obs = bool(obs_present[k])

        if use_obs:
            # Build synthetic XCO2 mask from GT next field
            gt_next_per_init = [data_loader.get_batch(init_indices[i] + k + 1, device=device) for i in range(n_inits)]
            gt_next_stacked = torch.cat(
                [b[target_var].expand(n_samples, *b[target_var].shape[1:]) for b in gt_next_per_init],
                dim=0,
            )
            obs_batch = {kk: vv for kk, vv in batch.items()}
            obs_batch[target_var] = gt_next_stacked
            om, ov = create_column_mask(
                obs_batch,
                target_var=target_var,
                obs_fraction=obs_fraction,
                mask_pattern=mask_pattern,
                nlat=nlat,
                nlon=nlon,
                ak_10=ak_10,
                soft_boundary_sigma=soft_boundary_sigma,
            )
            obs_batch["obs_mask"] = om
            obs_batch["obs_values"] = ov
            ov_normed = inner.normalize_observations(ov, obs_batch, target_var=target_var, targshift=False)
            obs_batch["obs_values"] = torch.nan_to_num(ov_normed, nan=0.0)
            for key in (
                "obs_mask",
                "obs_values",
                "obs_weight",
                "xco2_averaging_kernel",
                "xco2_apriori",
                "co2_profile_apriori",
                "pressure_weight",
            ):
                if key in obs_batch:
                    batch[key] = obs_batch[key]

            inner.generate_kwargs = sampler_generate_kwargs
            preds = _model_call_chunked(model, batch, chunk_size, BATCH, no_grad=False)
        else:
            if has_obs:
                inner.generate_kwargs = free_generate_kwargs
            else:
                # Unconditional: merge user-passed free_kwargs (e.g. noise_scale)
                # over the minimal default.
                base_uncond = {"n_samples": 1, "masking": False}
                if free_generate_kwargs:
                    base_uncond = {**base_uncond, **free_generate_kwargs}
                inner.generate_kwargs = base_uncond
            preds = _model_call_chunked(model, batch, chunk_size, BATCH, no_grad=True)

        pred_co2 = preds[target_var]
        current_co2 = pred_co2.detach()
        pred_np = current_co2.cpu().numpy().reshape(n_inits, n_samples, nlat, nlon, C)
        all_preds[:, :, k] = pred_np

    inner.generating = prev_generating

    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels
    ds = xr.Dataset(
        {
            target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
            "obs_present": (("lead",), obs_present),
        },
        coords={
            "init": _np.array(init_indices),
            "sample": _np.arange(n_samples),
            "lead": _np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def generate_trajectory_window_dflow(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps,
    obs_kwargs,
    free_kwargs=None,
    obs_every=1,
    obs_offset=0,
    window_size=4,
    window_stride=None,
    n_opt_steps=20,
    lr=1e-2,
    sigma_obs=0.1,
    reg_weight=0.0,
    target_var="co2massmix",
    chunk_size=None,
    device="cuda",
    seed=42,
    use_checkpointing=True,
    verbose=False,
):
    """Window D-Flow auto-regressive sampler (Phase 25h).

    For each window of ``window_size`` AR steps, optimize a per-window noise
    tensor ``z[W,BATCH,N,C]`` to fit observations across all W steps jointly.
    Backprops through the full chain residual_FM(z[w]) -> f_det(state_{w-1})
    -> ... at every AR step within the window.

    The optimizer steps ``n_opt_steps`` Adam iterations; gradient checkpointing
    on each per-step forward keeps memory near a single forward.

    Notes
    -----
    * Designed for ``ResidualFlowMatching``: requires ``model.model.f_det`` so
      gradient can chain across AR steps. The wrapper sets
      ``inner.enable_det_grad`` during the optimization phase.
    * obs_kwargs is the standard generate_kwargs dict (mask_pattern, obs_fraction,
      ak_10, soft_boundary_sigma); same key set as for the existing samplers.
    * Loss is computed in *normalized* observation space using the model's own
      forward_model (XCO2 column average). Equivalent to the dflow sampler's
      likelihood term but evaluated on the post-residual state, not the
      residual itself.
    * Stride: ``window_stride`` (default = window_size, non-overlapping). With
      stride < window_size, windows overlap and earlier steps get refined
      again from the next window.
    """
    import numpy as _np
    from torch.utils.checkpoint import checkpoint as _ckpt

    from neural_transport.forward_model import XCO2ForwardModel

    if window_stride is None:
        window_stride = window_size
    assert 1 <= window_stride <= window_size

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    prev_det_grad = getattr(inner, "enable_det_grad", False)
    inner.generating = True

    if not hasattr(inner, "f_det"):
        raise RuntimeError(
            "window_dflow requires a ResidualFlowMatching model (need f_det for gradient chaining across AR steps)."
        )

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon

    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    assert N == nlat * nlon

    BATCH = n_inits * n_samples
    if chunk_size is None or chunk_size >= BATCH:
        chunk_size = BATCH

    init_batches = [data_loader.get_batch(i, device=device) for i in init_indices]
    co2_shape = init_batches[0][target_var].shape[1:]
    current_co2 = torch.empty(BATCH, *co2_shape, device=device, dtype=init_batches[0][target_var].dtype)
    for i, ib in enumerate(init_batches):
        current_co2[i * n_samples : (i + 1) * n_samples] = ib[target_var]

    obs_present = _np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        obs_present[k] = ((k - obs_offset) >= 0) and ((k - obs_offset) % obs_every == 0)

    mask_pattern = obs_kwargs.get("mask_pattern", "satellite")
    obs_fraction = obs_kwargs.get("obs_fraction", 0.3)
    ak_10 = obs_kwargs.get("ak_10", None)
    soft_boundary_sigma = obs_kwargs.get("soft_boundary_sigma", 0.0)
    noise_scale = float(obs_kwargs.get("noise_scale", 1.0))

    # Free generate_kwargs used for the inner model.forward calls during opt
    # — we want unconditional generation per step (no internal posterior
    # conditioning); window-D-Flow handles obs externally.
    if free_kwargs is None:
        free_kwargs = {
            k: v
            for k, v in obs_kwargs.items()
            if k
            not in (
                "sampler",
                "sampler_params",
                "masking",
                "obs_mask",
                "obs_values",
                "obs_weight",
                "obs_fraction",
                "mask_source",
                "mask_pattern",
                "ak_10",
                "soft_boundary_sigma",
            )
        }
    free_kwargs = {**free_kwargs, "masking": False}

    all_preds = _np.full((n_inits, n_samples, n_steps, nlat, nlon, C), _np.nan, dtype=_np.float32)
    time_coord = _np.empty((n_inits, n_steps), dtype=times_axis.dtype)

    def _step_batch(k):
        """Build the same per-step batch dict that ``_batched`` uses, with
        ``current_co2`` inserted as target_var. Returns dict with all fields
        replicated over n_samples but WITHOUT the rolled state injected — caller
        should set ``out[target_var] = state``.
        """
        batches_per_init = []
        for i_pos, init_idx in enumerate(init_indices):
            idx = init_idx + k
            if idx + 1 >= T_total:
                return None, None
            b = data_loader.get_batch(idx, device=device)
            batches_per_init.append(b)
            time_coord[i_pos, k] = times_axis[idx + 1]
        batch = {}
        for key, v in batches_per_init[0].items():
            if isinstance(v, torch.Tensor):
                batch[key] = torch.cat(
                    [bb[key].expand(n_samples, *bb[key].shape[1:]) for bb in batches_per_init],
                    dim=0,
                )
            else:
                batch[key] = v
        # gt[t+1] for the obs ground truth
        gt_next_per_init = [data_loader.get_batch(init_indices[i] + k + 1, device=device) for i in range(n_inits)]
        gt_next = torch.cat(
            [b[target_var].expand(n_samples, *b[target_var].shape[1:]) for b in gt_next_per_init],
            dim=0,
        )
        return batch, gt_next

    def _build_obs(batch, gt_next):
        """Build forward_model + obs_values_norm + obs_weight_or_mask for this step."""
        obs_batch = {kk: vv for kk, vv in batch.items()}
        obs_batch[target_var] = gt_next
        om, ov = create_column_mask(
            obs_batch,
            target_var=target_var,
            obs_fraction=obs_fraction,
            mask_pattern=mask_pattern,
            nlat=nlat,
            nlon=nlon,
            ak_10=ak_10,
            soft_boundary_sigma=soft_boundary_sigma,
        )
        obs_batch["obs_mask"] = om
        obs_batch["obs_values"] = ov
        ov_norm = inner.normalize_observations(ov, obs_batch, target_var=target_var, targshift=False)
        ov_norm = torch.nan_to_num(ov_norm, nan=0.0)
        # Build a masking_config compatible with XCO2ForwardModel, mirroring
        # FlowMatching.prepare_masking_config().
        B_ = ov_norm.shape[0]
        if "xco2_averaging_kernel" in obs_batch:
            obs_mask = om.reshape(B_, nlat, nlon, 1).permute(0, 3, 1, 2)
            obs_values = ov_norm.reshape(B_, nlat, nlon, 1).permute(0, 3, 1, 2)
            ak = obs_batch["xco2_averaging_kernel"].reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
            xco2_prior = obs_batch["xco2_apriori"].reshape(B_, nlat, nlon, 1).permute(0, 3, 1, 2)
            co2_profile_prior = obs_batch["co2_profile_apriori"].reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
            pressure_weights = (
                obs_batch["pressure_weight"].reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
                if "pressure_weight" in obs_batch
                else None
            )
        else:
            obs_mask = om.reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
            obs_values = ov_norm.reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
            ak = xco2_prior = co2_profile_prior = pressure_weights = None
        obs_weight = None
        if "obs_weight" in obs_batch:
            ow = obs_batch["obs_weight"]
            if "xco2_averaging_kernel" in obs_batch:
                obs_weight = ow.reshape(B_, nlat, nlon, 1).permute(0, 3, 1, 2)
            else:
                obs_weight = ow.reshape(B_, nlat, nlon, C).permute(0, 3, 1, 2)
        target_mean = batch[f"{target_var}_offset"].reshape(B_, 1, 1, 1)
        target_std = batch[f"{target_var}_scale"].reshape(B_, 1, 1, 1)
        obs_mean = target_mean
        obs_std = target_std
        # We pass targshift=False to normalize_observations above, so the H
        # operator must NOT subtract targshift either — set to None.
        targshift_mean = None
        masking_config = dict(
            obs_mask=obs_mask,
            obs_values=obs_values,
            obs_mean=obs_mean,
            obs_std=obs_std,
            target_mean=target_mean,
            target_std=target_std,
            ak=ak,
            xco2_prior=xco2_prior,
            co2_profile_prior=co2_profile_prior,
            pressure_weights=pressure_weights,
            obs_weight=obs_weight,
            targshift_mean=targshift_mean,
        )
        fm = XCO2ForwardModel.from_masking_config(masking_config)
        return fm, obs_values, obs_mask, obs_weight, target_mean, target_std

    def _state_to_grid_norm(state_phys, target_mean, target_std):
        """[B, T=1, N, C] or [B, N, C] phys -> [B, C, Nlat, Nlon] normalized
        (no targshift; H operator handles it via masking_config.targshift_mean).
        """
        if state_phys.dim() == 4:
            state_phys = state_phys.squeeze(1)
        B_, N_, C_ = state_phys.shape
        x = (state_phys - target_mean.view(B_, 1, 1)) / target_std.view(B_, 1, 1)
        return x.reshape(B_, nlat, nlon, C_).permute(0, 3, 1, 2)

    def _step_forward(batch, state, z_w, *, with_grad=False):
        """One AR step with grad. state and z_w: [B, T=1, N, C].
        Returns next_state [B, T=1, N, C].

        We call ``inner`` (the FM/ResidualFM module) directly instead of the
        Lightning wrapper because the wrapper writes preds via in-place
        ``preds[v][:, t] = ...`` into a freshly-allocated empty tensor, which
        breaks the gradient chain (no grad_fn on the destination).
        """
        # Strip T dim for the inner module (which expects [B, N, C]).
        curr = {}
        T_dim = 1
        for k_, v_ in batch.items():
            if isinstance(v_, torch.Tensor) and v_.ndim >= 2 and v_.shape[1] == T_dim:
                curr[k_] = v_[:, 0]
            else:
                curr[k_] = v_
        curr[target_var] = state[:, 0] if state.dim() >= 4 else state
        if f"{target_var}_next" in curr:
            curr[f"{target_var}_next"] = curr[target_var]
        curr["noise"] = z_w[:, 0] if z_w.dim() >= 4 else z_w
        inner.generate_kwargs = {**free_kwargs, "enable_grad": True} if with_grad else free_kwargs
        out = inner(curr, mode="generate")
        # Re-add T dim to keep state shape consistent with current_co2 [B, T, N, C].
        nxt = out[target_var]
        if nxt.dim() == 3:
            nxt = nxt.unsqueeze(1)
        return nxt

    def _checkpointed_step(batch, state, z_w):
        if not use_checkpointing:
            return _step_forward(batch, state, z_w, with_grad=True)

        # Wrap so checkpoint sees Tensor inputs only; capture batch by closure.
        def _fn(state_, z_):
            return _step_forward(batch, state_, z_, with_grad=True)

        return _ckpt(_fn, state, z_w, use_reentrant=False)

    iters = range(0, n_steps, window_stride)
    if verbose:
        iters = tqdm(iters, desc="Window-DFlow rollout", total=(n_steps + window_stride - 1) // window_stride)

    for k0 in iters:
        W_eff = min(window_size, n_steps - k0)
        if W_eff <= 0:
            break

        # Pre-build per-step batches and obs (no grad needed for these).
        step_batches = []
        step_gt = []
        step_obs = []  # (forward_model, obs_values_norm, obs_mask, obs_weight, target_mean, target_std) or None
        valid_W = 0
        for w in range(W_eff):
            k = k0 + w
            with torch.no_grad():
                bw, gtw = _step_batch(k)
            if bw is None:
                break
            valid_W += 1
            step_batches.append(bw)
            step_gt.append(gtw)
            if obs_present[k]:
                with torch.no_grad():
                    step_obs.append(_build_obs(bw, gtw))
            else:
                step_obs.append(None)

        if valid_W == 0:
            break
        W_eff = valid_W

        # Process the BATCH dimension in chunks; init / sample axes are
        # independent so per-chunk Adam optimization is exact (not an
        # approximation). Within a chunk, z is shared across W AR steps.
        any_obs = any(o is not None for o in step_obs)
        carry_state = torch.empty_like(current_co2)
        commit_w = min(window_stride, W_eff) - 1  # 0-indexed step whose state we carry

        for s_ in range(0, BATCH, chunk_size):
            e_ = min(s_ + chunk_size, BATCH)
            chunk_B = e_ - s_

            def _slice_batch(b):
                out = {}
                for kk, vv in b.items():
                    if isinstance(vv, torch.Tensor) and vv.shape[0] == BATCH:
                        out[kk] = vv[s_:e_]
                    else:
                        out[kk] = vv
                return out

            chunk_step_batches = [_slice_batch(b) for b in step_batches]
            chunk_step_obs = []
            for o in step_obs:
                if o is None:
                    chunk_step_obs.append(None)
                    continue
                fm, ov_norm, om_grid, ow_grid, tm, ts = o
                fm_chunk = XCO2ForwardModel(
                    pressure_weights=fm.pressure_weights[s_:e_] if fm.pressure_weights is not None else None,
                    ak=fm.ak[s_:e_] if fm.ak is not None else None,
                    xco2_prior=fm.xco2_prior[s_:e_] if fm.xco2_prior is not None else None,
                    co2_profile_prior=fm.co2_profile_prior[s_:e_] if fm.co2_profile_prior is not None else None,
                    obs_mean=fm.obs_mean[s_:e_]
                    if fm.obs_mean is not None and fm.obs_mean.shape[0] == BATCH
                    else fm.obs_mean,
                    obs_std=fm.obs_std[s_:e_]
                    if fm.obs_std is not None and fm.obs_std.shape[0] == BATCH
                    else fm.obs_std,
                    target_mean=fm.target_mean[s_:e_]
                    if fm.target_mean is not None and fm.target_mean.shape[0] == BATCH
                    else fm.target_mean,
                    target_std=fm.target_std[s_:e_]
                    if fm.target_std is not None and fm.target_std.shape[0] == BATCH
                    else fm.target_std,
                    targshift_mean=fm.targshift_mean[s_:e_]
                    if fm.targshift_mean is not None and fm.targshift_mean.shape[0] == BATCH
                    else fm.targshift_mean,
                )
                chunk_step_obs.append(
                    (
                        fm_chunk,
                        ov_norm[s_:e_],
                        om_grid[s_:e_],
                        ow_grid[s_:e_] if ow_grid is not None else None,
                        tm[s_:e_],
                        ts[s_:e_],
                    )
                )

            gen = torch.Generator(device=device).manual_seed(seed + 1000 * k0 + s_)
            # Match data_loader's [B, T=1, N, C] layout — litmodule slices [:, t].
            z_param = torch.randn(W_eff, chunk_B, 1, N, C, device=device, generator=gen) * noise_scale
            z_param = z_param.contiguous().requires_grad_(True)
            opt = torch.optim.Adam([z_param], lr=lr)

            chunk_init_state = current_co2[s_:e_]

            if any_obs and n_opt_steps > 0:
                inner.enable_det_grad = True
                with torch.enable_grad():
                    for _ in range(n_opt_steps):
                        opt.zero_grad()
                        loss_total = z_param.new_zeros(())
                        state = chunk_init_state
                        for w in range(W_eff):
                            state = _checkpointed_step(chunk_step_batches[w], state, z_param[w])
                            if chunk_step_obs[w] is not None:
                                fm_, ov_, om_, ow_, _tm, _ts = chunk_step_obs[w]
                                x_grid = _state_to_grid_norm(state, _tm, _ts)
                                xco2_pred = fm_.forward(x_grid)
                                if ow_ is not None:
                                    resid = ow_ * (xco2_pred - ov_)
                                else:
                                    resid = torch.where(
                                        om_.bool(),
                                        xco2_pred - ov_,
                                        torch.zeros_like(xco2_pred),
                                    )
                                loss_total = loss_total + resid.pow(2).sum() / (2.0 * sigma_obs**2 * chunk_B)
                        if reg_weight > 0:
                            loss_total = loss_total + reg_weight * z_param.pow(2).sum() / (
                                chunk_B * z_param[0, 0].numel()
                            )
                        loss_total.backward()
                        opt.step()
                inner.enable_det_grad = prev_det_grad

            # Final no-grad forward; record per-step states for this chunk.
            inner.enable_det_grad = False
            z_final = z_param.detach()
            with torch.no_grad():
                state = chunk_init_state
                for w in range(W_eff):
                    state = _step_forward(chunk_step_batches[w], state, z_final[w])
                    state_flat = state if state.dim() == 3 else state.squeeze(1)
                    pred_np = state_flat.detach().cpu().numpy()
                    chunk_pred = pred_np.reshape(chunk_B, nlat, nlon, C)
                    # Map BATCH index back to (init, sample): BATCH = init * n_samples + sample
                    for bi in range(chunk_B):
                        global_b = s_ + bi
                        i_idx = global_b // n_samples
                        s_idx = global_b % n_samples
                        all_preds[i_idx, s_idx, k0 + w] = chunk_pred[bi]
                    if w == commit_w:
                        carry_state[s_:e_] = state.detach()
                # Last-window safeguard: if we never hit commit_w (W_eff < stride),
                # take final state.
                if commit_w >= W_eff:
                    carry_state[s_:e_] = state.detach()
            del z_param, z_final, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        current_co2 = carry_state

    inner.generating = prev_generating
    inner.enable_det_grad = prev_det_grad

    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels
    ds = xr.Dataset(
        {
            target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
            "obs_present": (("lead",), obs_present),
        },
        coords={
            "init": _np.array(init_indices),
            "sample": _np.arange(n_samples),
            "lead": _np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def generate_trajectory_enkf(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps,
    obs_kwargs,
    free_kwargs=None,
    sampler_kwargs=None,
    obs_every=1,
    obs_offset=0,
    sigma_obs=0.1,
    inflation=1.0,
    prior_inflation=1.0,
    loc_sigma=0.0,
    loc_min_presence=0.05,
    global_bias_correct=False,
    target_var="co2massmix",
    device="cuda",
    seed=42,
    chunk_size=None,
    verbose=False,
):
    """Phase 25i: Free-running ensemble + per-cell vertical EnKF post-hoc update.

    At each AR step, run the unconditional model to obtain a predicted ensemble.
    At observation steps, apply a stochastic (perturbed-obs) Ensemble Kalman
    update on every observed grid cell, using only that cell's column ensemble.

    The XCO2 forward operator is linear: y = sum_l(h_l * a_l * x_l). Per-cell
    update keeps the algorithm O(M * C^2) and avoids the global rank-deficiency
    problem of N_samp << N_grid sample covariances. No horizontal info is shared
    directly; obs information propagates spatially through the model dynamics.

    Args
    ----
    sigma_obs : float
        Observation error std in physical XCO2 units (kg/kg → ppm scale on
        co2massmix). Acts as a regularizer; pure-OSSE perfect obs only need
        small sigma_obs > 0 for stable Kalman gain.
    inflation : float
        Multiplicative inflation factor applied to ensemble anomalies right
        after every EnKF update to counter ensemble shrinkage.
    obs_kwargs : dict
        Sampler kwargs from configs (used for mask_pattern / obs_fraction /
        ak_10 / soft_boundary_sigma / noise_scale).
    free_kwargs : dict | None
        Unconditional generate_kwargs.
    """
    import numpy as _np

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    inner.generating = True

    if free_kwargs is None:
        free_kwargs = {"n_samples": 1, "masking": False}

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon

    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    assert N == nlat * nlon

    BATCH = n_inits * n_samples
    if chunk_size is None or chunk_size >= BATCH:
        chunk_size = BATCH

    init_batches = [data_loader.get_batch(i, device=device) for i in init_indices]
    co2_shape = init_batches[0][target_var].shape[1:]
    current_co2 = torch.empty(BATCH, *co2_shape, device=device, dtype=init_batches[0][target_var].dtype)
    for i, ib in enumerate(init_batches):
        current_co2[i * n_samples : (i + 1) * n_samples] = ib[target_var]

    obs_present = _np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        obs_present[k] = ((k - obs_offset) >= 0) and ((k - obs_offset) % obs_every == 0)

    mask_pattern = obs_kwargs.get("mask_pattern", "satellite")
    obs_fraction = obs_kwargs.get("obs_fraction", 0.3)
    ak_10 = obs_kwargs.get("ak_10", None)
    soft_boundary_sigma = obs_kwargs.get("soft_boundary_sigma", 0.0)

    all_preds = _np.full((n_inits, n_samples, n_steps, nlat, nlon, C), _np.nan, dtype=_np.float32)
    time_coord = _np.empty((n_inits, n_steps), dtype=times_axis.dtype)

    iters = range(n_steps)
    if verbose:
        iters = tqdm(iters, desc="EnKF trajectory", total=n_steps)

    enkf_rng = torch.Generator(device=device).manual_seed(seed + 1)

    for k in iters:
        batches_per_init = []
        for i_pos, init_idx in enumerate(init_indices):
            idx = init_idx + k
            if idx + 1 >= T_total:
                logger.warning("Init %d hit end at lead %d", init_idx, k)
                batches_per_init.append(None)
                continue
            b = data_loader.get_batch(idx, device=device)
            batches_per_init.append(b)
            time_coord[i_pos, k] = times_axis[idx + 1]

        if any(b is None for b in batches_per_init):
            break

        batch = {}
        for key, v in batches_per_init[0].items():
            if isinstance(v, torch.Tensor):
                batch[key] = torch.cat(
                    [bb[key].expand(n_samples, *bb[key].shape[1:]) for bb in batches_per_init],
                    dim=0,
                )
            else:
                batch[key] = v

        batch[target_var] = current_co2
        if f"{target_var}_next" in batch:
            batch[f"{target_var}_next"] = current_co2.clone()

        # If hybrid (FMPS prior + EnKF post), build obs BEFORE the model call
        # so the sampler can use them. Otherwise build them after the free
        # forward — needed only for the EnKF update.
        obs_built = False
        if obs_present[k] and sampler_kwargs is not None:
            gt_next = [data_loader.get_batch(init_indices[i] + k + 1, device=device) for i in range(n_inits)]
            gt_stacked = torch.cat(
                [b[target_var].expand(n_samples, *b[target_var].shape[1:]) for b in gt_next],
                dim=0,
            )
            obs_batch = {kk: vv for kk, vv in batch.items()}
            obs_batch[target_var] = gt_stacked
            om, ov = create_column_mask(
                obs_batch,
                target_var=target_var,
                obs_fraction=obs_fraction,
                mask_pattern=mask_pattern,
                nlat=nlat,
                nlon=nlon,
                ak_10=ak_10,
                soft_boundary_sigma=soft_boundary_sigma,
            )
            obs_batch["obs_mask"] = om
            ov_normed = inner.normalize_observations(ov, obs_batch, target_var=target_var, targshift=False)
            obs_batch["obs_values"] = torch.nan_to_num(ov_normed, nan=0.0)
            for key in (
                "obs_mask",
                "obs_values",
                "obs_weight",
                "xco2_averaging_kernel",
                "xco2_apriori",
                "co2_profile_apriori",
                "pressure_weight",
            ):
                if key in obs_batch:
                    batch[key] = obs_batch[key]
            inner.generate_kwargs = sampler_kwargs
            obs_built = True
        else:
            inner.generate_kwargs = free_kwargs

        preds = _model_call_chunked(model, batch, chunk_size, BATCH, no_grad=True)
        pred_co2 = preds[target_var].detach()  # [BATCH, N, C]

        if obs_present[k]:
            # Optional prior inflation: widen ensemble around its mean BEFORE
            # the EnKF update (Anderson 2007). Standard fix for ensemble
            # collapse over chained DA cycles. Applied per-init so each init's
            # ensemble inflates around its own mean.
            if prior_inflation != 1.0:
                pred_grid_pre = pred_co2.view(n_inits, n_samples, N, C)
                m_pre = pred_grid_pre.mean(dim=1, keepdim=True)
                pred_grid_pre = m_pre + prior_inflation * (pred_grid_pre - m_pre)
                pred_co2 = pred_grid_pre.view(*pred_co2.shape)
            if not obs_built:
                # Build GT obs and pressure_weight / averaging kernel for EnKF.
                gt_next = [data_loader.get_batch(init_indices[i] + k + 1, device=device) for i in range(n_inits)]
                gt_stacked = torch.cat(
                    [b[target_var].expand(n_samples, *b[target_var].shape[1:]) for b in gt_next],
                    dim=0,
                )
                obs_batch = {kk: vv for kk, vv in batch.items()}
                obs_batch[target_var] = gt_stacked
                om, ov = create_column_mask(
                    obs_batch,
                    target_var=target_var,
                    obs_fraction=obs_fraction,
                    mask_pattern=mask_pattern,
                    nlat=nlat,
                    nlon=nlon,
                    ak_10=ak_10,
                    soft_boundary_sigma=soft_boundary_sigma,
                )
            # om: [BATCH, T=1, N, 1] bool ; ov: [BATCH, T=1, N, 1] physical XCO2
            # pressure_weight & ak attached to obs_batch (and thus batch via shared dict)
            hak = (obs_batch["pressure_weight"] * obs_batch["xco2_averaging_kernel"]).squeeze(1)  # [BATCH, N, C]

            # Per-init slice: ensemble is consecutive n_samples. Mask & y_obs are
            # identical across the n_samples axis (built from GT, expanded), so
            # slice [0] for each init.
            mask_full = om.squeeze(-1).squeeze(1)  # [BATCH, N] bool
            y_full = ov.squeeze(-1).squeeze(1)  # [BATCH, N] physical XCO2 (NaN at unobserved)

            x_grid = pred_co2.view(n_inits, n_samples, N, C)
            for i in range(n_inits):
                mask_i = mask_full[i * n_samples]  # [N]
                y_i = y_full[i * n_samples]  # [N]
                hak_i = hak[i * n_samples]  # [N, C]
                obs_idx = mask_i.nonzero(as_tuple=True)[0]
                if obs_idx.numel() == 0:
                    continue
                col = x_grid[i, :, obs_idx, :]  # [n_samples, M, C]
                hak_obs = hak_i[obs_idx]  # [M, C]
                y_obs = y_i[obs_idx]  # [M]

                y_pred = (col * hak_obs.unsqueeze(0)).sum(dim=-1)  # [n_samples, M]
                y_mean = y_pred.mean(dim=0, keepdim=True)  # [1, M]
                y_anom = y_pred - y_mean  # [n_samples, M]
                x_mean = col.mean(dim=0, keepdim=True)  # [1, M, C]
                x_anom = col - x_mean  # [n_samples, M, C]

                denom = max(n_samples - 1, 1)
                Cov_xy = (x_anom * y_anom.unsqueeze(-1)).sum(dim=0) / denom  # [M, C]
                Var_y = (y_anom**2).sum(dim=0) / denom  # [M]
                K = Cov_xy / (Var_y.unsqueeze(-1) + sigma_obs**2)  # [M, C]

                eps = sigma_obs * torch.randn(n_samples, obs_idx.numel(), device=device, generator=enkf_rng)
                innovation = (y_obs.unsqueeze(0) + eps) - y_pred  # [n_samples, M]
                delta = K.unsqueeze(0) * innovation.unsqueeze(-1)  # [n_samples, M, C] — increment at obs cells

                if loc_sigma > 0.0:
                    # Horizontal localization: scatter increment onto full grid,
                    # Gaussian-smooth, normalize by smoothed obs-presence mask.
                    # This spreads each obs cell's increment to its neighbours
                    # so unobserved cells also get updated (FMPS does this
                    # implicitly via spatial_smoothing_sigma).
                    inc_grid = torch.zeros(n_samples, C, N, device=device, dtype=delta.dtype)
                    inc_grid[:, :, obs_idx] = delta.permute(0, 2, 1)  # [n_samples, C, M]
                    pres = torch.zeros(1, 1, N, device=device, dtype=delta.dtype)
                    pres[:, :, obs_idx] = 1.0
                    inc_grid = inc_grid.view(n_samples, C, nlat, nlon)
                    pres = pres.view(1, 1, nlat, nlon)
                    inc_smooth = gaussian_smooth_2d(inc_grid, loc_sigma)
                    pres_smooth = gaussian_smooth_2d(pres, loc_sigma)
                    inc_norm = inc_smooth / (pres_smooth + 1e-6)
                    if loc_min_presence > 0:
                        inc_norm = inc_norm * (pres_smooth > loc_min_presence).to(inc_norm.dtype)
                    inc_flat = inc_norm.view(n_samples, C, N).permute(0, 2, 1)  # [n_samples, N, C]
                    x_slab = x_grid[i] + inc_flat
                    if inflation != 1.0:
                        m = x_slab.mean(dim=0, keepdim=True)
                        x_slab = m + inflation * (x_slab - m)
                    x_grid[i] = x_slab
                else:
                    col_new = col + delta
                    if inflation != 1.0:
                        new_mean = col_new.mean(dim=0, keepdim=True)
                        col_new = new_mean + inflation * (col_new - new_mean)
                    x_grid[i, :, obs_idx, :] = col_new

            current_co2 = x_grid.view(*pred_co2.shape)

            if global_bias_correct:
                # Global drift correction: at each obs step, snap the
                # ensemble-mean global XCO2 mean to match observed XCO2 mean
                # (over obs cells). The residual-FM model has documented
                # ~0.029 ppm/step linear drift; per-cell EnKF only touches
                # obs cells, so global drift survives. We add a uniform
                # CO2 shift in mass-mixing-ratio units, computed per-init.
                x_grid_post = current_co2.view(n_inits, n_samples, N, C)
                hak = (obs_batch["pressure_weight"] * obs_batch["xco2_averaging_kernel"]).squeeze(1)
                hak_grid = hak.view(n_inits, n_samples, N, C)
                for i in range(n_inits):
                    mask_i = mask_full[i * n_samples]
                    obs_idx = mask_i.nonzero(as_tuple=True)[0]
                    if obs_idx.numel() == 0:
                        continue
                    # Predicted XCO2 over obs cells, ensemble-mean.
                    col = x_grid_post[i, :, obs_idx, :]  # [n_samples,M,C]
                    hak_obs = hak_grid[i, :, obs_idx, :]  # [n_samples,M,C]
                    y_pred_post = (col * hak_obs).sum(dim=-1).mean(dim=0)  # [M]
                    y_obs_post = y_full[i * n_samples][obs_idx]  # [M]
                    delta_y = (y_obs_post - y_pred_post).mean()  # scalar
                    sum_hak_mean = hak_obs.mean(dim=(0, 1)).sum()  # scalar
                    if sum_hak_mean.abs() > 1e-9:
                        delta_x = delta_y / sum_hak_mean
                        x_grid_post[i] = x_grid_post[i] + delta_x
                current_co2 = x_grid_post.view(*pred_co2.shape)
        else:
            current_co2 = pred_co2

        pred_np = current_co2.cpu().numpy().reshape(n_inits, n_samples, nlat, nlon, C)
        all_preds[:, :, k] = pred_np

    inner.generating = prev_generating

    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels
    ds = xr.Dataset(
        {
            target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
            "obs_present": (("lead",), obs_present),
        },
        coords={
            "init": _np.array(init_indices),
            "sample": _np.arange(n_samples),
            "lead": _np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def generate_trajectory_enks(
    model,
    data_loader,
    *,
    init_indices,
    n_samples,
    n_steps,
    obs_kwargs,
    free_kwargs=None,
    obs_every=1,
    obs_offset=0,
    sigma_obs=0.1,
    inflation=1.0,
    prior_inflation=1.0,
    loc_sigma=0.0,
    loc_min_presence=0.05,
    lag=24,
    damping=1.0,
    target_var="co2massmix",
    device="cuda",
    seed=42,
    chunk_size=None,
    verbose=False,
):
    """Phase 25n: Free-running ensemble + fixed-lag Ensemble Kalman Smoother.

    Same forward pass as ``generate_trajectory_enkf``: at every step the
    unconditional model produces a predicted ensemble. At obs step ``k`` we
    apply a per-cell vertical EnKF update to the *current* state AND propagate
    that increment backward through the last ``lag`` steps using the sample
    cross-covariance ``Cov(x_{k-l}, y_k)``. Each past state is smoothed by
    every observation that falls within ``lag`` steps after it.

    Memory: ``lag`` × BATCH × N × C float32 — ~2 GB for lag=120, BATCH=200.

    Per-cell + horizontal Gaussian localization mirror the EnKF. Forward
    propagation uses the EnKF-updated current state (so the dynamics see the
    latest information), while past states are corrected post-hoc only.
    """
    import numpy as _np

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model.eval()
    model.to(device)
    inner = getattr(model, "model", model)
    prev_generating = getattr(inner, "generating", False)
    inner.generating = True

    if free_kwargs is None:
        free_kwargs = {"n_samples": 1, "masking": False}
    inner.generate_kwargs = free_kwargs

    dataset = data_loader.dataset
    n_inits = len(init_indices)
    times_axis = dataset.ds.time.values
    T_total = len(data_loader)
    nlat = data_loader.grid_info.nlat
    nlon = data_loader.grid_info.nlon

    sample0 = dataset[init_indices[0]][target_var]
    if sample0.ndim == 3:
        _, N, C = sample0.shape
    else:
        N, C = sample0.shape
    assert N == nlat * nlon

    BATCH = n_inits * n_samples
    if chunk_size is None or chunk_size >= BATCH:
        chunk_size = BATCH

    init_batches = [data_loader.get_batch(i, device=device) for i in init_indices]
    co2_shape = init_batches[0][target_var].shape[1:]
    current_co2 = torch.empty(BATCH, *co2_shape, device=device, dtype=init_batches[0][target_var].dtype)
    for i, ib in enumerate(init_batches):
        current_co2[i * n_samples : (i + 1) * n_samples] = ib[target_var]

    obs_present = _np.zeros(n_steps, dtype=bool)
    for k in range(n_steps):
        obs_present[k] = ((k - obs_offset) >= 0) and ((k - obs_offset) % obs_every == 0)

    mask_pattern = obs_kwargs.get("mask_pattern", "satellite")
    obs_fraction = obs_kwargs.get("obs_fraction", 0.3)
    ak_10 = obs_kwargs.get("ak_10", None)
    soft_boundary_sigma = obs_kwargs.get("soft_boundary_sigma", 0.0)

    all_preds = _np.full((n_inits, n_samples, n_steps, nlat, nlon, C), _np.nan, dtype=_np.float32)
    time_coord = _np.empty((n_inits, n_steps), dtype=times_axis.dtype)

    # Rolling history of past ensemble states. Each entry is a [BATCH, N, C]
    # tensor on device representing the (possibly already-smoothed) state at
    # AR step k_step. We push at every step and update in place when later
    # obs arrive. A state is moved to ``all_preds`` once it's older than lag.
    history: list[tuple[int, torch.Tensor]] = []  # list of (k_step, x[BATCH,N,C])

    iters = range(n_steps)
    if verbose:
        iters = tqdm(iters, desc="EnKS trajectory", total=n_steps)

    enkf_rng = torch.Generator(device=device).manual_seed(seed + 1)

    def _flush(k_step: int, x: torch.Tensor):
        pred_np = x.detach().cpu().numpy().reshape(n_inits, n_samples, nlat, nlon, C)
        all_preds[:, :, k_step] = pred_np

    def _apply_increment(x_slab, mask_i, delta_obs, obs_idx):
        """Apply per-cell increment ``delta_obs`` (at masked cells) to a
        per-init ensemble slab ``x_slab[n_samples, N, C]``, optionally with
        horizontal Gaussian localization."""
        if loc_sigma > 0.0:
            inc_grid = torch.zeros(n_samples, C, N, device=device, dtype=delta_obs.dtype)
            inc_grid[:, :, obs_idx] = delta_obs.permute(0, 2, 1)
            pres = torch.zeros(1, 1, N, device=device, dtype=delta_obs.dtype)
            pres[:, :, obs_idx] = 1.0
            inc_grid = inc_grid.view(n_samples, C, nlat, nlon)
            pres = pres.view(1, 1, nlat, nlon)
            inc_smooth = gaussian_smooth_2d(inc_grid, loc_sigma)
            pres_smooth = gaussian_smooth_2d(pres, loc_sigma)
            inc_norm = inc_smooth / (pres_smooth + 1e-6)
            if loc_min_presence > 0:
                inc_norm = inc_norm * (pres_smooth > loc_min_presence).to(inc_norm.dtype)
            inc_flat = inc_norm.view(n_samples, C, N).permute(0, 2, 1)
            return x_slab + inc_flat
        else:
            x_slab = x_slab.clone()
            x_slab[:, obs_idx, :] = x_slab[:, obs_idx, :] + delta_obs
            return x_slab

    for k in iters:
        batches_per_init = []
        for i_pos, init_idx in enumerate(init_indices):
            idx = init_idx + k
            if idx + 1 >= T_total:
                logger.warning("Init %d hit end at lead %d", init_idx, k)
                batches_per_init.append(None)
                continue
            b = data_loader.get_batch(idx, device=device)
            batches_per_init.append(b)
            time_coord[i_pos, k] = times_axis[idx + 1]

        if any(b is None for b in batches_per_init):
            break

        batch = {}
        for key, v in batches_per_init[0].items():
            if isinstance(v, torch.Tensor):
                batch[key] = torch.cat(
                    [bb[key].expand(n_samples, *bb[key].shape[1:]) for bb in batches_per_init],
                    dim=0,
                )
            else:
                batch[key] = v

        batch[target_var] = current_co2
        if f"{target_var}_next" in batch:
            batch[f"{target_var}_next"] = current_co2.clone()

        preds = _model_call_chunked(model, batch, chunk_size, BATCH, no_grad=True)
        pred_co2 = preds[target_var].detach()  # [BATCH, N, C]

        if obs_present[k]:
            if prior_inflation != 1.0:
                pg = pred_co2.view(n_inits, n_samples, N, C)
                m_pre = pg.mean(dim=1, keepdim=True)
                pg = m_pre + prior_inflation * (pg - m_pre)
                pred_co2 = pg.view(*pred_co2.shape)

            gt_next = [data_loader.get_batch(init_indices[i] + k + 1, device=device) for i in range(n_inits)]
            gt_stacked = torch.cat(
                [b[target_var].expand(n_samples, *b[target_var].shape[1:]) for b in gt_next],
                dim=0,
            )
            obs_batch = {kk: vv for kk, vv in batch.items()}
            obs_batch[target_var] = gt_stacked
            om, ov = create_column_mask(
                obs_batch,
                target_var=target_var,
                obs_fraction=obs_fraction,
                mask_pattern=mask_pattern,
                nlat=nlat,
                nlon=nlon,
                ak_10=ak_10,
                soft_boundary_sigma=soft_boundary_sigma,
            )
            hak = (obs_batch["pressure_weight"] * obs_batch["xco2_averaging_kernel"]).squeeze(1)  # [BATCH,N,C]
            mask_full = om.squeeze(-1).squeeze(1)  # [BATCH, N]
            y_full = ov.squeeze(-1).squeeze(1)  # [BATCH, N]

            x_grid = pred_co2.view(n_inits, n_samples, N, C)

            # Past-state references for the smoother sweep (within lag).
            past_grids = [(k_step, x.view(n_inits, n_samples, N, C)) for (k_step, x) in history if k - k_step <= lag]

            for i in range(n_inits):
                mask_i = mask_full[i * n_samples]
                y_i = y_full[i * n_samples]
                hak_i = hak[i * n_samples]
                obs_idx = mask_i.nonzero(as_tuple=True)[0]
                if obs_idx.numel() == 0:
                    continue
                col = x_grid[i, :, obs_idx, :]  # [n_samples, M, C]
                hak_obs = hak_i[obs_idx]  # [M, C]
                y_obs = y_i[obs_idx]  # [M]

                y_pred = (col * hak_obs.unsqueeze(0)).sum(dim=-1)  # [n_samples, M]
                y_mean = y_pred.mean(dim=0, keepdim=True)
                y_anom = y_pred - y_mean
                denom = max(n_samples - 1, 1)
                Var_y = (y_anom**2).sum(dim=0) / denom  # [M]

                eps = sigma_obs * torch.randn(n_samples, obs_idx.numel(), device=device, generator=enkf_rng)
                innovation = (y_obs.unsqueeze(0) + eps) - y_pred  # [n_samples, M]

                # ── Filter update (current state) ──
                x_mean_cur = col.mean(dim=0, keepdim=True)
                x_anom_cur = col - x_mean_cur
                Cov_xy_cur = (x_anom_cur * y_anom.unsqueeze(-1)).sum(dim=0) / denom  # [M, C]
                K_cur = Cov_xy_cur / (Var_y.unsqueeze(-1) + sigma_obs**2)
                delta_cur = K_cur.unsqueeze(0) * innovation.unsqueeze(-1)  # [n_samples,M,C]

                x_slab = _apply_increment(x_grid[i], mask_i, delta_cur, obs_idx)
                if inflation != 1.0:
                    m_post = x_slab.mean(dim=0, keepdim=True)
                    x_slab = m_post + inflation * (x_slab - m_post)
                x_grid[i] = x_slab

                # ── Smoother updates (past states within lag) ──
                # Use cross-cov of past x-anom with CURRENT y-anom (anomalies
                # of current obs predictions). Same innovation, so the same
                # innovation realization gets propagated backward — guarantees
                # consistency between filter and smoother members.
                for _, past in past_grids:
                    col_past = past[i, :, obs_idx, :]  # [n_samples,M,C]
                    xm_p = col_past.mean(dim=0, keepdim=True)
                    xa_p = col_past - xm_p
                    Cov_xy_p = (xa_p * y_anom.unsqueeze(-1)).sum(dim=0) / denom  # [M,C]
                    K_p = damping * Cov_xy_p / (Var_y.unsqueeze(-1) + sigma_obs**2)
                    delta_p = K_p.unsqueeze(0) * innovation.unsqueeze(-1)
                    past_slab = _apply_increment(past[i], mask_i, delta_p, obs_idx)
                    past[i] = past_slab

            current_co2 = x_grid.view(*pred_co2.shape)
        else:
            current_co2 = pred_co2

        # Push current state to history; flush states older than lag.
        history.append((k, current_co2.clone()))
        while history and (k - history[0][0] > lag):
            old_k, old_x = history.pop(0)
            _flush(old_k, old_x)

    # Flush remaining history at end of trajectory.
    for old_k, old_x in history:
        _flush(old_k, old_x)
    history.clear()

    inner.generating = prev_generating

    lat_vals = data_loader.grid_info.lat
    lon_vals = data_loader.grid_info.lon
    levels = data_loader.grid_info.levels
    ds = xr.Dataset(
        {
            target_var: (("init", "sample", "lead", "lat", "lon", "level"), all_preds),
            "obs_present": (("lead",), obs_present),
        },
        coords={
            "init": _np.array(init_indices),
            "sample": _np.arange(n_samples),
            "lead": _np.arange(n_steps),
            "lat": lat_vals,
            "lon": lon_vals,
            "level": levels,
            "time": (("init", "lead"), time_coord),
        },
    )
    return ds


def generate_for_distributional_eval(
    model,
    dataset,
    outpath,
    n_ref_timesteps=20,
    n_gen_per_ref=10,
    device="cuda",
    target_vars_3d=None,
    generate_kwargs=None,
    seed=42,
    # Legacy aliases.
    n_gt_samples=None,
    n_gen_samples=None,
    batch_size=None,
):
    """Backward-compatible wrapper around GenerationPipeline.run_distributional()."""
    if n_gt_samples is not None:
        n_ref_timesteps = n_gt_samples
    if n_gen_samples is not None:
        n_gen_per_ref = max(1, n_gen_samples // n_ref_timesteps)
    pipeline = GenerationPipeline(
        model,
        dataset,
        target_vars_3d=target_vars_3d,
        device=device,
    )
    return pipeline.run_distributional(
        outpath,
        n_ref_timesteps=n_ref_timesteps,
        n_gen_per_ref=n_gen_per_ref,
        seed=seed,
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
