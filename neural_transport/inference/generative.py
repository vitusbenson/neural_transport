import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from cdo import Cdo
from tqdm import tqdm

from neural_transport.configs import DEFAULT_T
from neural_transport.inference.masking import (
    create_column_mask,
    create_mask,
    create_oco2_mask,
    create_oco2_mask_test,
)
from neural_transport.inference.noise import noise
from neural_transport.plots.plot_results import plot_masking_diagnostics


def get_zarrpath_obspath(out_path, rollout, freq, zarr_filename=None, zero_surfflux=False):
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
    """
    Collect batches over a time window.
    For sparse observation variables (target_vars_2d, xco2*): union all non-NaN observations
    For other variables: take mean over window
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
    return np.isnan(arr).any() or np.isinf(arr).any() or np.nanmax(np.abs(arr)) > thresh


def parse_freq(freq: str) -> int:
    """Convert `freq` (e.g.: '3h', '6h', '1D', ...) into appropriate integer hours."""
    num = int(''.join(filter(str.isdigit, freq)))
    unit = ''.join(filter(str.isalpha, freq))

    valid_units = {"h", "D"}
    if unit not in valid_units:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    if unit == "h":
        hours = num
    elif unit == "D":
        hours = num * 24
    return hours


def align_time(time, time_gen):
    """
    Find the index offset to align time of masking with trained dataset.
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
    n_samples = generate_kwargs.get("n_samples", 10)
    masking = generate_kwargs.get("masking", True)
    mask_pattern = generate_kwargs.get("mask_pattern", None)
    analyze_masking = generate_kwargs.get("analyze_masking", False)
    noise_pattern = generate_kwargs.get("noise_pattern", None)
    analyze_noise = generate_kwargs.get("analyze_noise", False)
    freq_int = parse_freq(generate_kwargs.get("generate_data_kwargs", {}).get("freq", "6h"))
    window_hours = generate_kwargs.get("window_hours", freq_int)

    nlat, nlon = model.model.in_nlat, model.model.in_nlon

    zarrpath, obspath = get_zarrpath_obspath(outpath, rollout, freq, zarr_filename, zero_surfflux=zero_surfflux)

    prototype_zarr = dataset.create_prototype_zarr(
        zarrpath,
        target_vars_3d=target_vars_3d,
        target_vars_2d=target_vars_2d,
        grid="default" if remap else None,
    )

    model = model.eval().to(device)
    model.return_intermediates = True
    model.model.generate_kwargs = generate_kwargs

    dss = []
    obss = []
    if mask_pattern is None:
        T = DEFAULT_T
        # T = len(dataset_gen)
    else:
        T = min(len(dataset), len(dataset_gen))
    offset = align_time(dataset.ds.time.values, dataset_gen.ds.time.values)
    print(f"Time alignment offset: {offset} timesteps")
    window_steps = max(1, window_hours // freq_int)
    print(f"Using observation window: {window_hours} hours = {window_steps} timesteps")

    for t in tqdm(range(T), desc="Generating") if verbose else range(T):
        # batches: dict of tensors [B T N C]
        batch, batch_gen = get_batches(t, offset, dataset, dataset_gen, window_steps, device)

        # Noise
        noise_list = noise(
            dataset_gen[0],  # only shape matters
            target_var=generate_kwargs["generate_data_kwargs"]["forcing_vars"][0],
            n_samples=n_samples,
            noise_pattern=noise_pattern,
            analyze_noise=analyze_noise,
            outpath=outpath,
        )

        # Masking
        if masking:
            target_var = target_vars_2d[0]
            if mask_pattern is None:
                obs_mask, obs_values = create_oco2_mask(batch_gen, target_var=target_var)
            else:
                obs_mask, obs_values = create_oco2_mask_test(
                    batch_gen, target_var=target_var, mask_pattern=mask_pattern, nlat=nlat, nlon=nlon
                )
            batch_gen["obs_mask_original"] = obs_mask.clone()
            batch_gen["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(
                obs_values, batch_gen, target_var=target_var, targshift=False
            )
            for k in (
                target_vars_2d
                + generate_kwargs["generate_data_kwargs"]["forcing_vars"]
                + ["obs_mask", "obs_mask_original"]
            ):
                batch[k] = batch_gen[k]
            batch["obs_values"] = obs_values_normed  # [B=1 T=1 N=2048 C=1]

        for k in batch.keys():
            batch[k] = batch[k].expand(n_samples, -1, -1, -1)  # [B=n_samples T N C]
        batch["noise"] = torch.cat(noise_list, dim=0).to(device)  # [B=n_samples T N C]
        if zero_surfflux:
            for var in ["co2flux_anthro", "co2flux_land", "co2flux_ocean"]:
                batch[var] = torch.zeros_like(batch[var])

        with torch.no_grad():
            preds = model(batch)
            traj = preds["trajectory"]

        preds_fixed = {}
        for k, v in preds.items():
            if isinstance(v, list):
                v = v[0]
            preds_fixed[k] = v

        if save_obs:
            preds_fixed["gph_bottom"] = batch["gph_bottom"]
            preds_fixed["gph_top"] = batch["gph_top"]

        ds = xr.Dataset({k: dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()})

        ds = ds.rename({"batch": "sample", "time": "trajectory_steps"})
        ds = ds.assign_coords(
            trajectory_steps=("trajectory_steps", np.arange(traj.shape[1])),
            sample=("sample", np.arange(n_samples)),
        )
        ds = ds.expand_dims(time=[prototype_zarr.isel(time=t).time.values])

        if masking:
            obs_mask_xr = dataset_gen.tensor_to_xarray(batch_gen["obs_mask"][:1])
            obs_values_xr = dataset_gen.tensor_to_xarray(obs_values[:1])
            ds["obs_mask"] = obs_mask_xr
            ds["obs_values"] = obs_values_xr

        if remap:
            ds = remap_with_cdo(dataset, prototype_zarr.isel(time=0), ds)

        if save_obs:
            obs = dataset.readout_stations(ds, grid="default" if remap else None)
            ds = ds.drop_vars(["gph_bottom", "gph_top"])
            obss.append(obs)

        dss.append(ds)

    ds_all = xr.concat(dss, dim="time")

    if save_obs:
        obs_all = xr.concat(obss, dim="time").fillna({"obs_filename": ""})
        obs_all.to_zarr(obspath, mode="w")
        ds_all = ds_all.fillna({"obs_filename": ""})

    ds_all.to_zarr(zarrpath, mode="w")

    if analyze_masking and masking:
        plot_masking_diagnostics(
            ds_all,
            str(outpath).replace("preds", "plots"),
            varnames=target_vars_3d,
            nlat=nlat,
            nlon=nlon,
            imgformats=["png"],
        )

    return ds_all


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
    condition_one_timestep = generate_kwargs.get("condition_one_timestep", True)
    n_samples = generate_kwargs.get("n_samples", 10)
    masking = generate_kwargs.get("masking", False)
    mask_source = generate_kwargs.get("mask_source", "3d")
    mask_pattern = generate_kwargs.get("mask_pattern", "vertical")
    analyze_masking = generate_kwargs.get("analyze_masking", False)
    obs_fraction = generate_kwargs.get("obs_fraction", 0.2)
    noise_pattern = generate_kwargs.get("noise_pattern", None)
    analyze_noise = generate_kwargs.get("analyze_noise", False)
    ak_10 = generate_kwargs.get("ak_10", None)

    nlat, nlon = model.model.in_nlat, model.model.in_nlon

    zarrpath, obspath = get_zarrpath_obspath(outpath, rollout, freq, zarr_filename, zero_surfflux=zero_surfflux)

    prototype_zarr = dataset.create_prototype_zarr(
        zarrpath,
        target_vars_3d=target_vars_3d,
        target_vars_2d=target_vars_2d,
        grid="default" if remap else None,
    )

    model = model.eval().to(device)
    model.return_intermediates = True
    model.model.generate_kwargs = generate_kwargs

    dss = []
    obss = []

    # Noise
    noise_list = noise(
        dataset[0],
        target_var=target_vars_3d[0],
        n_samples=n_samples,
        noise_pattern=noise_pattern,
        analyze_noise=analyze_noise,
        outpath=outpath,
    )

    if condition_one_timestep:
        base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items()}  # condition on the first timestep
        if masking:
            target_var = target_vars_3d[0]
            if mask_source in ("column", "xco2"):
                obs_mask, obs_values = create_column_mask(
                    base_batch,
                    target_var=target_var,
                    obs_fraction=obs_fraction,
                    mask_pattern=mask_pattern,
                    nlat=nlat,
                    nlon=nlon,
                    ak_10=ak_10,
                )
                base_batch["obs_mask"] = obs_mask
                obs_values_normed = model.model.normalize_observations(
                    obs_values, base_batch, target_var=target_var, targshift=False
                )
                base_batch["obs_values"] = obs_values_normed
            elif mask_pattern is None:
                obs_mask, obs_values = create_oco2_mask(base_batch, target_var=target_var)
                base_batch["obs_mask"] = obs_mask
                obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
                base_batch["obs_values"] = obs_values_normed
            else:
                obs_mask, obs_values = create_mask(
                    base_batch,
                    target_var=target_var,
                    obs_fraction=obs_fraction,
                    mask_pattern=mask_pattern,
                    nlat=nlat,
                    nlon=nlon,
                )
                base_batch["obs_mask"] = obs_mask
                obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
                base_batch["obs_values"] = obs_values_normed

    for i in tqdm(range(n_samples), desc="Generating samples") if verbose else range(n_samples):
        if not condition_one_timestep:
            base_batch = {
                k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()
            }  # condition on different timesteps
            if masking:
                target_var = target_vars_3d[0]
                if mask_source in ("column", "xco2"):
                    obs_mask, obs_values = create_column_mask(
                        base_batch,
                        target_var=target_var,
                        obs_fraction=obs_fraction,
                        mask_pattern=mask_pattern,
                        nlat=nlat,
                        nlon=nlon,
                        ak_10=ak_10,
                    )
                    base_batch["obs_mask"] = obs_mask
                    obs_values_normed = model.model.normalize_observations(
                        obs_values, base_batch, target_var=target_var, targshift=False
                    )
                    base_batch["obs_values"] = obs_values_normed
                else:
                    obs_mask, obs_values = create_mask(
                        base_batch,
                        target_var=target_var,
                        obs_fraction=obs_fraction,
                        mask_pattern=mask_pattern,
                        nlat=nlat,
                        nlon=nlon,
                    )
                    base_batch["obs_mask"] = obs_mask
                    obs_values_normed = model.model.normalize_observations(
                        obs_values, base_batch, target_var=target_var
                    )
                    base_batch["obs_values"] = obs_values_normed
        batch = {k: v.clone() for k, v in base_batch.items()}
        batch["noise"] = noise_list[i].to(device)

        if zero_surfflux:
            for var in ["co2flux_anthro", "co2flux_land", "co2flux_ocean"]:
                batch[var] = torch.zeros_like(batch[var])

        with torch.no_grad():
            preds = model(batch)
            traj = preds["trajectory"]

        preds_fixed = {}
        for k, v in preds.items():
            if isinstance(v, list):
                v = v[0]
            preds_fixed[k] = v

        if save_obs:
            preds_fixed["gph_bottom"] = batch["gph_bottom"]
            preds_fixed["gph_top"] = batch["gph_top"]

        ds = xr.Dataset({k: dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()})

        ds = ds.assign_coords(
            time=("time", prototype_zarr.time[: traj.shape[1]].values if traj.ndim > 0 else [0]),
            sample=("sample", [i]),
        )

        if masking:
            obs_mask_xr = dataset.tensor_to_xarray(batch["obs_mask"][:1])
            obs_values_xr = dataset.tensor_to_xarray(batch["obs_values"][:1])
            ds["obs_mask"] = obs_mask_xr
            ds["obs_values"] = obs_values_xr

        if remap:
            ds = remap_with_cdo(dataset, prototype_zarr.isel(time=0), ds)

        if save_obs:
            obs = dataset.readout_stations(ds, grid="default" if remap else None)
            ds = ds.drop_vars(["gph_bottom", "gph_top"])
            obss.append(obs)

        dss.append(ds)

    ds_all = xr.concat(dss, dim="sample")

    if save_obs:
        obs_all = xr.concat(obss, dim="sample").fillna({"obs_filename": ""})
        obs_all.to_zarr(obspath, mode="w")
        ds_all = ds_all.fillna({"obs_filename": ""})

    ds_all.to_zarr(zarrpath, mode="w")

    if analyze_masking and masking:
        plot_masking_diagnostics(
            ds_all,
            str(outpath).replace("preds", "plots"),
            varnames=target_vars_3d,
            nlat=nlat,
            nlon=nlon,
            imgformats=["png"],
        )

    return ds_all


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
    """Build GT and generated anomaly pools for distributional comparison.

    GT pool: sample n_gt_samples random timesteps from dataset,
             subtract per-timestep spatial mean -> anomaly patterns.
    Gen pool: generate n_gen_samples unconditional samples (batched),
              subtract per-sample spatial mean -> anomaly patterns.

    Args:
        model: NeuralTransport model (with FlowMatching inside).
        dataset: CarbonDataset.
        outpath: Path to save results.
        n_gt_samples: Number of GT samples to draw.
        n_gen_samples: Number of unconditional samples to generate.
        device: Device for generation.
        target_vars_3d: List of target variable names. Defaults to ["co2massmix"].
        generate_kwargs: kwargs for generation (steps, method, etc.).
        seed: Random seed for reproducibility.
        batch_size: Number of samples to generate per forward pass.

    Returns:
        gt_ds: xr.Dataset [sample, lat, lon, level] -- GT CO2 fields
        gen_ds: xr.Dataset [sample, lat, lon, level] -- generated CO2 fields
    """
    if target_vars_3d is None:
        target_vars_3d = ["co2massmix"]
    if generate_kwargs is None:
        generate_kwargs = {}

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    target_var = target_vars_3d[0]

    # --- GT pool ---
    n_gt = min(n_gt_samples, len(dataset))
    gt_indices = rng.choice(len(dataset), n_gt, replace=False)
    gt_fields = []
    for idx in gt_indices:
        batch = dataset[idx]
        field = batch[target_var]  # [T, N, C] or [N, C]
        if field.ndim == 3:
            field = field[0]  # first timestep: [N, C]
        gt_fields.append(field.numpy())
    gt_fields = np.stack(gt_fields)  # [n_gt, N, C]

    # Reshape to [n_gt, nlat, nlon, nlev]
    nlat = model.model.in_nlat
    nlon = model.model.in_nlon
    nlev = gt_fields.shape[-1]
    gt_fields = gt_fields.reshape(n_gt, nlat, nlon, nlev)

    # --- Gen pool ---
    model_eval = model.eval().to(device)
    model_eval.model.generating = True
    model_eval.model.generate_kwargs = generate_kwargs
    model_eval.return_intermediates = False
    model_eval.model.return_intermediates = False

    gen_fields = []
    n_remaining = n_gen_samples

    # Use first dataset sample as template for batch structure
    # dataset[i] returns {var: [T, N, C]}, unsqueeze(0) → [B=1, T, N, C]
    template = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items() if isinstance(v, torch.Tensor)}

    while n_remaining > 0:
        this_batch_size = min(batch_size, n_remaining)

        # Build batch: expand template to [batch_size, T, N, C]
        batch_in = {}
        for k, v in template.items():
            batch_in[k] = v.expand(this_batch_size, *v.shape[1:]).clone()

        with torch.no_grad():
            # Go through NeuralTransport.forward which handles T dimension
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
    gen_fields = gen_fields.reshape(len(gen_fields), nlat, nlon, nlev)

    # Build xarray datasets with proper 1D lat/lon coordinate arrays
    # Dataset stores per-cell lat/lon (flattened), so look up from grid prototypes
    from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS

    lat_vals = None
    for grid_name, coords in LATLON_PROTOTYPE_COORDS.items():
        if len(coords['lat']) == nlat and len(coords['lon']) == nlon:
            lat_vals = coords['lat']
            lon_vals = coords['lon']
            break
    if lat_vals is None:
        lat_vals = np.linspace(-90, 90, nlat)
        lon_vals = np.linspace(0, 360, nlon, endpoint=False)

    if hasattr(dataset, 'ds') and 'level' in dataset.ds:
        level_vals = dataset.ds.level.values[:nlev]
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

    # Save
    gt_ds.to_netcdf(outpath / "gt_pool.nc")
    gen_ds.to_netcdf(outpath / "gen_pool.nc")

    return gt_ds, gen_ds
