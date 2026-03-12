import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import xarray as xr
from cdo import Cdo

from neural_transport.plots.plot_results import (
    plot_noise_diagnostics, plot_masking_diagnostics
)
from neural_transport.tools.conversion import molemix_to_massmix

def get_zarrpath_obspath(out_path, rollout, freq, zarr_filename=None, zero_surfflux=False):
    if zarr_filename is None:
        zarr_filename = (
            f"co2_pred_rollout_{freq}.zarr" if rollout else "co2_pred_singlestep.zarr"
        )
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

    grid_temp_file = tempfile.NamedTemporaryFile(
        delete=True, prefix="grid_temp_file_", dir=tempfile.gettempdir()
    )
    prototype_zarr.to_netcdf(grid_temp_file.name)

    ds_temp_file = tempfile.NamedTemporaryFile(
        delete=True, prefix="ds_temp_file_", dir=tempfile.gettempdir()
    )
    ds.transpose("time", "height", "cell", "nv").to_netcdf(ds_temp_file.name)

    ds_remap = cdo.remapcon(
        grid_temp_file.name, input=ds_temp_file.name, returnXDataset=True
    )

    grid_temp_file.close()
    ds_temp_file.close()

    cdo.cleanTempDir()
    return ds_remap


def generate_noise(batch, target_var="co2massmix", n_samples=10, noise_pattern=None):

    all_levels = batch[target_var] # [T N C]
    all_levels = all_levels.unsqueeze(0) # [B T N C]

    if noise_pattern is None:
        return [torch.randn_like(all_levels) for _ in range(n_samples)]

    elif noise_pattern == "spiral_outward_noise":
        # Step 1: pick a reference noise vector
        x0 = torch.randn_like(all_levels)
        # Step 2: pick a direction vector (independent random noise)
        v = torch.randn_like(all_levels)
        # Step 3: create spiral path
        angles = torch.linspace(0, 4*torch.pi, n_samples)

        spiral_noises = []
        for a in angles:
            # Step 4: rotate between x0 and v
            x_init = torch.cos(a) * x0 + torch.sin(a) * v
            # Step 5: scale radius outward
            r = 1.0 + 0.2 * (a / angles[-1])  # gradually increase radius
            x_init_scaled = r * x_init
            spiral_noises.append(x_init_scaled)
        return spiral_noises

    elif noise_pattern == "spiral_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        angles = torch.linspace(0, 4*torch.pi, n_samples)

        spiral_noises = []
        for a in angles:
            x_init = torch.cos(a) * x0 + torch.sin(a) * v
            spiral_noises.append(x_init)
        return spiral_noises

    elif noise_pattern == "geodesic_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [torch.sqrt((1 - alpha)) * x0 + torch.sqrt(alpha) * v for alpha in alphas]

    elif noise_pattern == "linear_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [(1 - alpha) * x0 + alpha * v for alpha in alphas]

    elif noise_pattern == "antipodal_orthogonal_noise":
        x0 = torch.randn_like(all_levels)
        n_dirs = n_samples // 2  # each direction will yield a +v and -v pair
        # Start with random Gaussian directions
        dirs = [torch.randn_like(x0).flatten() for _ in range(n_dirs)]
        # Orthogonalize via Gram–Schmidt
        orth_dirs = []
        for v in dirs:
            for u in orth_dirs:
                v -= (v @ u) / (u @ u) * u
            orth_dirs.append(v)

        # Convert back to tensor shape and include antipodal pairs
        orth_dirs = [v.reshape_as(x0) for v in orth_dirs]
        all_noises = []
        for v in orth_dirs:
            all_noises.append(v)
            all_noises.append(-v)

        # If we have fewer than n_samples due to rounding
        if len(all_noises) < n_samples:
            all_noises.append(torch.randn_like(x0))

        return all_noises[:n_samples]

    else:
        raise ValueError(f"Unknown noise type: {noise_pattern}")


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


def noise(batch: dict,
          target_var: str | None = "co2massmix",
          n_samples: int | None = 10,
          noise_pattern: str | None = None,
          analyze_noise: bool = False,
          outpath: Path | str = None,
) -> list:
    noise_list = generate_noise(batch, target_var=target_var, n_samples=n_samples, noise_pattern=noise_pattern)
    if analyze_noise and noise_pattern is not None:
            if noise_pattern in ["spiral_noise", "spiral_outward_noise"]:
                angles = torch.linspace(0, 4*torch.pi, n_samples) # thetas
                param_name = "$\\theta$"
            elif noise_pattern in ["geodesic_noise", "linear_noise"]:
                angles = torch.linspace(0, 1, n_samples)  # alphas
                param_name = "$\\alpha$"
            elif noise_pattern == "antipodal_orthogonal_noise":
                labels = []
                for i in range(n_samples // 2):
                    labels += [f"{i+1}a", f"{i+1}b"]
                if n_samples % 2 == 1:
                    labels.append(f"{(n_samples // 2) + 1}a")
                angles = labels
                param_name = "Index pair"
            else:
                angles = torch.arange(n_samples)  # index
                param_name = "Index"
            plot_noise_diagnostics(noise_list, angles, str(outpath).replace("preds", "plots"),
                                   label=param_name, imgformats=["png"])
    return noise_list


def create_oco2_mask(batch, target_var="xco2_2019_scale"):
    """
    Create OCO-2 observation mask and values from sparse target_var field.
    Args:
      batch: dict of tensors, each of shape [B T N C]
      target_var: the variable to create the mask for

    Returns:
      obs_mask:  bool[B,T,N,C] with True where OCO-2 has data
      obs_values: float[B,T,N,C] containing observed values at mask locations,
                  NaN elsewhere
    """
    obs_mask = ~torch.isnan(batch[target_var])
    obs_values = batch[target_var].clone()
    obs_values = molemix_to_massmix(obs_values)
    batch[f"{target_var}_offset"] = molemix_to_massmix(batch[f"{target_var}_offset"])
    batch[f"{target_var}_scale"] = molemix_to_massmix(batch[f"{target_var}_scale"])

    # Handle xco2_averaging_kernel
    ak = batch["xco2_averaging_kernel"].clone()  # [B, T, N, C=10]
    n_levels = ak.shape[-1]
    ak_mask = obs_mask.expand(-1, -1, -1, n_levels)  # [B, T, N, C=10]
    valid_ak = ak[ak_mask]
    mean_ak_per_level = valid_ak.reshape(-1, n_levels).mean(dim=0)  # [C=10]
    mean_ak_full = mean_ak_per_level.view(1, 1, 1, -1).expand_as(ak)
    ak_cleaned = torch.where(ak_mask, ak, mean_ak_full)
    batch["xco2_averaging_kernel"] = ak_cleaned

    return obs_mask, obs_values  # [B T N C] each


def create_oco2_mask_test(
    batch,
    target_var="xco2_2019_scale",
    mask_pattern="diagonal",
    nlat=32,
    nlon=64,
):
    """
    Create synthetic OCO-2-like observation masks for testing.

    Args:
        batch: dict of tensors [B, T, N, C]
        target_var: variable used to infer shape
        mask_pattern:
            - "diagonal"   : diagonal stripe (lat = lon)
            - "leftright"  : left half observed
            - "topbottom"  : top half observed
            - "checkerboard"
            - "center_box" : central rectangle
        nlat, nlon: grid dimensions (must satisfy nlat * nlon == N)

    Returns:
        obs_mask   : bool [B, T, N, C]
        obs_values : float [B, T, N, C] (NaN outside mask)
    """

    B, T, N, C = batch[target_var].shape
    device = batch[target_var].device
    value = batch[f"{target_var}_offset"]
    value = molemix_to_massmix(value)

    assert nlat * nlon == N, "nlat * nlon must equal N"

    grid = torch.arange(N, device=device).reshape(nlat, nlon)

    mask2d = torch.zeros((nlat, nlon), dtype=torch.bool, device=device)

    if mask_pattern == "diagonal":
        for i in range(min(nlat, nlon)):
            mask2d[i, i] = True

    elif mask_pattern == "leftright":
        mask2d[:, : nlon // 2] = True

    elif mask_pattern == "topbottom":
        mask2d[: nlat // 2, :] = True

    elif mask_pattern == "checkerboard":
        mask2d = (
            (torch.arange(nlat, device=device)[:, None]
           + torch.arange(nlon, device=device)[None, :]) % 2 == 0
        )

    elif mask_pattern == "center_box":
        lat0, lat1 = nlat // 4, 3 * nlat // 4
        lon0, lon1 = nlon // 4, 3 * nlon // 4
        mask2d[lat0:lat1, lon0:lon1] = True

    else:
        raise ValueError(f"Unknown test pattern: {mask_pattern}")

    obs_indices = grid[mask2d].reshape(-1)

    obs_mask = torch.zeros((B, T, N, C), dtype=torch.bool, device=device)
    obs_values = torch.full(
        (B, T, N, C), float("nan"), device=device
    )

    obs_mask[:, :, obs_indices, :] = True
    obs_values[:, :, obs_indices, :] = value

    # Handle xco2_averaging_kernel
    ak = batch["xco2_averaging_kernel"].clone()  # [B, T, N, C=10]
    n_levels = ak.shape[-1]
    true_mask = ~torch.isnan(batch[target_var])
    ak_mask = true_mask.expand(-1, -1, -1, n_levels)  # [B, T, N, C=10]
    valid_ak = ak[ak_mask]
    mean_ak_per_level = valid_ak.reshape(-1, n_levels).mean(dim=0)  # [C=10]
    mean_ak_full = mean_ak_per_level.view(1, 1, 1, -1).expand_as(ak)
    batch["xco2_averaging_kernel"] = mean_ak_full

    return obs_mask, obs_values  # [B T N C] each


def create_mask(batch, target_var="co2massmix", obs_fraction=0.1, mask_pattern="random", nlat=32, nlon=64):
    """
    Create a random observation mask for the input batch.
    Args:
        batch: dict of tensors, each of shape [B T N C]
        target_var: the variable to create the mask for
        obs_fraction: fraction of points to keep as observations
        mask_pattern: "random", "vertical", "horizontal", "checkerboard", "satellite"
    """
    device = batch[target_var].device
    B, T, N, C = batch[target_var].shape

    obs_mask = torch.zeros((B, T, N, C), dtype=torch.bool, device=device)
    obs_values = torch.full_like(batch[target_var], float('nan'), device=device)

    for t in range(T):
        if mask_pattern == "random":
            num_obs = int(obs_fraction * N)
            obs_indices = torch.randperm(N, device=device)[:num_obs]

        elif mask_pattern == "vertical":
            # keep a fixed fraction of longitude columns
            num_cols = max(1, int(obs_fraction * nlon))
            cols = torch.arange(0, nlon, nlon // num_cols, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[:, cols].reshape(-1)

        elif mask_pattern == "horizontal":
            # keep a fixed fraction of latitude rows
            num_rows = max(1, int(obs_fraction * nlat))
            rows = torch.arange(0, nlat, nlat // num_rows, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[rows, :].reshape(-1)

        elif mask_pattern == "checkerboard":
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            mask2d = (torch.arange(nlat, device=device)[:, None] +
                      torch.arange(nlon, device=device)[None, :]) % 2 == 0
            obs_indices = grid[mask2d].reshape(-1)

        elif mask_pattern == "satellite":
            # grid = [nlat, nlon]
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            # choose swath width (fraction of nlon)
            swath_width = max(1, int(obs_fraction * nlon / 8))
            # tilt angle in radians (small tilt)
            tilt = -5 * np.pi / 180.0  
            cols = []
            for i in range(0, nlon, swath_width * 8):  # spacing between swaths
                for w in range(swath_width):
                    col_idx = i + w
                    if col_idx < nlon:
                        # apply tilt shift proportional to latitude
                        lat_offsets = ((torch.arange(nlat, device=device) * np.tan(tilt)).long()) % nlon
                        col_with_tilt = (col_idx + lat_offsets) % nlon
                        cols.append(col_with_tilt.unsqueeze(0))
            # stack and mask
            cols = torch.cat(cols, dim=0)  # shape [n_swath, nlat]
            obs_indices = grid[torch.arange(nlat).unsqueeze(0), cols].reshape(-1)

        else:
            raise ValueError(f"Unknown mask pattern: {mask_pattern}")

        # fill mask + values
        obs_mask[:, t, obs_indices, :] = True
        obs_values[:, t, obs_indices, :] = batch[target_var][:, t, obs_indices, :]

    return obs_mask, obs_values  # [B T N C] each


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
        offset = - (time == time_gen[0]).argmax().item()
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

    zarrpath, obspath = get_zarrpath_obspath(outpath, rollout, freq, zarr_filename, zero_surfflux = zero_surfflux)

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
        T = 5
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
        noise_list = noise(dataset_gen[0],  # only shape matters
                           target_var=generate_kwargs["generate_data_kwargs"]["forcing_vars"][0],n_samples=n_samples,
                           noise_pattern=noise_pattern,
                           analyze_noise=analyze_noise,
                           outpath=outpath)

        # Masking
        if masking:
            target_var = target_vars_2d[0]
            if mask_pattern is None:
                obs_mask, obs_values = create_oco2_mask(batch_gen, target_var=target_var)
            else:
                obs_mask, obs_values = create_oco2_mask_test(batch_gen, target_var=target_var, mask_pattern=mask_pattern, nlat=nlat, nlon=nlon)
            batch_gen["obs_mask_original"] = obs_mask.clone()
            ### End DEBUG
            batch_gen["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(obs_values, batch_gen, target_var=target_var, targshift=False)
            for k in target_vars_2d + generate_kwargs["generate_data_kwargs"]["forcing_vars"] + ["obs_mask", "obs_mask_original"]:
                batch[k] = batch_gen[k]
            batch["obs_values"] = obs_values_normed  # [B=1 T=1 N=2048 C=1]
            print(f"\nDEBUG iterative_generate_oco2 t={t}")
            print("  obs_values stats:")
            obs_valid = obs_values[~torch.isnan(obs_values)]
            print(f"    min={obs_valid.min().item():.6f}, max={obs_valid.max().item():.6f}")
            print(f"    mean={obs_valid.mean().item():.6f}, std={obs_valid.std().item():.6f}")
            print("  obs_values_normed stats:")
            obs_normed_valid = obs_values_normed[~torch.isnan(obs_values_normed)]
            print(f"  min={obs_normed_valid.min().item():.6f}, max={obs_normed_valid.max().item():.6f}")
            print(f"  mean={obs_normed_valid.mean().item():.6f}, std={obs_normed_valid.std().item():.6f}")
            # ### DEBUG: no masking
            # batch["obs_mask"] = torch.zeros_like(obs_mask, dtype=torch.bool)
            # ### DEBUG: test non tca masking_methods
            # batch["obs_mask"] = batch["obs_mask"].expand(-1, -1, -1, 10)
            # batch["obs_values"] = batch["obs_values"].expand(-1, -1, -1, 10)
            # ### End DEBUG

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

        ds = xr.Dataset(
            {k: dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()}
        )

        ds = ds.rename({"batch": "sample", "time": "trajectory_steps"})
        ds = ds.assign_coords(
            trajectory_steps=("trajectory_steps", np.arange(traj.shape[1])),
            sample=("sample", np.arange(n_samples)),
        )
        ds = ds.expand_dims(time=[prototype_zarr.isel(time=t).time.values])
        
        if remap:
            ds = remap_with_cdo(dataset, prototype_zarr.isel(time=0), ds)

        if save_obs:
            obs = dataset.readout_stations(ds, grid="default" if remap else None)
            ds = ds.drop_vars(["gph_bottom", "gph_top"])
            obss.append(obs)

        dss.append(ds)

        if t == 0 and analyze_masking and masking:
            batch_analyze = batch

    ### !!! Caution: need to fix this properly!!!
    good_dss = []
    for i, ds in enumerate(dss):
        if is_bad_sample(ds[target_vars_3d[0]].values):
            print(f"Skipping bad sample {i}")
            continue
        good_dss.append(ds)

    ds_all = xr.concat(good_dss, dim="time")
    ### !!!

    if save_obs:
        obs_all = xr.concat(obss, dim="time").fillna({"obs_filename": ""})
        obs_all.to_zarr(obspath, mode="w")
        ds_all = ds_all.fillna({"obs_filename": ""})

    ds_all.to_zarr(zarrpath, mode="w")

    if analyze_masking and masking:
        plot_masking_diagnostics(batch_analyze, ds_all,
                                 str(outpath).replace("preds", "plots"),
                                 varnames=target_vars_3d, nlat=nlat, nlon=nlon,
                                 imgformats=["png"])

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
    mask_pattern = generate_kwargs.get("mask_pattern", "vertical")
    analyze_masking = generate_kwargs.get("analyze_masking", False)
    obs_fraction = generate_kwargs.get("obs_fraction", 0.2)
    noise_pattern = generate_kwargs.get("noise_pattern", None)
    analyze_noise = generate_kwargs.get("analyze_noise", False)

    nlat, nlon = model.model.in_nlat, model.model.in_nlon

    zarrpath, obspath = get_zarrpath_obspath(outpath, rollout, freq, zarr_filename, zero_surfflux = zero_surfflux)

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
    noise_list = noise(dataset[0],
                       target_var=target_vars_3d[0],
                       n_samples=n_samples,
                       noise_pattern=noise_pattern,
                       analyze_noise=analyze_noise,
                       outpath=outpath)

    if condition_one_timestep:
        base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items()} # condition on the first timestep
        if masking:
            target_var = target_vars_3d[0]
            if mask_pattern is None:
                obs_mask, obs_values = create_oco2_mask(base_batch, target_var=target_var)
            else:
                obs_mask, obs_values = create_mask(base_batch, target_var=target_var, obs_fraction=obs_fraction, mask_pattern=mask_pattern, nlat=nlat, nlon=nlon)
            base_batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
            base_batch["obs_values"] = obs_values_normed

    for i in tqdm(range(n_samples), desc="Generating samples") if verbose else range(n_samples):
        if not condition_one_timestep:
            base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()} # condition on different timesteps
            if masking:
                target_var = target_vars_3d[0]
                obs_mask, obs_values = create_mask(base_batch, target_var=target_var, obs_fraction=obs_fraction, mask_pattern=mask_pattern, nlat=nlat, nlon=nlon)
                base_batch["obs_mask"] = obs_mask
                obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
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

        ds = xr.Dataset(
            {k: dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()}
        )

        ds = ds.rename({"time": "trajectory_steps"})
        ds = ds.assign_coords(
            trajectory_steps=("trajectory_steps", np.arange(traj.shape[1]) if traj.ndim > 0 else [0]),
            sample=("sample", [i]),
        )
        time_value = prototype_zarr.isel(time=i).time.values if not condition_one_timestep else prototype_zarr.isel(time=0).time.values
        ds = ds.expand_dims(time=[time_value])

        if remap:
            ds = remap_with_cdo(dataset, prototype_zarr.isel(time=0), ds)

        if save_obs:
            obs = dataset.readout_stations(ds, grid="default" if remap else None)
            ds = ds.drop_vars(["gph_bottom", "gph_top"])
            obss.append(obs)

        dss.append(ds)

    ### !!! Caution: need to fix this properly!!!
    good_dss = []
    for i, ds in enumerate(dss):
        if is_bad_sample(ds["co2massmix"].values):
            print(f"Skipping bad sample {i}")
            continue
        good_dss.append(ds)

    ds_all = xr.concat(good_dss, dim="sample")
    ### !!!

    if save_obs:
        obs_all = xr.concat(obss, dim="sample").fillna({"obs_filename": ""})
        obs_all.to_zarr(obspath, mode="w")
        ds_all = ds_all.fillna({"obs_filename": ""})

    ds_all.to_zarr(zarrpath, mode="w")

    if analyze_masking and masking:
        plot_masking_diagnostics(batch, ds_all,
                                 str(outpath).replace("preds", "plots"),
                                 varnames=target_vars_3d, nlat=nlat, nlon=nlon,
                                 imgformats=["png"])

    return ds_all
        