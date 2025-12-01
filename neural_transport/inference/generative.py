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


def generate_noise(batch, target_var="co2massmix", n_samples=10, noise=None):

    all_levels = batch[target_var] # [T N C]
    all_levels = all_levels.unsqueeze(0) # [B T N C]

    if noise is None:
        return [torch.randn_like(all_levels) for _ in range(n_samples)]

    elif noise == "spiral_outward_noise":
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

    elif noise == "spiral_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        angles = torch.linspace(0, 4*torch.pi, n_samples)

        spiral_noises = []
        for a in angles:
            x_init = torch.cos(a) * x0 + torch.sin(a) * v
            spiral_noises.append(x_init)
        return spiral_noises

    elif noise == "geodesic_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [torch.sqrt((1 - alpha)) * x0 + torch.sqrt(alpha) * v for alpha in alphas]

    elif noise == "linear_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [(1 - alpha) * x0 + alpha * v for alpha in alphas]

    elif noise == "antipodal_orthogonal_noise":
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
        raise ValueError(f"Unknown noise type: {noise}")


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
    # obs_values = torch.where(obs_mask, obs_values, torch.zeros_like(obs_values))
    obs_values = molemix_to_massmix(obs_values)

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


def create_mask(batch, target_var="co2massmix", obs_fraction=0.1, pattern="random", nlat=32, nlon=64):
    """
    Create a random observation mask for the input batch.
    Args:
        batch: dict of tensors, each of shape [B T N C]
        target_var: the variable to create the mask for
        obs_fraction: fraction of points to keep as observations
        pattern: "random", "vertical", "horizontal", "checkerboard", "satellite"
    """
    device = batch[target_var].device
    B, T, N, C = batch[target_var].shape

    obs_mask = torch.zeros((B, T, N, C), dtype=torch.bool, device=device)
    obs_values = torch.full_like(batch[target_var], float('nan'), device=device)

    for t in range(T):
        if pattern == "random":
            num_obs = int(obs_fraction * N)
            obs_indices = torch.randperm(N, device=device)[:num_obs]

        elif pattern == "vertical":
            # keep a fixed fraction of longitude columns
            num_cols = max(1, int(obs_fraction * nlon))
            cols = torch.arange(0, nlon, nlon // num_cols, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[:, cols].reshape(-1)

        elif pattern == "horizontal":
            # keep a fixed fraction of latitude rows
            num_rows = max(1, int(obs_fraction * nlat))
            rows = torch.arange(0, nlat, nlat // num_rows, device=device)
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            obs_indices = grid[rows, :].reshape(-1)

        elif pattern == "checkerboard":
            grid = torch.arange(N, device=device).reshape(nlat, nlon)
            mask2d = (torch.arange(nlat, device=device)[:, None] +
                      torch.arange(nlon, device=device)[None, :]) % 2 == 0
            obs_indices = grid[mask2d].reshape(-1)

        elif pattern == "satellite":
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
            raise ValueError(f"Unknown mask pattern: {pattern}")

        # fill mask + values
        obs_mask[:, t, obs_indices, :] = True
        obs_values[:, t, obs_indices, :] = batch[target_var][:, t, obs_indices, :]

    return obs_mask, obs_values  # [B T N C] each


def is_bad_sample(arr, thresh=1e6):
        return np.isnan(arr).any() or np.isinf(arr).any() or np.nanmax(np.abs(arr)) > thresh


def iterative_generate_oco2(
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
    forcing_vars_3d=[],
    target_vars_2d=[],
    save_obs=True,
    **generate_kwargs,
):
    n_samples = generate_kwargs.get("n_samples", 10)
    masking = generate_kwargs.get("masking", True)
    analyze_masking = generate_kwargs.get("analyze_masking", False)
    noise = generate_kwargs.get("noise", None)
    analyze_noise = generate_kwargs.get("analyze_noise", False)

    nlat, nlon = model.model.in_nlat, model.model.in_nlon

    zarrpath, obspath = get_zarrpath_obspath(outpath, rollout, freq, zarr_filename, zero_surfflux = zero_surfflux)

    prototype_zarr = dataset.create_prototype_zarr(
        zarrpath,
        target_vars_3d=forcing_vars_3d,
        target_vars_2d=target_vars_2d,
        grid="default" if remap else None,
    )

    model = model.eval().to(device)
    model.return_intermediates = True
    model.model.generate_kwargs = generate_kwargs

    dss = []
    obss = []
    T = len(dataset)
    T = 5 # for testing

    for t in tqdm(range(T), desc="Timestep") if verbose else range(T):
        batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[t].items()}

        # Noise
        noise_list = generate_noise(dataset[t],
                                    target_var=forcing_vars_3d[0],
                                    n_samples=n_samples,
                                    noise=noise)
        if analyze_noise and noise is not None:
            if noise in ["spiral_noise", "spiral_outward_noise"]:
                angles = torch.linspace(0, 4*torch.pi, n_samples) # thetas
                param_name = "$\\theta$"
            elif noise in ["geodesic_noise", "linear_noise"]:
                angles = torch.linspace(0, 1, n_samples)  # alphas
                param_name = "$\\alpha$"
            elif noise == "antipodal_orthogonal_noise":
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

        # Masking
        if masking:
            target_var = target_vars_2d[0]
            obs_mask, obs_values = create_oco2_mask(batch, target_var=target_var)

            batch["obs_mask"] = obs_mask
            obs_values_normed = model.model.normalize_observations(obs_values, batch, target_var=target_var, targshift=False)
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

        ds = xr.Dataset(
            {k: dataset.tensor_to_xarray(pred) for k, pred in preds_fixed.items()}
        )

        ds = ds.rename({"batch": "sample", "time": "trajectory_steps"})
        ds = ds.assign_coords(
            trajectory_steps=("trajectory_steps", np.arange(traj.shape[1])),
            sample=("sample", np.arange(n_samples)),
            time=("time", [prototype_zarr.isel(time=t).time.values]*n_samples),
        )
        
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
        # if is_bad_sample(ds[target_vars_2d[0]].values):
        #     print(f"Skipping bad sample {i}")
        #     continue
        good_dss.append(ds)

    ds_all = xr.concat(good_dss, dim="time")
    ### !!!

    if save_obs:
        obs_all = xr.concat(obss, dim="time").fillna({"obs_filename": ""})
        obs_all.to_zarr(obspath, mode="w")
        ds_all = ds_all.fillna({"obs_filename": ""})

    ds_all.to_zarr(zarrpath, mode="w")

    if analyze_masking and masking:
        plot_masking_diagnostics(batch, ds_all,
                                 str(outpath).replace("preds", "plots"),
                                 varnames=target_vars_2d, nlat=nlat, nlon=nlon,
                                 imgformats=["png"])

    return ds_all


def iterative_generate(
    model,
    dataset,
    outpath,
    rollout=False,
    device="cuda",
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
    masking = generate_kwargs.get("masking", False)
    pattern = generate_kwargs.get("pattern", "vertical")
    analyze_masking = generate_kwargs.get("analyze_masking", False)
    obs_fraction = generate_kwargs.get("obs_fraction", 0.2)
    noise = generate_kwargs.get("noise", None)
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

    dss = []
    obss = []

    # Noise
    noise_list = generate_noise(dataset[0], target_var=target_vars_3d[0], n_samples=n_samples, noise=noise)
    if analyze_noise and noise is not None:
        if noise in ["spiral_noise", "spiral_outward_noise"]:
            angles = torch.linspace(0, 4*torch.pi, n_samples) # thetas
            param_name = "$\\theta$"
        elif noise in ["geodesic_noise", "linear_noise"]:
            angles = torch.linspace(0, 1, n_samples)  # alphas
            param_name = "$\\alpha$"
        elif noise == "antipodal_orthogonal_noise":
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

    base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[0].items()} # condition on the first timestep

    if masking:
        target_var = target_vars_3d[0]
        if pattern == "oco2":
            obs_mask, obs_values = create_oco2_mask(base_batch, target_var=target_var)
        else:
            obs_mask, obs_values = create_mask(base_batch, target_var=target_var, obs_fraction=obs_fraction, pattern=pattern, nlat=nlat, nlon=nlon)
        base_batch["obs_mask"] = obs_mask
        obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
        base_batch["obs_values"] = obs_values_normed

    for i in range(n_samples):
        # base_batch = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items()} # condition on different timesteps

        # if masking:
        #     target_var = target_vars_3d[0]
        #     obs_mask, obs_values = create_mask(base_batch, target_var=target_var, obs_fraction=obs_fraction, pattern=pattern, nlat=nlat, nlon=nlon)
        #     base_batch["obs_mask"] = obs_mask
        #     obs_values_normed = model.model.normalize_observations(obs_values, base_batch, target_var=target_var)
        #     base_batch["obs_values"] = obs_values_normed
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

        ds = ds.assign_coords(
            time=("time", prototype_zarr.time[:traj.shape[1]].values if traj.ndim > 0 else [0]),
            sample=("sample", [i]),
        )
        print(ds.dims)
        
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
        