"""Prepare MIP OCO-2 dataset: download, regrid, resample, write, stats, obspack comparison."""

from pathlib import Path
import urllib.request
import tarfile

import numpy as np
import xarray as xr
from xarray.groupers import BinGrouper

from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS


def download_data(save_dir: str):
    """
    Download MIP OCO-2, OCO-3, TCCON, and (optionally) ObsPack datasets
    to the specified directory.

    Parameters
    ----------
    save_dir : str
        Directory where all datasets will be saved.
    """
    save_dir = Path(save_dir)
    base_dir = save_dir / "OCO2MIP"
    base_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "OCO2": [
            "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO2_b11.2_10sec_GOOD_r2.nc4"
        ],
        "OCO3": [
            "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO3_b11_10sec_GOOD_r2.nc4"
        ],
        "TCCON": [
            "https://data.caltech.edu/records/zr28z-s4y31/files/tccon_timeaverages_R20250609.tgz"
        ],
        # Uncomment and fill when access is granted
        # "ObsPack": [
        #     "<YOUR_PRIVATE_OBSPACK_URL>"
        # ]
    }

    for key, urls in datasets.items():
        target_dir = base_dir / key
        target_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            filename = Path(url).name
            outpath = target_dir / filename

            if not outpath.exists():
                print(f"Downloading {filename} to {target_dir}")
                try:
                    urllib.request.urlretrieve(url, outpath)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            else:
                print(f"{filename} already exists — skipping")

            # Extract TCCON archive if needed
            if key == "TCCON" and outpath.suffix == ".tgz":
                try:
                    print(f"Extracting {filename}...")
                    with tarfile.open(outpath, "r:gz") as tar:
                        tar.extractall(path=target_dir)
                except Exception as e:
                    print(f"Failed to extract {filename}: {e}")

    print("MIP OCO-2 downloads complete!")


def filter_trainset_mip_oco2(save_dir: str) -> xr.Dataset:
    """
    Filter MIP OCO-2 dataset to include only observations to assimilate and drop categorical/non-numeric variables.
    Converts NetCDF to Zarr format for faster access.

    Parameters
    ----------
    save_dir : str
        Directory where the raw OCO-2 dataset is stored.

    Returns
    -------
    ds : xr.Dataset
        Filtered dataset containing only assimilated observations.
    """
    save_dir = Path(save_dir)
    oco2_dir = save_dir / "OCO2MIP" / "OCO2"
    oco2_file = oco2_dir / "OCO2_b11.2_10sec_GOOD_r2.nc4"

    print(f"Converting OCO-2 to Zarr from {oco2_file}")
    ds_oco2 = xr.open_dataset(oco2_file)
    ds_oco2.to_zarr(f"{oco2_dir}/OCO2_b11.2_10sec_GOOD_r2.zarr", mode="w")

    flag = ds_oco2["assimilate_flag"].compute()
    ds_train = ds_oco2.where(flag == 1, drop=True)

    drop_vars = [
        "date", "assimilate_flag", "data_type",
        "xco2_quality_flag", "operation_mode",
        "land_water_indicator", "surface_type"
    ]
    ds_train = ds_train.drop_vars([v for v in drop_vars if v in ds_train.variables])

    ds_train.to_zarr(f"{oco2_dir}/oco2_train.zarr", mode="w")
    print(f"Filtered dataset written to {oco2_dir}/oco2_train.zarr")

    return ds_train


def reconstruct_pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    """
    Reconstructs vertical pressure levels for each OCO-2 sounding 
    using sigma_levels * psurf.

    Returns
    -------
    ds : xr.Dataset
        Input dataset with a new variable 'pressure_levels' [hPa].
    """
    sigma = ds["sigma_levels"]  # (level=20,)
    psurf = ds["psurf"]  # (sounding_id=3513503,)

    p_levels = (psurf * sigma).transpose("sounding_id", "levels")  # (sounding_id, levels)

    ds["pressure_levels"] = p_levels
    ds["pressure_levels"].attrs = {
        "units": "hPa",
        "long_name": "Pressure levels reconstructed from sigma_levels * psurf"
    }
    return ds


def reconstruct_co2_profile_from_xco2(ds: xr.Dataset) -> xr.Dataset:
    """
    Reconstruct per-sounding vertical CO2 profiles from OCO-2 XCO2 using
    the averaging kernel and the prior profile.

    Parameters
    ----------
    ds : xr.Dataset
        OCO-2 dataset containing variables:
          - xco2_var (sounding_id,)
          - prior_profile_var (sounding_id, levels)
          - ak_var (sounding_id, levels)
          - pressure_weight_var (sounding_id, levels)

    Returns
    -------
    ds : xr.Dataset
        Input dataset with a new variable 'co2_profile_retrieved' with dims ('sounding_id','levels') and units ppm.
    """
    xa = ds['co2_profile_apriori']       # ('sounding_id','levels')
    A  = ds['xco2_averaging_kernel']     # ('sounding_id','levels')
    w  = ds['pressure_weight']           # ('sounding_id','levels')
    X  = ds['xco2_raw']                  # ('sounding_id',)

    # compute prior column mean
    # note: pressure_weight is already a fraction (sums ~1 across levels)
    xa_col = (xa * w).sum(dim='levels')  # ('sounding_id',)
    xa_col.name = "co2_profile_apriori_colmean"

    delta = X - xa_col                   # ('sounding_id',)
    delta_exp = delta.expand_dims({'levels': xa.coords['levels']}) # ('sounding_id','levels')

    co2_retrieved = xa + A * delta_exp
    co2_retrieved.name = "co2_profile_retrieved"
    co2_retrieved.attrs = {
        "units": "ppm",
        "long_name": "Reconstructed retrieved CO2 profile from XCO2, prior and averaging kernel",
        "formula": "xa + A * (X - sum(w * xa))"
    }

    ds[co2_retrieved.name] = co2_retrieved

    return ds


def _parse_freq(freq: str) -> np.timedelta64:
    """Convert `freq` (e.g.: '3h', '6h', '1D', ...) into a numpy timedelta."""
    num = int(''.join(filter(str.isdigit, freq)))
    unit = ''.join(filter(str.isalpha, freq))

    valid_units = {"s", "m", "h", "D", "M", "Y"}
    if unit not in valid_units:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    return np.timedelta64(num, unit)


def _aligned_time_bins(t_min: np.datetime64, t_max: np.datetime64, delta_t: np.timedelta64) -> tuple[np.ndarray, np.ndarray]:
    """
    Align center datetimes to regular multiples of delta_t (e.g. 3h, 6h)
    and return bin edges + labels.
    """

    unit = str(delta_t.dtype).replace("timedelta64[", "").replace("]", "")
    step_int = int(delta_t.astype(f"timedelta64[{unit}]").astype(int))

    t_min_int = int(t_min.astype(f"datetime64[{unit}]").astype(int))
    t_max_int = int(t_max.astype(f"datetime64[{unit}]").astype(int))

    t_min_center = (t_min_int // step_int + 1) * step_int
    t_max_center = (t_max_int // step_int + 1) * step_int

    time_labels = np.arange(
        np.datetime64(t_min_center, unit),
        np.datetime64(t_max_center, unit),
        delta_t
    )

    half_step = delta_t / 2
    time_bins = np.concatenate(([time_labels[0] - half_step],
                                time_labels + half_step))

    time_bins = time_bins.astype("datetime64[ns]")
    time_labels = time_labels.astype("datetime64[ns]")
    return time_bins, time_labels


def _align_spatial_bins(centers: np.ndarray) -> np.ndarray:
    step = centers[1] - centers[0]
    edges = np.concatenate(([centers[0] - step/2],
                            centers + step/2))
    return edges


def regrid_temporal(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    freq: str | None = "3h",
    weights: np.ndarray | None = None
) -> xr.Dataset:
    """
    Temporally regrid OCO-2 soundings into regular time bins (default 3-hourly).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'time' and the specified variables.
    variables : list[str] | None, optional
        List of variable names to regrid. If None, all numeric variables 
        containing 'sounding_id' are included.
    freq : str | None, optional
        Target frequency for regridding (default "3h").
    weights : array-like | None, optional
        Optional weights for computing weighted means.

    Returns
    -------
    xr.Dataset
        Temporally regridded dataset with dimensions ['time', ('level')].
    """
    # Determine which variables to regrid
    if variables is None:
        variables = [
            var for var in ds.data_vars
            if "sounding_id" in ds[var].dims and var not in ["time", "date", "assimilate_flag", "data_type", "xco2_quality_flag", "operation_mode", "land_water_indicator", "surface_type"]
        ]

    # Compute time bin edges and labels
    delta_t = _parse_freq(freq)
    t_min, t_max = [np.datetime64(v, "ns") for v in (np.nanmin(ds["time"].values), np.nanmax(ds["time"].values))]
    time_bins, time_labels = _aligned_time_bins(t_min, t_max, delta_t)

    out_vars = {}

    for var in variables:
        print(f"Temporally regridding variable: {var}")
        da = ds[var]
        if "level" in da.dims:
            regridded_levels = []
            for lev in da.level:
                da_lev = da.sel(level=lev)
                tmp = xr.Dataset({"time": ds["time"], var: da_lev})
                if weights is None:
                    out = tmp.groupby_bins("time", bins=time_bins).mean()[var]
                else:
                    weighted_sum = (tmp[var] * weights).groupby_bins("time", bins=time_bins).sum()
                    sum_weights = xr.DataArray(weights).groupby_bins("time", bins=time_bins).sum()
                    out = weighted_sum / sum_weights
                out = out.rename({"time_bins": "time"})
                out = out.assign_coords(time=time_labels)
                out = out.expand_dims("level")
                out = out.assign_coords(level=[lev])
                regridded_levels.append(out)
            out_vars[var] = xr.concat(regridded_levels, dim="level")
        else:
            tmp = xr.Dataset({"time": ds["time"], var: da})
            if weights is None:
                out = tmp.groupby_bins("time", bins=time_bins).mean()[var]
            else:
                weighted_sum = (tmp[var] * weights).groupby_bins("time", bins=time_bins).sum()
                sum_weights = xr.DataArray(weights).groupby_bins("time", bins=time_bins).sum()
                out = weighted_sum / sum_weights
            out = out.rename({"time_bins": "time"})
            out = out.assign_coords(time=time_labels)
            out_vars[var] = out

    ds_temporal = xr.merge(out_vars.values()).sortby("time")

    ds_temporal.attrs.update({
        "title": f"OCO-2 regridded to {freq}",
        "temporal_frequency": freq,
        "time_method": "mean" if weights is None else "weighted mean",
        "time_range": f"{str(t_min)} to {str(t_max)}",
        "source": "NOAA GML / Caltech MIP OCO-2 products",
    })

    return ds_temporal


def regrid_spatial(ds: xr.Dataset,
                   variables: list[str] | None = None,
                   gridname: str = "latlon2x3",
                   weights: np.ndarray | None = None) -> xr.Dataset:
    """
    Spatially regrid OCO-2 soundings to a regular lat-lon grid.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'lat', 'lon', and the specified variables.
    variables : list[str], optional
        List of variable names to regrid. If None, all data variables containing "sounding_id" in dimensions except 'lat'/'lon' and categorical variables are regridded.
    gridname : str, optional
        Name of the target grid (must exist in LATLON_PROTOTYPE_COORDS).
    weights : array-like, optional
        Optional weights for computing weighted means.

    Returns
    -------
    xr.Dataset
        Regridded dataset with dimensions ['time', 'lat', 'lon', ('level')].
    """
    if variables is None:
        variables = [
            var for var in ds.data_vars
            if "sounding_id" in ds[var].dims and var not in ["lat", "lon", "time", "date", "assimilate_flag", "data_type", "xco2_quality_flag", "operation_mode", "land_water_indicator", "surface_type"]
        ]

    coords = LATLON_PROTOTYPE_COORDS[gridname]
    ds = ds.assign_coords(lon=((ds["lon"] + 360) % 360))  # Convert OCO-2 lon to [0, 360)
    lat_bins = np.linspace(coords["lat"].min(), coords["lat"].max(), len(coords["lat"]) + 1)
    lon_bins = np.linspace(coords["lon"].min(), coords["lon"].max(), len(coords["lon"]) + 1)

    lat_grouper = BinGrouper(bins=lat_bins)
    lon_grouper = BinGrouper(bins=lon_bins)

    out_vars = {}

    for var in variables:
        print(f"Spatially regridding variable: {var}")
        da = ds[var]

        # If variable has a "level" dimension, regrid per level
        if "level" in da.dims:
            regridded_levels = []
            for lev in da.level:
                values = da.sel(level=lev)
                tmp_ds = xr.Dataset({
                    "lat": (("obs",), ds["lat"].values),
                    "lon": (("obs",), ds["lon"].values),
                    var: (("obs",), values.values)
                })
                if weights is not None:
                    tmp_ds["weights"] = (("obs",), weights)

                if weights is None:
                    out = tmp_ds.groupby(lat=lat_grouper, lon=lon_grouper).mean()[var]
                else:
                    weighted_sum = (tmp_ds[var] * tmp_ds["weights"]).groupby(
                        lat=lat_grouper, lon=lon_grouper
                    ).sum()
                    sum_weights = tmp_ds["weights"].groupby(
                        lat=lat_grouper, lon=lon_grouper
                    ).sum()
                    out = weighted_sum / sum_weights

                out = out.expand_dims("level")
                out = out.rename({"lat_bins": "lat", "lon_bins": "lon"})
                out = out.assign_coords(
                    level=[lev],
                    lat=[(i.left + i.right)/2 for i in out["lat"].values],
                    lon=[(i.left + i.right)/2 for i in out["lon"].values],
                )
                regridded_levels.append(out)

            out_vars[var] = xr.concat(regridded_levels, dim="level")
        else:
            tmp_ds = xr.Dataset({
                "lat": (("obs",), ds["lat"].values),
                "lon": (("obs",), ds["lon"].values),
                var: (("obs",), da.values)
            })
            if weights is not None:
                tmp_ds["weights"] = (("obs",), weights)

            if weights is None:
                out = tmp_ds.groupby(lat=lat_grouper, lon=lon_grouper).mean()[var]
            else:
                weighted_sum = (tmp_ds[var] * tmp_ds["weights"]).groupby(
                    lat=lat_grouper, lon=lon_grouper
                ).sum()
                sum_weights = tmp_ds["weights"].groupby(
                    lat=lat_grouper, lon=lon_grouper
                ).sum()
                out = weighted_sum / sum_weights

            out = out.rename({"lat_bins": "lat", "lon_bins": "lon"})
            out = out.assign_coords(
                lat=[(i.left + i.right)/2 for i in out["lat"].values],
                lon=[(i.left + i.right)/2 for i in out["lon"].values],
            )
            out_vars[var] = out

    ds_spatial = xr.merge(list(out_vars.values()))
    # ds_spatial = ds_spatial.assign_coords(lon=((ds_spatial["lon"] + 180) % 360) - 180)
    # ds_spatial = ds_spatial.sortby("lon")

    ds_spatial.attrs.update({
        "title": f"OCO-2 regridded to {gridname}",
        "grid": gridname,
        "grid_method": "mean" if weights is None else "weighted mean",
        "source": "NOAA GML / Caltech MIP OCO-2 products",
    })

    return ds_spatial


def _agg_1d(
    ds: xr.Dataset,
    var: str,
    da: xr.DataArray,
    time_labels: np.ndarray,
    time_grouper: BinGrouper,
    lat_grouper: BinGrouper,
    lon_grouper: BinGrouper,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Aggregate a single 1D (or level-slice) variable into spatio-temporal bins."""
    tmp = xr.Dataset(
        {
            "time": ds["time"],
            "lat": ds["lat"],
            "lon": ds["lon"],
            var: da,
        }
    )
    if weights is not None:
        tmp["weights"] = weights

    if weights is None:
        grouped = tmp.groupby(time=time_grouper, lat=lat_grouper, lon=lon_grouper).mean()
        out = grouped[var]
    else:
        num = (tmp[var] * tmp["weights"]).groupby(
            time=time_grouper, lat=lat_grouper, lon=lon_grouper
        ).sum()
        den = tmp["weights"].groupby(
            time=time_grouper, lat=lat_grouper, lon=lon_grouper
        ).sum()
        out = num / den

    out = out.rename({"time_bins": "time", "lat_bins": "lat", "lon_bins": "lon"})
    out = out.assign_coords(time=time_labels)
    return out


def regrid_spatiotemporal(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    gridname: str = "latlon2x3",
    vertical_levels: str | None = "l34",
    freq: str = "3h",
    weights_var: str | None = None,
) -> xr.Dataset:
    """
    Joint spatio-temporal regridding: aggregate OCO-2 soundings into bins
    (time, lat, lon). Keeps level variables by aggregating each level.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with coords 'time', 'lat', 'lon' on dimension 'sounding_id'.
    variables : list[str] | None
        Variables to aggregate. If None, choose numeric vars on sounding_id.
    gridname : str
        Grid name in LATLON_PROTOTYPE_COORDS.
    freq : str
        Temporal frequency like '3h' or '6h'.
    weights_var : str | None
        Name of per-sounding weights variable (e.g. 1/uncertainty^2). If None,
        plain mean is used.

    Returns
    -------
    xr.Dataset
        Aggregated dataset with dims:
          - for 2D vars: (time, lat, lon)
          - for level vars: (time, level, lat, lon)
    """
    if variables is None:
        variables = [
            var for var in ds.data_vars
            if "sounding_id" in ds[var].dims and var not in ["time", "date", "assimilate_flag", "data_type", "xco2_quality_flag", "operation_mode", "land_water_indicator", "surface_type"]
        ]

    ds = ds.chunk({"sounding_id": min(ds.sounding_id.size, 500_000)})

    # prepare bins
    delta_t = _parse_freq(freq)
    t_min, t_max = [np.datetime64(v, "ns") for v in (np.nanmin(ds["time"].values), np.nanmax(ds["time"].values))]
    time_bins, time_labels = _aligned_time_bins(t_min, t_max, delta_t)

    coords = LATLON_PROTOTYPE_COORDS[gridname]
    ds = ds.assign_coords(lon=((ds["lon"] + 360) % 360))  # Convert OCO-2 lon to [0, 360)
    lat_centers = coords["lat"]
    lon_centers = coords["lon"]
    lat_bins = _align_spatial_bins(coords["lat"])
    lon_bins = _align_spatial_bins(coords["lon"])

    # groupers
    time_grouper = BinGrouper(bins=time_bins)
    lat_grouper = BinGrouper(bins=lat_bins)
    lon_grouper = BinGrouper(bins=lon_bins)

    weights = ds[weights_var] if weights_var is not None else None

    out_vars = {}
    for var in variables:
        da = ds[var]
        if "level" in da.dims:
            agg_levels = []
            for lev in da["level"].values:
                agg = _agg_1d(ds, var, da.sel(level=lev), time_labels,
                              time_grouper, lat_grouper, lon_grouper,
                              weights=weights)  # [time, lat, lon]
                agg = agg.expand_dims("level")  # [time, lat, lon, level=1]
                agg = agg.assign_coords(level=[lev])
                agg_levels.append(agg)
            stacked = xr.concat(agg_levels, dim="level")  # [level, time, lat, lon]
            stacked = stacked.transpose("time", "level", "lat", "lon")  # [time, level, lat, lon]
            stacked.name = var
            out_vars[var] = stacked
        else:
            agg = _agg_1d(ds, var, da, time_labels,
                          time_grouper, lat_grouper, lon_grouper,
                          weights=weights)  # [time, lat, lon]
            agg.name = var
            out_vars[var] = agg

    ds_regrid = xr.merge(list(out_vars.values()))
    ds_regrid = ds_regrid.assign_coords(time=time_labels, lat=lat_centers, lon=lon_centers)

    ds_regrid = ds_regrid.compute()

    # sort lon back to [-180,180) if desired (optional)
    # ds_regrid = ds_regrid.assign_coords(lon=((ds_regrid["lon"] + 180) % 360) - 180)
    # ds_regrid = ds_regrid.sortby("lon")

    ds_regrid.attrs.update(
        {
            "title": f"OCO-2 regridded to {gridname}_{vertical_levels}_{freq}",
            "grid": gridname,
            "grid_method": "mean" if weights is None else "weighted mean",
            "vertical_levels": vertical_levels,
            "temporal_frequency": freq,
            "time_method": "mean" if weights is None else "weighted mean",
            "time_range": f"{str(t_min)} to {str(t_max)}",
            "source": "NOAA GML / Caltech MIP OCO-2 products",
        }
    )

    return ds_regrid


def regrid_mip_oco2(
        save_dir: str,
        gridname: str | None = "latlon2x3",
        vertical_levels: str | None = "l34",
        freq: str | None = "3h"
) -> None:
    """
    Regrid MIP OCO-2 data to the specified grid and vertical levels.

    Parameters
    ----------
    save_dir : str
        Directory containing the raw OCO2MIP data (downloaded files).
    gridname : str | None
        Name of the target lat-lon grid (must exist in LATLON_PROTOTYPE_COORDS).
    vertical_levels : str | None
        Vertical level definition (default 'l34').
    freq : str | None
        Temporal frequency for resampling (default '3h').

    Returns
    -------
    None
        Writes regridded datasets to disk in Zarr format.
    """
    save_dir = Path(save_dir)
    base_dir = save_dir / "OCO2MIP"
    oco2_dir = base_dir / "OCO2"
    out_dir = oco2_dir / "OCO2_regrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    oco2_file = oco2_dir / "oco2_train.zarr"

    # --- Load datasets ---
    ds_oco2 = xr.open_zarr(oco2_file)

    ds = reconstruct_pressure_levels(ds_oco2)
    # ds = reconstruct_co2_profile_from_xco2(ds)
    ds = ds.rename({"latitude": "lat", "longitude": "lon", "levels": "level"})

    # --- Define regridding operation ---
    # print(f"Regridding temporally OCO-2 to {freq} frequency")
    # ds_temporal = regrid_temporal(
    #     ds,
    #     variables=["xco2_raw", "xco2_apriori", "xco2_2019_scale", "co2_profile_retrieved", "pressure_levels"],
    #     freq=freq,
    #     weights=None
    # )

    # print(f"Regridding Spatially OCO-2 to {gridname}")
    # ds_spatiotemporal = regrid_spatial(
    #     ds_temporal,
    #     variables=["xco2_raw", "xco2_apriori", "xco2_2019_scale", "co2_profile_retrieved", "pressure_levels"],
    #     gridname=gridname,
    #     weights=None
    # )

    print(f"Regridding spatiotemporally OCO-2 to {gridname}_{vertical_levels}_{freq}")
    ds_spatiotemporal = regrid_spatiotemporal(
        ds,
        variables=["xco2_raw", "xco2_apriori", "xco2_2019_scale", "co2_profile_retrieved", "pressure_levels"],
        gridname=gridname,
        vertical_levels=vertical_levels,
        freq=freq,
        weights_var=None,
    )

    # --- Write to disk ---
    out_path = out_dir / f"OCO2_regrid_{gridname}_{vertical_levels}_{freq}.zarr"
    print(f"Writing regridded dataset to {out_path}")
    ds_spatiotemporal.to_zarr(out_path)

    print("Regridding complete!")


def stats_mip_oco2(save_dir: str, gridname: str, vertical_levels: str, freq: str):
    """Compute statistics for MIP OCO-2 data."""
    # Implementation for computing statistics goes here
    pass


def obspack_mip_oco2(save_dir: str, gridname: str, vertical_levels: str, freq: str):
    """Compare MIP OCO-2 data with Obspack observations."""
    # Implementation for comparison with Obspack goes here
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gridname", type=str, default="latlon1x1")
    parser.add_argument("--vertical_levels", type=str, default="l34")
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    download_data(args.save_dir)


    filter_trainset_mip_oco2(args.save_dir)


    regrid_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )


    stats_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )

    obspack_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )
