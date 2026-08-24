"""Prepare MIP OCO-2 dataset: download, filter, regrid, write, stats."""

import tarfile
import urllib.request
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xarray.groupers import BinGrouper

from neural_transport.datasets.common import (
    compute_stats,
    optimize_zarr,
)
from neural_transport.datasets.grids import LATLON_PROTOTYPE_COORDS

dask.config.set(scheduler="threads")


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
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "OCO2MIP_OCO2": [
            "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO2_b11.2_10sec_GOOD_r2.nc4"
        ],
        "OCO2MIP_OCO3": [
            "https://gml.noaa.gov/aftp/user/andy/OCO-2/OCO3_b11_10sec_GOOD_r2.nc4"
        ],
        "OCO2MIP_TCCON": [
            "https://data.caltech.edu/records/zr28z-s4y31/files/tccon_timeaverages_R20250609.tgz"
        ],
        # Uncomment and fill when access is granted
        # "ObsPack": [
        #     "<YOUR_PRIVATE_OBSPACK_URL>"
        # ]
    }

    for key, urls in datasets.items():
        target_dir = save_dir / key
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
            if key == "OCO2MIP_TCCON" and outpath.suffix == ".tgz":
                nc_files = list(target_dir.glob("*.nc4"))
                if not nc_files:
                    try:
                        print(f"Extracting {filename}...")
                        with tarfile.open(outpath, "r:gz") as tar:
                            tar.extractall(path=target_dir)
                    except Exception as e:
                        print(f"Failed to extract {filename}: {e}")
                else:
                    print("TCCON NetCDF files already extracted — skipping")

    print("MIP OCO-2 downloads complete!")


def filter_mip_oco2(save_dir: str) -> xr.Dataset:
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
    oco2_dir = save_dir / "OCO2MIP_OCO2"
    oco2_file = oco2_dir / "OCO2_b11.2_10sec_GOOD_r2.nc4"
    zarr_file = oco2_dir / "OCO2_b11.2_10sec_GOOD_r2.zarr"
    filtered_dir = oco2_dir / "oco2_assimilate.zarr"

    if filtered_dir.is_dir() and (filtered_dir / ".zmetadata").exists():
        print(f"Skipping filtering — {filtered_dir} already exists.")
        return xr.open_zarr(filtered_dir)
    print(f"Opening {oco2_file}")
    ds = xr.open_dataset(oco2_file, chunks="auto")

    if not zarr_file.exists():
        print(f"Writing raw dataset to Zarr (once): {zarr_file}")
        ds.to_zarr(zarr_file, mode="w")

    flag = ds["assimilate_flag"].compute()
    ds_filtered = ds.where(flag == 1, drop=True)

    drop_vars = [
        "date", "assimilate_flag", "data_type",
        "xco2_quality_flag", "operation_mode",
        "land_water_indicator", "surface_type"
    ]
    ds_filtered = ds_filtered.drop_vars([v for v in drop_vars if v in ds_filtered])

    print(f"Writing filtered dataset to {filtered_dir}")
    ds_filtered.to_zarr(filtered_dir, mode="w")
    print("MIP OCO-2 filtering complete!")
    return ds_filtered


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
    tmp = da.to_dataset(name=var)
    tmp = tmp.assign_coords(time=ds["time"], lat=ds["lat"], lon=ds["lon"])
    tmp = tmp.compute()

    if weights is None:
        grouped = tmp.groupby(time=time_grouper, lat=lat_grouper, lon=lon_grouper).mean()
        out = grouped[var]
    else:
        tmp["weights"] = weights
        num = (tmp[var] * tmp["weights"]).groupby(
            time=time_grouper, lat=lat_grouper, lon=lon_grouper
        ).sum()
        den = tmp["weights"].groupby(
            time=time_grouper, lat=lat_grouper, lon=lon_grouper
        ).sum()
        out = num / den

    out = out.chunk({"time_bins": 1500, "lat_bins": -1, "lon_bins": -1})
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
    exclude_vars = [
        "lat", "lon", "time", "date",
        "assimilate_flag", "xco2_quality_flag",
        "data_type", "operation_mode",
        "land_water_indicator", "surface_type"
    ]  # First row are aggregation coords. Rest are categorical.
    if variables is None:
        variables = [
            var for var in ds.data_vars
            if "sounding_id" in ds[var].dims and var not in exclude_vars
        ]

    ds = ds.chunk({"sounding_id": min(ds.sounding_id.size, 64121)})

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


MIP_OCO2_HEIGHT = [
    0.0984, 52.6136, 105.2272, 156.4836, 210.4544, 256.2101, 312.9672, 370.0722, 420.9087, 472.6453, 512.4202, 578.6387, 625.9344, 663.7776, 740.1443, 770.1723, 841.8174, 884.8869, 945.2906, 997.8886
]  # mean of pressure levels in hPa obtained from "sigmal_levels" * "psurf" where assimilate_flag==1, rounded to 4 significant digits


MIP_OCO2_HEIGHT_STD = [
    0.0061, 3.2340, 6.4680, 9.6270, 12.9360, 16.5816, 19.2540, 22.9676, 25.8719, 29.0186, 33.1632, 35.5519, 38.5080, 43.7764, 45.9353, 49.3767, 51.7438, 54.5112, 58.0373, 61.2261
]  # std dev of pressure levels in hPa obtained from "sigmal_levels" * "psurf" where assimilate_flag==1, rounded to 4 significant digits


MIP_OCO2_LEVEL_AGG = dict(
    l20=[[i] for i in range(20)],  # native resolution
    l10=[
        [19],                   # 998 hPa (near surface)
        [18],                   # 945 (near surface)
        [17],                   # 885 (upper troposphere)
        [16],                   # 841 (upper troposphere)
        [15],                   # 770 (lower troposphere)
        [14],                   # 740 (lower troposphere)
        [13, 12, 11, 10, 9, 8], # 663-421 hPa (mid troposphere)
        [7, 6],                 # 370–312 hPa (mid troposphere)
        [5, 4, 3],              # 256–156 hPa (mid-upper troposphere)
        [2, 1, 0],              # 105-0.1 hPa (upper stratosphere)
    ][::-1],  # ordered such that averaging_kernel behaves linearly and somewhat resembles the Carbontracker l10 levels
    l5=[
        [0, 1, 2, 3, 4],      # upper stratosphere
        [5, 6, 7, 8],         # upper/mid-troposphere
        [9, 10, 11, 12],      # mid-troposphere
        [13, 14, 15],         # lower-mid troposphere
        [16, 17, 18, 19],     # near-surface
    ][::-1],
    l3=[[19], list(range(10, 19)), list(range(9))][::-1],  # ordered such that averaging_kernel behaves linearly
)


VERTICAL_LAYERS_OCO2MIP_COORDS = {
    "l20": dict(level=MIP_OCO2_HEIGHT),
    "l10": dict(level=[
        997.8886, 945.2906, 884.8869, 841.8174, 770.1723, 740.1443,
    ] + [
        np.mean([MIP_OCO2_HEIGHT[i] for i in group]) for group in MIP_OCO2_LEVEL_AGG["l10"][6:]
    ]),
    "l5": dict(level=[
        np.mean([MIP_OCO2_HEIGHT[i] for i in group]) for group in MIP_OCO2_LEVEL_AGG["l5"]
    ]),
    "l3": dict(level=[
        997.8886,
        np.mean([MIP_OCO2_HEIGHT[i] for i in range(10, 19)]),
        np.mean([MIP_OCO2_HEIGHT[i] for i in range(9)]),
    ]),
}


def vertical_aggregation_oco2(ds: xr.Dataset, levels: list[list[int]]) -> xr.Dataset:
    """
    Vertically aggregate OCO-2 profile variables according to specified level groupings.
    Uses pressure_weight as vertical weighting.
    """
    if "level" not in ds.dims:
        raise ValueError("Dataset must contain 'level' dimension (20 levels)")

    vertical_vars = [v for v in ds if "level" in ds[v].dims and v != "pressure_weight"]
    vertical_ds = ds[vertical_vars]

    aggregated_list = []
    aggregated_weights = []
    for i, lvl in enumerate(levels):
        pressure_weights = ds["pressure_weight"].isel(level=lvl)
        if len(lvl) > 1:
            pw_norm = pressure_weights / pressure_weights.sum("level")

            ds_aggregated = (vertical_ds.isel(level=lvl) * pw_norm).sum("level")
            ds_aggregated = ds_aggregated.assign_coords(dict(level=[i]))
            pw_out = pressure_weights.sum("level")
        else:
            ds_aggregated = vertical_ds.isel(level=lvl).assign_coords(dict(level=[i]))
            pw_out = pressure_weights
        if "level" not in pw_out.dims:
            pw_out = pw_out.expand_dims("level")
        pw_out = pw_out.assign_coords(dict(level=[i]))

        aggregated_list.append(ds_aggregated)
        aggregated_weights.append(pw_out)

    ds_agg = xr.concat(aggregated_list, dim="level")
    ds_agg["pressure_weight"] = xr.concat(aggregated_weights, dim="level")

    # Attach non-vertical variables
    for v in ds:
        if "level" not in ds[v].dims:
            ds_agg[v] = ds[v]

    return ds_agg


def regrid_mip_oco2(
        save_dir: str,
        gridname: str | None = "latlon2x3",
        vertical_levels: str | None = "l34",
        freq: str | None = "3h"
) -> xr.Dataset:
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
    oco2_dir = save_dir / "OCO2MIP_OCO2"
    out_dir = oco2_dir / "OCO2_regrid"
    regridded_dir = out_dir / f"OCO2_regrid_{gridname}_{vertical_levels}_{freq}.zarr"

    if regridded_dir.is_dir() and (regridded_dir / ".zmetadata").exists():
        print(f"Skipping regridding — {regridded_dir} already exists.")
        return xr.open_zarr(regridded_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oco2_file = oco2_dir / "oco2_assimilate.zarr"  # Input file

    # --- Load datasets ---
    ds_oco2 = xr.open_zarr(oco2_file)

    ds = reconstruct_pressure_levels(ds_oco2)
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
        variables=None,
        gridname=gridname,
        vertical_levels=vertical_levels,
        freq=freq,
        weights_var=None,
    )

    print(f"Aggregating vertically OCO-2 to {vertical_levels}")
    ds_full_regrid = vertical_aggregation_oco2(
        ds_spatiotemporal,
        levels=MIP_OCO2_LEVEL_AGG[vertical_levels]
    )
    ds_full_regrid["level"] = VERTICAL_LAYERS_OCO2MIP_COORDS[vertical_levels]["level"]

    # --- Write to disk ---
    ds_full_regrid = ds_full_regrid.assign_coords(
        time=ds_full_regrid["time"],
        lat=ds_full_regrid["lat"],
        lon=ds_full_regrid["lon"],
        level=ds_full_regrid["level"],
    )
    ds_full_regrid = ds_full_regrid.chunk(dict(time=-1, lat=-1, lon=-1, level=-1))
    print(f"Writing regridded dataset to {regridded_dir}")
    with ProgressBar():
        ds_full_regrid.to_zarr(regridded_dir, mode="w")
    print("MIP OCO-2 regridding complete!")
    return ds_full_regrid


def write_mip_oco2(
        save_dir: str,
        gridname: str | None = "latlon2x3",
        vertical_levels: str | None = "l34",
        freq: str | None = "3h"
) -> None:
    """Separate OCO-2 regridded data into train/val/test splits and write to disk."""
    save_dir = Path(save_dir)
    oco2_dir = save_dir / "OCO2MIP_OCO2"
    ds = xr.open_zarr(
        oco2_dir / "OCO2_regrid" / f"OCO2_regrid_{gridname}_{vertical_levels}_{freq}.zarr"
    )

    for split, timeslice in zip(
        ["val", "test", "train"],
        [
            slice("2021-01-01", "2021-12-31"),
            slice("2022-01-01", "2024-07-31"),
            slice(None, "2020-12-31"),
        ],
    ):
        print(f"Writing {split} to zarr")
        out_dir = oco2_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_opt = optimize_zarr(ds.sel(time=timeslice))
        ds_opt.to_zarr(
            out_dir / f"mip_oco2_{gridname}_{vertical_levels}_{freq}.zarr",
            mode="w",
        )
        print(f"MIP OCO-2 writing {split} complete!")


def stats_mip_oco2(
        save_dir: str, 
        gridname: str | None = "latlon2x3", 
        vertical_levels: str | None = "l34", 
        freq: str | None = "3h"
) -> None:
    """
    Compute and save statistics for MIP OCO-2 data (train/val/test).
    """
    save_dir = Path(save_dir)
    oco2_dir = save_dir / "OCO2MIP_OCO2"

    train_dir = oco2_dir / "train"
    val_dir = oco2_dir / "val"
    test_dir = oco2_dir / "test"

    ds = xr.open_zarr(
        train_dir / f"mip_oco2_{gridname}_{vertical_levels}_{freq}.zarr"
    )

    ds_stats = compute_stats(ds)

    for out_dir in [train_dir, val_dir, test_dir]:
        ds_stats.to_zarr(
            out_dir / f"mip_oco2_{gridname}_{vertical_levels}_{freq}_stats.zarr",
            mode="w",
        )
    print("MIP OCO-2 statistics computation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--gridname", type=str, default="latlon1x1")
    parser.add_argument("--vertical_levels", type=str, default="l34")
    parser.add_argument("--freq", type=str, default="3h")
    args = parser.parse_args()

    download_data(args.save_dir)


    filter_mip_oco2(args.save_dir)


    regrid_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )


    write_mip_oco2(
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