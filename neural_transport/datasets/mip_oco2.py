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


def regrid_spatial(ds: xr.Dataset,
                   variable: str | None = "xco2_raw",
                   gridname: str = "latlon2x3",
                   weights: np.ndarray | None = None) -> xr.Dataset:
    """
    Spatially regrid OCO-2 soundings to a regular lat-lon grid.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'lat', 'lon', and the specified variable.
    variable : str, optional
        Name of the variable to regrid (default is 'xco2_raw').
    gridname : str, optional
        Name of the target grid (must exist in LATLON_PROTOTYPE_COORDS).
    weights : array-like, optional
        Optional weights for computing weighted means.

    Returns
    -------
    xr.Dataset
        Regridded dataset with dimensions ('lat_bins', 'lon_bins').
    """
    coords = LATLON_PROTOTYPE_COORDS[gridname]
    lat_bins = np.linspace(coords["lat"].min(), coords["lat"].max(), len(coords["lat"]) + 1)
    lon_bins = np.linspace(coords["lon"].min(), coords["lon"].max(), len(coords["lon"]) + 1)
    print(f"Regridding to {len(lat_bins)-1} lat bins and {len(lon_bins)-1} lon bins.")
    print(f"Min lat/lon: {lat_bins.min()}/{lon_bins.min()}, Max lat/lon: {lat_bins.max()}/{lon_bins.max()}")

    ds = xr.Dataset({
        "lat": (("obs",), ds["lat"].values),
        "lon": (("obs",), ds["lon"].values),
        "values": (("obs",), ds[variable].values)
    })

    if weights is not None:
        ds["weights"] = (("obs",), weights)

    lat_grouper = BinGrouper(bins=lat_bins)
    lon_grouper = BinGrouper(bins=lon_bins)

    if weights is None:
        # Simple mean
        binned = ds.groupby(lat=lat_grouper, lon=lon_grouper).mean()
        out = binned["values"]
    else:
        # Weighted mean
        weighted_sum = (ds["values"] * ds["weights"]).groupby(
            lat=lat_grouper, lon=lon_grouper
        ).sum()
        sum_weights = ds["weights"].groupby(
            lat=lat_grouper, lon=lon_grouper
        ).sum()
        out = weighted_sum / sum_weights

    # Rename for clarity
    out.name = f"{variable}_regridded"
    out = out.rename({"lat_bins": "lat", "lon_bins": "lon"})

    out = out.assign_coords(
        lat=[(i.left + i.right) / 2 for i in out["lat"].values],
        lon=[(i.left + i.right) / 2 for i in out["lon"].values]
    )

    return out.to_dataset()


def regrid_mip_oco2(save_dir: str, gridname: str, vertical_levels: str):
    """
    Regrid MIP OCO-2 data to the specified grid and vertical levels.

    Parameters
    ----------
    save_dir : str
        Directory containing the raw OCO2MIP data (downloaded files).
    gridname : str
        Name of the target lat-lon grid (must exist in LATLON_PROTOTYPE_COORDS).
    vertical_levels : str
        Vertical level definition (e.g., 'l34' — kept for API consistency).

    Returns
    -------
    None
        Writes regridded datasets to disk in NetCDF or Zarr format.
    """
    save_dir = Path(save_dir)
    base_dir = save_dir / "OCO2MIP"
    out_dir = base_dir / "OCO2" / "OCO2_regrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    oco2_file = base_dir / "OCO2" / "OCO2_b11.2_10sec_GOOD_r2.nc4"

    # --- Load datasets ---
    print(f"Loading OCO-2 from {oco2_file}")
    ds_oco2 = xr.open_dataset(oco2_file)

    ds = reconstruct_pressure_levels(ds_oco2)
    ds = reconstruct_co2_profile_from_xco2(ds)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    # --- Filter only good-quality soundings ---
    if "xco2_quality_flag" in ds:
        ds = ds.where(ds["xco2_quality_flag"] == 0, drop=True)

    # --- Define regridding operation ---
    print(f"Regridding OCO-2 to {gridname}")
    ds_regrid = regrid_spatial(
        ds,
        variable="xco2_raw",
        gridname=gridname,
        weights=None
    )

    # --- Add metadata ---
    ds_regrid.attrs.update(
        {
            "title": f"OCO-2 regridded to {gridname}",
            "grid": gridname,
            "vertical_levels": vertical_levels,
            "source": "NOAA GML / Caltech MIP OCO-2 products",
        }
    )

    # --- Write to disk ---
    out_path = out_dir / f"OCO2_regrid_{gridname}_{vertical_levels}.nc"
    print(f"Writing regridded dataset to {out_path}")
    ds_regrid.to_netcdf(out_path)

    print("Regridding complete!")


def resample_mip_oco2(save_dir: str, gridname: str, vertical_levels: str, freq: str):
    """Resample MIP OCO-2 data to the specified frequency."""
    # Implementation for resampling data goes here
    pass


def write_mip_oco2(save_dir: str, gridname: str, vertical_levels: str, freq: str):
    """Write processed MIP OCO-2 data to disk."""
    # Implementation for writing data goes here
    pass


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
    parser.add_argument("--freq", type=str, default="1d")
    args = parser.parse_args()

    download_data(args.save_dir)

    regrid_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels
    )

    resample_mip_oco2(
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

    obspack_mip_oco2(
        args.save_dir,
        gridname=args.gridname,
        vertical_levels=args.vertical_levels,
        freq=args.freq
    )
