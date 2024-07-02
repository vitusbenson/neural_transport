import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.contrib.concurrent import process_map


def download_obspack(data_dir):
    """
    Download Data from https://gml.noaa.gov/ccgg/obspack/data.php
    """

    data_dir = Path(data_dir)

    fp = "http://gml.noaa.gov/ccgg/obspack/tmp/obspack_nbVFeb/obspack_co2_1_GLOBALVIEWplus_v9.1_2023-12-08.nc.tar.gz"

    outpath = data_dir / "Obspack" / (fp.split("/")[-1])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(fp, str(outpath))

    shutil.unpack_archive(outpath, data_dir / "Obspack")


def open_one_obspack(obspack_path):
    try:
        obspack_obs = xr.open_dataset(obspack_path)
        obspack_obs = (
            xr.Dataset(
                {
                    "lon": ("time", obspack_obs.longitude.values),
                    "lat": ("time", obspack_obs.latitude.values),
                    "co2molemix": ("time", obspack_obs.value.values * 1e6),
                    "height": ("time", obspack_obs.altitude.values),
                },
                coords={"time": ("time", obspack_obs.time.values)},
            )
            .drop_duplicates("time")
            .sortby("time")
        )
        obspack_obs = obspack_obs.resample(
            time="6h", origin="start_day", offset="3h", closed="left", label="left"
        ).mean()
        offset = pd.tseries.frequencies.to_offset("6h") / 2
        obspack_obs["time"] = obspack_obs.get_index("time") + offset

        return {obspack_path.stem: obspack_obs.to_array("vari")}
    except:
        print(f"Error with {obspack_path}")
        return None


def prepare_obspack_for_carboscope(data_dir):
    data_dir = Path(data_dir)

    obspack_dir = data_dir / "Obspack"

    obspack_paths = sorted(
        list((data_dir / "Obspack").glob("obspack_co2_*/data/nc/*.nc"))
    )

    all_obs = process_map(open_one_obspack, obspack_paths, max_workers=32, chunksize=1)
    all_obs = sorted(
        [a for a in all_obs if a is not None], key=lambda x: list(x.keys())[0]
    )

    obs = xr.merge(all_obs, join="outer")
    obs = obs.to_array("cell").to_dataset("vari")
    obs["filename"] = ("cell", obs.cell.values)
    obs["cell"] = ("cell", np.arange(len(obs.cell)))

    obs = obs.chunk({"time": 2000, "cell": -1})

    with ProgressBar():
        obs.to_zarr(obspack_dir / "obspack.zarr")

    attrs = []
    for obspack_path in obspack_paths:
        id = obspack_path.stem
        molecule, station, type, level, quality = id.split("_")
        attrs.append(
            dict(
                id=id,
                molecule=molecule,
                station=station,
                type=type,
                level=level,
                quality=quality,
            )
            | xr.open_dataset(obspack_path).attrs
        )

    df = pd.DataFrame.from_records(attrs)
    df.to_csv(obspack_dir / "obspack_metadata.csv", index=False)
