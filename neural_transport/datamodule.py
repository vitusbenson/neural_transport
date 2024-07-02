import shutil
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
import zarr
from numcodecs import blosc
from torch.utils.data import Dataset

from neural_transport.datasets.grids import *
from neural_transport.datasets.vars import *
from neural_transport.models.gnn.mesh import get_gridnc_from_grid
from neural_transport.tools.conversion import *
from neural_transport.tools.obspack_helper import extract_obspack_locs_from_xarray
from neural_transport.tools.xarray_helper import tensor_to_xarray

blosc.use_threads = False
dask.config.set(scheduler="synchronous")


class CarbonDataset(Dataset):
    def __init__(
        self,
        data_path="data/",
        dataset="egg4",
        grid="latlon1",
        vertical_levels="l10",
        freq="6h",
        n_timesteps=1,
        rename_dims={},
        target_vars=[],
        forcing_vars=[],
        time_interval=None,
        return_tuple=False,
        subsample_time=1,
        load_obspack=False,
        compute=False,
        new_zarr=False,
        use_fastaccess=False,
    ):
        super().__init__()

        self.n_timesteps = n_timesteps
        self.return_tuple = return_tuple

        self.vars = target_vars + forcing_vars
        self.vars_curr = target_vars + forcing_vars
        self.vars_next = target_vars

        self.data_path = Path(data_path)
        self.dataset = dataset
        self.grid = grid
        self.vertical_levels = vertical_levels
        self.freq = freq

        self.new_zarr = new_zarr
        self.use_fastaccess = use_fastaccess
        self.initial_time_idx = 0

        if new_zarr:
            ds = xr.open_zarr(
                self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}.zarr"
            )

            self.zarr = (
                zarr.load(
                    self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}.zarr"
                )
                if compute
                else zarr.open(
                    self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}.zarr",
                    mode="r",
                )
            )

            if time_interval:
                self.initial_time_idx = (
                    ds.indexes["time"].get_loc(time_interval[0]).start
                )
                ds = ds.sel(time=slice(*time_interval))

            ds_3d = ds["variables_3d"]
            ds_2d = ds["variables_2d"].compute()

            self.ds_3d_coords = ds_3d

            if "step" in ds.coords:

                if grid.startswith("latlon"):
                    ds_3d = ds_3d.transpose(
                        "time", "step", "lat", "lon", "level", "vari_3d"
                    ).stack({"cell": ["lat", "lon"]})
                    ds_2d = ds_2d.transpose(
                        "time", "step", "lat", "lon", "vari_2d"
                    ).stack({"cell": ["lat", "lon"]})
                    self.grid_ds = None
                else:
                    ds_3d = ds_3d.transpose("time", "step", "cell", "level", "vari_3d")
                    ds_2d = ds_2d.transpose("time", "step", "cell", "vari_2d")
                    self.grid_ds = get_gridnc_from_grid(grid)

            else:
                if grid.startswith("latlon"):
                    ds_3d = ds_3d.transpose(
                        "time", "lat", "lon", "level", "vari_3d"
                    ).stack({"cell": ["lat", "lon"]})
                    ds_2d = ds_2d.transpose("time", "lat", "lon", "vari_2d").stack(
                        {"cell": ["lat", "lon"]}
                    )
                    self.grid_ds = None
                else:
                    ds_3d = ds_3d.transpose("time", "cell", "level", "vari_3d")
                    ds_2d = ds_2d.transpose("time", "cell", "vari_2d")
                    self.grid_ds = get_gridnc_from_grid(grid)

            if compute:
                ds_2d = ds_2d.compute()

            ds = xr.merge(
                [
                    ds_3d.to_dataset("vari_3d"),
                    ds_2d.to_dataset("vari_2d"),
                ]
            )
            self.ds = ds
            self.ds_3d = ds_3d
            self.ds_2d = ds_2d

        else:
            ds = xr.open_zarr(self.data_path / f"{dataset}_{grid}.zarr")
            ds = ds.rename(rename_dims)
            if subsample_time > 1:
                ds = ds.isel(time=slice(None, None, subsample_time))

            if grid.startswith("latlon"):
                self.ds = ds.transpose("time", "lat", "lon", "level").stack(
                    {"cell": ["lat", "lon"]}
                )
                self.grid_ds = None
            else:
                self.ds = ds.transpose("time", "cell", "level")
                self.grid_ds = get_gridnc_from_grid(grid)

            if time_interval:
                self.ds = self.ds.sel(time=slice(*time_interval))

            if compute:
                self.ds = self.ds.compute()

        # with xr.open_dataset(
        #     self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}_stats.nc",
        #     # engine="h5netcdf",
        # ) as stats_ds:
        #     self.stats_ds = stats_ds.copy(deep=True).rename(rename_dims)

        # self.stats_ds = xr.open_dataset(
        #     self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}_stats.nc",
        #     engine="h5netcdf",
        # ).rename(rename_dims)

        self.stats_ds = xr.open_zarr(
            self.data_path / f"{dataset}_{grid}_{vertical_levels}_{freq}_stats.zarr",
        ).rename(rename_dims)

        if load_obspack:

            self.obspack_ds = (
                xr.open_zarr(self.data_path.parent.parent / "Obspack" / "obspack.zarr")
                .sel(time=ds.time, method="nearest")
                .compute()
            )

            self.obspack_ds["lat"] = self.obspack_ds.lat.interpolate_na(
                dim="time", method="nearest", fill_value="extrapolate"
            )
            self.obspack_ds["lon"] = self.obspack_ds.lon.interpolate_na(
                dim="time", method="nearest", fill_value="extrapolate"
            )
            self.obspack_ds["height"] = self.obspack_ds.height.interpolate_na(
                dim="time", method="nearest", fill_value="extrapolate"
            )
            self.obspack_metadata = pd.read_csv(
                self.data_path.parent.parent / "Obspack" / "obspack_metadata.csv"
            )

    def __len__(self):
        if "step" in self.ds.coords:
            return len(self.ds.time) * (len(self.ds.step) // (self.n_timesteps + 1))
        else:
            return len(self.ds.time) // self.n_timesteps - 1

    def __getitem__(self, t: int):

        if "step" in self.ds.coords:
            n_samples_per_startdate = len(self.ds.step) // (self.n_timesteps + 1)
            startdate_idx = t // n_samples_per_startdate
            step_idx = t % n_samples_per_startdate
            if step_idx + 1 >= len(self.ds.step):
                print("oh", step_idx, t, n_samples_per_startdate, len(self.ds.step))
            startdate_slice = slice(startdate_idx, startdate_idx + 1)
            step_slice = slice(
                step_idx * self.n_timesteps, (step_idx + 1) * self.n_timesteps + 1
            )
        else:
            timeslice_zarr = slice(
                self.initial_time_idx + t * self.n_timesteps,
                self.initial_time_idx + (t + 1) * self.n_timesteps + 1,
            )
            timeslice = slice(
                t * self.n_timesteps,
                (t + 1) * self.n_timesteps + 1,
            )
        # timeslice_next = slice((t + 1) * self.n_timesteps, (t + 2) * self.n_timesteps) # POSSIBLE BUG ???? For 4 timesteps, this gets Curr = (0, 4) and Next = (4, 8). But should get Curr = (0, 4) and Next = (1, 5) ???

        if self.new_zarr:
            if "step" in self.ds.coords:
                if self.grid.startswith("latlon"):
                    ds_3d = (
                        xr.DataArray(
                            self.zarr["variables_3d"][
                                self.initial_time_idx + startdate_idx, step_slice
                            ],
                            coords=self.ds_3d_coords.isel(
                                time=startdate_idx, step=step_slice
                            ).coords,
                            dims=("step", "vari_3d", "level", "lat", "lon"),
                        )
                        .transpose("step", "lat", "lon", "level", "vari_3d")
                        .stack({"cell": ["lat", "lon"]})
                        .to_dataset("vari_3d")
                    )

                else:
                    ds_3d = (
                        xr.DataArray(
                            self.zarr["variables_3d"][
                                self.initial_time_idx + startdate_idx, step_slice
                            ],
                            coords=self.ds_3d_coords.isel(
                                time=startdate_idx, step=step_slice
                            ).coords,
                            dims=("step", "vari_3d", "level", "cell"),
                        )
                        .transpose("step", "cell", "level", "vari_3d")
                        .to_dataset("vari_3d")
                    )

                ds_all = (
                    xr.merge(
                        [
                            ds_3d,
                            self.ds_2d.isel(
                                time=startdate_idx, step=step_slice
                            ).to_dataset("vari_2d"),
                        ]
                    )
                    .drop_vars("time")
                    .rename({"step": "time"})
                )
            else:

                if self.grid.startswith("latlon"):
                    ds_3d = (
                        xr.DataArray(
                            self.zarr["variables_3d"][timeslice_zarr],
                            coords=self.ds_3d_coords.isel(time=timeslice).coords,
                            dims=("time", "vari_3d", "level", "lat", "lon"),
                        )
                        .transpose("time", "lat", "lon", "level", "vari_3d")
                        .stack({"cell": ["lat", "lon"]})
                        .to_dataset("vari_3d")
                    )

                else:
                    ds_3d = (
                        xr.DataArray(
                            self.zarr["variables_3d"][timeslice_zarr],
                            coords=self.ds_3d_coords.isel(time=timeslice).coords,
                            dims=("time", "vari_3d", "level", "cell"),
                        )
                        .transpose("time", "cell", "level", "vari_3d")
                        .to_dataset("vari_3d")
                    )

                ds_all = xr.merge(
                    [
                        ds_3d,
                        self.ds_2d.isel(time=timeslice).to_dataset("vari_2d"),
                    ]
                )
        else:
            ds_all = self.ds.isel(time=timeslice).compute()

        ds = ds_all.isel(time=slice(0, self.n_timesteps))
        ds_next = ds_all.isel(time=slice(1, self.n_timesteps + 1))

        data = {
            k: torch.from_numpy(self.expand_dims(ds[k]).values.astype("float32"))
            for k in self.vars_curr
        }

        data |= {
            f"{k}_next": torch.from_numpy(
                self.expand_dims(ds_next[k]).values.astype("float32")
            )
            for k in self.vars_next
        }

        for molecule in ["ch4", "co2"]:
            density_var = f"{molecule}density"
            massmix_var = f"{molecule}massmix"
            if (
                density_var in self.vars_curr
                and massmix_var not in data
                and "airdensity" in ds.data_vars
            ):
                massmix = density_to_massmix(
                    ds[density_var],
                    ds["airdensity"],
                    ppm=True,
                    eps=1e-12,
                )
                data[massmix_var] = torch.from_numpy(
                    self.expand_dims(massmix).values.astype("float32")
                )
            if (
                density_var in self.vars_next
                and f"{massmix_var}_next" not in data
                and "airdensity" in ds.data_vars
            ):
                massmix_next = density_to_massmix(
                    ds_next[density_var],
                    ds_next["airdensity"],
                    ppm=True,
                    eps=1e-12,
                )
                data[f"{massmix_var}_next"] = torch.from_numpy(
                    self.expand_dims(massmix_next).values.astype("float32")
                )

            if (
                massmix_var in self.vars_curr
                and density_var not in data
                and "airdensity" in ds.data_vars
            ):
                density = massmix_to_density(
                    ds[massmix_var],
                    ds["airdensity"],
                    ppm=False,
                    eps=1e-12,
                )
                data[density_var] = torch.from_numpy(
                    self.expand_dims(density).values.astype("float32")
                )
            if (
                massmix_var in self.vars_next
                and f"{density_var}_next" not in data
                and "airdensity" in ds.data_vars
            ):
                density_next = massmix_to_density(
                    ds_next[massmix_var],
                    ds_next["airdensity"],
                    ppm=False,
                    eps=1e-12,
                )
                data[f"{density_var}_next"] = torch.from_numpy(
                    self.expand_dims(density_next).values.astype("float32")
                )

        data |= {
            f"{k}_offset": torch.from_numpy(
                self.expand_dims(self.stats_ds[k].sel(stats="mean")).values.astype(
                    "float32"
                )
            )
            for k in self.stats_ds.data_vars.keys()
        }
        data |= {
            f"{k}_scale": torch.from_numpy(
                self.expand_dims(self.stats_ds[k].sel(stats="std")).values.astype(
                    "float32"
                )
            )
            for k in self.stats_ds.data_vars.keys()
        }

        data = dict(sorted(data.items()))

        if self.return_tuple:
            return [v for k, v in data.items()]
        else:
            return data

    @staticmethod
    def expand_dims(arr):
        return arr.expand_dims(
            [d for d in ["time", "cell", "level"] if d not in arr.dims]
        ).transpose("time", "cell", "level")

    def readout_stations(self, ds, grid=None):
        grid = (
            self.grid
            if grid is None
            else (DEFAULT_GRIDS[self.dataset] if grid == "default" else grid)
        )
        return extract_obspack_locs_from_xarray(ds, self.obspack_ds, grid=grid)

    def create_prototype_zarr(
        self,
        zarrpath,
        target_vars_3d=None,
        target_vars_2d=None,
        grid=None,
        vertical_levels=None,
        overwrite=True,
    ):
        grid = (
            self.grid
            if grid is None
            else (DEFAULT_GRIDS[self.dataset] if grid == "default" else grid)
        )
        vertical_levels = (
            self.vertical_levels if vertical_levels is None else vertical_levels
        )

        if grid.startswith("latlon"):
            coords = {} | LATLON_PROTOTYPE_COORDS[grid]
            coords["level"] = VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"]
            coords["time"] = self.ds.time
            if "step" in self.ds.coords:
                coords["step"] = self.ds.step
                dims = ("time", "step", "lat", "lon", "level")
                chunking = {"time": 1, "step": 1, "lat": -1, "lon": -1, "level": -1}
            else:
                dims = ("time", "lat", "lon", "level")
                chunking = {"time": 10, "lat": -1, "lon": -1, "level": -1}
            nparr = np.NaN

        else:
            coords = dict(
                clon=self.ds.clon,
                clat=self.ds.clat,
                level=VERTICAL_LAYERS_PROTOTYPE_COORDS[vertical_levels]["level"],
                time=self.ds.time,
            )
            if "step" in self.ds.coords:
                coords["step"] = self.ds.step
                dims = ("time", "step", "cell", "level")
                chunking = {"time": 1, "step": 1, "cell": -1, "level": -1}
            else:
                dims = ("time", "cell", "level")
                chunking = {"time": 10, "cell": -1, "level": -1}

            nparr = np.full(
                (len(coords["time"]), len(coords["clon"]), len(coords["level"])),
                np.NaN,
            )

        arr = (
            xr.DataArray(
                nparr,
                coords=coords,
                dims=dims,
            )
            .astype(np.float32)
            .drop_encoding()
            .chunk(chunking)
        )

        target_vars = (
            self.vars_next
            # + [
            #     k.replace("density", "massmix")
            #     for k in self.vars_next
            #     if ("density" in k) and (k != "airdensity")
            # ]
            if target_vars_3d is None
            else target_vars_3d + target_vars_2d
        )

        prototype_zarr = xr.Dataset(
            {
                v: arr if v in target_vars_3d else arr.isel(level=0, drop=True)
                for v in target_vars
            },
            coords=coords,
        )

        if overwrite and zarrpath.exists():
            shutil.rmtree(zarrpath)
        prototype_zarr.to_zarr(zarrpath, compute=False)

        return prototype_zarr

    def tensor_to_xarray(self, tensor):
        return tensor_to_xarray(
            tensor, grid=self.grid, dataset=self.dataset, gridnc=self.grid_ds, vertical_levels=self.vertical_levels
        ).squeeze(drop=True)


class CarbonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        dataset,
        grid,
        vertical_levels,
        freq,
        n_timesteps,
        batch_size_train,
        batch_size_pred,
        target_vars=[],
        forcing_vars=[],
        val_rollout_n_timesteps=None,
        num_workers=16,
        time_interval=None,
        compute=False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.dataset = dataset
        self.grid = grid
        self.vertical_levels = vertical_levels
        self.freq = freq
        self.n_timesteps = n_timesteps
        self.val_rollout_n_timesteps = val_rollout_n_timesteps
        self.batch_size_train = batch_size_train
        self.batch_size_pred = batch_size_pred
        self.forcing_vars = forcing_vars
        self.target_vars = target_vars
        self.num_workers = num_workers
        self.time_interval = time_interval
        self.compute = compute

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = CarbonDataset(
                data_path=self.data_path / "train",
                dataset=self.dataset,
                grid=self.grid,
                vertical_levels=self.vertical_levels,
                freq=self.freq,
                n_timesteps=self.n_timesteps,
                new_zarr=True,
                forcing_vars=self.forcing_vars,
                target_vars=self.target_vars,
                use_fastaccess=False,
                time_interval=self.time_interval,
                compute=self.compute,
            )
            self.val_dataset = CarbonDataset(
                data_path=self.data_path / "val",
                dataset=self.dataset,
                grid=self.grid,
                vertical_levels=self.vertical_levels,
                freq=self.freq,
                n_timesteps=self.n_timesteps,
                new_zarr=True,
                forcing_vars=self.forcing_vars,
                target_vars=self.target_vars,
                use_fastaccess=False,
                compute=self.compute,
            )
            if self.val_rollout_n_timesteps is not None:
                self.val_rollout_dataset = CarbonDataset(
                    data_path=self.data_path / "val",
                    dataset=self.dataset,
                    grid=self.grid,
                    vertical_levels=self.vertical_levels,
                    freq=self.freq,
                    n_timesteps=self.val_rollout_n_timesteps,
                    new_zarr=True,
                    forcing_vars=self.forcing_vars,
                    target_vars=self.target_vars,
                    use_fastaccess=False,
                    compute=self.compute,
                )
        elif stage == "test":
            self.test_dataset = CarbonDataset(
                data_path=self.data_path / "test",
                dataset=self.dataset,
                grid=self.grid,
                vertical_levels=self.vertical_levels,
                freq=self.freq,
                n_timesteps=self.n_timesteps,
                new_zarr=True,
                forcing_vars=self.forcing_vars,
                target_vars=self.target_vars,
                compute=self.compute,
            )

    def get_dataloader(self, split="train"):
        dataset = getattr(self, f"{split}_dataset")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size={
                "train": self.batch_size_train,
                "val": self.batch_size_pred,
                "test": self.batch_size_pred,
                "val_rollout": 1,
            }[split],
            num_workers=self.num_workers,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            pin_memory=True,
            # persistent_workers=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(split="train")

    def val_dataloader(self):
        if self.val_rollout_n_timesteps:
            return [
                self.get_dataloader(split="val"),
                self.get_dataloader(split="val_rollout"),
            ]
        else:
            return self.get_dataloader(split="val")

    def test_dataloader(self):
        return self.get_dataloader(split="test")
