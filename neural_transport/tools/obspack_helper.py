import numpy as np
import scipy.spatial
import xarray as xr

from neural_transport.models.gnn.mesh import latlon_to_xyz


def extract_obspack_locs_from_xarray(ds, obs, grid="latlon4"):
    
    if grid.startswith("latlon"):
        obs_t = obs.sel(time=ds.time, method="nearest").dropna(
            "cell", how="all", subset=["lat", "lon", "height"]
        )
        ds_loc = ds.sel(lat=obs_t.lat, lon=obs_t.lon, method="nearest")

        height_idx = (
            (obs_t.height >= ds_loc.gph_bottom * 1000)
            & (obs_t.height <= ds_loc.gph_top * 1000)
        ).argmax(
            "level"
        )  # ((1000*ds_loc.gph - obs_t.height) ** 2).argmin("level")
        height_idx = height_idx.assign_coords(
            height=("cell", height_idx.values)
        ).compute()

        ds_cell = ds_loc.isel(level=height_idx)

        ds_out = xr.merge(
            [
                ds_cell.drop_vars(["level", "lat", "lon"]),
                obs_t.rename({k: f"obs_{k}" for k in obs_t.data_vars}),
            ]
        )

        return ds_out

    else:
        obs_t = obs.sel(time=ds.time, method="nearest").dropna(
            "cell", subset=["lat", "lon", "height"]
        )

        mesh_positions = np.stack(latlon_to_xyz(ds.clat, ds.clon), axis=-1)
        obs_positions = np.stack(latlon_to_xyz(obs_t.lat, obs_t.lon), axis=-1)
        kd_tree = scipy.spatial.cKDTree(mesh_positions)
        _, idxs = kd_tree.query(x=obs_positions)
        ds_loc = ds.isel(cell=idxs)

        height_idx = ((ds_loc.gph - obs_t.height) ** 2).argmin("level").compute()
        ds_cell = ds_loc.isel(level=height_idx.compute())

        ds_out = xr.merge(
            [
                ds_cell.drop_vars(["level", "clat", "clon"]),
                obs_t.rename({k: f"obs_{k}" for k in obs_t.data_vars}),
            ]
        )

        return ds_out
