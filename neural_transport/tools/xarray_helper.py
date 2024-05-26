import xarray as xr

from neural_transport.datasets.grids import *
from neural_transport.models.gnn.mesh import get_gridnc_from_grid


def tensor_to_xarray(arr, dataset="egg4", grid="latlon1", gridnc=None):
    coords = {}

    if len(arr.shape) == 5:
        B, T, n_cell, n_lev, C = arr.shape
    elif len(arr.shape) == 4:
        B, T, n_cell, n_lev = arr.shape
        C = 1
        arr = arr.unsqueeze(-1)

    coords["time"] = range(T)
    coords["batch"] = range(B)
    coords["vari"] = range(C)
    coords["height"] = HEIGHTS[dataset] if n_lev > 1 else range(1)

    if grid.startswith("latlon"):
        coords |= LATLON_PROTOTYPE_COORDS[grid]

        arr = arr.reshape(B, T, len(coords["lat"]), len(coords["lon"]), n_lev, C)

        dims = ("batch", "time", "lat", "lon", "height", "vari")
    else:
        ds = get_gridnc_from_grid(grid) if gridnc is None else gridnc

        coords["clat"] = ("cell", ds.clat.values)
        coords["clon"] = ("cell", ds.clon.values)

        dims = ("batch", "time", "cell", "height", "vari")

    ds = xr.DataArray(arr.detach().cpu().numpy(), coords=coords, dims=dims)
    if n_lev == 1:
        ds = ds.squeeze("height", drop=True)

    return ds
