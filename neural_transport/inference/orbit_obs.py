"""Real OCO-2 orbit-track observation provider for the MIP-style OSSE (P3).

The idealised OSSE (``create_column_mask``) samples synthetic swaths with a
uniform averaging kernel.  For a MIP-*realistic* OSSE we instead want, at each
6-hourly step, the *actual* OCO-2 sampling footprint and the *actual* 20-level
retrieval averaging kernels — sampled from a known CarbonTracker truth through
the corrected (interpolate-then-apply) forward operator from P1.

This module reads the staged, regridded MIP product
(``mip_oco2_latlon5.625_l20_6h.zarr``) and exposes, keyed by absolute
timestamp, the observed-cell mask plus the native-level (l20) averaging kernel,
pressure levels, and pressure weights on the model grid.  The values are
returned NaN-free (unobserved cells carry harmless finite dummies that the mask
excludes) so the consumer can call
:func:`neural_transport.forward_model.effective_column_kernel` grid-wide and
mask afterwards.

The XCO2 *values* are NOT read from this product — the OSSE synthesises them
from the CarbonTracker truth, so only the sampling geometry + kernels are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

# vari_3d entries we need from the regridded MIP product.
_AK = "xco2_averaging_kernel"
_PLEV = "pressure_levels"
_PW = "pressure_weight"


class OrbitObsProvider:
    """Serve real OCO-2 orbit masks + 20-level kernels keyed by timestamp.

    Parameters
    ----------
    zarr_path : str
        Path to a regridded MIP OCO-2 product (l20) with a ``variables_3d``
        DataArray indexed by a ``vari_3d`` coordinate that includes
        ``xco2_averaging_kernel``, ``pressure_levels``, ``pressure_weight``.
    nlat, nlon : int
        Model grid dimensions (must match the product's lat/lon).
    presence_eps : float
        A cell counts as observed at a step when its AK column is fully finite
        and ``max |ak| > presence_eps``.
    """

    def __init__(self, zarr_path: str, nlat: int, nlon: int, *, presence_eps: float = 1e-6) -> None:
        self.zarr_path = str(zarr_path)
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.presence_eps = float(presence_eps)

        ds = xr.open_zarr(self.zarr_path)
        if ds.sizes["lat"] != nlat or ds.sizes["lon"] != nlon:
            raise ValueError(f"grid mismatch: product is {ds.sizes['lat']}x{ds.sizes['lon']}, requested {nlat}x{nlon}")
        v3 = list(ds.vari_3d.values)
        self._ak_i = v3.index(_AK)
        self._plev_i = v3.index(_PLEV)
        self._pw_i = v3.index(_PW)
        self._v3d = ds.variables_3d  # lazy [time, vari_3d, level, lat, lon]
        self.nlev_ret = ds.sizes["level"]
        # Time index for fast exact lookup.
        self._times = pd.to_datetime(ds.time.values)
        self._time_to_idx = {np.datetime64(t): i for i, t in enumerate(ds.time.values)}
        self._cache: dict[int, dict] = {}

    # -- internals -------------------------------------------------------
    def _index_for(self, timestamp) -> int | None:
        key = np.datetime64(pd.Timestamp(timestamp))
        return self._time_to_idx.get(key)

    # -- public API ------------------------------------------------------
    def has(self, timestamp) -> bool:
        return self._index_for(timestamp) is not None

    @property
    def times(self) -> pd.DatetimeIndex:
        return self._times

    def get(self, timestamp) -> dict | None:
        """Return the orbit obs for ``timestamp`` or ``None`` if not present.

        Returns a dict with NaN-free arrays:
            ``mask``  : bool  [nlat, nlon]   — observed cells
            ``ak``    : float [nlev_ret, nlat, nlon]
            ``p_ret`` : float [nlev_ret, nlat, nlon] (hPa)
            ``pw``    : float [nlev_ret, nlat, nlon] (retrieval pressure weights)
            ``n_obs`` : int   — number of observed cells

        Unobserved cells carry finite dummies (ak=1, pw=1/nlev, p_ret=a monotone
        ramp) so downstream interpolation never sees NaNs; the mask excludes them.
        """
        idx = self._index_for(timestamp)
        if idx is None:
            return None
        if idx in self._cache:
            return self._cache[idx]

        ak = self._v3d.isel(time=idx, vari_3d=self._ak_i).values.astype(np.float32)  # [L,lat,lon]
        p_ret = self._v3d.isel(time=idx, vari_3d=self._plev_i).values.astype(np.float32)
        pw = self._v3d.isel(time=idx, vari_3d=self._pw_i).values.astype(np.float32)

        ak_finite = np.isfinite(ak)
        with np.errstate(invalid="ignore"):
            max_abs = np.max(np.where(ak_finite, np.abs(ak), 0.0), axis=0)
        mask = ak_finite.all(axis=0) & (max_abs > self.presence_eps)
        mask &= np.isfinite(p_ret).all(axis=0) & np.isfinite(pw).all(axis=0)

        L = ak.shape[0]
        # Fill unobserved (NaN) cells with harmless finite dummies.
        full = ~mask[None, :, :].repeat(L, axis=0)
        ak = np.where(full, 1.0, np.nan_to_num(ak, nan=1.0)).astype(np.float32)
        pw = np.where(full, 1.0 / L, np.nan_to_num(pw, nan=1.0 / L)).astype(np.float32)
        ramp = np.linspace(1.0, 1000.0, L, dtype=np.float32)[:, None, None]
        p_ret = np.where(full, ramp, np.nan_to_num(p_ret, nan=ramp)).astype(np.float32)

        out = {
            "mask": mask,
            "ak": ak,
            "p_ret": p_ret,
            "pw": pw,
            "n_obs": int(mask.sum()),
        }
        if len(self._cache) < 4096:
            self._cache[idx] = out
        return out

    def mean_obs_fraction(self, timestamps) -> float:
        """Mean fraction of grid cells observed over the given timestamps."""
        fr = []
        for t in timestamps:
            o = self.get(t)
            if o is not None:
                fr.append(o["n_obs"] / (self.nlat * self.nlon))
        return float(np.mean(fr)) if fr else 0.0
