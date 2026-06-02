"""Tests for the real OCO-2 orbit-obs provider (P3, MIP-style OSSE)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neural_transport.inference.generation import _build_orbit_enkf_obs
from neural_transport.inference.orbit_obs import OrbitObsProvider

MIP_L20_ZARR = Path(
    "/Net/Groups/BGI/tscratch/vbenson/graph_tm/data/OCO2MIP_OCO2/train/mip_oco2_latlon5.625_l20_6h.zarr"
)

pytestmark = pytest.mark.skipif(not MIP_L20_ZARR.exists(), reason="MIP OCO-2 l20 product not staged")


@pytest.fixture(scope="module")
def provider():
    return OrbitObsProvider(str(MIP_L20_ZARR), nlat=32, nlon=64)


def test_provider_loads_and_reports_levels(provider):
    assert provider.nlev_ret == 20
    assert provider.nlat == 32 and provider.nlon == 64
    assert len(provider.times) > 0


def test_known_mip_timestamp_present(provider):
    # A timestamp inside the MIP period (2015-2020) must be present.
    t = pd.Timestamp("2016-06-01 12:00:00")
    assert provider.has(t)
    o = provider.get(t)
    assert o is not None


def test_get_returns_finite_clean_arrays(provider):
    o = provider.get(pd.Timestamp("2016-06-01 12:00:00"))
    assert o["mask"].shape == (32, 64)
    assert o["ak"].shape == (20, 32, 64)
    assert o["p_ret"].shape == (20, 32, 64)
    assert o["pw"].shape == (20, 32, 64)
    # NaN-free everywhere (dummies fill unobserved cells).
    for k in ("ak", "p_ret", "pw"):
        assert np.isfinite(o[k]).all(), f"{k} has non-finite entries"


def test_obs_fraction_realistic(provider):
    # OCO-2 covers ~2-3% of model cells per 6h step over the MIP period.
    times = pd.date_range("2016-01-01", "2016-12-31", freq="6h")
    times = [t for t in times if provider.has(t)]
    frac = provider.mean_obs_fraction(times)
    assert 0.005 < frac < 0.08, f"obs fraction {frac:.3f} out of expected range"


def test_ak_shape_increases_toward_surface(provider):
    # OCO-2 AK is small at TOA (~0.3) and ~1 near the surface (level ordering TOA->surf).
    o = provider.get(pd.Timestamp("2016-06-01 12:00:00"))
    m = o["mask"]
    assert m.sum() > 0
    ak_obs = o["ak"][:, m]  # [20, n_obs]
    mean_ak = ak_obs.mean(axis=1)
    assert mean_ak[0] < 0.6, "TOA AK should be small"
    assert mean_ak[-1] > 0.8, "near-surface AK should be ~1"
    assert mean_ak[-1] > mean_ak[0]


def test_pressure_weights_sum_to_one(provider):
    o = provider.get(pd.Timestamp("2016-06-01 12:00:00"))
    m = o["mask"]
    pw_sum = o["pw"][:, m].sum(axis=0)  # [n_obs]
    assert np.allclose(pw_sum, 1.0, atol=1e-3)


def test_pressure_levels_increase_toward_surface(provider):
    o = provider.get(pd.Timestamp("2016-06-01 12:00:00"))
    m = o["mask"]
    p = o["p_ret"][:, m]  # [20, n_obs], TOA->surf
    assert (p[0] < p[-1]).all(), "p_ret should increase TOA->surface"
    assert p[-1].max() < 1100 and p[0].min() >= 0


def test_missing_timestamp_returns_none(provider):
    assert provider.get(pd.Timestamp("1850-01-01")) is None


# ── _build_orbit_enkf_obs: synthesis math (no staged data needed) ─────────


class _MockProvider:
    """Minimal OrbitObsProvider stand-in with a fixed orbit record."""

    def __init__(self, nlat, nlon, nlev_ret=20, seed=0):
        rng = np.random.RandomState(seed)
        self.nlat, self.nlon = nlat, nlon
        N = nlat * nlon
        mask = np.zeros(N, bool)
        mask[rng.choice(N, size=max(1, N // 5), replace=False)] = True
        self.mask2d = mask.reshape(nlat, nlon)
        # AK rising TOA->surface; retrieval pressure weights sum to 1.
        ak = np.linspace(0.3, 1.0, nlev_ret)[:, None, None] * np.ones((1, nlat, nlon))
        pw = np.ones((nlev_ret, nlat, nlon)) / nlev_ret
        p_ret = np.linspace(1.0, 1010.0, nlev_ret)[:, None, None] * np.ones((1, nlat, nlon))
        self._rec = {
            "mask": self.mask2d,
            "ak": ak.astype(np.float32),
            "pw": pw.astype(np.float32),
            "p_ret": p_ret.astype(np.float32),
            "n_obs": int(self.mask2d.sum()),
        }

    def get(self, _timestamp):
        return self._rec


def _edges(n_inits, N, C):
    # Monotone decreasing model level edges, surface->top (hPa).
    pb = torch.linspace(1000.0, 100.0, C + 1)[:-1].view(1, 1, C).expand(n_inits, N, C).contiguous()
    pt = torch.linspace(1000.0, 100.0, C + 1)[1:].view(1, 1, C).expand(n_inits, N, C).contiguous()
    return pb, pt


def test_orbit_obs_constant_profile_matches_kernel_sum():
    # For a constant truth profile x=c, y = c * sum_k(h_k a_k) at observed cells,
    # because the interpolation rows sum to 1 (so sum_l g_l = sum_k h_k a_k).
    nlat, nlon, C = 4, 8, 10
    N = nlat * nlon
    prov = _MockProvider(nlat, nlon)
    c = 400.0
    gt = torch.full((1, N, C), c)
    pb, pt = _edges(1, N, C)
    mask, y, g = _build_orbit_enkf_obs(prov, [np.datetime64("2016-06-01")], gt, pb, pt, nlat, nlon, device="cpu")
    expected = c * (prov._rec["pw"] * prov._rec["ak"]).sum(axis=0).reshape(N)  # per cell
    m = mask[0]
    assert m.sum() == prov._rec["n_obs"]
    torch.testing.assert_close(y[0][m], torch.as_tensor(expected, dtype=y.dtype)[m], rtol=1e-4, atol=1e-3)
    # y is NaN where unobserved.
    assert torch.isnan(y[0][~m]).all()


def test_orbit_obs_uniform_ak_recovers_column_mean():
    # ak_mode="uniform" with retrieval pw summing to 1 → y == c for constant x.
    nlat, nlon, C = 4, 8, 10
    N = nlat * nlon
    prov = _MockProvider(nlat, nlon)
    gt = torch.full((1, N, C), 400.0)
    pb, pt = _edges(1, N, C)
    mask, y, g = _build_orbit_enkf_obs(
        prov,
        [np.datetime64("2016-06-01")],
        gt,
        pb,
        pt,
        nlat,
        nlon,
        ak_mode="uniform",
        device="cpu",
    )
    m = mask[0]
    torch.testing.assert_close(y[0][m], torch.full((int(m.sum()),), 400.0), rtol=1e-4, atol=1e-3)


def test_orbit_obs_thin_fraction_reduces_count():
    nlat, nlon, C = 8, 16, 10
    N = nlat * nlon
    prov = _MockProvider(nlat, nlon, seed=1)
    gt = torch.full((1, N, C), 400.0)
    pb, pt = _edges(1, N, C)
    rng = torch.Generator().manual_seed(0)
    full, _, _ = _build_orbit_enkf_obs(prov, [np.datetime64("2016-06-01")], gt, pb, pt, nlat, nlon, device="cpu")
    thin, _, _ = _build_orbit_enkf_obs(
        prov,
        [np.datetime64("2016-06-01")],
        gt,
        pb,
        pt,
        nlat,
        nlon,
        thin_fraction=0.5,
        rng=rng,
        device="cpu",
    )
    assert thin[0].sum() < full[0].sum()
    assert thin[0].sum() >= 1


def test_orbit_obs_kernel_g_matches_direct_synthesis():
    # y must equal (g * x_truth).sum(level) exactly at observed cells.
    nlat, nlon, C = 4, 8, 10
    N = nlat * nlon
    prov = _MockProvider(nlat, nlon, seed=3)
    torch.manual_seed(7)
    gt = 400.0 + 3.0 * torch.randn(1, N, C)
    pb, pt = _edges(1, N, C)
    mask, y, g = _build_orbit_enkf_obs(prov, [np.datetime64("2016-06-01")], gt, pb, pt, nlat, nlon, device="cpu")
    direct = (g[0] * gt[0]).sum(dim=-1)
    m = mask[0]
    torch.testing.assert_close(y[0][m], direct[m], rtol=1e-5, atol=1e-4)
