"""Tests for generate_ensemble (Phase 24b ensemble-based eval).

Usage:
    pytest tests/test_generation_ensemble.py -v
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from neural_transport.inference.generation import generate_ensemble

# ── Mocks ────────────────────────────────────────────────────────────────


class _StochasticModel:
    """Mock NeuralTransport: prediction = mean(co2) + noise, so samples differ."""

    def __init__(self, nlat=4, nlon=8, nlev=3):
        self.model = SimpleNamespace(generating=False, generate_kwargs={})
        self._nlat = nlat
        self._nlon = nlon
        self._nlev = nlev

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, batch, *, mode=None):
        assert mode == "generate"
        x = batch["co2massmix"]  # [B, N, C]
        # Each sample: previous co2 mean + independent noise (torch.randn_like
        # is RNG-dependent → differs across samples within a batch).
        noise = torch.randn_like(x)
        out = x.mean(dim=(1, 2), keepdim=True) + 0.1 * noise
        return {"co2massmix": out}


class _MockDataset:
    def __init__(self, n=50, nlat=4, nlon=8, nlev=3):
        self.nlat = nlat
        self.nlon = nlon
        self.nlev = nlev
        self._n = n
        self._N = nlat * nlon
        times = pd.date_range("2019-01-01", periods=n, freq="6h")
        self.ds = xr.Dataset(coords={"time": times})
        torch.manual_seed(0)
        self._co2 = torch.randn(n, self._N, nlev)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {
            "co2massmix": self._co2[idx].clone(),
            "co2massmix_next": self._co2[min(idx + 1, self._n - 1)].clone(),
            "u": torch.zeros(self._N, self.nlev),
            "v": torch.zeros(self._N, self.nlev),
        }


class _MockLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.grid_info = SimpleNamespace(
            nlat=dataset.nlat,
            nlon=dataset.nlon,
            nlev=dataset.nlev,
            lat=np.linspace(-90, 90, dataset.nlat),
            lon=np.linspace(0, 360, dataset.nlon, endpoint=False),
            levels=np.arange(dataset.nlev),
        )

    def __len__(self):
        return len(self.dataset)

    def get_batch(self, idx, device="cpu"):
        return {k: v.unsqueeze(0) for k, v in self.dataset[idx].items()}


@pytest.fixture
def loader():
    return _MockLoader(_MockDataset(n=50))


@pytest.fixture
def model():
    return _StochasticModel()


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.quick
def test_shape_one_step(model, loader):
    ds = generate_ensemble(
        model,
        loader,
        init_indices=[0, 5, 10],
        n_samples=3,
        n_steps=1,
        device="cpu",
        verbose=False,
    )
    assert ds["co2massmix"].dims == ("init", "sample", "lead", "lat", "lon", "level")
    assert ds["co2massmix"].shape == (3, 3, 1, 4, 8, 3)


@pytest.mark.quick
def test_shape_trajectory(model, loader):
    ds = generate_ensemble(
        model,
        loader,
        init_indices=[0, 5],
        n_samples=2,
        n_steps=6,
        device="cpu",
        verbose=False,
    )
    assert ds["co2massmix"].shape == (2, 2, 6, 4, 8, 3)
    # Time coord matches init+lead+1.
    times_axis = loader.dataset.ds.time.values
    assert ds.coords["time"].values[0, 0] == times_axis[1]
    assert ds.coords["time"].values[1, 5] == times_axis[5 + 6]


@pytest.mark.quick
def test_reproducibility_same_seed(model, loader):
    kw = dict(init_indices=[3], n_samples=4, n_steps=2, device="cpu", verbose=False)
    a = generate_ensemble(model, loader, seed=42, **kw)["co2massmix"].values
    b = generate_ensemble(model, loader, seed=42, **kw)["co2massmix"].values
    np.testing.assert_allclose(a, b)


@pytest.mark.quick
def test_independent_noise_across_samples(model, loader):
    ds = generate_ensemble(
        model,
        loader,
        init_indices=[2],
        n_samples=5,
        n_steps=1,
        seed=42,
        device="cpu",
        verbose=False,
    )
    samples = ds["co2massmix"].values[0, :, 0]  # [5, lat, lon, level]
    # Pairwise differences are non-zero — samples must differ.
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.allclose(samples[i], samples[j])


@pytest.mark.quick
def test_feedback_differs_from_reinit_every_1(model, loader):
    kw = dict(init_indices=[4], n_samples=2, n_steps=4, seed=0, device="cpu", verbose=False)
    free = generate_ensemble(model, loader, reinit_every=None, **kw)["co2massmix"].values
    reinit = generate_ensemble(model, loader, reinit_every=1, **kw)["co2massmix"].values
    # lead=0 should match (no reinit yet); later leads differ.
    np.testing.assert_allclose(free[..., 0, :, :, :], reinit[..., 0, :, :, :])
    assert not np.allclose(free[..., 3, :, :, :], reinit[..., 3, :, :, :])
