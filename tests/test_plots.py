"""Tests for the plotting framework (Phase 12)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Import field_plots to ensure registration
import neural_transport.plots.field_plots  # noqa: E402, F401
from neural_transport.configs import PlotConfig  # noqa: E402
from neural_transport.evaluation.suite import EvalResult  # noqa: E402
from neural_transport.plots.base import (  # noqa: E402
    PLOT_REGISTRY,
    PlotContext,
    register_plot,
    run_plots,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_result(nlat=8, nlon=16, nlev=4, include_levels=True):
    """Create an EvalResult with synthetic 3D fields in metadata."""
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    pred = np.random.default_rng(42).normal(400, 5, (nlat, nlon, nlev))
    gt = np.random.default_rng(0).normal(400, 5, (nlat, nlon, nlev))
    meta = {"pred": pred, "gt": gt, "lat": lat, "lon": lon}
    if include_levels:
        meta["level_values"] = [1013, 843, 441, 73][:nlev]
    return EvalResult(pointwise={"rmse": 1.0}, metadata=meta)


def _make_2d_result(nlat=8, nlon=16):
    """Create an EvalResult with 2D fields (no level dimension)."""
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    pred = np.random.default_rng(42).normal(400, 5, (nlat, nlon))
    gt = np.random.default_rng(0).normal(400, 5, (nlat, nlon))
    return EvalResult(
        pointwise={"rmse": 1.0},
        metadata={"pred": pred, "gt": gt, "lat": lat, "lon": lon},
    )


# ── TestPlotContext ────────────────────────────────────────────────────────


class TestPlotContext:
    def test_savefig_creates_files(self, tmp_path):
        cfg = PlotConfig(imgformats=["png", "pdf"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ctx.savefig(fig, "test_save")
        assert (tmp_path / "test_save.png").exists()
        assert (tmp_path / "test_save.pdf").exists()

    def test_figure_context_manager_auto_saves(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        with ctx.figure("auto_test") as fig:
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1])
        assert (tmp_path / "auto_test.png").exists()

    def test_subplot_grid_respects_figsize_scale(self, tmp_path):
        cfg = PlotConfig(save_dir=str(tmp_path), figsize_scale=2.0)
        ctx = PlotContext(cfg)
        fig, axes = ctx.subplot_grid(2, 3)
        w, h = fig.get_size_inches()
        # base: 4*3*2=24 wide, 3.5*2*2=14 tall
        assert abs(w - 24.0) < 0.1
        assert abs(h - 14.0) < 0.1
        plt.close(fig)

    def test_save_dir_created(self, tmp_path):
        save_dir = tmp_path / "nested" / "plots"
        cfg = PlotConfig(save_dir=str(save_dir))
        PlotContext(cfg)
        assert save_dir.is_dir()


# ── TestRegistry ───────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_plot_adds_to_registry(self):
        name = "_test_dummy_plot_reg"
        try:

            @register_plot(name=name, categories=["always"], description="test")
            def dummy(result, ctx):
                pass

            assert name in PLOT_REGISTRY
            assert PLOT_REGISTRY[name]["categories"] == ["always"]
            assert PLOT_REGISTRY[name]["description"] == "test"
            assert PLOT_REGISTRY[name]["func"] is dummy
        finally:
            PLOT_REGISTRY.pop(name, None)

    def test_always_on_plots_registered(self):
        for plot_name in ["field_maps", "xco2_maps", "lat_height"]:
            assert plot_name in PLOT_REGISTRY, f"{plot_name} not registered"
            assert "always" in PLOT_REGISTRY[plot_name]["categories"]


# ── TestRunPlots ───────────────────────────────────────────────────────────


class TestRunPlots:
    def test_run_always_category(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_synthetic_result()
        called = run_plots(result, ctx, categories=["always"])
        assert set(called) == {"field_maps", "xco2_maps", "lat_height"}

    def test_auto_infer_always_only(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(pointwise={"rmse": 1.0}, metadata={})
        called = run_plots(result, ctx, categories=None)
        # "always" plots called, but they gracefully no-op (no pred/gt)
        assert "field_maps" in called

    def test_auto_infer_adds_ensemble(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(
            pointwise={"rmse": 1.0},
            ensemble={"crps": 0.5},
        )
        called = run_plots(result, ctx, categories=None)
        # Should include always + ensemble category names (though most aren't registered yet)
        assert "field_maps" in called

    def test_auto_infer_adds_distributional(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(
            pointwise={"rmse": 1.0},
            distributional={"energy_distance": 0.1},
        )
        called = run_plots(result, ctx, categories=None)
        assert "field_maps" in called

    def test_empty_category_calls_nothing(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_synthetic_result()
        called = run_plots(result, ctx, categories=["nonexistent_category"])
        assert called == []


# ── TestFieldPlots ─────────────────────────────────────────────────────────


class TestFieldPlots:
    def test_field_maps_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_synthetic_result()
        neural_transport.plots.field_plots.plot_field_maps(result, ctx)
        assert (tmp_path / "field_maps.png").exists()

    def test_xco2_maps_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_synthetic_result()
        neural_transport.plots.field_plots.plot_xco2_maps(result, ctx)
        assert (tmp_path / "xco2_maps.png").exists()

    def test_lat_height_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_synthetic_result()
        neural_transport.plots.field_plots.plot_lat_height(result, ctx)
        assert (tmp_path / "lat_height.png").exists()

    def test_graceful_skip_no_metadata(self, tmp_path):
        """Functions should no-op when metadata lacks pred/gt."""
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(pointwise={"rmse": 1.0})
        # Should not raise
        neural_transport.plots.field_plots.plot_field_maps(result, ctx)
        neural_transport.plots.field_plots.plot_xco2_maps(result, ctx)
        neural_transport.plots.field_plots.plot_lat_height(result, ctx)
        # No output files created
        assert not (tmp_path / "field_maps.png").exists()

    def test_2d_fields(self, tmp_path):
        """Works with 2D fields (no level dimension)."""
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_2d_result()
        neural_transport.plots.field_plots.plot_field_maps(result, ctx)
        assert (tmp_path / "field_maps.png").exists()
        # xco2 should also work with 2D
        neural_transport.plots.field_plots.plot_xco2_maps(result, ctx)
        assert (tmp_path / "xco2_maps.png").exists()

    def test_lat_height_skips_2d(self, tmp_path):
        """lat_height requires 3D fields, should skip for 2D."""
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_2d_result()
        neural_transport.plots.field_plots.plot_lat_height(result, ctx)
        assert not (tmp_path / "lat_height.png").exists()
