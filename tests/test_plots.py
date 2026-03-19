"""Tests for the plotting framework (Phases 12-13)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import neural_transport.plots.animation  # noqa: E402, F401
import neural_transport.plots.conditioning_diagnostics  # noqa: E402, F401
import neural_transport.plots.distributional_plots  # noqa: E402, F401
import neural_transport.plots.ensemble_plots  # noqa: E402, F401

# Import all plot modules to ensure registration
import neural_transport.plots.field_plots  # noqa: E402, F401
import neural_transport.plots.transport_plots  # noqa: E402, F401
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


# ---------------------------------------------------------------------------
# Phase 13 helpers
# ---------------------------------------------------------------------------


def _make_ensemble_result(nlat=8, nlon=16):
    """EvalResult with rank_histogram, calibration, spread_map."""
    rng = np.random.default_rng(42)
    n_bins = 11
    rh = rng.dirichlet(np.ones(n_bins))
    nominal = np.linspace(0.05, 0.95, 10)
    observed = nominal + rng.normal(0, 0.05, 10)
    observed = np.clip(observed, 0, 1)
    cal = {
        "nominal": nominal,
        "observed": observed,
        "calibration_error": float(np.mean(np.abs(nominal - observed))),
    }
    spread_map = rng.uniform(0.1, 2.0, (nlat, nlon))
    return EvalResult(
        pointwise={"rmse": 1.0},
        ensemble={"crps_mean": 0.5, "spread_skill_ratio": 1.1},
        maps={"spread_map": spread_map},
        diagnostics={"rank_histogram": rh, "calibration": cal},
    )


def _make_distributional_result(nlat=8, nlon=16, nlev=4):
    """EvalResult with gt_fields, gen_fields in metadata."""
    rng = np.random.default_rng(42)
    gt_fields = rng.normal(400, 5, (10, nlat, nlon, nlev))
    gen_fields = rng.normal(400, 5, (10, nlat, nlon, nlev))
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return EvalResult(
        pointwise={},
        distributional={"energy_distance": 0.1, "mmd_rbf": 0.05},
        metadata={
            "gt_fields": gt_fields,
            "gen_fields": gen_fields,
            "lat": lat,
            "lon": lon,
            "level_values": [1013, 843, 441, 73][:nlev],
        },
    )


def _make_conditioning_result(nlat=8, nlon=16, nlev=4):
    """EvalResult with pred, gt, mask_2d, and error maps."""
    rng = np.random.default_rng(42)
    pred = rng.normal(400, 5, (nlat, nlon, nlev))
    gt = rng.normal(400, 5, (nlat, nlon, nlev))
    mask_2d = rng.random((nlat, nlon)) > 0.5
    bias_map = pred.mean(axis=-1) - gt.mean(axis=-1)
    rmse_map = np.sqrt(((pred - gt) ** 2).mean(axis=-1))
    spread_map = rng.uniform(0.1, 2.0, (nlat, nlon))
    return EvalResult(
        pointwise={"rmse": 1.0},
        maps={"bias_map": bias_map, "rmse_map": rmse_map, "spread_map": spread_map},
        metadata={"pred": pred, "gt": gt, "mask_2d": mask_2d},
    )


# ── TestEnsemblePlots ──────────────────────────────────────────────────────


class TestEnsemblePlots:
    def test_ensemble_plots_registered(self):
        for name in ["rank_histogram", "calibration", "spread_maps"]:
            assert name in PLOT_REGISTRY, f"{name} not registered"
            assert "ensemble" in PLOT_REGISTRY[name]["categories"]

    def test_rank_histogram_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_ensemble_result()
        neural_transport.plots.ensemble_plots.plot_rank_histogram(result, ctx)
        assert (tmp_path / "rank_histogram.png").exists()

    def test_calibration_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_ensemble_result()
        neural_transport.plots.ensemble_plots.plot_calibration(result, ctx)
        assert (tmp_path / "calibration.png").exists()

    def test_spread_maps_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_ensemble_result()
        neural_transport.plots.ensemble_plots.plot_spread_maps(result, ctx)
        assert (tmp_path / "spread_maps.png").exists()

    def test_graceful_skip_missing_data(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(pointwise={"rmse": 1.0})
        # Should not raise
        neural_transport.plots.ensemble_plots.plot_rank_histogram(result, ctx)
        neural_transport.plots.ensemble_plots.plot_calibration(result, ctx)
        neural_transport.plots.ensemble_plots.plot_spread_maps(result, ctx)
        assert not (tmp_path / "rank_histogram.png").exists()
        assert not (tmp_path / "calibration.png").exists()
        assert not (tmp_path / "spread_maps.png").exists()


# ── TestDistributionalPlotsRegistered ──────────────────────────────────────


class TestDistributionalPlotsRegistered:
    def test_distributional_plots_registered(self):
        for name in [
            "marginals",
            "power_spectrum",
            "qq_plot",
            "sample_grid",
            "spatial_patterns",
            "lat_height_comparison",
            "distributional_summary",
        ]:
            assert name in PLOT_REGISTRY, f"{name} not registered"
            assert "distributional" in PLOT_REGISTRY[name]["categories"]

    def test_marginals_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_distributional_result()
        neural_transport.plots.distributional_plots.plot_marginals_registered(result, ctx)
        assert any(f.name.startswith("marginal_distributions") for f in tmp_path.iterdir())

    def test_qq_plot_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_distributional_result()
        neural_transport.plots.distributional_plots.plot_qq_registered(result, ctx)
        assert any(f.name.startswith("qq_plot") for f in tmp_path.iterdir())

    def test_distributional_summary_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_distributional_result()
        neural_transport.plots.distributional_plots.plot_distributional_summary_registered(result, ctx)
        assert any(f.name.startswith("distributional_metrics_summary") for f in tmp_path.iterdir())

    def test_graceful_skip_missing_metadata(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(pointwise={"rmse": 1.0})
        # Should not raise
        neural_transport.plots.distributional_plots.plot_marginals_registered(result, ctx)
        neural_transport.plots.distributional_plots.plot_qq_registered(result, ctx)
        neural_transport.plots.distributional_plots.plot_distributional_summary_registered(result, ctx)
        assert len(list(tmp_path.iterdir())) == 0


# ── TestConditioningPlotsRegistered ────────────────────────────────────────


class TestConditioningPlotsRegistered:
    def test_conditioning_plots_registered(self):
        for name in ["conditioning_comparison", "error_maps", "zonal_mean"]:
            assert name in PLOT_REGISTRY, f"{name} not registered"
            assert "conditioning" in PLOT_REGISTRY[name]["categories"]

    def test_conditioning_comparison_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_conditioning_result()
        neural_transport.plots.conditioning_diagnostics.plot_conditioning_comparison_single(result, ctx)
        assert (tmp_path / "conditioning_comparison.png").exists()

    def test_error_maps_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_conditioning_result()
        neural_transport.plots.conditioning_diagnostics.plot_error_maps_registered(result, ctx)
        assert (tmp_path / "error_maps.png").exists()

    def test_zonal_mean_produces_output(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_conditioning_result()
        neural_transport.plots.conditioning_diagnostics.plot_zonal_mean_single(result, ctx)
        assert (tmp_path / "zonal_mean.png").exists()

    def test_graceful_skip_missing_data(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(pointwise={"rmse": 1.0})
        neural_transport.plots.conditioning_diagnostics.plot_conditioning_comparison_single(result, ctx)
        neural_transport.plots.conditioning_diagnostics.plot_error_maps_registered(result, ctx)
        neural_transport.plots.conditioning_diagnostics.plot_zonal_mean_single(result, ctx)
        assert not (tmp_path / "conditioning_comparison.png").exists()
        assert not (tmp_path / "error_maps.png").exists()
        assert not (tmp_path / "zonal_mean.png").exists()


# ── TestSmartCategoryInference ─────────────────────────────────────────────


class TestSmartCategoryInference:
    def test_ensemble_result_infers_ensemble(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_ensemble_result()
        called = run_plots(result, ctx, categories=None)
        # Ensemble plots should be dispatched
        assert "rank_histogram" in called
        assert "calibration" in called
        assert "spread_maps" in called

    def test_conditioning_result_infers_conditioning(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_conditioning_result()
        called = run_plots(result, ctx, categories=None)
        assert "conditioning_comparison" in called
        assert "error_maps" in called
        assert "zonal_mean" in called

    def test_transport_result_infers_transport(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = EvalResult(
            pointwise={"rmse": 1.0},
            metadata={"experiment_type": "transport"},
        )
        called = run_plots(result, ctx, categories=None)
        # Transport plot names should be in called (even if they no-op)
        assert "metric_curves" in called
        assert "obspack_stations" in called

    def test_distributional_metadata_infers_distributional(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_distributional_result()
        called = run_plots(result, ctx, categories=None)
        assert "marginals" in called
        assert "qq_plot" in called
        assert "distributional_summary" in called


# ── TestRunPlotsIntegration ────────────────────────────────────────────────


class TestRunPlotsIntegration:
    def test_run_plots_ensemble_dispatches(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_ensemble_result()
        called = run_plots(result, ctx, categories=None)
        # All 3 ensemble + 3 always should be called
        for name in ["rank_histogram", "calibration", "spread_maps"]:
            assert name in called

    def test_run_plots_distributional_dispatches(self, tmp_path):
        cfg = PlotConfig(imgformats=["png"], save_dir=str(tmp_path))
        ctx = PlotContext(cfg)
        result = _make_distributional_result()
        called = run_plots(result, ctx, categories=None)
        for name in ["marginals", "qq_plot", "distributional_summary"]:
            assert name in called
