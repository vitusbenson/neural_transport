"""Pytest suite for the toy column OSSE.

All tests in this module are marked slow since they train models.
Run with: pytest -m slow

Quick tests: sanity checks with tiny model (500 samples, 5 epochs, 8x16 grid).
Slow tests: full validation with all 8 methods (10k samples, 50 epochs, 16x32 grid).
"""

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

from neural_transport.experiments.toy_column_osse import (
    CONDITIONING_METHODS,
    ToyVelocityWrapper,
    build_masking_config,
    create_column_obs,
    evaluate,
    generate_toy_data,
    get_column_weights,
    run_toy_osse,
    sample_conditioned,
    sample_unconditional,
    train_flow_matching,
)

# ── Expected metric keys ─────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {
    "rmse_3d_full",
    "rmse_3d_obs",
    "rmse_3d_away",
    "rmse_xco2_full",
    "rmse_xco2_obs",
    "rmse_xco2_away",
}

# ── RMSE thresholds (generous placeholders, tighten after first baseline run) ──

# Calibrated from baseline run (seed=42, 10k samples, 50 epochs, 16x32 grid).
# Thresholds set to ~1.5x observed values to catch regressions without being brittle.
RMSE_THRESHOLDS = {
    "unconditional": 100.0,
    "correction": 42.0,
    "correction_late": 60.0,
    "velocity_proj_late": 65.0,
    "guidance_0.5": 90.0,
    "guidance_1": 85.0,
    "guidance_1_late": 95.0,
    "repaint_late": 65.0,
}

# ── Quick fixtures (tiny model, shared across quick tests) ────────────────


@pytest.fixture(scope="module")
def quick_model_and_data():
    """Train a tiny model for quick sanity checks.

    500 samples, 5 epochs, 8x16 grid, 4 levels.
    """
    nlat, nlon, nlev = 8, 16, 4
    device = "cpu"

    data = generate_toy_data(n_samples=500, nlat=nlat, nlon=nlon, nlev=nlev, seed=42)
    column_weights = get_column_weights(nlev)

    conv_model, data_mean, data_std = train_flow_matching(data, nlev=nlev, epochs=5, lr=1e-3, device=device)
    velocity_wrapper = ToyVelocityWrapper(conv_model, nlev=nlev).to(device)

    # Ground truth and observations
    gt_norm = ((data[0:1] - data_mean) / data_std).to(device)
    gt_phys = data[0:1].to(device)
    obs_mask, obs_values = create_column_obs(gt_phys, column_weights, obs_fraction=0.3, seed=123)
    obs_mask = obs_mask.to(device)
    obs_values = obs_values.to(device)

    masking_config = build_masking_config(obs_mask, obs_values, data_mean, data_std, column_weights, device)

    return {
        "velocity_wrapper": velocity_wrapper,
        "masking_config": masking_config,
        "gt_norm": gt_norm,
        "obs_mask": obs_mask,
        "column_weights": column_weights,
        "data_mean": data_mean,
        "data_std": data_std,
        "nlat": nlat,
        "nlon": nlon,
        "nlev": nlev,
        "device": device,
    }


# ── Slow fixture (full OSSE with all methods) ────────────────────────────


@pytest.fixture(scope="module")
def full_osse_results(tmp_path_factory):
    """Run full toy OSSE with all 8 methods.

    10k samples, 50 epochs, 16x32 grid.
    """
    out_dir = tmp_path_factory.mktemp("full_osse")
    results = run_toy_osse(
        device="cpu",
        n_train=10000,
        epochs=50,
        n_samples=20,
        obs_fraction=0.3,
        methods=None,
        out_dir=str(out_dir),
        seed=42,
        nlat=16,
        nlon=32,
        nlev=4,
    )
    return {"results": results, "out_dir": out_dir}


# ── Quick tests ──────────────────────────────────────────────────────────


@pytest.mark.quick
def test_unconditional_no_nan(quick_model_and_data):
    """Unconditional samples contain no NaN/Inf and have correct shape."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    samples = sample_unconditional(ctx["velocity_wrapper"], 5, ctx["nlev"], ctx["nlat"], ctx["nlon"], ctx["device"])
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in unconditional samples"
    assert not torch.isinf(samples).any(), "Inf in unconditional samples"


@pytest.mark.quick
@pytest.mark.parametrize("method", ["guidance_1", "correction_late"])
def test_conditioned_no_nan(quick_model_and_data, method):
    """Conditioned samples contain no NaN/Inf and values < 1000 (no divergence)."""
    ctx = quick_model_and_data
    config = {k: v for k, v in CONDITIONING_METHODS[method].items()}
    config.pop("_unconditional", False)

    torch.manual_seed(42)
    samples = sample_conditioned(
        ctx["velocity_wrapper"],
        ctx["masking_config"],
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert not torch.isnan(samples).any(), f"NaN in {method} samples"
    assert not torch.isinf(samples).any(), f"Inf in {method} samples"
    assert samples.abs().max() < 1000, f"Divergence in {method}: max={samples.abs().max():.1f}"


@pytest.mark.quick
def test_evaluate_returns_all_keys(quick_model_and_data):
    """evaluate() returns all 6 expected metric keys."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    samples = sample_unconditional(ctx["velocity_wrapper"], 5, ctx["nlev"], ctx["nlat"], ctx["nlon"], ctx["device"])
    metrics = evaluate(
        samples,
        ctx["gt_norm"],
        ctx["obs_mask"],
        ctx["column_weights"],
        ctx["data_mean"],
        ctx["data_std"],
    )
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} is not float: {type(val)}"


@pytest.mark.quick
def test_run_toy_osse_smoke(tmp_path):
    """End-to-end run_toy_osse() with 2 methods, no crash."""
    results = run_toy_osse(
        device="cpu",
        n_train=500,
        epochs=5,
        n_samples=5,
        obs_fraction=0.3,
        methods=["unconditional", "guidance_1"],
        out_dir=str(tmp_path),
        seed=42,
        nlat=8,
        nlon=16,
        nlev=4,
    )
    assert "unconditional" in results
    assert "guidance_1" in results
    for name, res in results.items():
        assert "samples" in res
        assert "metrics" in res
        assert set(res["metrics"].keys()) == EXPECTED_METRIC_KEYS


# ── Slow tests ───────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("method", list(CONDITIONING_METHODS.keys()))
def test_no_nan(full_osse_results, method):
    """No NaN in metrics for any method."""
    results = full_osse_results["results"]
    assert method in results, f"Method {method} not in results"
    metrics = results[method]["metrics"]
    for key, val in metrics.items():
        assert not np.isnan(val), f"NaN in {method}/{key}"


@pytest.mark.slow
@pytest.mark.parametrize("method", list(CONDITIONING_METHODS.keys()))
def test_no_divergence(full_osse_results, method):
    """No divergence: abs().max() < 100 in normalized space."""
    results = full_osse_results["results"]
    samples = results[method]["samples"]
    max_val = samples.abs().max().item()
    assert max_val < 100, f"Divergence in {method}: max_abs={max_val:.1f}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    [m for m in CONDITIONING_METHODS.keys() if m != "unconditional"],
)
def test_xco2_rmse_threshold(full_osse_results, method):
    """RMSE XCO2 below generous threshold for each conditioned method."""
    results = full_osse_results["results"]
    metrics = results[method]["metrics"]
    threshold = RMSE_THRESHOLDS[method]
    rmse = metrics["rmse_xco2_full"]
    assert rmse < threshold, f"{method}: rmse_xco2_full={rmse:.4f} exceeds threshold={threshold}"


@pytest.mark.slow
def test_conditioned_beats_unconditional(full_osse_results):
    """At least one conditioned method beats unconditional on XCO2 RMSE."""
    results = full_osse_results["results"]
    uncond_rmse = results["unconditional"]["metrics"]["rmse_xco2_full"]

    conditioned_rmses = {
        name: res["metrics"]["rmse_xco2_full"] for name, res in results.items() if name != "unconditional"
    }

    best_method = min(conditioned_rmses, key=conditioned_rmses.get)
    best_rmse = conditioned_rmses[best_method]

    assert best_rmse < uncond_rmse, (
        f"No conditioned method beat unconditional ({uncond_rmse:.4f}). Best was {best_method} ({best_rmse:.4f})"
    )


@pytest.mark.slow
def test_metrics_json_saved(full_osse_results):
    """metrics_summary.json written with all 8 entries."""
    out_dir = full_osse_results["out_dir"]
    json_path = out_dir / "metrics_summary.json"
    assert json_path.exists(), f"metrics_summary.json not found at {json_path}"

    with open(json_path) as f:
        summary = json.load(f)

    assert len(summary) == len(CONDITIONING_METHODS), (
        f"Expected {len(CONDITIONING_METHODS)} entries, got {len(summary)}"
    )
    for method_name in CONDITIONING_METHODS:
        assert method_name in summary, f"Missing method {method_name} in JSON"
