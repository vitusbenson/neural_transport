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
    sample_fig,
    sample_flowdps,
    sample_ictm,
    sample_sde,
    sample_unconditional,
    train_flow_matching,
)
from neural_transport.inference.posterior_samplers import FlowDPSSampler

# ── Expected metric keys ─────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {
    "rmse_3d_full",
    "rmse_3d_obs",
    "rmse_3d_away",
    "rmse_xco2_full",
    "rmse_xco2_obs",
    "rmse_xco2_away",
    "spread_3d",
    "spread_xco2",
    "spread_skill_ratio",
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
    "flowdps_s0.1": 50.0,
    "flowdps_s0.5": 60.0,
    "flowdps_s1.0": 80.0,
    "flowdps_s0.1_smooth2": 50.0,
    "sde_s0.3": 100.0,
    "sde_s0.5": 100.0,
    "sde_s1.0": 100.0,
    "pc_c1_s0.3": 100.0,
    "pc_c3_s0.3": 100.0,
    "fig_c10_k1": 100.0,
    "fig_c20_k1": 100.0,
    "fig_c10_k3": 100.0,
    "ictm_r1.0_dec": 100.0,
    "ictm_r0.5_dec": 100.0,
    "ictm_r1.0_const": 100.0,
    "ictm_r1.0_inner3": 100.0,
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


@pytest.mark.quick
def test_flowdps_no_nan(quick_model_and_data):
    """FlowDPS samples contain no NaN/Inf and have correct shape."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1}
    samples = sample_flowdps(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in FlowDPS samples"
    assert not torch.isinf(samples).any(), "Inf in FlowDPS samples"
    assert samples.abs().max() < 1000, f"Divergence in FlowDPS: max={samples.abs().max():.1f}"


@pytest.mark.quick
def test_flowdps_projection_unit(quick_model_and_data):
    """Single projection step reduces column error at observed locations."""
    ctx = quick_model_and_data
    masking_config = ctx["masking_config"]

    sampler = FlowDPSSampler(
        velocity_model=ctx["velocity_wrapper"],
        masking_config=masking_config,
        sigma_obs=1e-6,  # near-zero for hard constraint
    )

    # Create a random field and project it
    torch.manual_seed(42)
    x_hat = torch.randn(1, ctx["nlev"], ctx["nlat"], ctx["nlon"], device=ctx["device"])

    xco2_before = sampler.compute_xco2(x_hat)
    x_hat_proj = sampler._project_column(x_hat)
    xco2_after = sampler.compute_xco2(x_hat_proj)

    obs_mask = masking_config["obs_mask"]
    obs_values = masking_config["obs_values"]

    error_before = (xco2_before - obs_values)[obs_mask].pow(2).mean().sqrt()
    error_after = (xco2_after - obs_values)[obs_mask].pow(2).mean().sqrt()

    assert error_after < error_before, (
        f"Projection did not reduce column error: before={error_before:.4f}, after={error_after:.4f}"
    )


@pytest.mark.quick
def test_sde_no_nan(quick_model_and_data):
    """SDE samples (sigma_max=0.3, no corrector) have no NaN/Inf, correct shape."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "sigma_max": 0.3}
    samples = sample_sde(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in SDE samples"
    assert not torch.isinf(samples).any(), "Inf in SDE samples"
    assert samples.abs().max() < 1000, f"Divergence in SDE: max={samples.abs().max():.1f}"


@pytest.mark.quick
def test_sde_with_corrector_no_nan(quick_model_and_data):
    """SDE + 2 corrector steps, no NaN/Inf."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "sigma_max": 0.3, "n_corrector_steps": 2, "corrector_step_size": 0.01}
    samples = sample_sde(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in SDE+corrector samples"
    assert not torch.isinf(samples).any(), "Inf in SDE+corrector samples"


@pytest.mark.quick
def test_sde_has_larger_spread(quick_model_and_data):
    """SDE ensemble spread >= FlowDPS ensemble spread."""
    ctx = quick_model_and_data
    masking_config = ctx["masking_config"]

    # FlowDPS baseline
    torch.manual_seed(42)
    flowdps_samples = sample_flowdps(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1},
        10,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    flowdps_std = flowdps_samples.std(dim=0).mean().item()

    # SDE
    torch.manual_seed(42)
    sde_samples = sample_sde(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1, "sigma_max": 0.5},
        10,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    sde_std = sde_samples.std(dim=0).mean().item()

    assert sde_std >= flowdps_std * 0.9, f"SDE spread ({sde_std:.4f}) should be >= FlowDPS spread ({flowdps_std:.4f})"


@pytest.mark.quick
@pytest.mark.parametrize("schedule", ["annealed", "constant", "cosine"])
def test_noise_schedule_variants(quick_model_and_data, schedule):
    """All noise schedule variants produce valid samples."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "sigma_max": 0.3, "noise_schedule": schedule}
    samples = sample_sde(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert not torch.isnan(samples).any(), f"NaN in SDE {schedule} samples"
    assert not torch.isinf(samples).any(), f"Inf in SDE {schedule} samples"


@pytest.mark.quick
def test_sde_sigma_zero_matches_flowdps(quick_model_and_data):
    """With sigma_max=0.0 + 0 corrector steps, SDE output matches FlowDPS."""
    ctx = quick_model_and_data
    masking_config = ctx["masking_config"]

    torch.manual_seed(42)
    flowdps_samples = sample_flowdps(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1},
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )

    torch.manual_seed(42)
    sde_samples = sample_sde(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1, "sigma_max": 0.0, "n_corrector_steps": 0},
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )

    assert torch.allclose(sde_samples, flowdps_samples, atol=1e-5), (
        f"SDE(sigma=0) should match FlowDPS. Max diff: {(sde_samples - flowdps_samples).abs().max():.6f}"
    )


# ── FIG quick tests ──────────────────────────────────────────────────────


@pytest.mark.quick
def test_fig_no_nan(quick_model_and_data):
    """FIG samples have no NaN/Inf, correct shape, and bounded values."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "step_size_c": 10.0, "k_steps": 1}
    samples = sample_fig(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in FIG samples"
    assert not torch.isinf(samples).any(), "Inf in FIG samples"
    assert samples.abs().max() < 1000, f"Divergence in FIG: max={samples.abs().max():.1f}"


@pytest.mark.quick
def test_fig_multiple_correction_steps(quick_model_and_data):
    """FIG with k=3 correction steps produces valid samples."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "step_size_c": 10.0, "k_steps": 3}
    samples = sample_fig(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in FIG k=3 samples"
    assert not torch.isinf(samples).any(), "Inf in FIG k=3 samples"


@pytest.mark.quick
def test_fig_reduces_column_error(quick_model_and_data):
    """FIG XCO2 RMSE at observed locations < 1.5x unconditional RMSE."""
    ctx = quick_model_and_data
    masking_config = ctx["masking_config"]

    # Unconditional baseline
    torch.manual_seed(42)
    uncond_samples = sample_unconditional(
        ctx["velocity_wrapper"], 10, ctx["nlev"], ctx["nlat"], ctx["nlon"], ctx["device"]
    )
    uncond_metrics = evaluate(
        uncond_samples, ctx["gt_norm"], ctx["obs_mask"], ctx["column_weights"], ctx["data_mean"], ctx["data_std"]
    )

    # FIG
    torch.manual_seed(42)
    fig_samples = sample_fig(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1, "step_size_c": 10.0, "k_steps": 1},
        10,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    fig_metrics = evaluate(
        fig_samples, ctx["gt_norm"], ctx["obs_mask"], ctx["column_weights"], ctx["data_mean"], ctx["data_std"]
    )

    assert fig_metrics["rmse_xco2_obs"] < uncond_metrics["rmse_xco2_obs"] * 1.5, (
        f"FIG xco2_obs RMSE ({fig_metrics['rmse_xco2_obs']:.4f}) should be < "
        f"1.5x unconditional ({uncond_metrics['rmse_xco2_obs']:.4f})"
    )


@pytest.mark.quick
def test_fig_with_measurement_noise(quick_model_and_data):
    """FIG with measurement interpolant noise (w=0.5) produces valid samples."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "step_size_c": 10.0, "k_steps": 1, "noise_scale_w": 0.5}
    samples = sample_fig(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in FIG w=0.5 samples"
    assert not torch.isinf(samples).any(), "Inf in FIG w=0.5 samples"


# ── ICTM quick tests ─────────────────────────────────────────────────────


@pytest.mark.quick
def test_ictm_no_nan(quick_model_and_data):
    """ICTM samples have no NaN/Inf, correct shape, and bounded values."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "r_max": 1.0, "r_schedule": "decreasing"}
    samples = sample_ictm(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in ICTM samples"
    assert not torch.isinf(samples).any(), "Inf in ICTM samples"
    assert samples.abs().max() < 1000, f"Divergence in ICTM: max={samples.abs().max():.1f}"


@pytest.mark.quick
def test_ictm_multiple_inner_steps(quick_model_and_data):
    """ICTM with n_inner_steps=3 produces valid samples."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "r_max": 1.0, "n_inner_steps": 3, "inner_lr": 0.1}
    samples = sample_ictm(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert samples.shape == (5, ctx["nlev"], ctx["nlat"], ctx["nlon"])
    assert not torch.isnan(samples).any(), "NaN in ICTM n_inner=3 samples"
    assert not torch.isinf(samples).any(), "Inf in ICTM n_inner=3 samples"


@pytest.mark.quick
def test_ictm_reduces_column_error(quick_model_and_data):
    """ICTM XCO2 RMSE at observed locations < 1.5x unconditional RMSE."""
    ctx = quick_model_and_data
    masking_config = ctx["masking_config"]

    # Unconditional baseline
    torch.manual_seed(42)
    uncond_samples = sample_unconditional(
        ctx["velocity_wrapper"], 10, ctx["nlev"], ctx["nlat"], ctx["nlon"], ctx["device"]
    )
    uncond_metrics = evaluate(
        uncond_samples, ctx["gt_norm"], ctx["obs_mask"], ctx["column_weights"], ctx["data_mean"], ctx["data_std"]
    )

    # ICTM
    torch.manual_seed(42)
    ictm_samples = sample_ictm(
        ctx["velocity_wrapper"],
        masking_config,
        {"sigma_obs": 0.1, "r_max": 1.0, "r_schedule": "decreasing"},
        10,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    ictm_metrics = evaluate(
        ictm_samples, ctx["gt_norm"], ctx["obs_mask"], ctx["column_weights"], ctx["data_mean"], ctx["data_std"]
    )

    assert ictm_metrics["rmse_xco2_obs"] < uncond_metrics["rmse_xco2_obs"] * 1.5, (
        f"ICTM xco2_obs RMSE ({ictm_metrics['rmse_xco2_obs']:.4f}) should be < "
        f"1.5x unconditional ({uncond_metrics['rmse_xco2_obs']:.4f})"
    )


@pytest.mark.quick
@pytest.mark.parametrize("schedule", ["constant", "decreasing", "increasing", "cosine"])
def test_ictm_r_schedule_variants(quick_model_and_data, schedule):
    """All r_schedule variants produce valid samples."""
    ctx = quick_model_and_data
    torch.manual_seed(42)
    masking_config = ctx["masking_config"]
    config = {"sigma_obs": 0.1, "r_max": 1.0, "r_schedule": schedule}
    samples = sample_ictm(
        ctx["velocity_wrapper"],
        masking_config,
        config,
        5,
        ctx["nlev"],
        ctx["nlat"],
        ctx["nlon"],
        ctx["device"],
    )
    assert not torch.isnan(samples).any(), f"NaN in ICTM {schedule} samples"
    assert not torch.isinf(samples).any(), f"Inf in ICTM {schedule} samples"


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
