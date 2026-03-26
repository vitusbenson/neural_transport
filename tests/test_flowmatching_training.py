"""Tests for OT coupling, time grid utilities, and mode dispatch.

Usage:
    pytest tests/test_flowmatching_training.py -v
    pytest tests/test_flowmatching_training.py -m quick -v
"""

import pytest
import torch
import torch.nn as nn

from neural_transport.models.flowmatching import FlowMatching, compute_ot_coupling

# ── OT coupling tests ────────────────────────────────────────────────────


@pytest.mark.quick
def test_ot_coupling_correct_shape():
    """Output has same shape as input."""
    x_0 = torch.randn(8, 4, 16, 32)
    x_1 = torch.randn(8, 4, 16, 32)
    result = compute_ot_coupling(x_0, x_1)
    assert result.shape == x_0.shape


@pytest.mark.quick
def test_ot_coupling_is_reordering():
    """Output rows are drawn from input rows (each output row matches some input row)."""
    B = 16
    x_0 = torch.randn(B, 4, 16, 32)
    x_1 = torch.randn(B, 4, 16, 32)
    result = compute_ot_coupling(x_0, x_1)

    # Each row in result should match at least one row in x_0
    x0_flat = x_0.reshape(B, -1)
    res_flat = result.reshape(B, -1)

    matched_outputs = 0
    for i in range(B):
        for j in range(B):
            if torch.allclose(res_flat[i], x0_flat[j], atol=1e-6):
                matched_outputs += 1
                break

    assert matched_outputs == B, f"Only {matched_outputs}/{B} output rows match input rows"


@pytest.mark.quick
def test_ot_coupling_reduces_cost():
    """OT coupling produces lower total L2 cost than random pairing."""
    torch.manual_seed(42)
    B = 32
    x_0 = torch.randn(B, 4, 8, 16)
    x_1 = torch.randn(B, 4, 8, 16)

    # Random pairing cost
    random_cost = (x_0 - x_1).reshape(B, -1).pow(2).sum(dim=1).mean().item()

    # OT pairing cost
    x_0_ot = compute_ot_coupling(x_0, x_1)
    ot_cost = (x_0_ot - x_1).reshape(B, -1).pow(2).sum(dim=1).mean().item()

    assert ot_cost <= random_cost, f"OT cost {ot_cost:.4f} > random cost {random_cost:.4f}"


@pytest.mark.quick
def test_ot_coupling_small_batch():
    """Works correctly with B=2."""
    x_0 = torch.randn(2, 4, 8, 16)
    x_1 = torch.randn(2, 4, 8, 16)
    result = compute_ot_coupling(x_0, x_1)
    assert result.shape == x_0.shape
    assert not torch.isnan(result).any()


@pytest.mark.quick
def test_ot_coupling_no_grad():
    """OT coupling should not track gradients."""
    x_0 = torch.randn(4, 2, 8, 8, requires_grad=True)
    x_1 = torch.randn(4, 2, 8, 8, requires_grad=True)
    result = compute_ot_coupling(x_0, x_1)
    assert not result.requires_grad


# ── Time grid tests ──────────────────────────────────────────────────────


@pytest.mark.quick
def test_time_grid_uniform():
    """Uniform grid has equal spacing."""
    grid = FlowMatching._build_time_grid(11, torch.device('cpu'), 'uniform')
    assert grid.shape == (11,)
    diffs = grid[1:] - grid[:-1]
    assert torch.allclose(diffs, diffs[0], atol=1e-6)


@pytest.mark.quick
def test_time_grid_cosine():
    """Cosine grid is denser at endpoints (larger steps in the middle)."""
    grid = FlowMatching._build_time_grid(21, torch.device('cpu'), 'cosine')
    diffs = grid[1:] - grid[:-1]
    # Middle step should be larger than first step
    assert diffs[len(diffs) // 2] > diffs[0]
    # Middle step should be larger than last step
    assert diffs[len(diffs) // 2] > diffs[-1]


@pytest.mark.quick
def test_time_grid_front_loaded():
    """Front-loaded grid is denser near t=0 (first step < last step)."""
    grid = FlowMatching._build_time_grid(21, torch.device('cpu'), 'front_loaded')
    diffs = grid[1:] - grid[:-1]
    assert diffs[0] < diffs[-1], "Front-loaded grid should have smaller steps near t=0"


@pytest.mark.quick
def test_all_grids_monotonic_0_to_1():
    """All grid types are monotonically increasing from 0 to 1."""
    for spacing in ['uniform', 'cosine', 'front_loaded']:
        grid = FlowMatching._build_time_grid(11, torch.device('cpu'), spacing)
        assert grid[0] == 0.0, f"{spacing}: grid doesn't start at 0"
        assert grid[-1] == 1.0, f"{spacing}: grid doesn't end at 1"
        diffs = grid[1:] - grid[:-1]
        assert (diffs > 0).all(), f"{spacing}: grid is not monotonically increasing"


# ── Mode dispatch tests ─────────────────────────────────────────────────


def _make_tiny_fm_model(norm="batch"):
    """Create a minimal FlowMatching model for testing mode dispatch."""
    NLAT, NLON, NLEV = 8, 16, 1

    regulargrid_kwargs = dict(
        input_vars=["co2massmix"],
        target_vars=["co2massmix"],
        nlat=NLAT,
        nlon=NLON,
        predict_delta=False,
        add_surfflux=False,
        dt=3600,
        massfixer="",
        targshift=False,
    )

    fm = FlowMatching(**regulargrid_kwargs)
    fm.init_model(
        submodel="unet",
        model_kwargs=dict(
            **regulargrid_kwargs,
            model_kwargs=dict(
                in_chans=NLEV + 1,
                out_chans=NLEV,
                embed_dim=16,
                act="leakyrelu",
                norm=norm,
                enc_filters=[[3], [3]],
                dec_filters=[[3], [3]],
            ),
        ),
        generating=False,
        return_intermediates=False,
        method="euler",
        nlev=NLEV,
        step_size=0.5,
        use_ot_coupling=False,
        time_sampling="uniform",
    )
    return fm


def _make_batch(nlat=8, nlon=16, nlev=1, B=4):
    """Create a synthetic batch dict for FlowMatching."""
    N = nlat * nlon
    return {
        "co2massmix": torch.randn(B, N, nlev),
        "co2massmix_next": torch.randn(B, N, nlev),
        "co2massmix_offset": torch.full((B, N, nlev), 400.0),
        "co2massmix_scale": torch.full((B, N, nlev), 10.0),
        "co2massmix_delta_offset": torch.zeros(B, N, nlev),
        "co2massmix_delta_scale": torch.ones(B, N, nlev),
    }


@pytest.mark.quick
def test_forward_mode_train_in_eval_mode():
    """mode='train' dispatches to training_forward even when model is in eval mode."""
    fm = _make_tiny_fm_model()
    fm.eval()  # Set model to eval mode

    batch = _make_batch()
    preds = fm(batch, mode="train")

    # training_forward produces dx_t key
    assert "dx_t" in preds, "mode='train' should use training_forward (dx_t missing)"


@pytest.mark.quick
def test_forward_mode_none_uses_self_training():
    """mode=None infers mode from self.training flag (backward compat)."""
    fm = _make_tiny_fm_model()

    batch = _make_batch()

    # In training mode, should use training_forward
    fm.train()
    preds = fm(batch, mode=None)
    assert "dx_t" in preds

    # In eval mode with generating=True, should use generation path
    fm.eval()
    fm.generating = True
    fm.return_intermediates = False
    fm.generate_kwargs = {}
    preds = fm(batch, mode=None)
    assert "dx_t" not in preds
    assert "co2massmix" in preds


@pytest.mark.quick
def test_forward_mode_generate_in_train_mode():
    """mode='generate' dispatches to inference_forward even when model is training."""
    fm = _make_tiny_fm_model()
    fm.train()
    fm.return_intermediates = False
    fm.generate_kwargs = {}

    batch = _make_batch()
    preds = fm(batch, mode="generate")

    # generation path produces co2massmix but no dx_t
    assert "co2massmix" in preds
    assert "dx_t" not in preds


@pytest.mark.quick
def test_mode_train_preserves_batchnorm_eval():
    """mode='train' must NOT set BatchNorm layers to training mode."""
    fm = _make_tiny_fm_model(norm="batch")
    fm.eval()  # All submodules in eval mode

    # Record BatchNorm eval states
    bn_modules = [m for m in fm.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_modules) > 0, "Model should contain BatchNorm layers"

    # Verify all BN layers are in eval mode
    for m in bn_modules:
        assert not m.training, "BatchNorm should be in eval mode before forward"

    batch = _make_batch()
    _ = fm(batch, mode="train")

    # After forward with mode="train", BN layers must STILL be in eval mode
    for m in bn_modules:
        assert not m.training, "mode='train' must not set BatchNorm to training mode"


@pytest.mark.quick
def test_batchnorm_running_stats_unchanged_with_mode_train():
    """BatchNorm running_mean/var must not change when using mode='train' in eval."""
    fm = _make_tiny_fm_model(norm="batch")

    # Run a few training steps to populate running stats
    fm.train()
    for _ in range(3):
        batch = _make_batch()
        _ = fm(batch)

    fm.eval()

    # Record running stats
    bn_modules = [m for m in fm.modules() if isinstance(m, nn.BatchNorm2d)]
    stats_before = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_modules]

    # Run forward with mode="train" (simulating validation_step)
    batch = _make_batch()
    _ = fm(batch, mode="train")

    # Running stats must be unchanged
    for i, m in enumerate(bn_modules):
        mean_before, var_before = stats_before[i]
        assert torch.allclose(m.running_mean, mean_before, atol=1e-7), (
            f"BatchNorm[{i}] running_mean changed during mode='train' forward"
        )
        assert torch.allclose(m.running_var, var_before, atol=1e-7), (
            f"BatchNorm[{i}] running_var changed during mode='train' forward"
        )


@pytest.mark.quick
def test_groupnorm_forward_mode_train():
    """FlowMatching with GroupNorm works correctly with mode='train'."""
    fm = _make_tiny_fm_model(norm="group")
    fm.eval()

    batch = _make_batch()
    preds = fm(batch, mode="train")

    assert "dx_t" in preds
    assert not torch.isnan(preds["co2massmix"]).any()


@pytest.mark.quick
def test_validation_step_does_not_corrupt_batchnorm():
    """NeuralTransport.validation_step uses mode='train', not model.train()."""
    from neural_transport.litmodule import NeuralTransport

    fm_kwargs = dict(
        model="flowmatching",
        model_kwargs=dict(
            input_vars=["co2massmix"],
            target_vars=["co2massmix"],
            nlat=8,
            nlon=16,
            predict_delta=False,
            add_surfflux=False,
            dt=3600,
            massfixer="",
            targshift=False,
            model_kwargs=dict(
                submodel="unet",
                model_kwargs=dict(
                    input_vars=["co2massmix"],
                    target_vars=["co2massmix"],
                    nlat=8,
                    nlon=16,
                    predict_delta=False,
                    add_surfflux=False,
                    dt=3600,
                    massfixer="",
                    targshift=False,
                    model_kwargs=dict(
                        in_chans=2,
                        out_chans=1,
                        embed_dim=16,
                        act="leakyrelu",
                        norm="batch",
                        enc_filters=[[3], [3]],
                        dec_filters=[[3], [3]],
                    ),
                ),
                generating=False,
                return_intermediates=False,
                method="euler",
                nlev=1,
                step_size=0.5,
                use_ot_coupling=False,
                time_sampling="uniform",
            ),
        ),
        loss="flowmatching_mse",
        loss_kwargs=dict(),
        metrics=[],
        no_grad_step_shedule=None,
        lr=1e-3,
        weight_decay=0.0,
        lr_shedule_kwargs=dict(
            warmup_steps=5,
            halfcosine_steps=50,
            min_lr=1e-6,
            max_lr=0.8,
        ),
        val_dataloader_names=["singlestep"],
        plot_kwargs=dict(
            variables=[],
            layer_idxs=[],
            n_samples=0,
            dataset="carbontracker",
            grid="latlon5.625",
            vertical_levels="l10",
            max_workers=1,
        ),
    )

    nt = NeuralTransport(**fm_kwargs)

    # Train for a few steps to populate BatchNorm running stats
    nt.train()
    N = 8 * 16
    for _ in range(3):
        batch = {
            "co2massmix": torch.randn(4, 1, N, 1),
            "co2massmix_next": torch.randn(4, 1, N, 1),
            "co2massmix_offset": torch.full((4, 1, N, 1), 400.0),
            "co2massmix_scale": torch.full((4, 1, N, 1), 10.0),
            "co2massmix_delta_offset": torch.zeros(4, 1, N, 1),
            "co2massmix_delta_scale": torch.ones(4, 1, N, 1),
        }
        nt.common_step(batch, mode="train")

    # Switch to eval mode (as Lightning does before validation)
    nt.eval()

    # Record BatchNorm running stats
    bn_modules = [m for m in nt.modules() if isinstance(m, nn.BatchNorm2d)]
    stats_before = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_modules]

    # Run validation_step
    val_batch = {
        "co2massmix": torch.randn(4, 1, N, 1),
        "co2massmix_next": torch.randn(4, 1, N, 1),
        "co2massmix_offset": torch.full((4, 1, N, 1), 400.0),
        "co2massmix_scale": torch.full((4, 1, N, 1), 10.0),
        "co2massmix_delta_offset": torch.zeros(4, 1, N, 1),
        "co2massmix_delta_scale": torch.ones(4, 1, N, 1),
    }
    nt.validation_step(val_batch, batch_idx=0, dataloader_idx=0)

    # BatchNorm running stats must be unchanged
    for i, m in enumerate(bn_modules):
        mean_before, var_before = stats_before[i]
        assert torch.allclose(m.running_mean, mean_before, atol=1e-7), (
            f"validation_step corrupted BatchNorm[{i}] running_mean"
        )
        assert torch.allclose(m.running_var, var_before, atol=1e-7), (
            f"validation_step corrupted BatchNorm[{i}] running_var"
        )
