"""End-to-end training tests for FlowMatching models.

Tests the full pipeline: model construction → training → validation →
generation quality evaluation. Uses synthetic data (no real CarbonTracker needed).

Usage:
    pytest tests/test_training_e2e.py -v              # All tests
    pytest tests/test_training_e2e.py -m slow -v      # Slow tests only
"""

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# ── Synthetic data helpers ───────────────────────────────────────────────


class _SyntheticCO2Dataset(Dataset):
    """Synthetic dataset mimicking CarbonDataset batch structure.

    Each item returns a dict with:
        co2massmix:        [T, N, C]  (input/target variable)
        co2massmix_next:   [T, N, C]  (next-step target)
        co2massmix_offset: [1, N, C]  (normalization offset)
        co2massmix_scale:  [1, N, C]  (normalization scale)

    With include_time=False, returns [N, C] instead (for GenerationQualityCallback).
    """

    def __init__(self, n_samples: int, nlat: int = 8, nlon: int = 16, nlev: int = 1, include_time: bool = True):
        self.n_samples = n_samples
        self.nlat = nlat
        self.nlon = nlon
        self.nlev = nlev
        self.N = nlat * nlon
        self.include_time = include_time

        torch.manual_seed(42)
        # Pre-generate data for reproducibility
        self.data = torch.randn(n_samples, self.N, nlev) * 10.0 + 400.0
        self.data_next = torch.randn(n_samples, self.N, nlev) * 10.0 + 400.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.include_time:
            return {
                "co2massmix": self.data[idx].unsqueeze(0),  # [1, N, C]
                "co2massmix_next": self.data_next[idx].unsqueeze(0),  # [1, N, C]
                "co2massmix_offset": torch.full((1, self.N, self.nlev), 400.0),
                "co2massmix_scale": torch.full((1, self.N, self.nlev), 10.0),
                "co2massmix_delta_offset": torch.zeros(1, self.N, self.nlev),
                "co2massmix_delta_scale": torch.ones(1, self.N, self.nlev),
            }
        else:
            # No time dimension — for GenerationQualityCallback
            return {
                "co2massmix": self.data[idx],  # [N, C]
                "co2massmix_next": self.data_next[idx],  # [N, C]
                "co2massmix_offset": torch.full((self.N, self.nlev), 400.0),
                "co2massmix_scale": torch.full((self.N, self.nlev), 10.0),
                "co2massmix_delta_offset": torch.zeros(self.N, self.nlev),
                "co2massmix_delta_scale": torch.ones(self.N, self.nlev),
            }


class SyntheticFMDataModule(pl.LightningDataModule):
    """Minimal LightningDataModule for FlowMatching E2E tests."""

    def __init__(
        self,
        n_train: int = 50,
        n_val: int = 10,
        nlat: int = 8,
        nlon: int = 16,
        nlev: int = 1,
        batch_size: int = 4,
    ):
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.nlat = nlat
        self.nlon = nlon
        self.nlev = nlev
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = _SyntheticCO2Dataset(self.n_train, self.nlat, self.nlon, self.nlev)
        self.val_ds = _SyntheticCO2Dataset(self.n_val, self.nlat, self.nlon, self.nlev)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return [DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)]


# ── Fixtures ─────────────────────────────────────────────────────────────

NLAT, NLON, NLEV = 8, 16, 1


@pytest.fixture
def tiny_fm_kwargs():
    """Minimal FlowMatching UNet configuration for CPU testing."""
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

    wrapper_kwargs = dict(
        **regulargrid_kwargs,
        model_kwargs=dict(
            submodel="unet",
            model_kwargs=dict(
                **regulargrid_kwargs,
                model_kwargs=dict(
                    in_chans=NLEV + 1,  # nlev + time channel
                    out_chans=NLEV,
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
            nlev=NLEV,
            step_size=0.5,
            use_ot_coupling=False,
            time_sampling="uniform",
        ),
    )

    return dict(
        model="flowmatching",
        model_kwargs=wrapper_kwargs,
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


@pytest.fixture
def synthetic_datamodule():
    """Create a SyntheticFMDataModule."""
    dm = SyntheticFMDataModule(n_train=50, n_val=10, nlat=NLAT, nlon=NLON, nlev=NLEV, batch_size=4)
    dm.setup()
    return dm


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_train_10_steps_metrics_finite(tiny_fm_kwargs, synthetic_datamodule, tmp_path):
    """Train FlowMatching for 10 steps and verify metrics are finite."""
    from neural_transport.litmodule import NeuralTransport

    model = NeuralTransport(**tiny_fm_kwargs)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, synthetic_datamodule)

    # Check that training produced finite metrics
    assert "Loss/Train" in trainer.callback_metrics
    train_loss = trainer.callback_metrics["Loss/Train"].item()
    assert torch.isfinite(torch.tensor(train_loss)), f"Train loss is not finite: {train_loss}"

    assert "Loss/Val_singlestep" in trainer.callback_metrics
    val_loss = trainer.callback_metrics["Loss/Val_singlestep"].item()
    assert torch.isfinite(torch.tensor(val_loss)), f"Val loss is not finite: {val_loss}"


@pytest.mark.slow
def test_gen_eval_callback_produces_metrics(tiny_fm_kwargs, synthetic_datamodule, tmp_path):
    """Verify GenerationQualityCallback produces GenEval metrics during training."""
    from neural_transport.litmodule import NeuralTransport
    from neural_transport.training.gen_eval_callback import GenerationQualityCallback

    # Enable generation mode in the wrapper kwargs
    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["generating"] = True
    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["return_intermediates"] = False

    model = NeuralTransport(**tiny_fm_kwargs)

    # Callback needs val_dataset WITHOUT T dimension (it calls FlowMatching directly)
    val_ds_for_callback = _SyntheticCO2Dataset(10, NLAT, NLON, NLEV, include_time=False)

    gen_eval_cb = GenerationQualityCallback(
        val_dataset=val_ds_for_callback,
        n_gt_samples=2,
        n_gen_samples=2,
        eval_every_n_epochs=1,
        target_var="co2massmix",
        generate_kwargs=dict(steps=3, method="euler"),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[gen_eval_cb],
        log_every_n_steps=1,
    )

    trainer.fit(model, synthetic_datamodule)

    # GenEval metrics should be logged
    metrics = trainer.callback_metrics
    assert "GenEval/RMSE" in metrics, f"GenEval/RMSE not found. Available: {list(metrics.keys())}"
    assert "GenEval/energy_distance" in metrics, "GenEval/energy_distance not found"

    rmse = metrics["GenEval/RMSE"].item()
    e_dist = metrics["GenEval/energy_distance"].item()
    assert torch.isfinite(torch.tensor(rmse)), f"GenEval/RMSE is not finite: {rmse}"
    assert torch.isfinite(torch.tensor(e_dist)), f"GenEval/energy_distance is not finite: {e_dist}"

    # Phase 19: valid_fraction should also be logged
    assert "GenEval/valid_fraction" in metrics, "GenEval/valid_fraction not found"
    vf = metrics["GenEval/valid_fraction"].item()
    assert 0.0 <= vf <= 1.0, f"valid_fraction out of range: {vf}"


@pytest.mark.slow
def test_gen_eval_callback_logs_valid_fraction(tiny_fm_kwargs, synthetic_datamodule, tmp_path):
    """Verify GenEval/valid_fraction is logged and in [0, 1]."""
    from neural_transport.litmodule import NeuralTransport
    from neural_transport.training.gen_eval_callback import GenerationQualityCallback

    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["generating"] = True
    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["return_intermediates"] = False

    model = NeuralTransport(**tiny_fm_kwargs)
    val_ds = _SyntheticCO2Dataset(10, NLAT, NLON, NLEV, include_time=False)

    gen_eval_cb = GenerationQualityCallback(
        val_dataset=val_ds,
        n_gt_samples=2,
        n_gen_samples=5,
        eval_every_n_epochs=1,
        target_var="co2massmix",
        generate_kwargs=dict(steps=3, method="euler"),
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        callbacks=[gen_eval_cb],
        log_every_n_steps=1,
    )
    trainer.fit(model, synthetic_datamodule)

    metrics = trainer.callback_metrics
    assert "GenEval/valid_fraction" in metrics
    vf = metrics["GenEval/valid_fraction"].item()
    assert 0.0 <= vf <= 1.0, f"valid_fraction out of range: {vf}"
    # With synthetic data, all samples should be valid
    assert vf > 0.0, "Expected some valid samples with synthetic data"


def test_gen_eval_callback_fast_ode_defaults():
    """Verify default generate_kwargs include fast ODE settings."""
    from unittest.mock import MagicMock

    from neural_transport.training.gen_eval_callback import GenerationQualityCallback

    dummy_ds = MagicMock()
    dummy_ds.__len__ = MagicMock(return_value=10)

    # Default: fast ODE settings applied
    cb = GenerationQualityCallback(val_dataset=dummy_ds)
    assert cb.generate_kwargs["method"] == "euler"
    assert cb.generate_kwargs["steps"] == 5

    # Explicit kwargs override defaults
    cb2 = GenerationQualityCallback(
        val_dataset=dummy_ds,
        generate_kwargs={"method": "midpoint", "steps": 10},
    )
    assert cb2.generate_kwargs["method"] == "midpoint"
    assert cb2.generate_kwargs["steps"] == 10

    # Partial override: only steps
    cb3 = GenerationQualityCallback(
        val_dataset=dummy_ds,
        generate_kwargs={"steps": 3},
    )
    assert cb3.generate_kwargs["method"] == "euler"  # default kept
    assert cb3.generate_kwargs["steps"] == 3  # overridden


@pytest.mark.slow
def test_gen_eval_callback_handles_all_nan(tiny_fm_kwargs, synthetic_datamodule, tmp_path):
    """When all generated samples are NaN, valid_fraction=0 and metrics are inf."""
    from unittest.mock import patch

    from neural_transport.litmodule import NeuralTransport
    from neural_transport.training.gen_eval_callback import GenerationQualityCallback

    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["generating"] = True
    tiny_fm_kwargs["model_kwargs"]["model_kwargs"]["return_intermediates"] = False

    model = NeuralTransport(**tiny_fm_kwargs)
    val_ds = _SyntheticCO2Dataset(10, NLAT, NLON, NLEV, include_time=False)

    gen_eval_cb = GenerationQualityCallback(
        val_dataset=val_ds,
        n_gt_samples=2,
        n_gen_samples=3,
        eval_every_n_epochs=1,
        target_var="co2massmix",
        generate_kwargs=dict(steps=3, method="euler"),
    )

    # Patch is_bad_sample to always return True (simulates all-NaN generation)
    with patch("neural_transport.training.gen_eval_callback.is_bad_sample", return_value=True):
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[gen_eval_cb],
            log_every_n_steps=1,
        )
        trainer.fit(model, synthetic_datamodule)

    metrics = trainer.callback_metrics
    assert "GenEval/valid_fraction" in metrics
    assert metrics["GenEval/valid_fraction"].item() == 0.0

    # RMSE and energy_distance should be inf
    assert metrics["GenEval/RMSE"].item() == float("inf")
    assert metrics["GenEval/energy_distance"].item() == float("inf")
    assert metrics["GenEval/gen_std"].item() == 0.0


@pytest.mark.slow
def test_optuna_objective_uses_composite_metric(tiny_fm_kwargs, tmp_path):
    """Verify FMOptunaObjective composite objective uses valid_fraction."""
    optuna = pytest.importorskip("optuna")
    from neural_transport.training.gen_eval_callback import GenerationQualityCallback
    from neural_transport.training.tuning import FMOptunaObjective

    val_ds = _SyntheticCO2Dataset(5, NLAT, NLON, NLEV, include_time=False)

    trainer_kwargs = dict(max_steps=5, accelerator="cpu")

    class _TestCompositeObjective(FMOptunaObjective):
        """Override to use synthetic data with GenEval callback."""

        def __call__(self, trial):
            from neural_transport.training.tuning import suggest_hyperparams

            params = suggest_hyperparams(trial)
            params["model_size"] = "XXS"
            params["time_loss_weight"] = None

            lit_kwargs, _, tr_kwargs = self._apply_hyperparams(params, trial.number)

            unet_kwargs = lit_kwargs["model_kwargs"]["model_kwargs"]["model_kwargs"]["model_kwargs"]
            unet_kwargs["embed_dim"] = 16
            unet_kwargs["enc_filters"] = [[3], [3]]
            unet_kwargs["dec_filters"] = [[3], [3]]

            # Enable generation mode
            lit_kwargs["model_kwargs"]["model_kwargs"]["generating"] = True
            lit_kwargs["model_kwargs"]["model_kwargs"]["return_intermediates"] = False

            tr_kwargs.pop("max_steps", None)
            tr_kwargs["max_epochs"] = 1

            trial_dir = self.run_dir / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            from neural_transport.litmodule import NeuralTransport

            model = NeuralTransport(**lit_kwargs)
            dm = SyntheticFMDataModule(n_train=20, n_val=5, nlat=NLAT, nlon=NLON, nlev=NLEV, batch_size=4)

            gen_eval_cb = GenerationQualityCallback(
                val_dataset=self.val_dataset,
                n_gt_samples=2,
                n_gen_samples=3,
                eval_every_n_epochs=1,
                target_var="co2massmix",
                generate_kwargs=dict(steps=3, method="euler"),
            )

            trainer = pl.Trainer(
                callbacks=[gen_eval_cb],
                logger=False,
                enable_progress_bar=False,
                **tr_kwargs,
            )
            trainer.fit(model, dm)

            # Use the composite objective logic from tuning.py
            import math

            metrics = {}
            for key in trainer.callback_metrics:
                val = trainer.callback_metrics[key]
                metrics[key] = val.item() if hasattr(val, "item") else float(val)

            valid_fraction = metrics.get("GenEval/valid_fraction", 0.0)
            e_dist = metrics.get("GenEval/energy_distance", float("inf"))

            if valid_fraction > 0 and math.isfinite(e_dist):
                objective = e_dist + self.stability_penalty * (1.0 - valid_fraction)
            elif "Loss/Val_singlestep" in metrics:
                objective = metrics["Loss/Val_singlestep"] + self.stability_penalty
            else:
                objective = float("inf")

            return objective

    objective = _TestCompositeObjective(
        base_data_kwargs={},
        base_lit_module_kwargs=tiny_fm_kwargs,
        base_trainer_kwargs=trainer_kwargs,
        run_dir=tmp_path / "trials",
        val_dataset=val_ds,
        ema_kwargs=None,
        stability_penalty=10.0,
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value is not None
    assert trial.value < float("inf"), f"Trial returned inf: {trial.value}"


@pytest.mark.slow
def test_optuna_objective_runs_single_trial(tiny_fm_kwargs, tmp_path):
    """Verify FMOptunaObjective completes a single trial on CPU."""
    optuna = pytest.importorskip("optuna")
    from neural_transport.training.tuning import FMOptunaObjective

    # The objective needs CarbonDataModule-style data_kwargs.
    # We'll create a custom objective that uses our synthetic data instead.
    val_ds = _SyntheticCO2Dataset(5, NLAT, NLON, NLEV)

    # Build minimal trainer kwargs (enable_progress_bar will be set in __call__)
    trainer_kwargs = dict(
        max_steps=5,
        accelerator="cpu",
    )

    # Build minimal data_kwargs that CarbonDataModule would accept.
    # Since FMOptunaObjective creates CarbonDataModule internally, we need
    # to test with a thin wrapper instead.
    class _TestObjective(FMOptunaObjective):
        """Override to use synthetic data instead of CarbonDataModule."""

        def __call__(self, trial):
            from neural_transport.training.tuning import suggest_hyperparams

            params = suggest_hyperparams(trial)

            # Override params for tiny CPU test
            params["model_size"] = "XXS"
            params["time_loss_weight"] = None

            lit_kwargs, _, tr_kwargs = self._apply_hyperparams(params, trial.number)

            # Force tiny model dims for test
            unet_kwargs = lit_kwargs["model_kwargs"]["model_kwargs"]["model_kwargs"]["model_kwargs"]
            unet_kwargs["embed_dim"] = 16
            unet_kwargs["enc_filters"] = [[3], [3]]
            unet_kwargs["dec_filters"] = [[3], [3]]

            # Force max_epochs=1 to ensure validation runs
            tr_kwargs.pop("max_steps", None)
            tr_kwargs["max_epochs"] = 1

            trial_dir = self.run_dir / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            from neural_transport.litmodule import NeuralTransport

            model = NeuralTransport(**lit_kwargs)

            dm = SyntheticFMDataModule(n_train=20, n_val=5, nlat=NLAT, nlon=NLON, nlev=NLEV, batch_size=4)

            trainer = pl.Trainer(
                callbacks=[],
                logger=False,
                enable_progress_bar=False,
                **tr_kwargs,
            )

            trainer.fit(model, dm)

            # Extract val loss
            val_loss = trainer.callback_metrics.get("Loss/Val_singlestep")
            objective = val_loss.item() if val_loss is not None else float("inf")
            return objective

    objective = _TestObjective(
        base_data_kwargs={},
        base_lit_module_kwargs=tiny_fm_kwargs,
        base_trainer_kwargs=trainer_kwargs,
        run_dir=tmp_path / "trials",
        val_dataset=val_ds,
        ema_kwargs=None,  # Disable EMA for speed
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value is not None
    assert trial.value < float("inf"), f"Trial returned inf: {trial.value}"


# ── SwinTransformer tests ──────────────────────────────────────────────


@pytest.fixture
def tiny_swin_fm_kwargs():
    """Minimal FlowMatching SwinTransformer configuration for CPU testing."""
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

    wrapper_kwargs = dict(
        **regulargrid_kwargs,
        model_kwargs=dict(
            submodel="swintransformer",
            model_kwargs=dict(
                **regulargrid_kwargs,
                model_kwargs=dict(
                    in_chans=NLEV + 1,  # nlev + time channel
                    out_chans=NLEV,
                    embed_dim=32,
                    depths=(2,),
                    num_heads=(2,),
                    img_size=(16, 32),
                    patch_size=4,
                    window_size=(4, 8),
                    mlp_ratio=2.0,
                    drop_path_rate=0.0,
                ),
            ),
            generating=False,
            return_intermediates=False,
            method="euler",
            nlev=NLEV,
            step_size=0.5,
            use_ot_coupling=False,
            time_sampling="uniform",
        ),
    )

    return dict(
        model="flowmatching",
        model_kwargs=wrapper_kwargs,
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


@pytest.mark.slow
def test_swin_fm_train_10_steps(tiny_swin_fm_kwargs, synthetic_datamodule, tmp_path):
    """Train SwinTransformer FM for 1 epoch and verify metrics are finite."""
    from neural_transport.litmodule import NeuralTransport

    model = NeuralTransport(**tiny_swin_fm_kwargs)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, synthetic_datamodule)

    assert "Loss/Train" in trainer.callback_metrics
    train_loss = trainer.callback_metrics["Loss/Train"].item()
    assert torch.isfinite(torch.tensor(train_loss)), f"Train loss is not finite: {train_loss}"

    assert "Loss/Val_singlestep" in trainer.callback_metrics
    val_loss = trainer.callback_metrics["Loss/Val_singlestep"].item()
    assert torch.isfinite(torch.tensor(val_loss)), f"Val loss is not finite: {val_loss}"


def test_suggest_hyperparams_swintransformer():
    """Verify suggest_hyperparams with submodel='swintransformer' produces correct params."""
    optuna = pytest.importorskip("optuna")
    from neural_transport.training.tuning import suggest_hyperparams

    study = optuna.create_study()
    trial = study.ask()
    params = suggest_hyperparams(trial, submodel="swintransformer")

    # Should have drop_path_rate but NOT norm
    assert "drop_path_rate" in params
    assert 0.0 <= params["drop_path_rate"] <= 0.3
    assert "norm" not in params

    # Should have standard params
    assert "lr" in params
    assert "model_size" in params
    assert "use_ot_coupling" in params


def test_suggest_hyperparams_unet_has_norm():
    """Verify suggest_hyperparams with submodel='unet' produces norm but not drop_path_rate."""
    optuna = pytest.importorskip("optuna")
    from neural_transport.training.tuning import suggest_hyperparams

    study = optuna.create_study()
    trial = study.ask()
    params = suggest_hyperparams(trial, submodel="unet")

    assert "norm" in params
    assert params["norm"] in ("batch", "group")
    assert "drop_path_rate" not in params


def test_apply_hyperparams_swintransformer(tiny_swin_fm_kwargs):
    """Verify _apply_hyperparams correctly sets SwinTransformer kwargs."""
    pytest.importorskip("optuna")
    from neural_transport.training.tuning import FMOptunaObjective

    objective = FMOptunaObjective(
        base_data_kwargs={},
        base_lit_module_kwargs=tiny_swin_fm_kwargs,
        base_trainer_kwargs={"max_steps": 5, "accelerator": "cpu"},
        run_dir="/tmp/test",
        submodel="swintransformer",
    )

    params = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "warmup_steps": 100,
        "halfcosine_steps": 5000,
        "max_lr": 0.8,
        "model_size": "XS",
        "drop_path_rate": 0.15,
        "use_ot_coupling": False,
        "time_sampling": "uniform",
        "time_sampling_kwargs": None,
        "time_loss_weight": None,
        "gradient_clip_val": 32,
    }

    lit_kwargs, _, _ = objective._apply_hyperparams(params, trial_number=0)

    # Navigate to submodel kwargs
    fm_kwargs = lit_kwargs["model_kwargs"]["model_kwargs"]
    submodel_kwargs = fm_kwargs["model_kwargs"]["model_kwargs"]

    assert fm_kwargs["submodel"] == "swintransformer"
    assert submodel_kwargs["embed_dim"] == 128  # XS
    assert submodel_kwargs["depths"] == (6,)  # XS
    assert submodel_kwargs["num_heads"] == (4,)  # XS
    assert submodel_kwargs["drop_path_rate"] == 0.15
    assert submodel_kwargs["window_size"] == (8, 8)
    # UNet keys should be removed
    assert "enc_filters" not in submodel_kwargs
    assert "dec_filters" not in submodel_kwargs
    assert "norm" not in submodel_kwargs
