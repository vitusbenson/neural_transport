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
