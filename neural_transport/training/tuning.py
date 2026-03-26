"""Optuna-based hyperparameter tuning for Flow Matching models.

Provides a structured search over optimizer, scheduler, architecture, and
FM-specific training parameters (OT coupling, time sampling, loss weighting).
Uses PyTorch Lightning integration for pruning and metric reporting.

Usage:
    from neural_transport.training.tuning import run_optuna_study, get_best_config

    study = run_optuna_study(
        study_name="fm_tuning",
        storage="sqlite:///optuna_fm_study.db",
        base_data_kwargs=data_kwargs,
        base_lit_module_kwargs=lit_module_kwargs,
        base_trainer_kwargs=trainer_kwargs,
        run_dir=Path("./tuning_runs"),
    )
    best_config = get_best_config(study)
"""

from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl

from neural_transport.litmodule import NeuralTransport
from neural_transport.training.ema import EMACallback
from neural_transport.training.gen_eval_callback import GenerationQualityCallback

logger = logging.getLogger(__name__)

# ── Model size definitions ──────────────────────────────────────────────
# Simultaneously scale width (embed_dim) and depth (enc/dec stages).
# Approximate parameter counts for in_chans=11, out_chans=10:

MODEL_SIZES: dict[str, dict[str, Any]] = {
    "XXS": dict(
        embed_dim=32,
        enc_filters=[[3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3]],
    ),
    "XS": dict(
        embed_dim=64,
        enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
    ),
    "S": dict(
        embed_dim=96,
        enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
    ),
    "M": dict(
        embed_dim=192,
        enc_filters=[[7], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3]],
    ),
    "L": dict(
        embed_dim=256,
        enc_filters=[[7], [3, 3], [3, 3], [3, 3], [3, 3]],
        dec_filters=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
    ),
}


def suggest_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest FM training hyperparameters.

    Search space covers:
    - Optimizer: lr, weight_decay
    - LR schedule: warmup_steps, halfcosine_steps, max_lr
    - Architecture: model_size (XXS/XS/S/M/L)
    - FM training: use_ot_coupling, time_sampling, time_loss_weight
    - Training: gradient_clip_val

    Args:
        trial: Optuna trial object.

    Returns:
        Dict of suggested hyperparameters.
    """
    params: dict[str, Any] = {
        # Optimizer
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.5),
        # LR schedule
        "warmup_steps": trial.suggest_int("warmup_steps", 100, 5000),
        "halfcosine_steps": trial.suggest_int("halfcosine_steps", 3000, 50000),
        "max_lr": trial.suggest_float("max_lr", 0.3, 1.0),
        # Architecture
        "model_size": trial.suggest_categorical("model_size", list(MODEL_SIZES.keys())),
        # FM training
        "use_ot_coupling": trial.suggest_categorical("use_ot_coupling", [True, False]),
        "time_sampling": trial.suggest_categorical("time_sampling", ["uniform", "logit_normal", "beta"]),
        "time_loss_weight": trial.suggest_categorical("time_loss_weight", ["none", "snr", "sigma_inv"]),
        # Training
        "gradient_clip_val": trial.suggest_categorical("gradient_clip_val", [16, 32, 64]),
    }

    # Conditional params for time_sampling distributions
    if params["time_sampling"] == "logit_normal":
        params["time_sampling_kwargs"] = {
            "mean": trial.suggest_float("logit_normal_mean", -1.0, 1.0),
            "std": trial.suggest_float("logit_normal_std", 0.3, 2.0),
        }
    elif params["time_sampling"] == "beta":
        params["time_sampling_kwargs"] = {
            "a": trial.suggest_float("beta_a", 0.5, 5.0),
            "b": trial.suggest_float("beta_b", 0.5, 5.0),
        }
    else:
        params["time_sampling_kwargs"] = None

    # Convert "none" string to None for time_loss_weight
    if params["time_loss_weight"] == "none":
        params["time_loss_weight"] = None

    return params


class _OptunaPruningCallback(pl.Callback):
    """Manual Optuna pruning callback compatible with pytorch_lightning imports.

    Reports the monitored metric to Optuna after each validation epoch and
    raises ``optuna.exceptions.TrialPruned`` if the trial should be stopped.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "Loss/Val_singlestep"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self._epoch = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=self._epoch)
        self._epoch += 1
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Pruned at epoch {self._epoch}")


def _deep_copy_kwargs(kwargs: dict) -> dict:
    """Deep copy a kwargs dict, handling non-serializable values."""
    return copy.deepcopy(kwargs)


class FMOptunaObjective:
    """Optuna objective for FlowMatching hyperparameter search.

    Mirrors the setup in ``train_singlestep()`` but constructs the trainer
    directly to access callback metrics after training.

    Args:
        base_data_kwargs: Base data configuration (CarbonDataModule kwargs).
        base_lit_module_kwargs: Base model configuration (NeuralTransport kwargs).
        base_trainer_kwargs: Base trainer configuration (pl.Trainer kwargs).
        run_dir: Root directory for trial outputs.
        val_dataset: Pre-loaded validation dataset for GenerationQualityCallback.
            If None, the callback is not used.
        gen_eval_kwargs: Override kwargs for GenerationQualityCallback.
        ema_kwargs: EMA callback kwargs. Set to None to disable EMA.
        stability_penalty: Penalty weight for (1 - valid_fraction) in the
            composite objective. A model with valid_fraction=0.5 gets a penalty
            of stability_penalty * 0.5 added to its energy_distance.
    """

    def __init__(
        self,
        base_data_kwargs: dict,
        base_lit_module_kwargs: dict,
        base_trainer_kwargs: dict,
        run_dir: Path | str,
        val_dataset: Any = None,
        gen_eval_kwargs: dict | None = None,
        ema_kwargs: dict | None = None,
        shared_datamodule: Any = None,
        stability_penalty: float = 10.0,
    ):
        self.base_data_kwargs = base_data_kwargs
        self.base_lit_module_kwargs = base_lit_module_kwargs
        self.base_trainer_kwargs = base_trainer_kwargs
        self.run_dir = Path(run_dir)
        self.val_dataset = val_dataset
        self.gen_eval_kwargs = gen_eval_kwargs or {}
        self.ema_kwargs = ema_kwargs if ema_kwargs is not None else {"decay": 0.9999, "ema_start_step": 1000}
        self.shared_datamodule = shared_datamodule
        self.stability_penalty = stability_penalty

    def _apply_hyperparams(self, params: dict[str, Any], trial_number: int) -> tuple[dict, dict, dict]:
        """Merge suggested hyperparameters into base configs.

        Returns:
            Tuple of (lit_module_kwargs, data_kwargs, trainer_kwargs).
        """
        lit_kwargs = _deep_copy_kwargs(self.base_lit_module_kwargs)
        data_kwargs = _deep_copy_kwargs(self.base_data_kwargs)
        trainer_kwargs = _deep_copy_kwargs(self.base_trainer_kwargs)

        # Optimizer params
        lit_kwargs["lr"] = params["lr"]
        lit_kwargs["weight_decay"] = params["weight_decay"]

        # LR schedule
        if "lr_shedule_kwargs" not in lit_kwargs:
            lit_kwargs["lr_shedule_kwargs"] = {}
        lit_kwargs["lr_shedule_kwargs"]["warmup_steps"] = params["warmup_steps"]
        lit_kwargs["lr_shedule_kwargs"]["halfcosine_steps"] = params["halfcosine_steps"]
        lit_kwargs["lr_shedule_kwargs"]["max_lr"] = params["max_lr"]

        # Architecture — update model size in the nested model_kwargs
        model_size_config = MODEL_SIZES[params["model_size"]]
        wrapper_kwargs = lit_kwargs.get("model_kwargs", {})
        fm_kwargs = wrapper_kwargs.get("model_kwargs", {})
        inner_model_kwargs = fm_kwargs.get("model_kwargs", {})
        unet_kwargs = inner_model_kwargs.get("model_kwargs", {})

        unet_kwargs["embed_dim"] = model_size_config["embed_dim"]
        unet_kwargs["enc_filters"] = model_size_config["enc_filters"]
        unet_kwargs["dec_filters"] = model_size_config["dec_filters"]

        # FM training params
        fm_kwargs["use_ot_coupling"] = params["use_ot_coupling"]
        fm_kwargs["time_sampling"] = params["time_sampling"]
        fm_kwargs["time_sampling_kwargs"] = params["time_sampling_kwargs"]
        fm_kwargs["time_loss_weight"] = params["time_loss_weight"]

        # Loss function depends on time_loss_weight
        if params["time_loss_weight"] is not None:
            lit_kwargs["loss"] = "flowmatching_weighted_mse"
        else:
            lit_kwargs["loss"] = "flowmatching_mse"

        # Trainer params
        trainer_kwargs["gradient_clip_val"] = params["gradient_clip_val"]

        return lit_kwargs, data_kwargs, trainer_kwargs

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single Optuna trial.

        Args:
            trial: Optuna trial.

        Returns:
            Objective value (GenEval/energy_distance or Loss/Val_singlestep).
        """
        params = suggest_hyperparams(trial)
        lit_kwargs, data_kwargs, trainer_kwargs = self._apply_hyperparams(params, trial.number)

        trial_dir = self.run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save trial config
        trial_config = {"trial_number": trial.number, "params": _make_json_safe(params)}
        with open(trial_dir / "trial_config.json", "w") as f:
            json.dump(trial_config, f, indent=2)

        logger.info("Trial %d: %s", trial.number, params)

        try:
            # Build model; reuse shared datamodule to avoid reloading ~40GB per trial
            from neural_transport.datamodule import CarbonDataModule, PreBatchedShuffleCallback

            model = NeuralTransport(**lit_kwargs)
            if self.shared_datamodule is not None:
                dset = self.shared_datamodule
            else:
                dset = CarbonDataModule(**data_kwargs)

            # Build callbacks
            tb_logger = pl.loggers.tensorboard.TensorBoardLogger(trial_dir, name="", version="singlestep")

            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                save_top_k=1,
                save_last=True,
                monitor="Loss/Val_singlestep",
                filename="Epoch={epoch}-Step={step}-LossVal={Loss/Val_singlestep:.6f}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
            )
            lr_monitor = pl.callbacks.LearningRateMonitor()

            callbacks = [checkpoint_callback, lr_monitor]

            # Manual pruning callback (avoids lightning/pytorch_lightning import mismatch)
            callbacks.append(_OptunaPruningCallback(trial, monitor="Loss/Val_singlestep"))

            # EMA
            if self.ema_kwargs:
                callbacks.append(EMACallback(**self.ema_kwargs))

            # Generation quality callback
            if self.val_dataset is not None:
                gen_eval_kwargs = {
                    "val_dataset": self.val_dataset,
                    "eval_every_n_epochs": 1,
                    "n_gt_samples": 5,
                    "n_gen_samples": 20,
                    **self.gen_eval_kwargs,
                }
                callbacks.append(GenerationQualityCallback(**gen_eval_kwargs))

            # PreBatchedShuffleCallback if needed
            if data_kwargs.get("compute", False):
                callbacks.append(PreBatchedShuffleCallback())

            # Remove logger from trainer_kwargs if present (we set our own)
            trainer_kwargs.pop("logger", None)

            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=tb_logger,
                enable_progress_bar=False,
                **trainer_kwargs,
            )

            trainer.fit(model, dset)

            # Extract metrics
            metrics = {}
            for key in trainer.callback_metrics:
                val = trainer.callback_metrics[key]
                if hasattr(val, "item"):
                    metrics[key] = val.item()
                else:
                    metrics[key] = float(val)

            # Save trial results
            trial_info = {
                "trial_number": trial.number,
                "params": _make_json_safe(params),
                "metrics": metrics,
                "status": "complete",
            }
            with open(trial_dir / "trial_info.json", "w") as f:
                json.dump(trial_info, f, indent=2)

            # Return composite objective: energy_distance + penalty * (1 - valid_fraction)
            valid_fraction = metrics.get("GenEval/valid_fraction", 0.0)
            e_dist = metrics.get("GenEval/energy_distance", float("inf"))

            if valid_fraction > 0 and math.isfinite(e_dist):
                objective = e_dist + self.stability_penalty * (1.0 - valid_fraction)
            elif "Loss/Val_singlestep" in metrics:
                # Fallback: no valid generations, use val loss with full penalty
                objective = metrics["Loss/Val_singlestep"] + self.stability_penalty
            else:
                objective = float("inf")

            logger.info("Trial %d complete: objective=%.6f", trial.number, objective)
            return objective

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e, exc_info=True)
            # Save failure info
            with open(trial_dir / "trial_info.json", "w") as f:
                json.dump(
                    {
                        "trial_number": trial.number,
                        "params": _make_json_safe(params),
                        "status": "failed",
                        "error": str(e),
                    },
                    f,
                    indent=2,
                )
            return float("inf")


def run_optuna_study(
    study_name: str,
    storage: str,
    base_data_kwargs: dict,
    base_lit_module_kwargs: dict,
    base_trainer_kwargs: dict,
    run_dir: Path | str,
    n_trials: int = 200,
    pruner_n_startup: int = 10,
    pruner_n_warmup: int = 5,
    val_dataset: Any = None,
    gen_eval_kwargs: dict | None = None,
    ema_kwargs: dict | None = None,
    seed: int = 42,
    shared_datamodule: Any = None,
    stability_penalty: float = 10.0,
) -> optuna.Study:
    """Create and run an Optuna study for FM hyperparameter tuning.

    Args:
        study_name: Name for the Optuna study.
        storage: Storage URL (e.g. "sqlite:///study.db").
        base_data_kwargs: Base CarbonDataModule kwargs.
        base_lit_module_kwargs: Base NeuralTransport kwargs.
        base_trainer_kwargs: Base pl.Trainer kwargs.
        run_dir: Root directory for trial outputs.
        n_trials: Number of trials to run.
        pruner_n_startup: Trials before pruning starts.
        pruner_n_warmup: Validation epochs before pruning in each trial.
        val_dataset: Pre-loaded validation dataset for GenEval callback.
        gen_eval_kwargs: Override kwargs for GenerationQualityCallback.
        ema_kwargs: EMA callback kwargs. None to disable.
        seed: Random seed for reproducibility.
        shared_datamodule: Pre-loaded CarbonDataModule to reuse across trials.
            Avoids reloading ~40GB of data per trial.
        stability_penalty: Penalty weight for (1 - valid_fraction) in composite objective.

    Returns:
        Completed Optuna Study.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_n_startup,
        n_warmup_steps=pruner_n_warmup,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    objective = FMOptunaObjective(
        base_data_kwargs=base_data_kwargs,
        base_lit_module_kwargs=base_lit_module_kwargs,
        base_trainer_kwargs=base_trainer_kwargs,
        run_dir=run_dir,
        val_dataset=val_dataset,
        gen_eval_kwargs=gen_eval_kwargs,
        ema_kwargs=ema_kwargs,
        shared_datamodule=shared_datamodule,
        stability_penalty=stability_penalty,
    )

    study.optimize(objective, n_trials=n_trials)

    # Export results
    export_study_results(study, run_dir / "study_results")

    return study


def get_best_config(study: optuna.Study) -> dict[str, Any]:
    """Extract the best trial's hyperparameters.

    Args:
        study: Completed Optuna study.

    Returns:
        Dict with the best hyperparameters (same format as suggest_hyperparams output).
    """
    best = study.best_params.copy()

    # Reconstruct conditional params
    if best.get("time_sampling") == "logit_normal":
        best["time_sampling_kwargs"] = {
            "mean": best.pop("logit_normal_mean", 0.0),
            "std": best.pop("logit_normal_std", 1.0),
        }
    elif best.get("time_sampling") == "beta":
        best["time_sampling_kwargs"] = {
            "a": best.pop("beta_a", 1.0),
            "b": best.pop("beta_b", 1.0),
        }
    else:
        best["time_sampling_kwargs"] = None
        # Clean up stale conditional params
        best.pop("logit_normal_mean", None)
        best.pop("logit_normal_std", None)
        best.pop("beta_a", None)
        best.pop("beta_b", None)

    # Convert "none" to None
    if best.get("time_loss_weight") == "none":
        best["time_loss_weight"] = None

    best["best_value"] = study.best_value
    best["best_trial_number"] = study.best_trial.number

    return best


def export_study_results(study: optuna.Study, out_dir: Path | str) -> Path:
    """Export study results to CSV and JSON.

    Args:
        study: Optuna study.
        out_dir: Output directory.

    Returns:
        Path to the output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame export
    df = study.trials_dataframe()
    df.to_csv(out_dir / "trials.csv", index=False)

    # Best config
    best = get_best_config(study)
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best, f, indent=2, default=str)

    # Study summary
    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "best_params": _make_json_safe(best),
    }
    with open(out_dir / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Study results exported to %s", out_dir)
    return out_dir


def _make_json_safe(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
