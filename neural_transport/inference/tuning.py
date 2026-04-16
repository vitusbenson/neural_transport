"""Optuna-based hyperparameter tuning for posterior sampling methods.

Provides per-method search spaces and an objective function that evaluates
posterior conditioning quality via pressure-weighted RMSE of ensemble mean
vs ground truth, averaged over multiple target samples.

Usage:
    from neural_transport.inference.tuning import run_posterior_study

    study = run_posterior_study(
        method="flowdps",
        model=model,
        dataset=dataset,
        grid_info=grid_info,
        study_name="flowdps_tuning",
        storage="sqlite:///posterior_tuning.db",
        run_dir=Path("./tuning_runs"),
    )
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from neural_transport.configs import GenerateConfig, compat_to_generate_kwargs

logger = logging.getLogger(__name__)


# ── Pressure weights ─────────────────────────────────────────────────────


def compute_pressure_weights(levels: np.ndarray) -> np.ndarray:
    """Compute normalized layer thickness weights from pressure levels.

    Uses centered finite differences at interior levels and one-sided
    differences at boundaries. Weights are normalized to sum to 1.

    Parameters
    ----------
    levels : np.ndarray
        Pressure levels in hPa, ordered from surface (high) to top (low).
        E.g. [1013, 1005, 995, 971, 943, 843, 642, 441, 243, 73].

    Returns
    -------
    np.ndarray
        Normalized weights with same shape as levels, summing to 1.
    """
    levels = np.asarray(levels, dtype=np.float64)
    n = len(levels)
    if n == 1:
        return np.array([1.0])

    dp = np.zeros(n)
    # Boundaries: half-interval
    dp[0] = abs(levels[0] - levels[1]) / 2.0
    dp[-1] = abs(levels[-2] - levels[-1]) / 2.0
    # Interior: centered differences
    for i in range(1, n - 1):
        dp[i] = abs(levels[i - 1] - levels[i + 1]) / 2.0

    return dp / dp.sum()


def pressure_weighted_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    pressure_weights: np.ndarray,
    cos_lat_weights: np.ndarray | None = None,
) -> float:
    """Compute pressure-weighted (and optionally lat-weighted) RMSE.

    Parameters
    ----------
    pred : np.ndarray
        Prediction, shape [nlat, nlon, nlev].
    target : np.ndarray
        Ground truth, same shape as pred.
    pressure_weights : np.ndarray
        Per-level weights [nlev], summing to 1.
    cos_lat_weights : np.ndarray or None
        Per-latitude weights [nlat], will be normalized internally.

    Returns
    -------
    float
        Weighted RMSE scalar.
    """
    diff_sq = (pred - target) ** 2  # [nlat, nlon, nlev]

    # Pressure weighting over levels
    pw = pressure_weights / pressure_weights.sum()
    weighted = (diff_sq * pw[np.newaxis, np.newaxis, :]).sum(axis=2)  # [nlat, nlon]

    # Latitude weighting
    if cos_lat_weights is not None:
        clw = cos_lat_weights / cos_lat_weights.sum()
        weighted = (weighted * clw[:, np.newaxis]).sum(axis=0).mean()  # scalar
    else:
        weighted = weighted.mean()

    return float(np.sqrt(weighted))


# ── Per-method search spaces ─────────────────────────────────────────────


def suggest_dps_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest DPS guidance hyperparameters."""
    params: dict[str, Any] = {
        "sampler": None,
        "conditioning_mode": "guidance",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.05, 5.0, log=True),
        "guidance_scale": trial.suggest_float("guidance_scale", 0.1, 10.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
    }
    masking_time = trial.suggest_categorical("masking_time", ["none", "smooth_late_masking"])
    if masking_time == "none":
        params["masking_time"] = None
    else:
        params["masking_time"] = masking_time
        params["t_threshold"] = trial.suggest_float("t_threshold", 0.5, 0.95)
    return params


def suggest_flowdps_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest FlowDPS hyperparameters."""
    return {
        "sampler": "flowdps",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.005, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "fresh_noise": trial.suggest_categorical("fresh_noise", [True, False]),
        "steps": trial.suggest_categorical("steps", [11, 21, 51]),
    }


def suggest_sde_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest SDE posterior sampler hyperparameters."""
    params: dict[str, Any] = {
        "sampler": "sde",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.01, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "sigma_max": trial.suggest_float("sigma_max", 0.05, 2.0, log=True),
        "noise_schedule": trial.suggest_categorical("noise_schedule", ["annealed", "constant", "cosine"]),
        "use_projection": trial.suggest_categorical("use_projection", [True, False]),
    }
    n_corrector = trial.suggest_int("n_corrector_steps", 0, 5)
    params["n_corrector_steps"] = n_corrector
    if n_corrector > 0:
        params["corrector_step_size"] = trial.suggest_float("corrector_step_size", 0.001, 0.1, log=True)
    else:
        params["corrector_step_size"] = 0.01
    return params


def suggest_fig_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest FIG hyperparameters."""
    return {
        "sampler": "fig",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.01, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "step_size_c": trial.suggest_float("step_size_c", 0.5, 100.0, log=True),
        "k_steps": trial.suggest_int("k_steps", 1, 10),
        "noise_scale_w": trial.suggest_float("noise_scale_w", 0.0, 2.0),
        "skip_first_last": trial.suggest_categorical("skip_first_last", [True, False]),
    }


def suggest_ictm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest ICTM hyperparameters."""
    return {
        "sampler": "ictm",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.01, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "r_max": trial.suggest_float("r_max", 0.05, 5.0, log=True),
        "r_schedule": trial.suggest_categorical("r_schedule", ["constant", "decreasing", "increasing", "cosine"]),
        "n_inner_steps": trial.suggest_int("n_inner_steps", 1, 10),
        "inner_lr": trial.suggest_float("inner_lr", 0.005, 1.0, log=True),
    }


def suggest_mcg_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest MCG (Manifold Constrained Gradient) hyperparameters."""
    return {
        "sampler": "mcg",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.005, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "fresh_noise": trial.suggest_categorical("fresh_noise", [True, False]),
        "n_forward_steps": trial.suggest_int("n_forward_steps", 1, 5),
    }


def suggest_pcfm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest PCFM (Physics-Constrained Flow Matching) hyperparameters."""
    return {
        "sampler": "pcfm",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.005, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "fresh_noise": trial.suggest_categorical("fresh_noise", [True, False]),
        "n_forward_steps": trial.suggest_int("n_forward_steps", 1, 5),
        "lambda_penalty": trial.suggest_float("lambda_penalty", 0.1, 1.0),
    }


def suggest_fmps_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest FMPS (Flow Matching Posterior Sampling) hyperparameters."""
    params: dict[str, Any] = {
        "sampler": "fmps",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.01, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "guidance_strength": trial.suggest_float("guidance_strength", 0.1, 50.0, log=True),
        "r_schedule": trial.suggest_categorical("r_schedule", ["linear", "cosine", "constant"]),
        "svd_rank": trial.suggest_int("svd_rank", 0, 10),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.1, 10.0, log=True),
    }
    use_spectral = trial.suggest_categorical("use_spectral_filter", [True, False])
    if use_spectral:
        params["spectral_k_low"] = trial.suggest_int("spectral_k_low", 2, 8)
        params["spectral_k_high"] = trial.suggest_int("spectral_k_high", 8, 16)
    else:
        params["spectral_k_low"] = 0
        params["spectral_k_high"] = 0
    return params


def suggest_dflow_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest D-Flow (source optimization) hyperparameters."""
    return {
        "sampler": "dflow",
        "sigma_obs": trial.suggest_float("sigma_obs", 0.01, 2.0, log=True),
        "spatial_smoothing_sigma": trial.suggest_float("spatial_smoothing_sigma", 0.0, 5.0),
        "soft_boundary_sigma": trial.suggest_float("soft_boundary_sigma", 0.0, 3.0),
        "n_opt_steps": trial.suggest_int("n_opt_steps", 10, 200),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "reg_weight": trial.suggest_float("reg_weight", 1e-3, 10.0, log=True),
        "reg_type": trial.suggest_categorical("reg_type", ["l2", "norm_diff", "chi_prior"]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "lbfgs"]),
        "use_checkpointing": trial.suggest_categorical("use_checkpointing", [True, False]),
    }


SUGGEST_FUNCS: dict[str, Any] = {
    "dps": suggest_dps_params,
    "flowdps": suggest_flowdps_params,
    "sde": suggest_sde_params,
    "fig": suggest_fig_params,
    "ictm": suggest_ictm_params,
    "mcg": suggest_mcg_params,
    "pcfm": suggest_pcfm_params,
    "fmps": suggest_fmps_params,
    "dflow": suggest_dflow_params,
}

METHODS = list(SUGGEST_FUNCS.keys())


# ── Objective function ───────────────────────────────────────────────────


def _build_generate_config(
    method_params: dict[str, Any],
    base_config: GenerateConfig,
) -> GenerateConfig:
    """Merge suggested method params into a base GenerateConfig.

    Distributes keys to the correct nested dataclass (sampler_params,
    conditioning, or top-level).
    """
    from neural_transport.configs import ConditioningConfig, SamplerParams

    overrides = {}
    sampler_overrides = {}
    conditioning_overrides = {}

    sampler_fields = {f.name for f in dataclasses.fields(SamplerParams)}
    conditioning_fields = {f.name for f in dataclasses.fields(ConditioningConfig)}
    top_fields = {f.name for f in dataclasses.fields(GenerateConfig)}

    for k, v in method_params.items():
        if k in sampler_fields:
            sampler_overrides[k] = v
        elif k in conditioning_fields:
            conditioning_overrides[k] = v
        elif k in top_fields:
            overrides[k] = v
        else:
            # Unknown key — try sampler_params as fallback
            sampler_overrides[k] = v

    # Build merged config via dotted-key merge
    dotted: dict[str, Any] = {}
    for k, v in overrides.items():
        dotted[k] = v
    for k, v in sampler_overrides.items():
        dotted[f"sampler_params.{k}"] = v
    for k, v in conditioning_overrides.items():
        dotted[f"conditioning.{k}"] = v

    return base_config.merge(**dotted)


class PosteriorSamplerObjective:
    """Optuna objective for posterior sampler hyperparameter search.

    Evaluates conditioning quality by generating ensemble members for
    multiple target samples and computing pressure-weighted RMSE of the
    ensemble mean vs the ground truth.

    Parameters
    ----------
    model : NeuralTransport
        Pre-loaded FM model (shared across trials).
    dataset : CarbonDataset
        Test dataset providing target samples.
    method : str
        Sampler method name: "dps", "flowdps", "sde", "fig", "ictm".
    base_config : GenerateConfig
        Base generation config with mask/conditioning settings.
        Method-specific params will be overridden per trial.
    grid_info : GridInfo
        Grid metadata (lat, lon, levels, nlat, nlon).
    target_vars : list[str]
        Target variable names (e.g. ["co2massmix"]).
    n_targets : int
        Number of target samples to evaluate per trial.
    n_samples : int
        Number of ensemble members per target.
    nan_penalty : float
        Penalty weight for NaN fraction in objective.
    run_dir : Path
        Output directory for trial info files.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        method: str,
        base_config: GenerateConfig,
        grid_info: Any,
        target_vars: list[str] | None = None,
        n_targets: int = 10,
        n_samples: int = 20,
        nan_penalty: float = 100.0,
        run_dir: Path | str = "tuning_runs",
        device: str = "cuda",
    ):
        if method not in SUGGEST_FUNCS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")

        self.model = model
        self.dataset = dataset
        self.method = method
        self.base_config = base_config
        self.grid_info = grid_info
        self.target_vars = target_vars or ["co2massmix"]
        self.n_targets = min(n_targets, len(dataset))
        self.n_samples = n_samples
        self.nan_penalty = nan_penalty
        self.run_dir = Path(run_dir)
        self.device = device

        # Pre-select diverse target indices (fixed across trials for fair comparison)
        rng = np.random.RandomState(42)
        self.target_indices = sorted(rng.choice(len(dataset), size=self.n_targets, replace=False).tolist())
        logger.info("Evaluation targets (dataset indices): %s", self.target_indices)

        # Precompute weights
        self.pressure_weights = compute_pressure_weights(grid_info.levels)
        self.cos_lat_weights = np.cos(np.deg2rad(grid_info.lat))

    def __call__(self, trial: optuna.Trial) -> float:
        """Run a single Optuna trial."""
        suggest_fn = SUGGEST_FUNCS[self.method]
        method_params = suggest_fn(trial)

        config = _build_generate_config(method_params, self.base_config)
        generate_kwargs = compat_to_generate_kwargs(config)

        trial_dir = self.run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save trial config
        trial_config = {"trial_number": trial.number, "method": self.method, "params": method_params}
        with open(trial_dir / "trial_config.json", "w") as f:
            json.dump(trial_config, f, indent=2, default=str)

        logger.info("Trial %d [%s]: %s", trial.number, self.method, method_params)

        try:
            rmses = []
            total_samples = 0
            nan_samples = 0

            for dataset_idx in self.target_indices:
                rmse, n_valid, n_total = self._evaluate_target(dataset_idx, generate_kwargs, trial_dir)
                if rmse is not None:
                    rmses.append(rmse)
                total_samples += n_total
                nan_samples += n_total - n_valid

            nan_fraction = nan_samples / max(total_samples, 1)

            if rmses:
                mean_rmse = float(np.mean(rmses))
            else:
                mean_rmse = float("inf")

            objective = mean_rmse + self.nan_penalty * nan_fraction

            trial_info = {
                "trial_number": trial.number,
                "method": self.method,
                "params": method_params,
                "mean_rmse": mean_rmse,
                "nan_fraction": nan_fraction,
                "objective": objective,
                "n_targets_evaluated": len(rmses),
                "status": "complete",
            }
            with open(trial_dir / "trial_info.json", "w") as f:
                json.dump(trial_info, f, indent=2, default=str)

            logger.info(
                "Trial %d complete: rmse=%.4f, nan_frac=%.3f, objective=%.4f",
                trial.number,
                mean_rmse,
                nan_fraction,
                objective,
            )
            return objective

        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e, exc_info=True)
            with open(trial_dir / "trial_info.json", "w") as f:
                json.dump(
                    {
                        "trial_number": trial.number,
                        "method": self.method,
                        "params": method_params,
                        "status": "failed",
                        "error": str(e),
                    },
                    f,
                    indent=2,
                    default=str,
                )
            return float("inf")

    def _evaluate_target(
        self,
        dataset_idx: int,
        generate_kwargs: dict,
        trial_dir: Path,
    ) -> tuple[float | None, int, int]:
        """Generate ensemble for one target and compute pw-RMSE.

        Uses generate_multi_target (which correctly handles per-target
        conditioning) on a single target index.

        Args:
            dataset_idx: Index into self.dataset for this target.

        Returns (rmse_or_None, n_valid_samples, n_total_samples).
        """
        from neural_transport.inference.generation import generate_multi_target

        target_var = self.target_vars[0]

        with tempfile.TemporaryDirectory(prefix=f"trial{trial_dir.name}_t{dataset_idx}_") as tmp:
            zarr_path = generate_multi_target(
                model=self.model,
                dataset=self.dataset,
                target_indices=[dataset_idx],
                n_samples_per_target=self.n_samples,
                batch_size=self.n_samples,
                generate_kwargs=generate_kwargs,
                out_dir=tmp,
                device=self.device,
                target_var=target_var,
                verbose=False,
                seed=dataset_idx,  # different seed per target for diversity
            )

            import zarr as zarr_lib

            store = zarr_lib.open_group(str(zarr_path), mode="r")
            samples_np = np.array(store["predictions"])[0]  # [n_samples, nlat, nlon, nlev]
            gt_np = np.array(store["gt"])[0]  # [nlat, nlon, nlev]

        n_total = samples_np.shape[0]

        # Check for NaN/Inf
        valid_mask = np.isfinite(samples_np).all(axis=(1, 2, 3))
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            return None, n_valid, n_total

        # Ensemble mean of valid samples
        ensemble_mean = samples_np[valid_mask].mean(axis=0)  # [nlat, nlon, nlev]

        rmse = pressure_weighted_rmse(
            ensemble_mean,
            gt_np,
            self.pressure_weights,
            self.cos_lat_weights,
        )

        return rmse, n_valid, n_total


# ── Study runner ─────────────────────────────────────────────────────────


def run_posterior_study(
    method: str,
    model: Any,
    dataset: Any,
    grid_info: Any,
    base_config: GenerateConfig,
    study_name: str,
    storage: str,
    run_dir: Path | str,
    *,
    target_vars: list[str] | None = None,
    n_trials: int = 50,
    n_targets: int = 10,
    n_samples: int = 20,
    nan_penalty: float = 100.0,
    device: str = "cuda",
    seed: int = 42,
) -> optuna.Study:
    """Create and run an Optuna study for one posterior sampling method.

    Parameters
    ----------
    method : str
        Sampler method: "dps", "flowdps", "sde", "fig", "ictm".
    model : NeuralTransport
        Pre-loaded FM model.
    dataset : CarbonDataset
        Test dataset.
    grid_info : GridInfo
        Grid metadata.
    base_config : GenerateConfig
        Base config with mask/conditioning settings.
    study_name : str
        Optuna study name.
    storage : str
        Optuna storage URL (e.g. "sqlite:///study.db").
    run_dir : Path
        Root directory for trial outputs.
    target_vars : list[str], optional
        Target variable names. Default: ["co2massmix"].
    n_trials : int
        Number of trials.
    n_targets : int
        Target samples per trial.
    n_samples : int
        Ensemble members per target.
    nan_penalty : float
        Penalty for NaN fraction.
    device : str
        Torch device.
    seed : int
        Random seed.

    Returns
    -------
    optuna.Study
        Completed study.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed)

    # Retry study creation with random backoff for SQLite race conditions
    import random
    import time

    for attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                sampler=sampler,
                load_if_exists=True,
            )
            break
        except Exception as e:
            if attempt == 9:
                raise
            wait = random.uniform(1, 5 * (attempt + 1))
            logger.warning("Study creation attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)

    objective = PosteriorSamplerObjective(
        model=model,
        dataset=dataset,
        method=method,
        base_config=base_config,
        grid_info=grid_info,
        target_vars=target_vars,
        n_targets=n_targets,
        n_samples=n_samples,
        nan_penalty=nan_penalty,
        run_dir=run_dir,
        device=device,
    )

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    # Save study summary
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(
        "Study '%s' done: %d trials (%d complete, %d pruned, %d failed)",
        study_name,
        len(study.trials),
        n_complete,
        len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    )

    if n_complete > 0:
        best = study.best_trial
        logger.info("Best trial %d: objective=%.4f, params=%s", best.number, best.value, best.params)

        summary = {
            "study_name": study_name,
            "method": method,
            "n_trials": len(study.trials),
            "n_complete": n_complete,
            "best_trial": best.number,
            "best_objective": best.value,
            "best_params": best.params,
        }
        with open(run_dir / "study_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return study


def get_best_config(
    study: optuna.Study,
    method: str,
    base_config: GenerateConfig,
) -> GenerateConfig:
    """Extract the best GenerateConfig from a completed study.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    method : str
        Method name (needed to reconstruct suggest params).
    base_config : GenerateConfig
        Base config to merge best params into.

    Returns
    -------
    GenerateConfig
        Config with best hyperparameters applied.
    """
    best_params = study.best_params
    method_params = _reconstruct_method_params(method, best_params)

    return _build_generate_config(method_params, base_config)


def _reconstruct_method_params(method: str, optuna_params: dict) -> dict[str, Any]:
    """Reconstruct structured method params from flat Optuna best_params.

    Maps Optuna's flat param names back to the method_params dict that
    _build_generate_config expects.
    """
    p = dict(optuna_params)

    if method == "dps":
        result: dict[str, Any] = {
            "sampler": None,
            "conditioning_mode": "guidance",
            "sigma_obs": p["sigma_obs"],
            "guidance_scale": p["guidance_scale"],
            "spatial_smoothing_sigma": p["spatial_smoothing_sigma"],
        }
        masking_time = p.get("masking_time", "none")
        if masking_time == "none":
            result["masking_time"] = None
        else:
            result["masking_time"] = masking_time
            result["t_threshold"] = p.get("t_threshold", 0.9)
        return result

    elif method == "flowdps":
        return {
            "sampler": "flowdps",
            "sigma_obs": p["sigma_obs"],
            "spatial_smoothing_sigma": p["spatial_smoothing_sigma"],
            "fresh_noise": p["fresh_noise"],
            "steps": p["steps"],
        }

    elif method == "sde":
        result = {
            "sampler": "sde",
            "sigma_obs": p["sigma_obs"],
            "sigma_max": p["sigma_max"],
            "noise_schedule": p["noise_schedule"],
            "use_projection": p["use_projection"],
            "n_corrector_steps": p["n_corrector_steps"],
            "corrector_step_size": p.get("corrector_step_size", 0.01),
        }
        return result

    elif method == "fig":
        return {
            "sampler": "fig",
            "sigma_obs": p["sigma_obs"],
            "step_size_c": p["step_size_c"],
            "k_steps": p["k_steps"],
            "noise_scale_w": p["noise_scale_w"],
            "skip_first_last": p["skip_first_last"],
        }

    elif method == "ictm":
        return {
            "sampler": "ictm",
            "sigma_obs": p["sigma_obs"],
            "r_max": p["r_max"],
            "r_schedule": p["r_schedule"],
            "n_inner_steps": p["n_inner_steps"],
            "inner_lr": p["inner_lr"],
        }

    elif method == "mcg":
        return {
            "sampler": "mcg",
            "sigma_obs": p["sigma_obs"],
            "spatial_smoothing_sigma": p.get("spatial_smoothing_sigma", 0.0),
            "soft_boundary_sigma": p.get("soft_boundary_sigma", 0.0),
            "fresh_noise": p.get("fresh_noise", True),
            "n_forward_steps": p.get("n_forward_steps", 1),
        }

    elif method == "pcfm":
        return {
            "sampler": "pcfm",
            "sigma_obs": p["sigma_obs"],
            "spatial_smoothing_sigma": p.get("spatial_smoothing_sigma", 0.0),
            "soft_boundary_sigma": p.get("soft_boundary_sigma", 0.0),
            "fresh_noise": p.get("fresh_noise", True),
            "n_forward_steps": p.get("n_forward_steps", 1),
            "lambda_penalty": p.get("lambda_penalty", 1.0),
        }

    elif method == "fmps":
        result = {
            "sampler": "fmps",
            "sigma_obs": p["sigma_obs"],
            "spatial_smoothing_sigma": p.get("spatial_smoothing_sigma", 0.0),
            "soft_boundary_sigma": p.get("soft_boundary_sigma", 0.0),
            "guidance_strength": p.get("guidance_strength", 1.0),
            "r_schedule": p.get("r_schedule", "linear"),
            "svd_rank": p.get("svd_rank", 0),
            "grad_clip_norm": p.get("grad_clip_norm", 1.0),
            "spectral_k_low": p.get("spectral_k_low", 0),
            "spectral_k_high": p.get("spectral_k_high", 0),
        }
        return result

    elif method == "dflow":
        return {
            "sampler": "dflow",
            "sigma_obs": p["sigma_obs"],
            "spatial_smoothing_sigma": p.get("spatial_smoothing_sigma", 0.0),
            "soft_boundary_sigma": p.get("soft_boundary_sigma", 0.0),
            "n_opt_steps": p.get("n_opt_steps", 50),
            "lr": p.get("lr", 1e-2),
            "reg_weight": p.get("reg_weight", 1.0),
            "reg_type": p.get("reg_type", "l2"),
            "optimizer": p.get("optimizer", "adam"),
            "use_checkpointing": p.get("use_checkpointing", False),
        }

    else:
        raise ValueError(f"Unknown method: {method}")


# ── Multi-objective tuning (NSGA-II) ───────────────────────────────────


class MultiObjectivePosteriorObjective:
    """NSGA-II multi-objective for posterior sampler tuning.

    Returns 3 objectives (all minimize):
    - pw_rmse_away: pressure-weighted RMSE at unobserved locations
    - obs_residual_xco2: column XCO2 RMSE at observed locations
    - roughness: spatial roughness of column XCO2 ensemble mean

    Parameters
    ----------
    model, dataset, method, base_config, grid_info, target_vars,
    n_targets, n_samples, run_dir, device: same as PosteriorSamplerObjective.
    nan_penalty : float
        Additive penalty per objective for NaN fraction.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        method: str,
        base_config: GenerateConfig,
        grid_info: Any,
        target_vars: list[str] | None = None,
        n_targets: int = 10,
        n_samples: int = 10,
        nan_penalty: float = 100.0,
        run_dir: Path | str = "tuning_runs",
        device: str = "cuda",
    ):
        if method not in SUGGEST_FUNCS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")

        self.model = model
        self.dataset = dataset
        self.method = method
        self.base_config = base_config
        self.grid_info = grid_info
        self.target_vars = target_vars or ["co2massmix"]
        self.n_targets = min(n_targets, len(dataset))
        self.n_samples = n_samples
        self.nan_penalty = nan_penalty
        self.run_dir = Path(run_dir)
        self.device = device

        # Pre-select diverse target indices (fixed across trials)
        rng = np.random.RandomState(42)
        self.target_indices = sorted(rng.choice(len(dataset), size=self.n_targets, replace=False).tolist())

        self.pressure_weights = compute_pressure_weights(grid_info.levels)
        self.cos_lat_weights = np.cos(np.deg2rad(grid_info.lat))

    def __call__(self, trial: optuna.Trial) -> tuple[float, float, float]:
        """Run a single Optuna trial. Returns (rmse_away, obs_residual, roughness)."""
        suggest_fn = SUGGEST_FUNCS[self.method]
        method_params = suggest_fn(trial)

        config = _build_generate_config(method_params, self.base_config)
        generate_kwargs = compat_to_generate_kwargs(config)

        trial_dir = self.run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_config = {"trial_number": trial.number, "method": self.method, "params": method_params}
        with open(trial_dir / "trial_config.json", "w") as f:
            json.dump(trial_config, f, indent=2, default=str)

        logger.info("Trial %d [%s]: %s", trial.number, self.method, method_params)

        try:
            rmses_away = []
            obs_residuals = []
            roughness_values = []
            total_samples = 0
            nan_samples = 0

            for dataset_idx in self.target_indices:
                result = self._evaluate_target(dataset_idx, generate_kwargs, trial_dir)
                total_samples += result["n_total"]
                nan_samples += result["n_total"] - result["n_valid"]
                if result["rmse_away"] is not None:
                    rmses_away.append(result["rmse_away"])
                if result["obs_residual"] is not None:
                    obs_residuals.append(result["obs_residual"])
                if result["roughness"] is not None:
                    roughness_values.append(result["roughness"])

            nan_fraction = nan_samples / max(total_samples, 1)
            penalty = self.nan_penalty * nan_fraction

            obj_rmse_away = (float(np.mean(rmses_away)) if rmses_away else float("inf")) + penalty
            obj_obs_residual = (float(np.mean(obs_residuals)) if obs_residuals else float("inf")) + penalty
            obj_roughness = (float(np.mean(roughness_values)) if roughness_values else float("inf")) + penalty

            trial_info = {
                "trial_number": trial.number,
                "method": self.method,
                "params": method_params,
                "rmse_away": obj_rmse_away,
                "obs_residual": obj_obs_residual,
                "roughness": obj_roughness,
                "nan_fraction": nan_fraction,
                "status": "complete",
            }
            with open(trial_dir / "trial_info.json", "w") as f:
                json.dump(trial_info, f, indent=2, default=str)

            logger.info(
                "Trial %d: rmse_away=%.4f, obs_res=%.4f, rough=%.4f",
                trial.number,
                obj_rmse_away,
                obj_obs_residual,
                obj_roughness,
            )
            return obj_rmse_away, obj_obs_residual, obj_roughness

        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e, exc_info=True)
            return float("inf"), float("inf"), float("inf")

    def _evaluate_target(
        self,
        dataset_idx: int,
        generate_kwargs: dict,
        trial_dir: Path,
    ) -> dict:
        """Generate ensemble for one target and compute multi-objective metrics."""
        import tempfile

        import zarr as zarr_lib

        from neural_transport.inference.generation import generate_multi_target
        from neural_transport.inference.metrics import (
            compute_xco2_column,
            spatial_roughness,
        )
        from neural_transport.inference.metrics import (
            rmse_away as _rmse_away,
        )

        target_var = self.target_vars[0]

        with tempfile.TemporaryDirectory(prefix=f"trial{trial_dir.name}_t{dataset_idx}_") as tmp:
            zarr_path = generate_multi_target(
                model=self.model,
                dataset=self.dataset,
                target_indices=[dataset_idx],
                n_samples_per_target=self.n_samples,
                batch_size=self.n_samples,
                generate_kwargs=generate_kwargs,
                out_dir=tmp,
                device=self.device,
                target_var=target_var,
                verbose=False,
                seed=dataset_idx,
            )

            store = zarr_lib.open_group(str(zarr_path), mode="r")
            samples_np = np.array(store["predictions"])[0]  # [n_samples, nlat, nlon, nlev]
            gt_2d = np.array(store["gt"])[0]  # [nlat, nlon, nlev]
            mask_raw = np.array(store["obs_mask"])[0]  # [nlat, nlon] or similar

        n_total = samples_np.shape[0]

        valid_mask = np.isfinite(samples_np).all(axis=(1, 2, 3))
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            return {"rmse_away": None, "obs_residual": None, "roughness": None, "n_valid": 0, "n_total": n_total}

        ensemble_mean = samples_np[valid_mask].mean(axis=0)

        # Obs mask
        mask_2d = None
        if mask_raw.size == self.grid_info.nlat * self.grid_info.nlon:
            mask_2d = mask_raw.reshape(self.grid_info.nlat, self.grid_info.nlon).astype(bool)

        # 1. RMSE at unobserved locations
        val_rmse_away = _rmse_away(ensemble_mean, gt_2d, mask_2d)

        # 2. Column XCO2 obs residual
        val_obs_residual = None
        if mask_2d is not None and mask_2d.any():
            xco2_pred = compute_xco2_column(ensemble_mean, self.pressure_weights, np.ones_like(self.pressure_weights))
            xco2_gt = compute_xco2_column(gt_2d, self.pressure_weights, np.ones_like(self.pressure_weights))
            diff = xco2_pred - xco2_gt
            val_obs_residual = float(np.sqrt(np.mean(diff[mask_2d] ** 2)))

        # 3. Spatial roughness of column XCO2
        val_roughness = None
        xco2_ens = compute_xco2_column(ensemble_mean, self.pressure_weights, np.ones_like(self.pressure_weights))
        if xco2_ens.ndim == 2:
            rough = spatial_roughness(xco2_ens)
            val_roughness = (rough["roughness_lat"] + rough["roughness_lon"]) / 2.0

        return {
            "rmse_away": val_rmse_away if np.isfinite(val_rmse_away) else None,
            "obs_residual": val_obs_residual,
            "roughness": val_roughness,
            "n_valid": n_valid,
            "n_total": n_total,
        }


def select_from_pareto(
    study: optuna.Study,
    obs_residual_threshold: float = 0.5,
) -> list[optuna.trial.FrozenTrial]:
    """Select trials from Pareto front satisfying obs_residual constraint.

    Filters Pareto-optimal trials where obs_residual (objective 1) is below
    the threshold, then sorts by rmse_away (objective 0).

    Parameters
    ----------
    study : optuna.Study
        Multi-objective study with 3 directions.
    obs_residual_threshold : float
        Maximum allowed obs_residual_xco2.

    Returns
    -------
    list[FrozenTrial] — filtered and sorted Pareto-optimal trials.
    """
    pareto_trials = study.best_trials  # Pareto front
    filtered = [t for t in pareto_trials if t.values[1] <= obs_residual_threshold]
    filtered.sort(key=lambda t: t.values[0])  # sort by rmse_away
    return filtered


def run_multi_objective_study(
    method: str,
    model: Any,
    dataset: Any,
    grid_info: Any,
    base_config: GenerateConfig,
    study_name: str,
    storage: str,
    run_dir: Path | str,
    *,
    target_vars: list[str] | None = None,
    n_trials: int = 100,
    n_targets: int = 10,
    n_samples: int = 10,
    nan_penalty: float = 100.0,
    device: str = "cuda",
    seed: int = 42,
) -> optuna.Study:
    """Create and run an NSGA-II multi-objective study for posterior sampling.

    Returns
    -------
    optuna.Study with 3 objectives: rmse_away, obs_residual, roughness.
    """
    import random
    import time

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.NSGAIISampler(seed=seed)

    for attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                directions=["minimize", "minimize", "minimize"],
                sampler=sampler,
                load_if_exists=True,
            )
            break
        except Exception as e:
            if attempt == 9:
                raise
            wait = random.uniform(1, 5 * (attempt + 1))
            logger.warning("Study creation attempt %d failed (%s), retrying in %.1fs", attempt + 1, e, wait)
            time.sleep(wait)

    objective = MultiObjectivePosteriorObjective(
        model=model,
        dataset=dataset,
        method=method,
        base_config=base_config,
        grid_info=grid_info,
        target_vars=target_vars,
        n_targets=n_targets,
        n_samples=n_samples,
        nan_penalty=nan_penalty,
        run_dir=run_dir,
        device=device,
    )

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(
        "Multi-objective study '%s' done: %d trials (%d complete)",
        study_name,
        len(study.trials),
        n_complete,
    )

    if n_complete > 0:
        pareto = study.best_trials
        logger.info("Pareto front: %d trials", len(pareto))

        summary = {
            "study_name": study_name,
            "method": method,
            "n_trials": len(study.trials),
            "n_complete": n_complete,
            "n_pareto": len(pareto),
            "pareto_trials": [
                {"number": t.number, "values": t.values, "params": t.params}
                for t in pareto[:10]  # top 10
            ],
        }
        with open(run_dir / "study_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return study
