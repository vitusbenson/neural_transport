"""EvaluationSuite orchestrator and EvalResult container.

Provides a numpy-level API for running the right metrics for any experiment
type (deterministic, ensemble, distributional). Wraps functions from
evaluation.pointwise, evaluation.ensemble, and evaluation.distributional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_transport.configs import EvalConfig
from neural_transport.evaluation.distributional import compute_distributional_metrics
from neural_transport.evaluation.ensemble import (
    calibration_score,
    crps_ensemble,
    rank_histogram,
    spread_skill_ratio,
)
from neural_transport.evaluation.pointwise import (
    METRICS_NP,
    compute_error_maps,
    compute_error_scalars,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_json_safe(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.complexfloating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Structured container for evaluation results."""

    pointwise: dict[str, float] = field(default_factory=dict)
    ensemble: dict[str, float] | None = None
    distributional: dict[str, float] | None = None
    maps: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        """Flatten all scalar sections into a single dict with section prefixes."""
        flat: dict[str, float] = {}
        for key, val in self.pointwise.items():
            flat[f"pointwise/{key}"] = val
        if self.ensemble is not None:
            for key, val in self.ensemble.items():
                flat[f"ensemble/{key}"] = val
        if self.distributional is not None:
            for key, val in self.distributional.items():
                flat[f"distributional/{key}"] = val
        return flat

    def to_json(self, path: Path) -> None:
        """Write scalar results to JSON (maps and large arrays excluded)."""
        data: dict[str, Any] = {"pointwise": _make_json_safe(self.pointwise)}
        if self.ensemble is not None:
            data["ensemble"] = _make_json_safe(self.ensemble)
        if self.distributional is not None:
            data["distributional"] = _make_json_safe(self.distributional)
        if self.diagnostics:
            data["diagnostics"] = _make_json_safe(self.diagnostics)
        if self.metadata:
            # Skip large arrays (>1000 elements) to prevent enormous JSON files
            filtered = {}
            for k, v in self.metadata.items():
                if isinstance(v, np.ndarray) and v.size > 1000:
                    continue
                filtered[k] = v
            if filtered:
                data["metadata"] = _make_json_safe(filtered)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> EvalResult:
        """Load an EvalResult from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            pointwise=data.get("pointwise", {}),
            ensemble=data.get("ensemble"),
            distributional=data.get("distributional"),
            diagnostics=data.get("diagnostics", {}),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# EvaluationSuite
# ---------------------------------------------------------------------------


class EvaluationSuite:
    """Unified numpy-level evaluation orchestrator."""

    def __init__(self, config: EvalConfig | None = None):
        self.config = config or EvalConfig()

    def evaluate_deterministic(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        lat_weights: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> EvalResult:
        """Run all pointwise metrics on a single prediction vs ground truth."""
        pw = {}
        for name, fn in METRICS_NP.items():
            pw[name] = fn(pred, gt, weights=lat_weights)
        return EvalResult(
            pointwise=pw,
            metadata=metadata or {},
        )

    def evaluate_ensemble(
        self,
        samples: np.ndarray,
        gt: np.ndarray,
        lat_weights: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> EvalResult:
        """Run pointwise + ensemble metrics on ensemble samples vs ground truth.

        Parameters
        ----------
        samples : np.ndarray, shape [n_samples, nlat, nlon, (nlev)]
        gt : np.ndarray, shape [nlat, nlon, (nlev)]
        lat_weights : np.ndarray, optional
        metadata : dict, optional
        """
        # 1. Pointwise on ensemble mean
        ens_mean = samples.mean(axis=0)
        pw_result = self.evaluate_deterministic(ens_mean, gt, lat_weights=lat_weights)

        # 2. Error maps
        bias_map, rmse_map, mean_map, spread_map = compute_error_maps(samples, gt)
        maps = {
            "bias_map": bias_map,
            "rmse_map": rmse_map,
            "mean_map": mean_map,
            "spread_map": spread_map,
        }

        # 3. Error scalars
        bias_s, rmse_s, mean_s, spread_s = compute_error_scalars(
            bias_map, rmse_map, mean_map, spread_map, weights=lat_weights
        )

        # 4. CRPS
        crps_map, crps_mean = crps_ensemble(samples, gt)
        maps["crps_map"] = crps_map

        # 5. Spread-skill
        ss_ratio = spread_skill_ratio(samples, gt)

        # 6. Calibration
        cal = calibration_score(samples, gt)

        # 7. Rank histogram
        rh = rank_histogram(samples, gt)

        ensemble_dict = {
            "crps_mean": crps_mean,
            "spread_skill_ratio": ss_ratio,
            "bias_scalar": float(bias_s),
            "rmse_scalar": float(rmse_s),
            "mean_scalar": float(mean_s),
            "spread_scalar": float(spread_s),
            "calibration_error": cal["calibration_error"],
        }

        diagnostics = {
            "rank_histogram": rh,
            "calibration": cal,
        }

        return EvalResult(
            pointwise=pw_result.pointwise,
            ensemble=ensemble_dict,
            maps=maps,
            diagnostics=diagnostics,
            metadata=metadata or {},
        )

    def evaluate_distributional(
        self,
        gt_pool: np.ndarray,
        gen_pool: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        metadata: dict | None = None,
    ) -> EvalResult:
        """Run distributional metrics comparing two pools of fields.

        Parameters
        ----------
        gt_pool : np.ndarray, shape [N, nlat, nlon, (nlev)]
        gen_pool : np.ndarray, shape [M, nlat, nlon, (nlev)]
        lat, lon : 1D arrays
        metadata : dict, optional
        """
        metrics = compute_distributional_metrics(gt_pool, gen_pool, lat, lon)
        meta = dict(metadata) if metadata else {}
        meta.setdefault("gt_fields", gt_pool)
        meta.setdefault("gen_fields", gen_pool)
        meta.setdefault("lat", lat)
        meta.setdefault("lon", lon)
        return EvalResult(
            pointwise={},
            distributional=metrics,
            metadata=meta,
        )

    def to_dataframe(self, result: EvalResult) -> pd.DataFrame:
        """Convert an EvalResult to a single-row DataFrame."""
        return pd.DataFrame([result.to_flat_dict()])
