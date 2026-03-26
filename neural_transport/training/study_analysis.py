"""Publication-quality analysis and visualization of Optuna tuning studies.

Produces a comprehensive set of plots and tables for understanding
hyperparameter search results, parameter importance, and training dynamics.

Usage:
    from neural_transport.training.study_analysis import analyze_study

    analyze_study("sqlite:///optuna_fm_study.db", "fm_tuning", out_dir="analysis/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Publication-quality matplotlib defaults
ANALYSIS_RC_PARAMS = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def _save_fig(fig: plt.Figure, out_dir: Path, name: str, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save figure in multiple formats."""
    for fmt in imgformats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_study(
    study_or_db: Any,
    study_name: str | None = None,
    out_dir: str | Path = "analysis",
    imgformats: tuple[str, ...] = ("pdf", "png"),
    run_dir: str | Path | None = None,
) -> Path:
    """Full analysis of an Optuna study with publication-quality plots.

    Args:
        study_or_db: Either an optuna.Study object or a path/URL to the storage
            (e.g. "sqlite:///study.db").
        study_name: Study name (required if study_or_db is a storage path).
        out_dir: Output directory for plots and tables.
        imgformats: Image formats to save.
        run_dir: Directory containing trial subdirectories (for training curves).

    Returns:
        Path to the output directory.
    """
    import optuna

    plt.rcParams.update(ANALYSIS_RC_PARAMS)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load study if needed
    if isinstance(study_or_db, optuna.Study):
        study = study_or_db
    else:
        study = optuna.load_study(study_name=study_name, storage=str(study_or_db))

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_complete == 0:
        logger.warning("No completed trials found in study '%s'", study.study_name)
        return out_dir

    logger.info(
        "Analyzing study '%s': %d trials (%d complete, %d pruned, %d failed)",
        study.study_name,
        len(study.trials),
        n_complete,
        len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    )

    # Generate all plots
    plot_optimization_history(study, out_dir, imgformats)
    plot_parameter_importance(study, out_dir, imgformats)
    plot_parallel_coordinates(study, out_dir, imgformats)
    plot_contour_top_params(study, out_dir, imgformats)
    plot_slice_plots(study, out_dir, imgformats)
    generate_summary_table(study, out_dir)

    if run_dir is not None:
        plot_best_trial_training_curves(study, Path(run_dir), out_dir, imgformats)

    logger.info("Analysis complete. Results saved to %s", out_dir)
    return out_dir


def plot_optimization_history(study: Any, out_dir: Path, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Plot objective value vs trial number with running best."""
    import optuna

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        return

    trial_numbers = [t.number for t in complete_trials]
    values = [t.value for t in complete_trials]

    # Running best
    running_best = []
    best_so_far = float("inf")
    for v in values:
        best_so_far = min(best_so_far, v)
        running_best.append(best_so_far)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_numbers, values, alpha=0.5, s=20, c="C0", label="Trial value")
    ax.plot(trial_numbers, running_best, color="C1", linewidth=2, label="Running best")

    # Mark pruned trials
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned_trials:
        pruned_numbers = [t.number for t in pruned_trials]
        ax.scatter(
            pruned_numbers,
            [ax.get_ylim()[1] * 0.95] * len(pruned_numbers),
            marker="x",
            color="gray",
            alpha=0.3,
            s=15,
            label=f"Pruned ({len(pruned_trials)})",
        )

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"Optimization History — {study.study_name}")
    ax.legend()
    _save_fig(fig, out_dir, "optimization_history", imgformats)


def plot_parameter_importance(study: Any, out_dir: Path, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Plot fANOVA-based parameter importance."""
    import optuna

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning("Could not compute parameter importance: %s", e)
        return

    if not importances:
        return

    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    y_pos = np.arange(len(names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Hyperparameter Importance (fANOVA) — {study.study_name}")
    ax.invert_yaxis()
    _save_fig(fig, out_dir, "parameter_importance", imgformats)

    # Save as JSON too
    with open(out_dir / "parameter_importance.json", "w") as f:
        json.dump(importances, f, indent=2)


def plot_parallel_coordinates(study: Any, out_dir: Path, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Plot parallel coordinate plot of all parameters colored by objective."""
    import optuna

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) < 3:
        return

    # Get all parameter names
    param_names = sorted(set().union(*(t.params.keys() for t in complete_trials)))

    # Build data matrix
    data = []
    for t in complete_trials:
        row = {"objective": t.value}
        for p in param_names:
            val = t.params.get(p, None)
            if val is not None:
                row[p] = val
        data.append(row)

    df = pd.DataFrame(data)

    # Select numeric (non-boolean) columns for parallel coordinates
    numeric_cols = [
        c
        for c in df.columns
        if c != "objective" and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

    if len(numeric_cols) < 2:
        return

    # Normalize columns to [0, 1] for parallel coordinates
    fig, ax = plt.subplots(figsize=(14, 6))
    norm_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
    norm_df = norm_df.dropna(axis=1, how="all")
    numeric_cols = list(norm_df.columns)
    if len(numeric_cols) < 2:
        plt.close(fig)
        return
    for col in numeric_cols:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5

    # Color by objective (lower = better = darker blue)
    obj_values = df["objective"].values
    obj_norm = (obj_values - obj_values.min()) / (obj_values.max() - obj_values.min() + 1e-10)
    cmap = plt.cm.viridis_r

    x = np.arange(len(numeric_cols))
    for i in range(len(norm_df)):
        ax.plot(x, norm_df.iloc[i].values, color=cmap(obj_norm[i]), alpha=0.4, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized Value")
    ax.set_title(f"Parallel Coordinates — {study.study_name}")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=obj_values.min(), vmax=obj_values.max())
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Objective Value")

    _save_fig(fig, out_dir, "parallel_coordinates", imgformats)


def plot_contour_top_params(study: Any, out_dir: Path, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Plot 2D contour plots for the top-2 most important parameter pairs."""
    import optuna

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) < 5:
        return

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        return

    # Get top parameters (only numeric ones that vary)
    numeric_params = []
    for name in importances:
        values = [t.params.get(name) for t in complete_trials if name in t.params]
        values = [v for v in values if isinstance(v, int | float)]
        if len(set(values)) > 1:
            numeric_params.append(name)

    if len(numeric_params) < 2:
        return

    top_params = numeric_params[: min(4, len(numeric_params))]

    n_pairs = len(top_params) * (len(top_params) - 1) // 2
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    pair_idx = 0
    for i in range(len(top_params)):
        for j in range(i + 1, len(top_params)):
            row, col = pair_idx // n_cols, pair_idx % n_cols
            ax = axes[row, col]

            x_vals, y_vals, obj_vals = [], [], []
            for t in complete_trials:
                if top_params[i] in t.params and top_params[j] in t.params:
                    xi = t.params[top_params[i]]
                    yi = t.params[top_params[j]]
                    if isinstance(xi, int | float) and isinstance(yi, int | float):
                        x_vals.append(xi)
                        y_vals.append(yi)
                        obj_vals.append(t.value)

            if len(x_vals) >= 3:
                sc = ax.scatter(x_vals, y_vals, c=obj_vals, cmap="viridis_r", alpha=0.7, s=30)
                plt.colorbar(sc, ax=ax, label="Objective")

            ax.set_xlabel(top_params[i])
            ax.set_ylabel(top_params[j])
            pair_idx += 1

    # Hide empty axes
    for idx in range(pair_idx, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Parameter Pair Contours — {study.study_name}")
    fig.tight_layout()
    _save_fig(fig, out_dir, "contour_top_params", imgformats)


def plot_slice_plots(study: Any, out_dir: Path, imgformats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Plot objective vs each individual parameter (slice plots)."""
    import optuna

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) < 3:
        return

    param_names = sorted(set().union(*(t.params.keys() for t in complete_trials)))
    n_params = len(param_names)

    if n_params == 0:
        return

    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, pname in enumerate(param_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        param_vals = []
        obj_vals = []
        for t in complete_trials:
            if pname in t.params:
                param_vals.append(t.params[pname])
                obj_vals.append(t.value)

        if not param_vals:
            ax.set_visible(False)
            continue

        # Check if categorical
        if isinstance(param_vals[0], str) or isinstance(param_vals[0], bool):
            categories = sorted(set(str(v) for v in param_vals))
            cat_data = {c: [] for c in categories}
            for pv, ov in zip(param_vals, obj_vals):
                cat_data[str(pv)].append(ov)

            positions = range(len(categories))
            bp = ax.boxplot(
                [cat_data[c] for c in categories],
                positions=list(positions),
                widths=0.6,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("C0")
                patch.set_alpha(0.5)
            ax.set_xticks(list(positions))
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
        else:
            ax.scatter(param_vals, obj_vals, alpha=0.5, s=15, c="C0")

        ax.set_xlabel(pname, fontsize=8)
        ax.set_ylabel("Objective", fontsize=8)

    # Hide extra axes
    for idx in range(n_params, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"Parameter Slice Plots — {study.study_name}")
    fig.tight_layout()
    _save_fig(fig, out_dir, "slice_plots", imgformats)


def generate_summary_table(study: Any, out_dir: Path) -> pd.DataFrame:
    """Generate and save a summary table of all trials.

    Returns:
        DataFrame with trial summaries.
    """

    df = study.trials_dataframe()
    df.to_csv(out_dir / "all_trials.csv", index=False)

    # Best trials summary (top 10)
    complete_df = df[df["state"] == "COMPLETE"].copy()
    if len(complete_df) > 0:
        complete_df = complete_df.sort_values("value")
        top10 = complete_df.head(10)
        top10.to_csv(out_dir / "top10_trials.csv", index=False)

        # Print summary
        summary_lines = [
            f"Study: {study.study_name}",
            f"Total trials: {len(study.trials)}",
            f"Complete: {len(complete_df)}",
            f"Pruned: {len(df[df['state'] == 'PRUNED'])}",
            f"Failed: {len(df[df['state'] == 'FAIL'])}",
            "",
            f"Best value: {study.best_value:.6f} (trial {study.best_trial.number})",
            "Best params:",
        ]
        for k, v in study.best_params.items():
            summary_lines.append(f"  {k}: {v}")

        summary_text = "\n".join(summary_lines)
        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text)

        logger.info("\n%s", summary_text)

    return df


def plot_best_trial_training_curves(
    study: Any,
    run_dir: Path,
    out_dir: Path,
    imgformats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Plot training curves from the best trial's TensorBoard logs."""

    best_trial = study.best_trial
    trial_dir = run_dir / f"trial_{best_trial.number:04d}" / "singlestep"

    # Try to read TensorBoard events
    events_dirs = list(trial_dir.rglob("events.out.tfevents.*")) if trial_dir.exists() else []

    if not events_dirs:
        logger.info("No TensorBoard events found for best trial %d", best_trial.number)
        return

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(events_dirs[0].parent))
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        metrics_to_plot = [t for t in tags if any(k in t for k in ["Loss/", "GenEval/", "lr-"])]

        if not metrics_to_plot:
            return

        n_metrics = len(metrics_to_plot)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        axes = np.atleast_2d(axes)

        for idx, tag in enumerate(metrics_to_plot):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, linewidth=1)
            ax.set_xlabel("Step")
            ax.set_ylabel(tag.split("/")[-1])
            ax.set_title(tag, fontsize=9)

        for idx in range(n_metrics, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)

        fig.suptitle(f"Best Trial #{best_trial.number} Training Curves")
        fig.tight_layout()
        _save_fig(fig, out_dir, "best_trial_training_curves", imgformats)

    except ImportError:
        logger.info("tensorboard not available for training curve plotting")
    except Exception as e:
        logger.warning("Failed to plot training curves: %s", e)
