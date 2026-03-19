"""Ablation and summary utility plots (NOT registered — different signatures).

These functions take multiple configs/results and don't fit the standard
(result, ctx) signature used by @register_plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_transport.plots.utilities.plot_utils import mpl_rc_params, save_figure


def load_ablation_results(results_dir: str | Path) -> dict:
    """Load ablation_summary.json from a results directory."""
    path = Path(results_dir) / "ablation_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def plot_ablation_sweep(
    configs: dict[str, dict],
    metric_key: str,
    param_extractor: callable,
    out_dir: str | Path,
    *,
    title: str = "",
    xlabel: str = "Parameter",
    ylabel: str | None = None,
    log_x: bool = False,
    second_metric_key: str | None = None,
    second_ylabel: str | None = None,
    imgformats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Generic line/scatter for metric vs hyperparameter.

    Parameters
    ----------
    configs : dict mapping config name to metrics dict
    metric_key : key to extract from each metrics dict
    param_extractor : callable(config_name) -> numeric param value
    out_dir : output directory
    title : plot title
    xlabel, ylabel : axis labels
    log_x : use log scale for x-axis
    second_metric_key : optional second metric for dual y-axis
    second_ylabel : label for second y-axis
    imgformats : output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    params = []
    values = []
    values2 = []
    for name, metrics in configs.items():
        try:
            p = param_extractor(name)
        except (ValueError, KeyError, IndexError):
            continue
        if metric_key not in metrics:
            continue
        params.append(p)
        values.append(metrics[metric_key])
        if second_metric_key and second_metric_key in metrics:
            values2.append(metrics[second_metric_key])

    if not params:
        return

    # Sort by parameter value
    order = np.argsort(params)
    params = np.array(params)[order]
    values = np.array(values)[order]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(params, values, "o-", color="C0", label=metric_key)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel or metric_key)
    if log_x:
        ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    if values2 and len(values2) == len(params):
        values2 = np.array(values2)[order]
        ax2 = ax1.twinx()
        ax2.plot(params, values2, "s--", color="C1", label=second_metric_key)
        ax2.set_ylabel(second_ylabel or second_metric_key)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    else:
        ax1.legend(fontsize=7)

    ax1.set_title(title or f"{metric_key} vs {xlabel}")
    fig.tight_layout()
    save_figure(fig, out_dir, f"ablation_sweep_{metric_key}", imgformats=imgformats)


def plot_summary_bars(
    results: dict[str, dict],
    metric_key: str,
    out_dir: str | Path,
    *,
    title: str = "",
    highlight_best: bool = True,
    lower_is_better: bool = True,
    imgformats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Horizontal bar chart of configs ranked by metric.

    Parameters
    ----------
    results : dict mapping config name to metrics dict
    metric_key : key to extract and rank by
    out_dir : output directory
    title : plot title
    highlight_best : color-code best config
    lower_is_better : determines which config is "best"
    imgformats : output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    names = []
    values = []
    for name, metrics in results.items():
        if metric_key in metrics:
            names.append(name)
            values.append(metrics[metric_key])

    if not names:
        return

    # Sort by value
    order = np.argsort(values)
    if not lower_is_better:
        order = order[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    y_pos = np.arange(len(names))

    colors = []
    best_idx = 0
    for i, name in enumerate(names):
        if "unconditional" in name.lower():
            colors.append("gray")
        elif highlight_best and i == best_idx:
            colors.append("C2")
        else:
            colors.append("C0")

    ax.barh(y_pos, values, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(metric_key)
    ax.set_title(title or f"Configs ranked by {metric_key}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir, f"summary_bars_{metric_key}", imgformats=imgformats)


def plot_pareto_front(
    results: dict[str, dict],
    x_metric: str,
    y_metric: str,
    out_dir: str | Path,
    *,
    title: str = "",
    imgformats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    """Scatter plot of cost/time vs accuracy metric with annotated config names.

    Parameters
    ----------
    results : dict mapping config name to metrics dict
    x_metric : key for x-axis (e.g., wall_time)
    y_metric : key for y-axis (e.g., rmse)
    out_dir : output directory
    title : plot title
    imgformats : output formats
    """
    plt.rcParams.update(mpl_rc_params)
    out_dir = Path(out_dir)

    names = []
    xs = []
    ys = []
    for name, metrics in results.items():
        if x_metric in metrics and y_metric in metrics:
            names.append(name)
            xs.append(metrics[x_metric])
            ys.append(metrics[y_metric])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=40, color="C0", zorder=3)

    for name, x, y in zip(names, xs, ys):
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6,
            alpha=0.8,
        )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title or f"{y_metric} vs {x_metric}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir, f"pareto_{x_metric}_vs_{y_metric}", imgformats=imgformats)
