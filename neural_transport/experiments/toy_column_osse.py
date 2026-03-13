"""
Toy column OSSE for debugging column conditioning in isolation.

Self-contained script: generates synthetic 3D fields, trains a tiny flow matching
model, then tests all conditioning methods for column (XCO2) observations.

Key design: no targshift, simple standardization, minimal model. If column
conditioning works here, the real-data bugs are in the normalization pipeline.
If it doesn't, the algorithms themselves need fixing.

Usage:
    python -m neural_transport.experiments.toy_column_osse
    python -m neural_transport.experiments.toy_column_osse --device cuda
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader, TensorDataset

from neural_transport.inference.posterior_samplers import FlowDPSSampler
from neural_transport.models.flowmatching import MaskedVelocityWrapper, compute_ot_coupling

# ── Data generation ──────────────────────────────────────────────────────


def generate_toy_data(n_samples=10000, nlat=16, nlon=32, nlev=4, seed=42):
    """Generate 3D fields as mixtures of spatially-correlated Gaussians.

    Each sample is a [nlev, nlat, nlon] field built from 5 Gaussian bumps
    with per-level amplitude modulation.
    """
    rng = np.random.RandomState(seed)

    lat_grid = np.linspace(-90, 90, nlat)
    lon_grid = np.linspace(0, 360, nlon, endpoint=False)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)  # [nlat, nlon]

    n_components = 5
    fields = np.zeros((n_samples, nlev, nlat, nlon), dtype=np.float32)

    for i in range(n_samples):
        for k in range(n_components):
            center_lat = rng.uniform(-60, 60)
            center_lon = rng.uniform(0, 360)
            sigma_lat = rng.uniform(15, 40)
            sigma_lon = rng.uniform(20, 60)
            amplitude = rng.uniform(380, 420)

            bump = amplitude * np.exp(
                -0.5 * ((lat_mesh - center_lat) / sigma_lat) ** 2 - 0.5 * ((lon_mesh - center_lon) / sigma_lon) ** 2
            )  # [nlat, nlon]

            # Per-level modulation (surface has more variability)
            level_weights = np.array([1.0, 0.8, 0.5, 0.3])[:nlev]
            level_offsets = rng.normal(0, 2, nlev)

            for lev in range(nlev):
                fields[i, lev] += level_weights[lev] * bump + level_offsets[lev]

        # Add background
        fields[i] += 400.0

    return torch.from_numpy(fields)


# ── Column forward model ────────────────────────────────────────────────


def get_column_weights(nlev=4):
    """Fixed column weights mimicking pressure-weighted AK.

    w_k = [0.05, 0.15, 0.30, 0.50] (surface-heavy, like real atmosphere).
    """
    weights = torch.tensor([0.05, 0.15, 0.30, 0.50][:nlev], dtype=torch.float32)
    return weights


def compute_xco2_toy(fields, weights):
    """Compute column XCO2 from 3D fields.

    Args:
        fields: [B, nlev, nlat, nlon]
        weights: [nlev]

    Returns:
        xco2: [B, 1, nlat, nlon]
    """
    w = weights.view(1, -1, 1, 1).to(fields.device)
    return (w * fields).sum(dim=1, keepdim=True)


def create_column_obs(fields, weights, obs_fraction=0.3, seed=None):
    """Create column observations at random spatial subset.

    Returns:
        obs_mask: [B, 1, nlat, nlon] bool
        obs_values: [B, 1, nlat, nlon] XCO2 at observed locations
    """
    B, C, nlat, nlon = fields.shape
    N = nlat * nlon
    n_obs = int(obs_fraction * N)

    if seed is not None:
        torch.manual_seed(seed)
    obs_indices = torch.randperm(N)[:n_obs]

    mask_flat = torch.zeros(N, dtype=torch.bool, device=fields.device)
    mask_flat[obs_indices] = True
    obs_mask = mask_flat.view(1, 1, nlat, nlon).expand(B, 1, -1, -1)

    xco2 = compute_xco2_toy(fields, weights)
    obs_values = torch.where(obs_mask, xco2, torch.zeros_like(xco2))

    return obs_mask, obs_values


# ── Tiny flow matching model ────────────────────────────────────────────


class TinyConvNet(nn.Module):
    """Small ConvNet for flow matching velocity prediction (~50k params)."""

    def __init__(self, nlev=4, hidden=64):
        super().__init__()
        # Input: [B, nlev+1, nlat, nlon] (field + time channel)
        self.net = nn.Sequential(
            nn.Conv2d(nlev + 1, hidden, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, nlev, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ToyVelocityWrapper(nn.Module):
    """Wraps TinyConvNet for ODESolver interface."""

    def __init__(self, model, nlev=4):
        super().__init__()
        self.model = model
        self.nlev = nlev
        # Attributes for MaskedVelocityWrapper compatibility
        self.targshift = False
        self.target_vars = ["toy_co2"]

    def forward(self, x, t):
        B, C, H, W = x.shape
        t_ch = t.view(-1, 1, 1, 1).expand(B, 1, H, W)
        x_in = torch.cat([x, t_ch], dim=1)
        return self.model(x_in)


# ── Training ────────────────────────────────────────────────────────────


def train_flow_matching(data, nlev=4, epochs=50, lr=1e-3, batch_size=128, device="cpu", use_ot_coupling=False):
    """Train unconditional flow matching model on toy data."""
    model = TinyConvNet(nlev=nlev).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Standardize
    data_mean = data.mean()
    data_std = data.std()
    data_norm = (data - data_mean) / data_std

    dataset = TensorDataset(data_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x_1,) in loader:
            x_1 = x_1.to(device)
            B = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            if use_ot_coupling:
                x_0 = compute_ot_coupling(x_0, x_1)
            t = torch.rand(B, device=device)

            path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = path_sample.x_t

            t_ch = path_sample.t.view(-1, 1, 1, 1).expand_as(x_t[:, :1, :, :])
            x_in = torch.cat([x_t, t_ch], dim=1)
            v_pred = model(x_in)

            loss = F.mse_loss(v_pred, path_sample.dx_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(dataset)
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg:.6f}")

    print(f"  Training took {time.time() - t0:.1f}s")
    model.eval()
    return model, data_mean.item(), data_std.item()


# ── Conditioning test ───────────────────────────────────────────────────


def sample_unconditional(velocity_wrapper, n_samples, nlev, nlat, nlon, device, steps=20):
    """Generate unconditional samples."""
    x_init = torch.randn(n_samples, nlev, nlat, nlon, device=device)
    time_grid = torch.linspace(0, 1, steps, device=device)

    solver = ODESolver(velocity_model=velocity_wrapper)
    traj = solver.sample(
        time_grid=time_grid, x_init=x_init, method="midpoint", step_size=None, return_intermediates=False
    )
    return traj


def sample_conditioned(model_wrapper, masking_config, generate_kwargs, n_samples, nlev, nlat, nlon, device, steps=20):
    """Generate conditioned samples using MaskedVelocityWrapper."""
    x_init = torch.randn(n_samples, nlev, nlat, nlon, device=device)
    time_grid = torch.linspace(0, 1, steps, device=device)
    masking_config["time_grid"] = time_grid

    masked_wrapper = MaskedVelocityWrapper(
        submodel=model_wrapper,
        masking_config=masking_config,
        nlev=nlev,
        **generate_kwargs,
    )

    solver = ODESolver(velocity_model=masked_wrapper)
    traj = solver.sample(
        time_grid=time_grid, x_init=x_init, method="midpoint", step_size=None, return_intermediates=False
    )
    return traj


def sample_flowdps(model_wrapper, masking_config, generate_kwargs, n_samples, nlev, nlat, nlon, device, steps=20):
    """Generate conditioned samples using FlowDPS (projection-based posterior sampling)."""
    x_init = torch.randn(n_samples, nlev, nlat, nlon, device=device)
    time_grid = torch.linspace(0, 1, steps, device=device)

    sampler = FlowDPSSampler(
        velocity_model=model_wrapper,
        masking_config=masking_config,
        sigma_obs=generate_kwargs.get("sigma_obs", 0.1),
        spatial_smoothing_sigma=generate_kwargs.get("spatial_smoothing_sigma", 0.0),
        fresh_noise=generate_kwargs.get("fresh_noise", True),
    )

    samples = sampler.sample(x_init, time_grid, return_intermediates=False)
    return samples


def build_masking_config(obs_mask, obs_values, data_mean, data_std, column_weights, device):
    """Build masking config for MaskedVelocityWrapper."""
    obs_mean = torch.tensor(data_mean, device=device).view(1, 1, 1, 1)
    obs_std = torch.tensor(data_std, device=device).view(1, 1, 1, 1)
    target_mean = torch.tensor(data_mean, device=device).view(1, 1, 1, 1)
    target_std = torch.tensor(data_std, device=device).view(1, 1, 1, 1)

    nlev = column_weights.shape[0]
    B = obs_mask.shape[0]
    nlat, nlon = obs_mask.shape[2], obs_mask.shape[3]

    # Pressure weights = column_weights (in our toy model, h_k = w_k)
    h = column_weights.view(1, -1, 1, 1).expand(B, nlev, nlat, nlon).to(device)
    # AK = 1.0 (no averaging kernel correction in toy model; XCO2 = sum(h_k * x_k))
    ak = torch.ones(B, nlev, nlat, nlon, device=device)

    # Normalize obs values
    obs_values_norm = (obs_values.to(device) - obs_mean) / obs_std

    return {
        "obs_mask": obs_mask.to(device),
        "obs_values": obs_values_norm,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "ak": ak,
        "pressure_weights": h,
        "xco2_prior": None,
        "co2_profile_prior": None,
    }


# ── Evaluation ──────────────────────────────────────────────────────────


def evaluate(samples, gt, obs_mask, column_weights, data_mean, data_std):
    """Compute metrics for conditioned samples.

    Args:
        samples: [n_samples, nlev, nlat, nlon] normalized
        gt: [1, nlev, nlat, nlon] normalized ground truth
        obs_mask: [1, 1, nlat, nlon] bool
        column_weights: [nlev]
    """
    # Denormalize
    samples_phys = samples * data_std + data_mean
    gt_phys = gt * data_std + data_mean

    ens_mean = samples_phys.mean(dim=0, keepdim=True)  # [1, nlev, nlat, nlon]
    diff_3d = ens_mean - gt_phys  # [1, nlev, nlat, nlon]

    # 3D field RMSE
    rmse_3d_full = float(diff_3d.pow(2).mean().sqrt())

    # XCO2
    w = column_weights.view(1, -1, 1, 1).to(samples.device)
    xco2_pred = (w * ens_mean).sum(dim=1, keepdim=True)
    xco2_gt = (w * gt_phys).sum(dim=1, keepdim=True)
    xco2_diff = xco2_pred - xco2_gt

    rmse_xco2_full = float(xco2_diff.pow(2).mean().sqrt())

    mask = obs_mask.to(samples.device)
    mask_3d = mask.expand_as(diff_3d)

    rmse_3d_obs = float(diff_3d[mask_3d].pow(2).mean().sqrt()) if mask_3d.any() else float('nan')
    rmse_3d_away = float(diff_3d[~mask_3d].pow(2).mean().sqrt()) if (~mask_3d).any() else float('nan')
    rmse_xco2_obs = float(xco2_diff[mask].pow(2).mean().sqrt()) if mask.any() else float('nan')
    rmse_xco2_away = float(xco2_diff[~mask].pow(2).mean().sqrt()) if (~mask).any() else float('nan')

    return {
        "rmse_3d_full": rmse_3d_full,
        "rmse_3d_obs": rmse_3d_obs,
        "rmse_3d_away": rmse_3d_away,
        "rmse_xco2_full": rmse_xco2_full,
        "rmse_xco2_obs": rmse_xco2_obs,
        "rmse_xco2_away": rmse_xco2_away,
    }


# ── Plotting ────────────────────────────────────────────────────────────


def plot_results(all_results, gt_field, obs_mask, column_weights, data_mean, data_std, out_dir):
    """Plot comparison: GT vs each conditioning method."""
    gt_phys = gt_field * data_std + data_mean
    w = column_weights.view(1, -1, 1, 1)
    gt_xco2 = (w * gt_phys).sum(dim=1)[0].cpu().numpy()  # [nlat, nlon]
    mask_2d = obs_mask[0, 0].cpu().numpy()

    n_methods = len(all_results)
    fig, axes = plt.subplots(n_methods + 1, 4, figsize=(16, 3.5 * (n_methods + 1)))

    # GT row
    vmin, vmax = np.nanpercentile(gt_xco2, [2, 98])
    axes[0, 0].imshow(gt_xco2, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 0].set_title("GT XCO2")
    obs_display = np.where(mask_2d, gt_xco2, np.nan)
    axes[0, 1].imshow(obs_display, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0, 1].set_title("Observed XCO2")
    gt_lev0 = gt_phys[0, 0].cpu().numpy()
    axes[0, 2].imshow(gt_lev0, origin="lower", cmap="cividis", aspect="auto")
    axes[0, 2].set_title("GT Level 0")
    axes[0, 3].axis("off")
    axes[0, 0].set_ylabel("Ground\nTruth", fontsize=10, rotation=0, labelpad=60, va="center")

    for i, (method_name, result) in enumerate(all_results.items()):
        row = i + 1
        samples = result["samples"]  # [n, nlev, nlat, nlon] normalized
        samples_phys = samples * data_std + data_mean
        ens_mean = samples_phys.mean(dim=0)  # [nlev, nlat, nlon]
        pred_xco2 = (w[0] * ens_mean).sum(dim=0).cpu().numpy()  # [nlat, nlon]

        axes[row, 0].imshow(pred_xco2, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 0].set_title("Pred XCO2")

        xco2_diff = np.abs(pred_xco2 - gt_xco2)
        dmax = max(np.nanpercentile(xco2_diff, 98), 1e-6)
        axes[row, 1].imshow(xco2_diff, origin="lower", cmap="Reds", vmin=0, vmax=dmax, aspect="auto")
        axes[row, 1].set_title("|XCO2 Diff|")

        pred_lev0 = ens_mean[0].cpu().numpy()
        axes[row, 2].imshow(pred_lev0, origin="lower", cmap="cividis", aspect="auto")
        axes[row, 2].set_title("Pred Level 0")

        metrics = result["metrics"]
        metrics_str = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        axes[row, 3].text(
            0.1, 0.5, metrics_str, transform=axes[row, 3].transAxes, fontsize=8, va="center", family="monospace"
        )
        axes[row, 3].axis("off")

        axes[row, 0].set_ylabel(method_name.replace("_", "\n"), fontsize=9, rotation=0, labelpad=60, va="center")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_dir / "toy_column_osse_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_dir / 'toy_column_osse_results.png'}")


# ── Main ────────────────────────────────────────────────────────────────

CONDITIONING_METHODS = {
    "unconditional": dict(_unconditional=True),
    "correction": dict(
        conditioning_mode="correction",
        masking_method="total_column_average_simple",
        masking_time=None,
    ),
    "correction_late": dict(
        conditioning_mode="correction",
        masking_method="total_column_average_simple",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "velocity_proj_late": dict(
        conditioning_mode="velocity_projection",
        masking_method="total_column_average_simple",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "guidance_0.5": dict(
        conditioning_mode="guidance",
        guidance_scale=0.5,
        masking_method="total_column_average_simple",
        masking_time=None,
    ),
    "guidance_1": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        masking_time=None,
    ),
    "guidance_1_late": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    "repaint_late": dict(
        conditioning_mode="repaint",
        masking_method="total_column_average_simple",
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    # DPS guidance ablation (Phase 5)
    "dps_s1.0": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        sigma_obs=1.0,
    ),
    "dps_s0.5": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        sigma_obs=0.5,
    ),
    "dps_s0.1": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        sigma_obs=0.1,
    ),
    "dps_s0.5_smooth2": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        sigma_obs=0.5,
        spatial_smoothing_sigma=2.0,
    ),
    "dps_s0.5_smooth2_late": dict(
        conditioning_mode="guidance",
        guidance_scale=1.0,
        masking_method="total_column_average_simple",
        sigma_obs=0.5,
        spatial_smoothing_sigma=2.0,
        masking_time="smooth_late_masking",
        t_threshold=0.8,
    ),
    # FlowDPS (Phase 6) — projection-based posterior sampling
    "flowdps_s0.1": dict(
        _sampler="flowdps",
        sigma_obs=0.1,
    ),
    "flowdps_s0.5": dict(
        _sampler="flowdps",
        sigma_obs=0.5,
    ),
    "flowdps_s1.0": dict(
        _sampler="flowdps",
        sigma_obs=1.0,
    ),
    "flowdps_s0.1_smooth2": dict(
        _sampler="flowdps",
        sigma_obs=0.1,
        spatial_smoothing_sigma=2.0,
    ),
}


def _print_summary_table(all_results):
    """Print a formatted summary table of results."""
    print(f"\n{'=' * 100}")
    header = f"{'Method':<22} {'RMSE_3d':>10} {'RMSE_3d_o':>10} {'RMSE_3d_a':>10} {'RMSE_xco2':>10} {'RMSE_xo':>10} {'RMSE_xa':>10}"
    print(header)
    print(f"{'-' * 100}")
    for name, res in all_results.items():
        m = res["metrics"]

        def _f(v):
            return f"{v:.4f}" if not np.isnan(v) else "N/A"

        print(
            f"{name:<22} {_f(m['rmse_3d_full']):>10} {_f(m['rmse_3d_obs']):>10} "
            f"{_f(m['rmse_3d_away']):>10} {_f(m['rmse_xco2_full']):>10} "
            f"{_f(m['rmse_xco2_obs']):>10} {_f(m['rmse_xco2_away']):>10}"
        )
    print(f"{'=' * 100}")


def run_toy_osse(
    device="cpu",
    n_train=10000,
    epochs=50,
    n_samples=20,
    obs_fraction=0.3,
    methods=None,
    out_dir=None,
    seed=42,
    nlat=16,
    nlon=32,
    nlev=4,
):
    """Run toy column OSSE end-to-end.

    Parameters
    ----------
    device : str
        Device to use for training and sampling.
    n_train : int
        Number of training samples to generate.
    epochs : int
        Number of training epochs.
    n_samples : int
        Number of ensemble samples per conditioning method.
    obs_fraction : float
        Fraction of spatial locations observed.
    methods : list[str] or None
        Subset of CONDITIONING_METHODS to run. None runs all.
    out_dir : str or Path or None
        Output directory for plots/JSON. None skips saving.
    seed : int
        Random seed for data generation.
    nlat, nlon, nlev : int
        Grid dimensions.

    Returns
    -------
    dict[str, dict]
        Maps method_name -> {"samples": Tensor, "metrics": dict}.
    """
    # 1. Generate data
    print("Generating toy data...")
    data = generate_toy_data(n_samples=n_train, nlat=nlat, nlon=nlon, nlev=nlev, seed=seed)
    column_weights = get_column_weights(nlev)

    # 2. Train flow matching model
    print("Training flow matching model...")
    conv_model, data_mean, data_std = train_flow_matching(data, nlev=nlev, epochs=epochs, lr=1e-3, device=device)

    velocity_wrapper = ToyVelocityWrapper(conv_model, nlev=nlev).to(device)

    # 3. Pick a ground truth sample and create column obs
    gt_idx = 0
    gt_norm = ((data[gt_idx : gt_idx + 1] - data_mean) / data_std).to(device)
    gt_phys = data[gt_idx : gt_idx + 1].to(device)

    obs_mask, obs_values = create_column_obs(gt_phys, column_weights, obs_fraction=obs_fraction, seed=123)
    obs_mask = obs_mask.to(device)
    obs_values = obs_values.to(device)

    print(f"GT field shape: {gt_norm.shape}")
    print(f"Obs mask: {obs_mask.sum().item()}/{obs_mask.numel()} locations observed")
    print(f"Column weights: {column_weights.tolist()}")

    # 4. Select conditioning methods
    if methods is not None:
        selected = {k: v for k, v in CONDITIONING_METHODS.items() if k in methods}
        if not selected:
            raise ValueError(f"No matching methods. Available: {list(CONDITIONING_METHODS.keys())}")
    else:
        selected = CONDITIONING_METHODS

    # 5. Test each conditioning method
    all_results = {}

    for method_name, method_config in selected.items():
        print(f"\nTesting: {method_name}")
        config = {k: v for k, v in method_config.items()}
        is_unconditional = config.pop("_unconditional", False)
        sampler_type = config.pop("_sampler", None)

        torch.manual_seed(42)

        if is_unconditional:
            samples = sample_unconditional(velocity_wrapper, n_samples, nlev, nlat, nlon, device)
        elif sampler_type == "flowdps":
            masking_config = build_masking_config(obs_mask, obs_values, data_mean, data_std, column_weights, device)
            samples = sample_flowdps(velocity_wrapper, masking_config, config, n_samples, nlev, nlat, nlon, device)
        else:
            masking_config = build_masking_config(obs_mask, obs_values, data_mean, data_std, column_weights, device)
            samples = sample_conditioned(velocity_wrapper, masking_config, config, n_samples, nlev, nlat, nlon, device)

        metrics = evaluate(samples, gt_norm, obs_mask, column_weights, data_mean, data_std)
        all_results[method_name] = {"samples": samples.cpu(), "metrics": metrics}
        print(f"  {metrics}")

    # 6. Save results (if out_dir provided)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_summary = {name: res["metrics"] for name, res in all_results.items()}
        with open(out_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        plot_results(all_results, gt_norm.cpu(), obs_mask.cpu(), column_weights, data_mean, data_std, out_dir)

    # 7. Print summary table
    _print_summary_table(all_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Toy column OSSE for debugging")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--obs_fraction", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        out_dir = repo_root / "experiments" / "toy_column_osse_output"
    else:
        out_dir = args.out_dir

    run_toy_osse(
        device=args.device,
        n_train=args.n_train,
        epochs=args.epochs,
        n_samples=args.n_samples,
        obs_fraction=args.obs_fraction,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
