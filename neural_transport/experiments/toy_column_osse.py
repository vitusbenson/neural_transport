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
from torch.utils.data import DataLoader, TensorDataset

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

from neural_transport.models.flowmatching import MaskedVelocityWrapper


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
                -0.5 * ((lat_mesh - center_lat) / sigma_lat) ** 2
                - 0.5 * ((lon_mesh - center_lon) / sigma_lon) ** 2
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

def train_flow_matching(data, nlev=4, epochs=50, lr=1e-3, batch_size=128, device="cpu"):
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
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    print(f"  Training took {time.time()-t0:.1f}s")
    model.eval()
    return model, data_mean.item(), data_std.item()


# ── Conditioning test ───────────────────────────────────────────────────

def sample_unconditional(velocity_wrapper, n_samples, nlev, nlat, nlon, device, steps=20):
    """Generate unconditional samples."""
    x_init = torch.randn(n_samples, nlev, nlat, nlon, device=device)
    time_grid = torch.linspace(0, 1, steps, device=device)

    solver = ODESolver(velocity_model=velocity_wrapper)
    traj = solver.sample(time_grid=time_grid, x_init=x_init,
                         method="midpoint", step_size=None,
                         return_intermediates=False)
    return traj


def sample_conditioned(model_wrapper, masking_config, generate_kwargs,
                       n_samples, nlev, nlat, nlon, device, steps=20):
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
    traj = solver.sample(time_grid=time_grid, x_init=x_init,
                         method="midpoint", step_size=None,
                         return_intermediates=False)
    return traj


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

def plot_results(all_results, gt_field, obs_mask, column_weights, data_mean, data_std,
                 out_dir):
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
        axes[row, 0].set_title(f"Pred XCO2")

        xco2_diff = np.abs(pred_xco2 - gt_xco2)
        dmax = max(np.nanpercentile(xco2_diff, 98), 1e-6)
        axes[row, 1].imshow(xco2_diff, origin="lower", cmap="Reds", vmin=0, vmax=dmax, aspect="auto")
        axes[row, 1].set_title("|XCO2 Diff|")

        pred_lev0 = ens_mean[0].cpu().numpy()
        axes[row, 2].imshow(pred_lev0, origin="lower", cmap="cividis", aspect="auto")
        axes[row, 2].set_title("Pred Level 0")

        metrics = result["metrics"]
        metrics_str = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        axes[row, 3].text(0.1, 0.5, metrics_str, transform=axes[row, 3].transAxes,
                          fontsize=8, va="center", family="monospace")
        axes[row, 3].axis("off")

        axes[row, 0].set_ylabel(method_name.replace("_", "\n"), fontsize=9,
                                rotation=0, labelpad=60, va="center")

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
}


def main():
    parser = argparse.ArgumentParser(description="Toy column OSSE for debugging")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--obs_fraction", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    nlat, nlon, nlev = 16, 32, 4

    if args.out_dir is None:
        out_dir = Path(__file__).resolve().parent / "toy_column_osse_output"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate data
    print("Generating toy data...")
    data = generate_toy_data(n_samples=args.n_train, nlat=nlat, nlon=nlon, nlev=nlev)
    column_weights = get_column_weights(nlev)

    # 2. Train flow matching model
    print("Training flow matching model...")
    conv_model, data_mean, data_std = train_flow_matching(
        data, nlev=nlev, epochs=args.epochs, lr=1e-3, device=device)

    velocity_wrapper = ToyVelocityWrapper(conv_model, nlev=nlev).to(device)

    # 3. Pick a ground truth sample and create column obs
    gt_idx = 0
    gt_norm = ((data[gt_idx:gt_idx+1] - data_mean) / data_std).to(device)
    gt_phys = data[gt_idx:gt_idx+1].to(device)

    obs_mask, obs_values = create_column_obs(
        gt_phys, column_weights, obs_fraction=args.obs_fraction, seed=123)
    obs_mask = obs_mask.to(device)
    obs_values = obs_values.to(device)

    print(f"GT field shape: {gt_norm.shape}")
    print(f"Obs mask: {obs_mask.sum().item()}/{obs_mask.numel()} locations observed")
    print(f"Column weights: {column_weights.tolist()}")

    # 4. Test each conditioning method
    all_results = {}

    for method_name, method_config in CONDITIONING_METHODS.items():
        print(f"\nTesting: {method_name}")
        config = {k: v for k, v in method_config.items()}
        is_unconditional = config.pop("_unconditional", False)

        torch.manual_seed(42)

        if is_unconditional:
            samples = sample_unconditional(
                velocity_wrapper, args.n_samples, nlev, nlat, nlon, device)
        else:
            masking_config = build_masking_config(
                obs_mask, obs_values, data_mean, data_std, column_weights, device)
            samples = sample_conditioned(
                velocity_wrapper, masking_config, config,
                args.n_samples, nlev, nlat, nlon, device)

        metrics = evaluate(samples, gt_norm, obs_mask, column_weights, data_mean, data_std)
        all_results[method_name] = {"samples": samples.cpu(), "metrics": metrics}
        print(f"  {metrics}")

    # 5. Save results
    metrics_summary = {name: res["metrics"] for name, res in all_results.items()}
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    plot_results(all_results, gt_norm.cpu(), obs_mask.cpu(), column_weights,
                 data_mean, data_std, out_dir)

    # 6. Print summary table
    print(f"\n{'='*100}")
    header = f"{'Method':<22} {'RMSE_3d':>10} {'RMSE_3d_o':>10} {'RMSE_3d_a':>10} {'RMSE_xco2':>10} {'RMSE_xo':>10} {'RMSE_xa':>10}"
    print(header)
    print(f"{'-'*100}")
    for name, res in all_results.items():
        m = res["metrics"]
        def _f(v):
            return f"{v:.4f}" if not np.isnan(v) else "N/A"
        print(f"{name:<22} {_f(m['rmse_3d_full']):>10} {_f(m['rmse_3d_obs']):>10} "
              f"{_f(m['rmse_3d_away']):>10} {_f(m['rmse_xco2_full']):>10} "
              f"{_f(m['rmse_xco2_obs']):>10} {_f(m['rmse_xco2_away']):>10}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
