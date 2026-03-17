"""Noise generation utilities for posterior sampling.

Extracted from generative.py — provides structured noise patterns
(spiral, geodesic, antipodal, etc.) for flow matching inference.
"""

from pathlib import Path

import torch


def generate_noise(batch, target_var="co2massmix", n_samples=10, noise_pattern=None):
    all_levels = batch[target_var]  # [T N C]
    all_levels = all_levels.unsqueeze(0)  # [B T N C]

    if noise_pattern is None:
        return [torch.randn_like(all_levels) for _ in range(n_samples)]

    elif noise_pattern == "spiral_outward_noise":
        # Step 1: pick a reference noise vector
        x0 = torch.randn_like(all_levels)
        # Step 2: pick a direction vector (independent random noise)
        v = torch.randn_like(all_levels)
        # Step 3: create spiral path
        angles = torch.linspace(0, 4 * torch.pi, n_samples)

        spiral_noises = []
        for a in angles:
            # Step 4: rotate between x0 and v
            x_init = torch.cos(a) * x0 + torch.sin(a) * v
            # Step 5: scale radius outward
            r = 1.0 + 0.2 * (a / angles[-1])  # gradually increase radius
            x_init_scaled = r * x_init
            spiral_noises.append(x_init_scaled)
        return spiral_noises

    elif noise_pattern == "spiral_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        angles = torch.linspace(0, 4 * torch.pi, n_samples)

        spiral_noises = []
        for a in angles:
            x_init = torch.cos(a) * x0 + torch.sin(a) * v
            spiral_noises.append(x_init)
        return spiral_noises

    elif noise_pattern == "geodesic_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [torch.sqrt(1 - alpha) * x0 + torch.sqrt(alpha) * v for alpha in alphas]

    elif noise_pattern == "linear_noise":
        x0 = torch.randn_like(all_levels)
        v = torch.randn_like(all_levels)
        alphas = torch.linspace(0, 1, n_samples)
        return [(1 - alpha) * x0 + alpha * v for alpha in alphas]

    elif noise_pattern == "antipodal_orthogonal_noise":
        x0 = torch.randn_like(all_levels)
        n_dirs = n_samples // 2  # each direction will yield a +v and -v pair
        # Start with random Gaussian directions
        dirs = [torch.randn_like(x0).flatten() for _ in range(n_dirs)]
        # Orthogonalize via Gram–Schmidt
        orth_dirs = []
        for v in dirs:
            for u in orth_dirs:
                v -= (v @ u) / (u @ u) * u
            orth_dirs.append(v)

        # Convert back to tensor shape and include antipodal pairs
        orth_dirs = [v.reshape_as(x0) for v in orth_dirs]
        all_noises = []
        for v in orth_dirs:
            all_noises.append(v)
            all_noises.append(-v)

        # If we have fewer than n_samples due to rounding
        if len(all_noises) < n_samples:
            all_noises.append(torch.randn_like(x0))

        return all_noises[:n_samples]

    else:
        raise ValueError(f"Unknown noise type: {noise_pattern}")


def noise(
    batch: dict,
    target_var: str | None = "co2massmix",
    n_samples: int | None = 10,
    noise_pattern: str | None = None,
    analyze_noise: bool = False,
    outpath: Path | str = None,
) -> list:
    noise_list = generate_noise(batch, target_var=target_var, n_samples=n_samples, noise_pattern=noise_pattern)
    if analyze_noise and noise_pattern is not None:
        from neural_transport.plots.plot_results import plot_noise_diagnostics

        if noise_pattern in ["spiral_noise", "spiral_outward_noise"]:
            angles = torch.linspace(0, 4 * torch.pi, n_samples)  # thetas
            param_name = "$\\theta$"
        elif noise_pattern in ["geodesic_noise", "linear_noise"]:
            angles = torch.linspace(0, 1, n_samples)  # alphas
            param_name = "$\\alpha$"
        elif noise_pattern == "antipodal_orthogonal_noise":
            labels = []
            for i in range(n_samples // 2):
                labels += [f"{i + 1}a", f"{i + 1}b"]
            if n_samples % 2 == 1:
                labels.append(f"{(n_samples // 2) + 1}a")
            angles = labels
            param_name = "Index pair"
        else:
            angles = torch.arange(n_samples)  # index
            param_name = "Index"
        plot_noise_diagnostics(
            noise_list, angles, str(outpath).replace("preds", "plots"), label=param_name, imgformats=["png"]
        )
    return noise_list
