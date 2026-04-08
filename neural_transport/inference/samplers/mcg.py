"""MCG (Manifold Constrained Gradient) posterior sampler — fixed OT-interpolant version.

Replaces the original manifold projection (Chung et al., NeurIPS 2022) with a
PCFM-style OT-interpolant approach (Utkarsh et al., 2025, arXiv 2506.04171).

Instead of re-noising + re-velocity (which destroys the conditioning signal),
we forward-shoot to estimate the clean sample, project onto the column
measurement manifold, and OT-interpolant back.  With ``n_forward_steps=1``
this is equivalent to FlowDPS; with ``n_forward_steps>1`` we use multi-step
Euler integration for a better clean estimate before projection.
"""

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler


class MCGSampler(PosteriorSampler):
    """Manifold-Constrained Gradient posterior sampler (OT-interpolant version).

    At each step t_n -> t_{n+1}:
      1. Forward shoot: estimate clean sample x_hat via Tweedie (n_forward_steps=1)
         or multi-step Euler integration to t=1 (n_forward_steps>1).
      2. Column projection: x_hat_proj = project(x_hat, y)
      3. OT interpolant back: x_{t+1} = (1-t_{n+1})*z + t_{n+1}*x_hat_proj

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        fresh_noise: If True, draw fresh z each step for re-noising.
        n_forward_steps: Number of Euler steps for forward shooting (1 = single Tweedie).
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        fresh_noise=True,
        n_forward_steps=1,
        # Deprecated — kept for backward compat with old configs/Optuna studies
        manifold_alpha=0.5,
        n_manifold_steps=1,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.fresh_noise = fresh_noise
        self.n_forward_steps = max(1, int(n_forward_steps))

    def _forward_shoot(self, x_t: Tensor, t: float) -> Tensor:
        """Estimate clean sample by integrating ODE from t to 1.

        Args:
            x_t: [B, C, Nlat, Nlon] current noisy state.
            t: current time (scalar).

        Returns:
            x_hat_1: [B, C, Nlat, Nlon] estimated clean sample at t=1.
        """
        if self.n_forward_steps == 1:
            t_tensor = t * torch.ones(x_t.shape[0], device=x_t.device)
            v_theta = self.velocity_model(x_t, t_tensor)
            return self._tweedie_estimate(x_t, t_tensor, v_theta)

        # Multi-step Euler from t to 1
        x = x_t
        dt = (1.0 - t) / self.n_forward_steps
        for k in range(self.n_forward_steps):
            t_k = t + k * dt
            t_tensor = t_k * torch.ones(x.shape[0], device=x.device)
            v = self.velocity_model(x, t_tensor)
            x = x + v * dt
        return x

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run MCG sampling loop.

        Args:
            x_init: [B, C, Nlat, Nlon] initial noise.
            time_grid: [T] time points from 0 to 1.
            return_intermediates: If True, return [T, B, C, Nlat, Nlon].

        Returns:
            Final samples [B, C, Nlat, Nlon], or trajectory if return_intermediates.
        """
        x_t = x_init
        z = x_init.clone()

        intermediates = [x_t] if return_intermediates else None

        for i in range(len(time_grid) - 1):
            t_n = time_grid[i]
            t_next = time_grid[i + 1]

            # 1. Forward shoot: estimate clean sample
            x_hat = self._forward_shoot(x_t, t_n)

            # 2. Column projection
            x_hat_proj = self._project_column(x_hat)

            # 3. Re-noise with fresh or fixed noise (OT interpolant back)
            if self.fresh_noise and i < len(time_grid) - 2:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat_proj, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
