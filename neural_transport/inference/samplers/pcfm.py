"""PCFM (Physics-Constrained Flow Matching) posterior sampler.

Reference: Utkarsh et al., 2025, arXiv 2506.04171.

Forward-shoot → column projection → lambda-blend → OT-interpolant re-noise.
With ``lambda_penalty=1.0`` this is equivalent to MCG (hard projection).
With ``lambda_penalty<1.0`` the projection is softened, blending the
projected and unprojected clean estimates.
"""

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler


class PCFMSampler(PosteriorSampler):
    """Physics-Constrained Flow Matching posterior sampler.

    At each step t_n -> t_{n+1}:
      1. Forward shoot: estimate clean sample x_hat via Tweedie or multi-step Euler.
      2. Column projection: x_hat_proj = project(x_hat, y).
      3. Lambda blend: x_hat_soft = (1 - lambda) * x_hat + lambda * x_hat_proj.
      4. OT interpolant back: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_soft.

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        fresh_noise: If True, draw fresh z each step for re-noising.
        n_forward_steps: Number of Euler steps for forward shooting (1 = single Tweedie).
        lambda_penalty: Blending weight for projection (1.0 = hard, 0.0 = none).
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        fresh_noise=True,
        n_forward_steps=1,
        lambda_penalty=1.0,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.fresh_noise = fresh_noise
        self.n_forward_steps = max(1, int(n_forward_steps))
        self.lambda_penalty = float(lambda_penalty)

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run PCFM sampling loop.

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
            x_hat = self._forward_shoot(x_t, t_n, self.n_forward_steps)

            # 2. Column projection
            x_hat_proj = self._project_column(x_hat)

            # 3. Lambda blend: soft constraint relaxation
            if self.lambda_penalty < 1.0:
                x_hat_proj = (1.0 - self.lambda_penalty) * x_hat + self.lambda_penalty * x_hat_proj

            # 4. Re-noise with fresh or fixed noise (OT interpolant back)
            if self.fresh_noise and i < len(time_grid) - 2:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat_proj, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
