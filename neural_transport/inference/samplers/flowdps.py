"""FlowDPS posterior sampler (Kim et al., ICCV 2025).

Tweedie estimate -> column projection -> re-noise cycle.
Inherits shared infrastructure from PosteriorSampler.
"""

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler


class FlowDPSSampler(PosteriorSampler):
    """FlowDPS posterior sampler via projection.

    For CondOT flow matching (x_t = (1-t)*x_0 + t*x_1), at each step t_n -> t_{n+1}:
      1. Tweedie estimate: x_hat_1 = x_t + (1 - t_n) * v_theta(x_t, t_n)
      2. Column projection: adjust x_hat_1 at observed locations
      3. Re-noise: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_1_proj

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization. Smaller = harder constraint.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        fresh_noise: If True, draw fresh z each step. If False, reuse initial noise.
    """

    def __init__(self, velocity_model, masking_config, sigma_obs=0.1, spatial_smoothing_sigma=0.0, fresh_noise=True):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.fresh_noise = fresh_noise

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run FlowDPS sampling loop.

        Args:
            x_init: [B, C, Nlat, Nlon] initial noise.
            time_grid: [T] time points from 0 to 1.
            return_intermediates: If True, return [T, B, C, Nlat, Nlon].

        Returns:
            Final samples [B, C, Nlat, Nlon], or trajectory if return_intermediates.
        """
        x_t = x_init
        z = x_init.clone()  # initial noise for re-noising

        intermediates = [x_t] if return_intermediates else None

        for i in range(len(time_grid) - 1):
            t_n = time_grid[i]
            t_next = time_grid[i + 1]

            # Ensure t is a tensor
            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)

            # 1. Velocity prediction
            v_theta = self.velocity_model(x_t, t_tensor)

            # 2. Tweedie estimate
            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)

            # 3. Column projection
            x_hat_proj = self._project_column(x_hat)

            # 4. Re-noise with fresh or fixed noise
            if self.fresh_noise and i < len(time_grid) - 2:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat_proj, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)  # [T, B, C, Nlat, Nlon]
        return x_t
