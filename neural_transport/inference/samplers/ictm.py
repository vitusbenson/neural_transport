"""ICTM (Iterative Corrupted Trajectory Matching) posterior sampler.

Implements arXiv 2405.18816: Tweedie estimate + local MAP optimization
with time-varying regularization r(t).
"""

import math

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler
from neural_transport.models.flowmatching import _gaussian_smooth_2d


class ICTMSampler(PosteriorSampler):
    """ICTM posterior sampler with time-varying regularization.

    At each step t_n -> t_{n+1}:
      1. Tweedie estimate: x_hat_1 = x_t + (1 - t_n) * v_theta(x_t, t_n)
      2. MAP solve: adjust x_hat_1 with time-varying regularization r(t)
      3. Re-noise: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_proj

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        fresh_noise: If True, draw fresh z each step for re-noising.
        r_max: Maximum regularization strength.
        r_schedule: One of "constant", "decreasing", "increasing", "cosine".
        n_inner_steps: MAP optimization steps (1 = closed-form for linear H).
        inner_lr: Learning rate for gradient descent (only used when n_inner_steps > 1).
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        fresh_noise=True,
        r_max=1.0,
        r_schedule="decreasing",
        n_inner_steps=1,
        inner_lr=0.1,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.fresh_noise = fresh_noise
        self.r_max = r_max
        self.r_schedule = r_schedule
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

    def r(self, t):
        """Time-varying regularization schedule r(t).

        Args:
            t: scalar or tensor time in [0, 1].

        Returns:
            Regularization strength at time t (clamped to min=1e-6).
        """
        t_val = t.item() if isinstance(t, torch.Tensor) and t.dim() == 0 else t
        if not isinstance(t_val, int | float):
            t_val = float(t_val)

        if self.r_schedule == "constant":
            val = self.r_max
        elif self.r_schedule == "decreasing":
            val = self.r_max * (1.0 - t_val)
        elif self.r_schedule == "increasing":
            val = self.r_max * t_val
        elif self.r_schedule == "cosine":
            val = self.r_max * math.cos(math.pi * t_val / 2.0)
        else:
            raise ValueError(f"Unknown r_schedule: {self.r_schedule}")

        return max(val, 1e-6)

    def _linear_map_solve(self, x_hat, t):
        """Closed-form MAP solve for linear forward model H.

        Same as FlowDPS projection but with sigma_eff = sigma_obs / r(t).
        """
        sigma_eff = self.sigma_obs / self.r(t)
        return self.forward_model.project(
            x_hat,
            self.obs_values,
            self.obs_mask,
            sigma_eff,
            self.spatial_smoothing_sigma,
        )

    def _nonlinear_map_solve(self, x_hat, t):
        """Gradient descent MAP solve for general (nonlinear) forward model H.

        Minimizes: ||H(x) - y||^2 / (2*sigma_obs^2) + ||x - x_hat||^2 / (2*r(t)^2)
        """
        r_t = self.r(t)
        fm = self.forward_model
        h_ak = fm._get_h_ak_for_x(x_hat)

        x = x_hat.clone()

        for _ in range(self.n_inner_steps):
            # Observation gradient: H^T(H(x) - y) / sigma_obs^2
            xco2_x = fm.forward(x)
            obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2_x))
            column_error = torch.where(self.obs_mask, xco2_x - obs_safe, torch.zeros_like(xco2_x))

            if self.spatial_smoothing_sigma > 0:
                column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)

            grad_obs = h_ak * column_error / (self.sigma_obs**2)

            # Regularization gradient: (x - x_hat) / r(t)^2
            grad_prior = (x - x_hat) / (r_t**2)

            # Total gradient
            grad = grad_obs + grad_prior

            # Clip gradient norm
            grad_norm = grad.norm()
            if grad_norm > 1.0:
                grad = grad / grad_norm

            x = x - self.inner_lr * grad

        return x

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run ICTM sampling loop.

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

            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)

            # 1. Velocity prediction
            v_theta = self.velocity_model(x_t, t_tensor)

            # 2. Tweedie estimate
            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)

            # 3. MAP solve with time-varying regularization
            if self.n_inner_steps <= 1:
                x_hat_proj = self._linear_map_solve(x_hat, t_n)
            else:
                x_hat_proj = self._nonlinear_map_solve(x_hat, t_n)

            # 4. Re-noise with fresh or fixed noise
            if self.fresh_noise and i < len(time_grid) - 2:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat_proj, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
