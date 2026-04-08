"""SDE posterior sampler with optional Langevin corrector.

Combines FlowDPS projection with stochastic noise injection for better
posterior exploration.
"""

import math

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler
from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d


class StochasticPosteriorSampler(PosteriorSampler):
    """SDE posterior sampler with optional Langevin corrector.

    Combines FlowDPS projection with stochastic noise injection for better
    posterior exploration. The SDE formulation:
        dx = v_theta(x,t)dt + sigma(t)dW_t

    At each step:
      1. Tweedie estimate via velocity model
      2. Column projection (FlowDPS)
      3. Re-noise with CondOT interpolation
      4. Add SDE diffusion noise: sigma(t) * sqrt(dt) * z
      5. Optional Langevin corrector steps targeting p(x_t|y)

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        fresh_noise: If True, draw fresh z each step for re-noising.
        sigma_max: Maximum SDE noise scale.
        noise_schedule: Schedule type: "annealed", "constant", or "cosine".
        n_corrector_steps: Number of Langevin corrector steps per predictor step (0 = pure SDE).
        corrector_step_size: Step size epsilon for Langevin corrector.
        corrector_snr: Signal-to-noise ratio for SNR-based step size (unused if corrector_step_size > 0).
        use_projection: If True, apply column projection after Tweedie estimate.
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        fresh_noise=True,
        sigma_max=0.5,
        noise_schedule="annealed",
        n_corrector_steps=0,
        corrector_step_size=0.01,
        corrector_snr=0.16,
        use_projection=True,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.fresh_noise = fresh_noise
        self.sigma_max = sigma_max
        self.noise_schedule = noise_schedule
        self.n_corrector_steps = n_corrector_steps
        self.corrector_step_size = corrector_step_size
        self.corrector_snr = corrector_snr
        self.use_projection = use_projection

    def sigma(self, t):
        """Noise schedule sigma(t).

        Args:
            t: scalar or tensor time in [0, 1].

        Returns:
            Noise level at time t.
        """
        if self.noise_schedule == "annealed":
            return self.sigma_max * (1.0 - t)
        elif self.noise_schedule == "constant":
            return self.sigma_max
        elif self.noise_schedule == "cosine":
            t_val = t.item() if isinstance(t, torch.Tensor) and t.dim() == 0 else t
            if isinstance(t_val, int | float):
                return self.sigma_max * math.cos(math.pi * t_val / 2.0)
            return self.sigma_max * torch.cos(torch.tensor(math.pi) * t / 2.0)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def _langevin_corrector(self, x_t, t, z):
        """K Langevin MCMC corrector steps targeting p(x_t|y).

        Score = prior score (Tweedie) + likelihood gradient (DPS).
        Step: x <- x + eps * (score + lik_grad) + sqrt(2*eps) * noise

        Args:
            x_t: Current state [B, C, Nlat, Nlon].
            t: Current time (scalar tensor).
            z: Base noise for re-noising.

        Returns:
            Corrected x_t.
        """
        eps = self.corrector_step_size
        t_tensor = t * torch.ones(x_t.shape[0], device=x_t.device)
        t_scalar = t_tensor.view(-1, 1, 1, 1)

        for _ in range(self.n_corrector_steps):
            v_theta = self.velocity_model(x_t, t_tensor)

            # Prior score from Tweedie: score = v_theta / (1 - t)
            score = v_theta / (1.0 - t_scalar).clamp(min=1e-6)

            # Likelihood gradient (same as DPS Phase 5)
            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)

            fm = self.forward_model
            h_ak = fm._get_h_ak_for_x(x_t)

            xco2_hat = fm.forward(x_hat)
            if self.obs_weight is not None:
                column_error = self.obs_weight * (self.obs_values.detach() - xco2_hat)
            else:
                obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2_hat))
                column_error = torch.where(self.obs_mask, obs_safe - xco2_hat, torch.zeros_like(xco2_hat))

            if self.spatial_smoothing_sigma > 0:
                column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)

            lik_grad = h_ak * column_error / (self.sigma_obs**2)

            # Langevin step
            noise = torch.randn_like(x_t)
            x_t = x_t + eps * (score + lik_grad) + (2.0 * eps) ** 0.5 * noise

        return x_t

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run SDE sampling loop with optional Langevin corrector.

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
            dt = t_next - t_n
            is_final = i == len(time_grid) - 2

            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)

            # 1. Velocity prediction
            v_theta = self.velocity_model(x_t, t_tensor)

            # 2. Tweedie estimate
            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)

            # 3. Column projection (optional)
            if self.use_projection:
                x_hat = self._project_column(x_hat)

            # 4. Re-noise with fresh or fixed noise
            if self.fresh_noise and not is_final:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self._renoise(x_hat, t_next_tensor, z)

            # 5. Add SDE noise (skip at final step)
            if not is_final and self.sigma_max > 0:
                sig = self.sigma(t_n)
                x_t = x_t + sig * dt.sqrt() * torch.randn_like(x_t)

            # 6. Langevin corrector (skip at final step)
            if not is_final and self.n_corrector_steps > 0:
                x_t = self._langevin_corrector(x_t, t_next, z)

                # Re-project after corrector to maintain column constraint
                if self.use_projection:
                    v_corr = self.velocity_model(x_t, t_next * torch.ones(x_t.shape[0], device=x_t.device))
                    x_hat_corr = self._tweedie_estimate(
                        x_t, t_next * torch.ones(x_t.shape[0], device=x_t.device), v_corr
                    )
                    x_hat_corr = self._project_column(x_hat_corr)
                    if self.fresh_noise:
                        z = torch.randn_like(x_t)
                    x_t = self._renoise(x_hat_corr, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
