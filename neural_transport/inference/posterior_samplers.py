"""
Posterior samplers for flow matching models.

FlowDPSSampler implements the FlowDPS algorithm (Kim et al., ICCV 2025):
Tweedie estimate -> column projection -> re-noise cycle.

This REPLACES the ODE solver loop — no velocity integration.
"""

import torch

from neural_transport.models.flowmatching import _gaussian_smooth_2d


class FlowDPSSampler:
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
        self.velocity_model = velocity_model
        self.sigma_obs = sigma_obs
        self.spatial_smoothing_sigma = spatial_smoothing_sigma
        self.fresh_noise = fresh_noise

        # Unpack masking config
        self.obs_mask = masking_config.get("obs_mask", None)
        self.obs_values = masking_config.get("obs_values", None)
        self.obs_mean = masking_config.get("obs_mean", None)
        self.obs_std = masking_config.get("obs_std", None)
        self.target_mean = masking_config.get("target_mean", None)
        self.target_std = masking_config.get("target_std", None)
        self.ak = masking_config.get("ak", None)
        self.pressure_weights = masking_config.get("pressure_weights", None)
        self.xco2_prior = masking_config.get("xco2_prior", None)
        self.co2_profile_prior = masking_config.get("co2_profile_prior", None)
        self.targshift_mean = masking_config.get("targshift_mean", None)

    def compute_xco2(self, x):
        """OCO-2 forward model: XCO2 = xco2_prior + sum(h * a * (x - x_prior)).

        Copied from MaskedVelocityWrapper.compute_xco2 for consistency.
        """
        h = self.pressure_weights
        if h is None:
            h = 1.0 / x.shape[1]

        if self.xco2_prior is not None and self.co2_profile_prior is not None:
            x_corrected = x + self.targshift_mean if self.targshift_mean is not None else x
            x_phys = x_corrected * self.target_std + self.target_mean
            xco2 = self.xco2_prior + (h * self.ak * (x_phys - self.co2_profile_prior)).sum(dim=1, keepdim=True)
            return (xco2 - self.obs_mean) / self.obs_std
        else:
            h_ak = h * self.ak if self.ak is not None else h
            h_ak_sum = h_ak.sum(dim=1, keepdim=True).clamp(min=1e-12)
            result = (h_ak * x).sum(dim=1, keepdim=True) + (self.target_mean / self.target_std) * (h_ak_sum - 1.0)
            if self.targshift_mean is not None:
                result = result + self.targshift_mean * h_ak_sum
            return result

    def _tweedie_estimate(self, x_t, t, v_theta):
        """Tweedie denoising estimate: x_hat_1 = x_t + (1 - t) * v_theta(x_t, t).

        For CondOT: x_t = (1-t)*x_0 + t*x_1, so the clean estimate at t is:
        x_hat_1 = x_t + (1 - t) * v, where v approximates (x_1 - x_0).
        """
        t_scalar = t.view(-1, 1, 1, 1) if t.dim() == 1 else t
        return x_t + (1.0 - t_scalar) * v_theta

    def _project_column(self, x_hat):
        """Project Tweedie estimate onto column measurement manifold.

        x_hat_k += (h_k * a_k) * (y - H(x_hat)) / (||h*a||_2^2 + sigma^2)

        Only modifies at observed locations (via obs_mask).
        """
        h = self.pressure_weights
        if h is None:
            h = 1.0 / x_hat.shape[1]

        h_ak = h * self.ak if self.ak is not None else h

        # Forward model: H(x_hat)
        xco2_hat = self.compute_xco2(x_hat)  # [B, 1, Nlat, Nlon]

        # Column error: y - H(x_hat), zeroed at unobserved locations
        obs_safe = torch.where(self.obs_mask, self.obs_values.detach(), torch.zeros_like(xco2_hat))
        column_error = torch.where(self.obs_mask, obs_safe - xco2_hat, torch.zeros_like(xco2_hat))

        # Optional spatial smoothing of column error
        if self.spatial_smoothing_sigma > 0:
            column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)

        # Denominator: sum_j (h_j * a_j)^2 per spatial location
        # h_ak is [B, nlev, Nlat, Nlon], square and sum over level dim
        h_ak_sq_sum = (h_ak**2).sum(dim=1, keepdim=True)  # [B, 1, Nlat, Nlon]
        denom = h_ak_sq_sum + self.sigma_obs**2  # [B, 1, Nlat, Nlon]

        # Correction: (h_k * a_k) * column_error / denom
        # h_ak is [B, nlev, Nlat, Nlon], column_error/denom are [B, 1, Nlat, Nlon]
        correction = h_ak * column_error / denom  # [B, nlev, Nlat, Nlon]

        return x_hat + correction

    def _renoise(self, x_hat_proj, t_next, z):
        """Re-noise: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_proj.

        At t_next=1.0, this gives pure signal (no noise).
        """
        t_next_scalar = t_next.view(-1, 1, 1, 1) if t_next.dim() >= 1 else t_next
        return (1.0 - t_next_scalar) * z + t_next_scalar * x_hat_proj

    @torch.no_grad()
    def sample(self, x_init, time_grid, return_intermediates=False):
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


class StochasticPosteriorSampler:
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
        self.flowdps = FlowDPSSampler(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma, fresh_noise)
        self.velocity_model = velocity_model
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
            import math

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
            x_hat = self.flowdps._tweedie_estimate(x_t, t_tensor, v_theta)

            h = self.flowdps.pressure_weights
            if h is None:
                h = 1.0 / x_t.shape[1]
            h_ak = h * self.flowdps.ak if self.flowdps.ak is not None else h

            xco2_hat = self.flowdps.compute_xco2(x_hat)
            obs_safe = torch.where(self.flowdps.obs_mask, self.flowdps.obs_values.detach(), torch.zeros_like(xco2_hat))
            column_error = torch.where(self.flowdps.obs_mask, obs_safe - xco2_hat, torch.zeros_like(xco2_hat))

            if self.flowdps.spatial_smoothing_sigma > 0:
                column_error = _gaussian_smooth_2d(column_error, self.flowdps.spatial_smoothing_sigma)

            lik_grad = h_ak * column_error / (self.flowdps.sigma_obs**2)

            # Langevin step
            noise = torch.randn_like(x_t)
            x_t = x_t + eps * (score + lik_grad) + (2.0 * eps) ** 0.5 * noise

        return x_t

    @torch.no_grad()
    def sample(self, x_init, time_grid, return_intermediates=False):
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
            x_hat = self.flowdps._tweedie_estimate(x_t, t_tensor, v_theta)

            # 3. Column projection (optional)
            if self.use_projection:
                x_hat = self.flowdps._project_column(x_hat)

            # 4. Re-noise with fresh or fixed noise
            if self.flowdps.fresh_noise and not is_final:
                z = torch.randn_like(x_t)

            t_next_tensor = t_next * torch.ones(x_t.shape[0], device=x_t.device)
            x_t = self.flowdps._renoise(x_hat, t_next_tensor, z)

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
                    x_hat_corr = self.flowdps._tweedie_estimate(
                        x_t, t_next * torch.ones(x_t.shape[0], device=x_t.device), v_corr
                    )
                    x_hat_corr = self.flowdps._project_column(x_hat_corr)
                    if self.flowdps.fresh_noise:
                        z = torch.randn_like(x_t)
                    x_t = self.flowdps._renoise(x_hat_corr, t_next_tensor, z)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
