"""FIG (Flow with Interpolant Guidance) posterior sampler.

Implements Ricci et al. (ICLR 2025): standard Euler ODE step followed by
gradient correction via a measurement interpolant.
"""

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler
from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d


class FIGSampler(PosteriorSampler):
    """FIG posterior sampler: Euler ODE + measurement interpolant guidance.

    For each ODE step i (t -> next_t):
      1. Euler step: x_next = x + v_theta(x, t) * dt
      2. Measurement interpolant: y_t = next_t * y + w * (1-t) * H(noise)
      3. K gradient corrections (optionally skip first & last step):
         grad = H^T(H(x_next) - y_t) / (||h*a||^2 + sigma_obs^2)
         x_next = x_next - c * (1-t)/t * grad

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for gradient regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        k_steps: Number of gradient correction steps per ODE step.
        step_size_c: Step size multiplier for gradient correction.
        noise_scale_w: Scale of noise in measurement interpolant (0 = noiseless).
        skip_first_last: If True, skip gradient correction at first and last steps.
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        k_steps=1,
        step_size_c=10.0,
        noise_scale_w=0.0,
        skip_first_last=True,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.k_steps = k_steps
        self.step_size_c = step_size_c
        self.noise_scale_w = noise_scale_w
        self.skip_first_last = skip_first_last

    def _measurement_interpolant(self, y, t, next_t, noise):
        """Compute measurement interpolant target.

        y_t = next_t * y + w * (1-t) * H(noise)

        Smoothly ramps from noise toward the true observation as t -> 1.
        """
        h_noise = self.forward_model.forward(noise)  # [B, 1, Nlat, Nlon]
        return next_t * y + self.noise_scale_w * (1.0 - t) * h_noise

    def _compute_gradient(self, x, y_t):
        """Compute analytical gradient for linear forward model.

        grad_k = h_k * a_k * (H(x) - y_t) / (||h*a||^2 + sigma_obs^2)

        Only applies at observed locations (via obs_mask).
        Note: FIG uses H(x) - y_t (positive error sign), so cannot use project() directly.
        """
        fm = self.forward_model
        h_ak = fm._get_h_ak_for_x(x)

        # Forward model: H(x)
        xco2_x = fm.forward(x)  # [B, 1, Nlat, Nlon]

        # Column error at observed locations: H(x) - y_t
        column_error = torch.where(
            self.obs_mask,
            xco2_x - y_t,
            torch.zeros_like(xco2_x),
        )

        # Optional spatial smoothing
        if self.spatial_smoothing_sigma > 0:
            column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)

        # Denominator: sum_j (h_j * a_j)^2 + sigma_obs^2
        h_ak_sq_sum = (h_ak**2).sum(dim=1, keepdim=True)
        denom = h_ak_sq_sum + self.sigma_obs**2

        return h_ak * column_error / denom  # [B, nlev, Nlat, Nlon]

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run FIG sampling loop: Euler ODE + measurement interpolant guidance.

        Args:
            x_init: [B, C, Nlat, Nlon] initial noise.
            time_grid: [T] time points from 0 to 1.
            return_intermediates: If True, return [T, B, C, Nlat, Nlon].

        Returns:
            Final samples [B, C, Nlat, Nlon], or trajectory if return_intermediates.
        """
        x_t = x_init
        noise = torch.randn_like(x_init)  # fixed noise for interpolant
        n_steps = len(time_grid) - 1

        # Observation target (normalized)
        obs_safe = torch.where(
            self.obs_mask,
            self.obs_values.detach(),
            torch.zeros_like(self.obs_values),
        )

        intermediates = [x_t] if return_intermediates else None

        for i in range(n_steps):
            t_n = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t_n

            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)

            # 1. Euler step: x_next = x + v_theta(x, t) * dt
            v_theta = self.velocity_model(x_t, t_tensor)
            x_next = x_t + v_theta * dt

            # 2. Gradient correction (skip first and last if configured)
            skip = self.skip_first_last and (i == 0 or i == n_steps - 1)
            if not skip:
                # Use next_t for the schedule ratio (post-step time), clamp to avoid blow-up
                t_next_val = float(t_next.item() if isinstance(t_next, torch.Tensor) else t_next)
                t_next_clamped = max(t_next_val, 1e-2)
                ratio = min((1.0 - t_next_clamped) / t_next_clamped, 10.0)

                # Measurement interpolant target
                y_t = self._measurement_interpolant(obs_safe, t_next_clamped, t_next_val, noise)

                # K gradient correction steps
                step_scale = self.step_size_c * ratio
                for _ in range(self.k_steps):
                    grad = self._compute_gradient(x_next, y_t)
                    # Clip gradient norm to prevent divergence
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    x_next = x_next - step_scale * grad

            x_t = x_next

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
