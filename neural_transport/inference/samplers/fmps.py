"""FMPS (Flow Matching Posterior Sampling) sampler.

Reference: arXiv 2411.07625.

Steers the velocity field directly toward observation consistency by adding
a scaled likelihood gradient correction term:
    dx = [v_theta(x, t) - r(t) * grad_likelihood] dt

Unlike project-renoise methods (FlowDPS, MCG, PCFM), FMPS modifies the
velocity field and uses a pure Euler step. Optionally integrates
DiffStateGrad (SVD subspace projection) and FGPS (spectral filtering).
"""

import math

import torch
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler


class FMPSSampler(PosteriorSampler):
    """Flow Matching Posterior Sampling via velocity correction.

    At each step t_n -> t_{n+1}:
      1. Velocity prediction: v = model(x_t, t).
      2. Tweedie estimate: x_hat = x_t + (1 - t) * v.
      3. Likelihood gradient: grad = h_ak * (H(x_hat) - y) / sigma_obs^2.
      4. Optional SVD projection of gradient (DiffStateGrad).
      5. Corrected velocity: v_corr = v - r(t) * grad.
      6. Euler step: x_{t+dt} = x_t + v_corr * dt.
      7. Optional spectral filtering (FGPS).

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for pseudoinverse regularization.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        guidance_strength: Scale factor for the correction term.
        r_schedule: Schedule for r(t): "linear" = strength*(1-t),
            "cosine" = strength*cos(pi*t/2), "constant" = strength.
        svd_rank: If > 0, project gradient onto top-k SVD subspace (DiffStateGrad).
        spectral_k_low: Min frequency cutoff for FGPS (0 = disabled).
        spectral_k_high: Max frequency cutoff for FGPS (0 = disabled).
        grad_clip_norm: Clip gradient norm to this value (0 = no clipping).
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        guidance_strength=1.0,
        r_schedule="linear",
        svd_rank=0,
        spectral_k_low=0,
        spectral_k_high=0,
        grad_clip_norm=1.0,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.guidance_strength = guidance_strength
        self.r_schedule = r_schedule
        self.svd_rank = svd_rank
        self.spectral_k_low = spectral_k_low
        self.spectral_k_high = spectral_k_high
        self.grad_clip_norm = grad_clip_norm

    def _r(self, t: float) -> float:
        """Compute time-dependent guidance scale r(t).

        Args:
            t: current time in [0, 1].

        Returns:
            Scale factor for likelihood gradient.
        """
        if self.r_schedule == "linear":
            return self.guidance_strength * (1.0 - t)
        elif self.r_schedule == "cosine":
            return self.guidance_strength * math.cos(math.pi * t / 2.0)
        elif self.r_schedule == "constant":
            return self.guidance_strength
        else:
            raise ValueError(f"Unknown r_schedule: {self.r_schedule}")

    @torch.no_grad()
    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run FMPS sampling loop.

        Args:
            x_init: [B, C, Nlat, Nlon] initial noise.
            time_grid: [T] time points from 0 to 1.
            return_intermediates: If True, return [T, B, C, Nlat, Nlon].

        Returns:
            Final samples [B, C, Nlat, Nlon], or trajectory if return_intermediates.
        """
        x_t = x_init

        intermediates = [x_t] if return_intermediates else None

        for i in range(len(time_grid) - 1):
            t_n = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t_n

            t_tensor = t_n * torch.ones(x_t.shape[0], device=x_t.device)

            # 1. Velocity prediction
            v_theta = self.velocity_model(x_t, t_tensor)

            # 2. Tweedie estimate for likelihood gradient
            x_hat = self._tweedie_estimate(x_t, t_tensor, v_theta)

            # 3. Likelihood gradient (points away from observations)
            grad = self._compute_likelihood_gradient(x_hat)

            # 4. Optional SVD projection (DiffStateGrad)
            if self.svd_rank > 0:
                grad = self._project_to_svd_subspace(grad, x_t, self.svd_rank)

            # 5. Gradient norm clipping to prevent divergence
            if self.grad_clip_norm > 0:
                grad_norm = grad.norm()
                if grad_norm > self.grad_clip_norm:
                    grad = grad * (self.grad_clip_norm / grad_norm)

            # 6. Corrected velocity: subtract gradient (it points away from obs)
            r_t = self._r(float(t_n))
            v_corrected = v_theta - r_t * grad

            # 7. Euler step
            x_t = x_t + v_corrected * dt

            # 8. Optional spectral filtering (FGPS)
            if self.spectral_k_low > 0 and self.spectral_k_high > 0:
                x_t = self._spectral_filter(x_t, float(t_next), self.spectral_k_low, self.spectral_k_high)

            if return_intermediates:
                intermediates.append(x_t)

        if return_intermediates:
            return torch.stack(intermediates, dim=0)
        return x_t
