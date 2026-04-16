"""Abstract base classes for flow matching samplers.

BaseSampler: minimal interface — just sample().
PosteriorSampler: adds shared Tweedie/project/renoise logic used by
FlowDPS, SDE, FIG, and ICTM posterior samplers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from neural_transport.forward_model import XCO2ForwardModel
from neural_transport.tools.spatial import gaussian_smooth_2d as _gaussian_smooth_2d


class BaseSampler(ABC):
    """Abstract base for all samplers (ODE, posterior, etc.).

    Contract:
        sample(x_init, time_grid, return_intermediates=False) -> Tensor
            x_init:  [B, C, Nlat, Nlon]
            returns: [B, C, Nlat, Nlon]  or  [T, B, C, Nlat, Nlon] if intermediates
    """

    @abstractmethod
    def sample(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor: ...


class PosteriorSampler(BaseSampler):
    """Abstract base for posterior samplers that condition on observations.

    Provides shared infrastructure:
    - masking_config unpacking
    - XCO2ForwardModel construction
    - _tweedie_estimate, _project_column, _renoise
    """

    def __init__(
        self,
        velocity_model: Any,
        masking_config: dict[str, Any],
        sigma_obs: float = 0.1,
        spatial_smoothing_sigma: float = 0.0,
    ) -> None:
        self.velocity_model = velocity_model
        self.sigma_obs = sigma_obs
        self.spatial_smoothing_sigma = spatial_smoothing_sigma

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
        self.obs_weight = masking_config.get("obs_weight", None)

        self.forward_model = XCO2ForwardModel.from_masking_config(masking_config)

        # Ensure obs_mask is bool for torch.where operations
        if self.obs_mask is not None and self.obs_mask.dtype != torch.bool:
            self.obs_mask = self.obs_mask.bool()

    def _tweedie_estimate(self, x_t: Tensor, t: Tensor, v_theta: Tensor) -> Tensor:
        """Tweedie denoising estimate: x_hat_1 = x_t + (1 - t) * v_theta.

        For CondOT: x_t = (1-t)*x_0 + t*x_1, so the clean estimate at t is:
        x_hat_1 = x_t + (1 - t) * v, where v approximates (x_1 - x_0).
        """
        t_scalar = t.view(-1, 1, 1, 1) if t.dim() == 1 else t
        return x_t + (1.0 - t_scalar) * v_theta

    def _project_column(self, x_hat: Tensor) -> Tensor:
        """Project Tweedie estimate onto column measurement manifold."""
        return self.forward_model.project(
            x_hat,
            self.obs_values,
            self.obs_mask,
            self.sigma_obs,
            self.spatial_smoothing_sigma,
            obs_weight=self.obs_weight,
        )

    def _renoise(self, x_hat_proj: Tensor, t_next: Tensor, z: Tensor) -> Tensor:
        """Re-noise: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_proj.

        At t_next=1.0, this gives pure signal (no noise).
        """
        t_next_scalar = t_next.view(-1, 1, 1, 1) if t_next.dim() >= 1 else t_next
        return (1.0 - t_next_scalar) * z + t_next_scalar * x_hat_proj

    def _forward_shoot(self, x_t: Tensor, t: float, n_steps: int = 1) -> Tensor:
        """Estimate clean sample by integrating ODE from t to 1.

        With n_steps=1, this is a single Tweedie estimate.
        With n_steps>1, uses multi-step Euler integration.

        Args:
            x_t: [B, C, Nlat, Nlon] current noisy state.
            t: current time (scalar).
            n_steps: number of Euler steps (1 = single Tweedie).

        Returns:
            x_hat_1: [B, C, Nlat, Nlon] estimated clean sample at t=1.
        """
        if n_steps == 1:
            t_tensor = t * torch.ones(x_t.shape[0], device=x_t.device)
            v_theta = self.velocity_model(x_t, t_tensor)
            return self._tweedie_estimate(x_t, t_tensor, v_theta)

        # Multi-step Euler from t to 1
        x = x_t
        dt = (1.0 - t) / n_steps
        for k in range(n_steps):
            t_k = t + k * dt
            t_tensor = t_k * torch.ones(x.shape[0], device=x.device)
            v = self.velocity_model(x, t_tensor)
            x = x + v * dt
        return x

    def _compute_likelihood_gradient(self, x_hat: Tensor) -> Tensor:
        """Analytical likelihood gradient for linear column forward model.

        Computes: h_ak * (H(x_hat) - y) / sigma_obs^2
        Only at observed locations (via obs_mask or obs_weight).

        Args:
            x_hat: [B, C, Nlat, Nlon] estimated clean sample.

        Returns:
            Gradient [B, C, Nlat, Nlon] pointing away from observations.
        """
        fm = self.forward_model
        h_ak = fm._get_h_ak_for_x(x_hat)
        xco2_hat = fm.forward(x_hat)

        if self.obs_weight is not None:
            column_error = self.obs_weight * (xco2_hat - self.obs_values.detach())
        else:
            column_error = torch.where(
                self.obs_mask,
                xco2_hat - self.obs_values.detach(),
                torch.zeros_like(xco2_hat),
            )

        if self.spatial_smoothing_sigma > 0:
            column_error = _gaussian_smooth_2d(column_error, self.spatial_smoothing_sigma)

        return h_ak * column_error / (self.sigma_obs**2)

    def _project_to_svd_subspace(self, grad: Tensor, x_batch: Tensor, k: int) -> Tensor:
        """Project gradient onto top-k SVD subspace of batch states (DiffStateGrad).

        Constrains corrections to the data manifold to prevent artifacts.

        Args:
            grad: [B, C, H, W] gradient to project.
            x_batch: [B, C, H, W] current batch of states.
            k: number of singular vectors to keep.

        Returns:
            Projected gradient [B, C, H, W].
        """
        B, C, H, W = x_batch.shape
        D = C * H * W

        if k <= 0 or k >= min(B, D):
            return grad

        # Flatten: [B, D]
        x_flat = x_batch.reshape(B, D)
        grad_flat = grad.reshape(B, D)

        # Low-rank SVD: x_flat = U @ diag(S) @ V^T, keep top-k of V
        # V_k: [D, k] — top-k right singular vectors (principal directions)
        _, _, V = torch.svd_lowrank(x_flat, q=k)  # V: [D, k]

        # Project: grad_proj = (grad @ V) @ V^T
        coeffs = grad_flat @ V  # [B, k]
        grad_proj = coeffs @ V.T  # [B, D]

        return grad_proj.reshape(B, C, H, W)

    def _spectral_filter(self, x: Tensor, t: float, k_low: int, k_high: int) -> Tensor:
        """Time-varying low-pass spectral filter (FGPS).

        At time t, allows frequencies up to k_max = k_low + t * (k_high - k_low).
        Early steps suppress high frequencies; later steps allow full spectrum.

        Args:
            x: [B, C, H, W] spatial field.
            t: current time in [0, 1].
            k_low: minimum frequency cutoff (at t=0).
            k_high: maximum frequency cutoff (at t=1).

        Returns:
            Filtered field [B, C, H, W].
        """
        B, C, H, W = x.shape
        k_max = k_low + t * (k_high - k_low)

        # Build radial frequency grid (same for all batch/channel)
        fy = torch.fft.fftfreq(H, device=x.device).unsqueeze(1) * H  # [H, 1]
        fx = torch.fft.fftfreq(W, device=x.device).unsqueeze(0) * W  # [1, W]
        k_rad = torch.sqrt(fy**2 + fx**2)  # [H, W]

        # Smooth cutoff via sigmoid to avoid ringing
        sharpness = 2.0
        mask = torch.sigmoid(sharpness * (k_max - k_rad))  # [H, W]

        # Apply per batch/channel
        X_freq = torch.fft.fft2(x)
        X_filtered = X_freq * mask.unsqueeze(0).unsqueeze(0)
        return torch.fft.ifft2(X_filtered).real
