"""Abstract base classes for flow matching samplers.

BaseSampler: minimal interface — just sample().
PosteriorSampler: adds shared Tweedie/project/renoise logic used by
FlowDPS, SDE, FIG, and ICTM posterior samplers.
"""

from abc import ABC, abstractmethod

from torch import Tensor

from neural_transport.forward_model import XCO2ForwardModel


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
        velocity_model,
        masking_config: dict,
        sigma_obs: float = 0.1,
        spatial_smoothing_sigma: float = 0.0,
    ):
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

        self.forward_model = XCO2ForwardModel.from_masking_config(masking_config)

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
        )

    def _renoise(self, x_hat_proj: Tensor, t_next: Tensor, z: Tensor) -> Tensor:
        """Re-noise: x_{t+1} = (1 - t_{n+1}) * z + t_{n+1} * x_hat_proj.

        At t_next=1.0, this gives pure signal (no noise).
        """
        t_next_scalar = t_next.view(-1, 1, 1, 1) if t_next.dim() >= 1 else t_next
        return (1.0 - t_next_scalar) * z + t_next_scalar * x_hat_proj
