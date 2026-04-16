"""D-Flow posterior sampler via source-space optimization.

Reference: arXiv 2402.14017v2, arXiv 2602.21469v1.

Optimizes the initial noise x_0 by backpropagating through the ODE solver
to minimize a measurement loss + regularization:
    min_{x_0}  ||H(ODE(x_0)) - y||^2 / (2*sigma_obs^2)  +  lambda * R(x_0)

Unlike velocity-field methods (FMPS, DPS) or project-renoise methods
(FlowDPS, MCG, PCFM), D-Flow keeps the learned dynamics completely frozen
and performs all conditioning at the source level.
"""

import math

import torch
from flow_matching.solver import ODESolver
from torch import Tensor

from neural_transport.inference.samplers.base import PosteriorSampler


class DFlowSampler(PosteriorSampler):
    """D-Flow: posterior sampling via source-point optimization.

    At each call to sample():
      1. Initialize x_0 from x_init (Gaussian noise).
      2. For n_opt_steps iterations:
         a. Integrate ODE from t=0 to t=1 with enable_grad=True.
         b. Compute likelihood loss + regularization on x_0.
         c. Backprop and update x_0.
      3. Final ODE solve (no grad) to produce output.

    Args:
        velocity_model: Unconditional velocity model (forward(x, t) -> v).
        masking_config: Dict with obs_mask, obs_values, ak, pressure_weights, etc.
        sigma_obs: Observation noise for likelihood weighting.
        spatial_smoothing_sigma: Gaussian smoothing of column error (0 = none).
        n_opt_steps: Number of optimization iterations on x_0.
        lr: Optimizer learning rate.
        reg_weight: Regularization strength lambda.
        reg_type: Regularization type: "l2", "norm_diff", or "chi_prior".
        optimizer: Optimizer type: "adam" or "lbfgs".
        use_checkpointing: Use gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        velocity_model,
        masking_config,
        sigma_obs=0.1,
        spatial_smoothing_sigma=0.0,
        n_opt_steps=50,
        lr=1e-2,
        reg_weight=1.0,
        reg_type="l2",
        optimizer="adam",
        use_checkpointing=False,
    ):
        super().__init__(velocity_model, masking_config, sigma_obs, spatial_smoothing_sigma)
        self.n_opt_steps = n_opt_steps
        self.lr = lr
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.optimizer_type = optimizer
        self.use_checkpointing = use_checkpointing

    def _regularization(self, x_0: Tensor) -> Tensor:
        """Compute regularization term on source point x_0.

        Args:
            x_0: [B, C, H, W] current source point.

        Returns:
            Scalar regularization loss.
        """
        d = x_0[0].numel()  # dimension per sample
        if self.reg_type == "l2":
            # Gaussian prior: -log p(x_0) ~ ||x_0||^2 / 2, normalized by d
            return x_0.pow(2).sum() / (x_0.shape[0] * d)
        elif self.reg_type == "norm_diff":
            # Typical-set: encourage ||x_0|| ~ sqrt(d)
            norms = x_0.reshape(x_0.shape[0], -1).norm(dim=1)
            return ((norms - math.sqrt(d)) ** 2).mean()
        elif self.reg_type == "chi_prior":
            # Chi-squared concentration: ||x_0||^2 ~ d
            sq_norms = x_0.reshape(x_0.shape[0], -1).pow(2).sum(dim=1)
            return ((sq_norms - d) ** 2 / d).mean()
        else:
            raise ValueError(f"Unknown reg_type: {self.reg_type}")

    def _compute_loss(self, x_1: Tensor, x_0: Tensor) -> Tensor:
        """Compute total loss: likelihood + regularization.

        Args:
            x_1: [B, C, H, W] ODE-solved sample at t=1.
            x_0: [B, C, H, W] current source point.

        Returns:
            Scalar total loss.
        """
        # Likelihood: ||H(x_1) - y||^2 / (2 * sigma_obs^2) at observed locations
        xco2_pred = self.forward_model.forward(x_1)

        if self.obs_weight is not None:
            residual = self.obs_weight * (xco2_pred - self.obs_values)
        else:
            residual = torch.where(
                self.obs_mask,
                xco2_pred - self.obs_values,
                torch.zeros_like(xco2_pred),
            )

        likelihood = residual.pow(2).sum() / (2.0 * self.sigma_obs**2 * x_1.shape[0])

        # Regularization
        reg = self._regularization(x_0)

        return likelihood + self.reg_weight * reg

    def _ode_solve(self, x_0: Tensor, time_grid: Tensor, enable_grad: bool = True) -> Tensor:
        """Solve ODE from t=0 to t=1 using flow_matching ODESolver.

        Args:
            x_0: [B, C, H, W] initial noise.
            time_grid: [T] time points from 0 to 1.
            enable_grad: Whether to compute gradients through the solve.

        Returns:
            x_1: [B, C, H, W] sample at t=1.
        """
        solver = ODESolver(velocity_model=self.velocity_model)
        return solver.sample(
            x_init=x_0,
            time_grid=time_grid,
            method="euler",
            step_size=None,
            enable_grad=enable_grad,
            return_intermediates=False,
        )

    def sample(self, x_init: Tensor, time_grid: Tensor, return_intermediates: bool = False) -> Tensor:
        """Run D-Flow source optimization.

        Args:
            x_init: [B, C, Nlat, Nlon] initial noise.
            time_grid: [T] time points from 0 to 1.
            return_intermediates: If True, return [N_opt, B, C, Nlat, Nlon]
                snapshots of x_1 at each optimization step.

        Returns:
            Final samples [B, C, Nlat, Nlon], or trajectory if return_intermediates.
        """
        # Clone x_init as the optimizable source point.
        # .contiguous() is needed for L-BFGS which calls .view(-1) on gradients.
        x_0 = x_init.detach().clone().contiguous().requires_grad_(True)

        # Create optimizer
        if self.optimizer_type == "adam":
            opt = torch.optim.Adam([x_0], lr=self.lr)
        elif self.optimizer_type == "lbfgs":
            opt = torch.optim.LBFGS([x_0], lr=self.lr, max_iter=20, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        intermediates = [] if return_intermediates else None

        # Enable gradients for the optimization loop — the caller
        # (generate_multi_target) wraps model calls in torch.no_grad(),
        # but D-Flow needs gradients to backprop through the ODE solver.
        with torch.enable_grad():
            for _ in range(self.n_opt_steps):
                if self.optimizer_type == "lbfgs":
                    # L-BFGS requires a closure
                    def closure():
                        opt.zero_grad()
                        x_1 = self._ode_solve(x_0, time_grid, enable_grad=True)
                        loss = self._compute_loss(x_1, x_0)
                        loss.backward()
                        return loss

                    opt.step(closure)

                    if return_intermediates:
                        with torch.no_grad():
                            x_1_snap = self._ode_solve(x_0, time_grid, enable_grad=False)
                        intermediates.append(x_1_snap)
                else:
                    # Adam: standard forward-backward
                    opt.zero_grad()
                    x_1 = self._ode_solve(x_0, time_grid, enable_grad=True)
                    loss = self._compute_loss(x_1, x_0)
                    loss.backward()
                    opt.step()

                    if return_intermediates:
                        intermediates.append(x_1.detach())

        # Final forward pass without gradients
        with torch.no_grad():
            x_final = self._ode_solve(x_0, time_grid, enable_grad=False)

        if return_intermediates:
            intermediates.append(x_final)
            return torch.stack(intermediates, dim=0)
        return x_final
