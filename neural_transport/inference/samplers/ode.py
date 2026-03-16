"""ODE sampler wrapping flow_matching.solver.ODESolver."""

from flow_matching.solver import ODESolver
from torch import Tensor

from neural_transport.inference.samplers.base import BaseSampler


class ODESampler(BaseSampler):
    """ODE-based sampler for unconditional/masked-conditional generation.

    Wraps the flow_matching ODESolver in a BaseSampler-conforming interface.
    Conditioning (if any) is handled by the velocity_model itself
    (e.g. MaskedVelocityWrapper), not by this class.

    Args:
        velocity_model: Model with forward(x, t) -> v.
        method: ODE solver method (e.g. "midpoint", "euler").
        step_size: Fixed step size, or None for adaptive.
        atol: Absolute tolerance for adaptive solvers.
        rtol: Relative tolerance for adaptive solvers.
    """

    def __init__(
        self,
        velocity_model,
        method: str = "midpoint",
        step_size: float | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        self.velocity_model = velocity_model
        self.method = method
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol

    def sample(
        self,
        x_init: Tensor,
        time_grid: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor:
        solver = ODESolver(velocity_model=self.velocity_model)
        return solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method=self.method,
            step_size=self.step_size,
            return_intermediates=return_intermediates,
            atol=self.atol,
            rtol=self.rtol,
        )
