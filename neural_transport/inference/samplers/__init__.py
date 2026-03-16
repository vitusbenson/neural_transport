from neural_transport.inference.samplers.base import BaseSampler, PosteriorSampler
from neural_transport.inference.samplers.fig import FIGSampler
from neural_transport.inference.samplers.flowdps import FlowDPSSampler
from neural_transport.inference.samplers.ictm import ICTMSampler
from neural_transport.inference.samplers.ode import ODESampler
from neural_transport.inference.samplers.sde import StochasticPosteriorSampler

SAMPLER_REGISTRY: dict[str, type[BaseSampler]] = {
    "ode": ODESampler,
    "flowdps": FlowDPSSampler,
    "sde": StochasticPosteriorSampler,
    "fig": FIGSampler,
    "ictm": ICTMSampler,
}


def create_sampler(name, velocity_model, masking_config=None, **kwargs):
    """Factory: instantiate sampler by name. Raises KeyError/TypeError on bad input."""
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown sampler {name!r}. Available: {sorted(SAMPLER_REGISTRY)}")
    cls = SAMPLER_REGISTRY[name]
    if name == "ode":
        return cls(velocity_model=velocity_model, **kwargs)
    if masking_config is None:
        raise TypeError(f"Posterior sampler {name!r} requires masking_config")
    return cls(velocity_model=velocity_model, masking_config=masking_config, **kwargs)


__all__ = [
    "BaseSampler",
    "PosteriorSampler",
    "ODESampler",
    "FlowDPSSampler",
    "StochasticPosteriorSampler",
    "FIGSampler",
    "ICTMSampler",
    "SAMPLER_REGISTRY",
    "create_sampler",
]
