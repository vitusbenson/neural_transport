from neural_transport.models.amortized_residual_flowmatching import (
    AmortizedResidualFlowMatching,
)
from neural_transport.models.flowmatching import FlowMatching
from neural_transport.models.residual_flowmatching import ResidualFlowMatching

# from neural_transport.models.flowmatching import VelocityWrapper

MODELWRAPPERS = {
    "flowmatching": FlowMatching,
    "residual_flowmatching": ResidualFlowMatching,
    "amortized_residual_flowmatching": AmortizedResidualFlowMatching,
    # "velocitywrapper": VelocityWrapper,
}
