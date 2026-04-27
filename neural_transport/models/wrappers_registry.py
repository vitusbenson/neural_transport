from neural_transport.models.flowmatching import FlowMatching
from neural_transport.models.residual_flowmatching import ResidualFlowMatching

# from neural_transport.models.flowmatching import VelocityWrapper

MODELWRAPPERS = {
    "flowmatching": FlowMatching,
    "residual_flowmatching": ResidualFlowMatching,
    # "velocitywrapper": VelocityWrapper,
}
