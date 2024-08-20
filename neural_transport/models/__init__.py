from neural_transport.models.gnn.gnn import GraphTM
from neural_transport.models.gnn.graphcast import GraphCast
from neural_transport.models.hybridsfno import HybridSFNO
from neural_transport.models.sfno import SFNO

# from neural_transport.models.sfno_v2 import SFNOv2
from neural_transport.models.swintransformer import SwinTransformer
from neural_transport.models.unet import UNet

MODELS = {
    "gnn": GraphTM,
    "unet": UNet,
    "sfno": SFNO,
    "swintransformer": SwinTransformer,
    "hybridsfno": HybridSFNO,
    "sfnov2": SFNOv2,
    "graphcast": GraphCast,
}
