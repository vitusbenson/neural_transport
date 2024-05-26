from neural_transport.models.gnn.gnn import GraphTM
from neural_transport.models.hybridsfno import HybridSFNO
from neural_transport.models.sfno import SFNO
from neural_transport.models.swintransformer import SwinTransformer
from neural_transport.models.unet import UNet

MODELS = {
    "gnn": GraphTM,
    "unet": UNet,
    "sfno": SFNO,
    "swintransformer": SwinTransformer,
    "hybridsfno": HybridSFNO,
}
