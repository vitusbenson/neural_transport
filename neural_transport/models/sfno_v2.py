from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet

from neural_transport.models.regulargrid import RegularGridModel


class SFNOv2(RegularGridModel):

    def init_model(
        self,
        scale_factor=1,
        in_chans=193,
        out_chans=19,
        embed_dim=256,
        num_layers=8,
        spectral_layers=3,
    ) -> None:

        self.sfnonet = SphericalFourierNeuralOperatorNet(
            spectral_transform="sht",
            model_grid_type="equiangular",
            sht_grid_type="legendre-gauss",
            filter_type="linear",
            operator_type="dhconv",
            scale_factor=scale_factor,
            inp_shape=(self.nlat, self.nlon),
            out_shape=(self.nlat, self.nlon),
            inp_chans=in_chans,
            out_chans=out_chans,
            embed_dim=embed_dim,
            num_layers=num_layers,
            repeat_layers=1,
            use_mlp=True,
            mlp_ratio=2.0,
            encoder_ratio=1,
            decoder_ratio=1,
            activation_function="gelu",
            encoder_layers=1,
            pos_embed="none",  # "frequency", "direct"
            pos_drop_rate=0.0,
            path_drop_rate=0.0,
            mlp_drop_rate=0.0,
            normalization_layer="instance_norm",
            max_modes=None,
            hard_thresholding_fraction=1.0,
            big_skip=False,  # Changed
            rank=1.0,
            factorization=None,
            separable=False,
            complex_activation="real",
            spectral_layers=spectral_layers,
            bias=False,
            checkpointing=0,
        )

    def model(self, x_in):

        x_out = self.sfnonet(x_in)

        return x_out
