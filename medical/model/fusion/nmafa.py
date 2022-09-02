import torch.nn as nn
import torch
from einops import rearrange
from .multi_spatial_att import MultiSpatialFusion
from .cross_modality_att import CrossModalityFusion

class NMaFaLayer(nn.Module):
    def __init__(self, model_num,
                 in_channels,
                 hidden_size,
                 img_size,
                 mlp_size=256,
                 self_num_layer=2,
                 window_size=(2, 4, 4),
                 token_mixer_size=32,
                 token_learner=False):
        super().__init__()
        self.img_size = img_size
        self.hidden_size = hidden_size

        self.spatial_att = MultiSpatialFusion(in_channels=model_num*in_channels,
                                                hidden_size=hidden_size,
                                                img_size=img_size,
                                                mlp_size=mlp_size,
                                                num_layers=self_num_layer,
                                                window_size=window_size)

        self.modality_att = CrossModalityFusion(model_num=model_num,
                                               in_channels=in_channels,
                                               hidden_size=hidden_size,
                                               img_size=img_size,
                                               mlp_size=mlp_size,
                                               token_mixer_size=token_mixer_size,
                                               token_learner=token_learner)

    def forward(self, x):
        # x: (batch, modal_num, hidden_size, d, w, h)
        q = rearrange(x, "b m f d w h -> b (m f) d w h")
        q = self.spatial_att(q)
        fusion_out = self.modality_att(q, x)
        return fusion_out


if __name__ == '__main__':
    t1 = torch.rand(1, 4, 16, 16, 16, 16)

    model = NMaFaLayer(4, 16, 16, (16, 16, 16))

    out = model(t1)
    print(out.shape)