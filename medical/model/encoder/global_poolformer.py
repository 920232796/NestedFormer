from typing import Sequence, Union
import ml_collections
import torch
import torch.nn as nn
from medical.model.fusion.layers import get_config
from einops import rearrange

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(Convolution, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride_size, padding_size),
                                    nn.InstanceNorm3d(out_channels),
                                    nn.ReLU())
    def forward(self, x):
        x = self.conv_1(x)
        return x

class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_size, padding_size):
        super(TwoConv, self).__init__()
        self.conv_1 = Convolution(in_channels, out_channels, kernel_size, stride_size, padding_size)
        self.conv_2 = Convolution(out_channels, out_channels, kernel_size, stride_size, padding_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            pool_size = (2, 2, 2)
    ):

        super().__init__()

        up_chns = in_chns // 2
        self.upsample = torch.nn.ConvTranspose3d(in_chns, up_chns, kernel_size=pool_size, stride=pool_size, padding=0)

        self.convs = TwoConv(cat_chns + up_chns, out_chns, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        x_0 = self.upsample(x)

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class MlpChannel(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Conv3d(config.hidden_size, config.mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(config.mlp_dim, config.hidden_size, 1)
        self.drop = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.norm = LayerNormChannel(num_channels=config.hidden_size)

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = self.norm(x)

        return x

class GlobalPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        all_size = self.img_size[0] * self.img_size[1] * self.img_size[2]
        self.global_layer = nn.Linear(1, all_size)


    def forward(self, x):
        x = rearrange(x, "b c d w h -> b c (d w h)")

        x = x.mean(dim=-1, keepdims=True)

        x = self.global_layer(x)
        x = rearrange(x, "b c (d w h) -> b c d w h", d=self.img_size[0], w=self.img_size[1], h=self.img_size[2])
        return x


class BlockPool(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNormChannel(config.hidden_size, eps=1e-6)
        self.ffn = MlpChannel(config)
        self.attn = GlobalPool(config)
        # self.attn = nn.AvgPool3d(3, 1, padding=1) # poolformer

    def forward(self, x):
        h = x
        x = self.attention_norm(x)

        x = self.attn(x) + x
        # x = self.attn(x) - x # poolformer
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class GlobalPoolformer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 img_size,
                 patch_size,
                 mlp_size=256,
                 num_layers=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = get_config(in_channels=in_channels,
                                 hidden_size=out_channels,
                                 patch_size=patch_size,
                                 mlp_dim=mlp_size,
                                 img_size=img_size)

        self.block_list = nn.ModuleList([BlockPool(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config)

    def forward(self, x, out_hidden=False):
        x = self.embeddings(x)
        hidden_state = []
        for l in self.block_list:
            x = l(x)
            hidden_state.append(x)
        if out_hidden:
            return x, hidden_state
        return x

class GlobalPoolformerEncoder(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            features: Sequence[int],
            pool_size,

    ):
        super().__init__()

        fea = features
        self.drop = nn.Dropout()
        self.in_channels = in_channels
        self.features = features
        self.img_size = img_size
        self.conv_0 = TwoConv(in_channels, features[0], 3, 1, 1)
        self.down_1 = GlobalPoolformer(fea[0], fea[1], img_size=img_size[0], patch_size=pool_size[0], mlp_size=fea[1]*2, num_layers=2)
        self.down_2 = GlobalPoolformer(fea[1], fea[2], img_size=img_size[1], patch_size=pool_size[1], mlp_size=fea[2]*2, num_layers=2)
        self.down_3 = GlobalPoolformer(fea[2], fea[3], img_size=img_size[2], patch_size=pool_size[2], mlp_size=fea[3]*2, num_layers=2)
        self.down_4 = GlobalPoolformer(fea[3], fea[4], img_size=img_size[3], patch_size=pool_size[3], mlp_size=fea[4]*2, num_layers=2)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        return x4, x3, x2, x1, x0

class Encoder(nn.Module):

    def __init__(self, model_num,
                 img_size,
                 fea,
                 pool_size,
                 ):

        super().__init__()
        self.model_num = model_num
        self.encoders = nn.ModuleList([])
        for i in range(model_num):
            encoder = GlobalPoolformerEncoder(
                                               img_size=img_size,
                                               in_channels=1,
                                               pool_size=pool_size,
                                               features=fea)

            self.encoders.append(encoder)

    def forward(self, x):
        encoder_out = []
        x = x.unsqueeze(dim=2)
        for i in range(self.model_num):
            encoder_out.append(self.encoders[i](x[:, i]))

        return encoder_out