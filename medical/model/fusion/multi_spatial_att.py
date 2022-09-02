
import numpy as np
import math
from torch import nn, einsum
import torch
from einops import rearrange
from .layers import Attention, ACT2FN, PostionEmbedding, get_config, Mlp

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, types=0, is_window=False):
        super(Embeddings, self).__init__()
        self.is_window = is_window
        self.types = types
        self.config = config
        in_channels = config.in_channels
        patch_size = config.patch_size

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        return x

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            min_indice = self.relative_indices.min()
            self.relative_indices += (-min_indice)
            max_indice = self.relative_indices.max().item()
            self.pos_embedding = nn.Parameter(torch.randn(max_indice + 1, max_indice + 1, max_indice + 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]
        nw_d = n_d // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1], self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d = self.window_size[2], nw_h=nw_h, nw_w=nw_w, nw_d=nw_d)
        out = self.to_out(out)

        return out

class MultiAttention(nn.Module):
    def __init__(self, config, is_position=False):
        super().__init__()
        self.config = config
        self.is_position = is_position
        self.v_attention = Attention(config)
        self.h_attention = Attention(config)
        self.window_attention = WindowAttention(config.hidden_size,
                                                config.num_heads, config.hidden_size // config.num_heads,
                                                config.window_size, relative_pos_embedding=True)
        if is_position:
            self.pos_embedding_1 = PostionEmbedding(config, types=1)
            self.pos_embedding_2 = PostionEmbedding(config, types=2)

    def forward(self, x):
        batch_size, hidden_size, D, W, H = x.shape

        x_1 = rearrange(x, "b c d w h -> (b d) (w h) c")
        x_2 = rearrange(x, "b c d w h -> (b w h) d c")
        x_3 = x.permute(0, 2, 3, 4, 1)

        if self.is_position:
            x_1 = self.pos_embedding_1(x_1)
            x_2 = self.pos_embedding_2(x_2)

        x_1 = self.v_attention(x_1)
        x_2 = self.h_attention(x_2)
        x_3 = self.window_attention(x_3)

        x_3 = rearrange(x_3, "b d w h c -> b (d w h) c", d=D, w=W, h=H)

        x_1 = rearrange(x_1, "(b d) (w h) c -> b (d w h) c", d=D, w=W, h=H)

        x_2 = rearrange(x_2, "(b w h) d c -> b (d w h) c", d=D, w=W, h=H)

        return x_1 + x_2 + x_3

class MultiAttBlock(nn.Module):
    def __init__(self, config, is_position=False):
        super(MultiAttBlock, self).__init__()
        self.config = config
        self.input_shape = config.img_size
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = MultiAttention(config, is_position=is_position)

    def forward(self, x):
        batch_size, hidden_size, D, W, H = x.shape
        x = rearrange(x, "b c d w h -> b (d w h) c")
        h = x
        x = self.attention_norm(x)

        x = rearrange(x, "b (d w h) c -> b c d w h", d=D, w=W, h=H)

        x = self.attn(x)

        x = x + h

        h = x

        x = self.ffn_norm(x)

        x = self.ffn(x)

        x = x + h

        x = x.transpose(-1, -2)
        out_size = (self.input_shape[0] // self.config.patch_size[0],
                    self.input_shape[1] // self.config.patch_size[1],
                    self.input_shape[2] // self.config.patch_size[2],)
        x = x.view((batch_size, self.config.hidden_size, out_size[0], out_size[1], out_size[2])).contiguous()

        return x

class MultiSpatialFusion(nn.Module):
    def __init__(self, in_channels,
                 hidden_size,
                 img_size,
                 num_heads=8,
                 mlp_size=256,
                 num_layers=1,
                 window_size=(8, 8, 8),
                 out_hidden=False):
        super().__init__()
        self.config = get_config(in_channels=in_channels, hidden_size=hidden_size,
                                 patch_size=(1, 1, 1), img_size=img_size, mlp_dim=mlp_size, num_heads=num_heads, window_size=window_size)
        self.block_list = nn.ModuleList([MultiAttBlock(self.config, is_position=True)if i == 0 else MultiAttBlock(self.config) for i in range(num_layers)])

        self.embeddings = Embeddings(self.config)
        self.out_hidden = out_hidden

    def forward(self, x):

        x = self.embeddings(x)
        hidden_states = []
        for l in self.block_list:

            x = l(x)
            if self.out_hidden:
                hidden_states.append(x)

        if self.out_hidden:
            return x, hidden_states

        return x

if __name__ == '__main__':

    t1 = torch.rand(1, 3, 16, 16, 16)

    model = MultiSpatialFusion(in_channels=3, hidden_size=64, img_size=(16, 16, 16))
    out = model(t1)

    print(out.shape)

