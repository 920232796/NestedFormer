
import math
from einops import rearrange
import torch.nn as nn
import torch
from .layers import get_config, Mlp

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = config.img_size
        in_channels = config.in_channels
        patch_size = config.patch_size
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class AttentionCrossModal(nn.Module):
    def __init__(self, config):
        super(AttentionCrossModal, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, kv):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(kv)
        mixed_value_layer = self.value(kv)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class CrossAttBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.attention_norm_cross = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn_cross = AttentionCrossModal(config)

    def forward(self, q, kv):
        # q是其他模态特征。
        h = q

        x = self.attn_cross(q, kv)
        x = x + h
        x = self.attention_norm_cross(x)

        h = x
        x = self.ffn(x)
        x = x + h
        x = self.ffn_norm(x)

        return x

class TokenLearner(nn.Module):
    def __init__(self, in_channels, S):

        super().__init__()
        self.token_conv = nn.Conv3d(in_channels=in_channels, out_channels=S, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        selected = self.token_conv(x)
        selected = rearrange(selected, "b s d w h -> b s (d w h) 1")
        selected = torch.sigmoid(selected)

        x = rearrange(x, "b c d w h -> b 1 (d w h) c")

        out = (x * selected).mean(dim=2)
        return out

class CrossModalityFusion(nn.Module):
    def __init__(self, model_num, in_channels,
                 hidden_size,
                 img_size, mlp_size=256,
                 token_mixer_size=32,
                 token_learner=False):
        super().__init__()
        self.embeddings = nn.ModuleList([])
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        patch_size = (1, 1, 1)
        self.config = get_config(in_channels=in_channels, hidden_size=hidden_size, patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)
        self.model_num = model_num
        self.img_size = img_size
        patch_num = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.token_learner = token_learner
        if token_learner:
            self.token_mixer = TokenLearner(in_channels=in_channels, S = token_mixer_size)
        else :
            self.token_mixer = nn.Linear(patch_num, token_mixer_size)

        for i in range(model_num):
            self.embeddings.append(Embeddings(self.config))

        self.cross_attention = CrossAttBlock(config=self.config)

    def forward(self, q, kv):

        q = rearrange(q, "b c d w h -> b (d w h) c")
        embed_x = []
        for i in range(self.model_num):
            x = self.embeddings[i](kv[:, i])
            if self.token_learner:
                x = rearrange(x, "b (d w h) c -> b c d w h", d=self.img_size[0], w=self.img_size[1], h=self.img_size[2])
                x = self.token_mixer(x)

            else :
                x = x.transpose(-1, -2)
                x = self.token_mixer(x)
                x = x.transpose(-1, -2)

            embed_x.append(x)

        embed_x = torch.cat(embed_x, dim=1)
        batch_size = embed_x.shape[0]

        corss_out = self.cross_attention(q, embed_x)
        corss_out = corss_out.transpose(-1, -2)
        corss_out = corss_out.view((batch_size, self.hidden_size, self.img_size[0], self.img_size[1], self.img_size[2]))

        return corss_out

if __name__ == '__main__':

    q = torch.rand(1, 64, 16, 16, 16)
    kv = torch.rand(1, 3, 64, 16, 16, 16)

    model = CrossModalityFusion(model_num=3,
                                in_channels=64,
                                hidden_size=64,
                                img_size=(16,16,16),
                                token_learner=True,
                                token_mixer_size=32)

    out = model(q, kv)

    print(out.shape)


