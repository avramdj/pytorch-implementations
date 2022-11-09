import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].view(*x.shape[1:])
        return self.dropout(x)


class MhaBlock(nn.Module):
    def __init__(self, d_model, n_heads=10, mask=None):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), "embedding size `d_model` must be divisible by the number of attention heads `n_heads`"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_h = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        return (
            x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        )

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.d_k)
        a = F.softmax(scores, dim=-1)
        x = torch.matmul(a, v)
        return x

    def forward(self, x, q=None):
        q = q or x
        batch_size = q.size(0)
        q = self.w_q(q)
        k = self.w_k(x)
        v = self.w_v(x)

        sq = self.split_heads(q, batch_size)
        sk = self.split_heads(k, batch_size)
        sv = self.split_heads(v, batch_size)

        sh = self.attention(sq, sk, sv)
        gh = self.group_heads(sh, batch_size)
        out = self.w_h(gh)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Linear(d_model, d_model)
        self.proj_2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.proj_2(self.act(self.proj_1(x)))


class Sublayer(nn.Module):
    def __init__(self, layer, d_model, dropout=0.1):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        l = self.layer(x, **kwargs)
        l = self.dropout(l)
        return self.norm(x + l)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha_sublayer = Sublayer(MhaBlock(d_model, n_heads), d_model, dropout)
        self.ffn_sublayer = Sublayer(PositionWiseFFN(d_model), d_model, dropout)

    def forward(self, x):
        x = self.mha_sublayer(x)
        x = self.ffn_sublayer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mask=None, dropout=0.1):
        super().__init__()
        self.mmha_sublayer = Sublayer(MhaBlock(d_model, n_heads), d_model, dropout)
        self.mha_sublayer = Sublayer(MhaBlock(d_model, n_heads, mask=mask), d_model, dropout)
        self.ffn_sublayer = Sublayer(PositionWiseFFN(d_model), d_model, dropout)

    def forward(self, x):
        masked_q = self.mmha_sublayer(x)
        x = self.mha_sublayer(x, q=masked_q)
        x = self.ffn_sublayer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, mask=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, mask=mask) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
