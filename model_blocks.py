import torch
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_ffwd: int,
        block_size: int,
        dropout: float,
        mask_future: bool,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(
            n_heads, d_model, block_size, dropout, mask_future
        )
        self.ffwd = FeedForward(d_model, d_ffwd, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layer_norm1(x)
        z = z + self.mha(z, z, z)

        z = self.layer_norm2(z)
        z = z + self.ffwd(z)
        return z


class Decoder(nn.Module):
    """Transformer decoder."""

    def __init__(
        self, n_heads: int, d_model: int, d_ffwd: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(
            n_heads, d_model, block_size, dropout, mask_future=True
        )

        self.layer_norm2 = nn.LayerNorm(d_model)
        # No future masking b/c kv comes from encoder.
        self.cross_attention = MultiHeadAttention(
            n_heads, d_model, block_size, dropout, mask_future=False
        )

        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ffwd, dropout)

    def forward(
        self, x: torch.Tensor, encoder_k: torch.Tensor, encoder_v: torch.Tensor
    ) -> torch.Tensor:
        z = self.layer_norm1(x)
        z = z + self.self_attention(z, z, z)

        z = self.layer_norm2(z)
        z = z + self.cross_attention(z, encoder_k, encoder_v)

        z = self.layer_norm3(z)
        z = z + self.ffwd(z)
        return z


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        block_size: int,
        dropout: float,
        mask_future: bool,
    ) -> None:
        super().__init__()

        assert (
            d_model % n_heads == 0
        ), f"d_model {d_model} must be divisible by n_heads {n_heads}."
        self.head_size = d_model // n_heads

        self.n_heads = n_heads

        self.k_net = nn.Linear(d_model, d_model, bias=False)
        self.q_net = nn.Linear(d_model, d_model, bias=False)
        self.v_net = nn.Linear(d_model, d_model, bias=False)

        if mask_future:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.mask_future = mask_future

        self.wei_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        Q = self.q_net(q)
        K = self.k_net(k)
        V = self.v_net(v)

        Q, K, V = (self.split_heads(i) for i in (Q, K, V))

        z = self.scaled_dot_product_attention(Q, K, V)

        z = self.combine_heads(z)

        z = self.linear(z)
        z = self.output_dropout(z)
        return z

    def scaled_dot_product_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        wei = Q @ K.transpose(-2, -1)  # Calculate affinities.
        wei = wei * (self.head_size**-0.5)  # Scale by 1/sqrt(head_size).

        if self.mask_future:
            # Mask to time length (for short sequence generation).
            T = wei.shape[-1]
            wei = wei.masked_fill((self.tril == 0)[:T, :T], float("-inf"))

        wei = F.softmax(wei, dim=-1)  # Softmax smoothing.
        wei = self.wei_dropout(wei)

        out = wei @ V  # Take weighted mean of values.
        return out

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, n_heads, T, head_size = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, int(n_heads * head_size))


class FeedForward(nn.Module):
    """A simple MLP feed-forward module."""

    def __init__(self, d_model: int, d_ffwd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffwd, bias=True),
            nn.GELU(),
            nn.Linear(d_ffwd, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, block_size: int, embed_size: int) -> None:
        super().__init__()

        pos_encoding = torch.zeros(block_size, embed_size, dtype=torch.float32)

        pos_arange = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
        dim_arange = torch.arange(embed_size // 2, dtype=torch.float32)

        div_factor = 10_000 ** ((2 * dim_arange) / embed_size)
        pos_encoding[:, ::2] = torch.sin(pos_arange / div_factor)
        pos_encoding[:, 1::2] = torch.cos(pos_arange / div_factor)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding[:, : x.shape[-2]]
