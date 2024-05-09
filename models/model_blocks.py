import torch
import torch.nn as nn
from torch.nn import functional as F

from train_funcs import HyperParams


class Encoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_ffwd: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(n_heads, d_model, dropout)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ffwd, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        z = self.layer_norm1(x)
        z = z + self.mha(z, z, z, attention_mask)

        z = self.layer_norm2(z)
        z = z + self.ffwd(z)
        return z


class Decoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, n_heads: int, d_model: int, d_ffwd: int, dropout: float) -> None:
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(n_heads, d_model, dropout)

        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ffwd, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_k: torch.Tensor,
        encoder_v: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = self.layer_norm1(x)
        z = z + self.self_attention(z, z, z, self_attention_mask)

        z = self.layer_norm2(z)
        z = z + self.cross_attention(z, encoder_k, encoder_v, cross_attention_mask)

        z = self.layer_norm3(z)
        z = z + self.ffwd(z)
        return z


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        dropout: float,
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

        self.wei_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        Q = self.q_net(q)
        K = self.k_net(k)
        V = self.v_net(v)

        Q, K, V = (self.split_heads(i) for i in (Q, K, V))

        z = self.scaled_dot_product_attention(Q, K, V, attention_mask)

        z = self.combine_heads(z)

        z = self.linear(z)
        z = self.output_dropout(z)
        return z

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        wei = Q @ K.transpose(-2, -1)  # Calculate affinities.
        wei = wei * (self.head_size**-0.5)

        wei = wei + attention_mask

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


def get_attention_mask(
    q_tokens: torch.Tensor,
    k_tokens: torch.Tensor,
    q_pad_idx: int | None,
    k_pad_idx: int | None,
    mask_future: bool,
) -> torch.Tensor:
    """Return an addative attention mask. Supports pad and future masking."""
    B, T = q_tokens.shape
    mask = torch.zeros((B, T, T), dtype=torch.float32)

    # Pad mask.
    if None not in (q_pad_idx, k_pad_idx):
        q_mask = (q_tokens == q_pad_idx).unsqueeze(-1)
        k_mask = (k_tokens == k_pad_idx).unsqueeze(-2)

        # HACK: cannot be -inf or else becomes nan in softmax for queries.
        mask[q_mask | k_mask] = torch.finfo(torch.float32).min

    # Future mask.
    if mask_future:
        future_mask = torch.triu(
            torch.full((T, T), fill_value=True, dtype=torch.bool), diagonal=1
        )
        mask.masked_fill_(future_mask, float("-inf"))

    mask.unsqueeze_(1)  # Convert to multi-headed mask.
    return mask.to(device=HyperParams.DEVICE)


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


class TokenEncoding(nn.Module):
    """Token embedding + sin position encoding."""

    def __init__(self, vocab_size: int, d_model: int, block_size: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinPositionalEncoding(block_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.token_embedding(x)
        z = z + self.pos_encoding(z)
        return z


class SinPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, block_size: int, d_model: int) -> None:
        super().__init__()

        pos_encoding = torch.zeros(block_size, d_model, dtype=torch.float32)

        pos_arange = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
        dim_arange = torch.arange(d_model // 2, dtype=torch.float32)

        div_factor = 10_000 ** ((2 * dim_arange) / d_model)
        pos_encoding[:, ::2] = torch.sin(pos_arange / div_factor)
        pos_encoding[:, 1::2] = torch.cos(pos_arange / div_factor)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding[:, : x.shape[-2]]
