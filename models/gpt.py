import torch
import torch.nn as nn

from models.model_blocks import TokenEncoding, Encoder, get_attention_mask


class GPT(nn.Module):
    """GPT model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ffwd: int,
        block_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_encoding = TokenEncoding(vocab_size, d_model, block_size)
        self.encoders = nn.ModuleList(
            [Encoder(n_heads, d_model, d_ffwd, dropout) for i in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.model_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.token_encoding(x)  # (B, T, d_model).

        attention_mask = get_attention_mask(
            q_tokens=x, k_tokens=x, q_pad_idx=None, k_pad_idx=None, mask_future=True
        )
        for encoder in self.encoders:
            z = encoder(z, attention_mask)

        z = self.layer_norm(z)
        logits = self.model_head(z)  # (B, T, vocab_size).
        return logits
