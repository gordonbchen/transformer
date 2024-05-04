import torch
import torch.nn as nn

from models.model_blocks import TokenEncoding, Encoder


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

        self.block_size = block_size

        self.token_encoding = TokenEncoding(vocab_size, d_model, block_size)
        self.attention_blocks = nn.Sequential(
            *(
                Encoder(n_heads, d_model, d_ffwd, block_size, dropout, mask_future=True)
                for i in range(n_layers)
            )
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.model_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.token_encoding(x)  # (B, T, d_model).
        z = self.attention_blocks(z)

        z = self.layer_norm(z)
        logits = self.model_head(z)  # (B, T, vocab_size).
        return logits
