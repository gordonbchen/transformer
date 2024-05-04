import torch
import torch.nn as nn
import torch.nn.functional as F

from model_blocks import TokenEncoding, Encoder


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

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_tokens: int) -> torch.Tensor:
        for i in range(n_tokens):
            logits = self(tokens[:, -self.block_size :])  # (B, T, C)
            logits = logits[:, -1, :]  # only use last pred col. (B, C)

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_tokens), dim=-1)  # append. (B, T+1)

        return tokens
