import torch
import torch.nn as nn
import torch.nn.functional as F

from model_blocks import SinPositionalEncoding, Encoder


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

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = SinPositionalEncoding(block_size, d_model)

        self.attention_blocks = nn.Sequential(
            *(
                Encoder(n_heads, d_model, d_ffwd, block_size, dropout, mask_future=True)
                for i in range(n_layers)
            )
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.model_head = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(inputs)  # (B, T, embed_size).
        embeddings = self.position_embedding(token_embeddings)

        z = self.attention_blocks(embeddings)
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
