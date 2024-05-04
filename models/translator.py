import torch
import torch.nn as nn

from models.model_blocks import TokenEncoding, Encoder, Decoder


class Translator(nn.Module):
    """A NMT transformer."""

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
        self.encoder_token_encoding = TokenEncoding(vocab_size, d_model, block_size)
        self.encoder_blocks = nn.Sequential(
            *(
                Encoder(
                    n_heads, d_model, d_ffwd, block_size, dropout, mask_future=False
                )
                for i in range(n_layers)
            )
        )

        self.decoder_token_encoding = TokenEncoding(vocab_size, d_model, block_size)
        self.decoder_blocks = nn.ModuleList(
            [
                Decoder(n_heads, d_model, d_ffwd, block_size, dropout)
                for i in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.model_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_z = self.encoder_token_encoding(x)
        encoder_z = self.encoder_blocks(encoder_z)

        decoder_z = self.decoder_token_encoding(x)
        for decoder in self.decoder_blocks:
            decoder_z = decoder(decoder_z, encoder_z, encoder_z)

        z = self.layer_norm(decoder_z)
        logits = self.model_head(z)
        return logits
