import torch
import torch.nn as nn

from models.model_blocks import TokenEncoding, Encoder, Decoder, get_attention_mask


class Translator(nn.Module):
    """A NMT transformer."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int,
        d_ffwd: int,
        block_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        source_pad_idx: int,
        target_pad_idx: int,
    ) -> None:
        super().__init__()

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

        self.encoder_token_encoding = TokenEncoding(
            source_vocab_size, d_model, block_size
        )
        self.encoders = nn.ModuleList(
            [Encoder(n_heads, d_model, d_ffwd, dropout) for i in range(n_layers)]
        )

        self.decoder_token_encoding = TokenEncoding(
            target_vocab_size, d_model, block_size
        )
        self.decoders = nn.ModuleList(
            [Decoder(n_heads, d_model, d_ffwd, dropout) for i in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.model_head = nn.Linear(d_model, target_vocab_size)

    def forward(
        self, encoder_decoder_xs: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        encoder_x, decoder_x = encoder_decoder_xs

        # Encoder.
        encoder_z = self.encoder_token_encoding(encoder_x)

        encoder_attention_mask = get_attention_mask(
            encoder_x,
            encoder_x,
            self.source_pad_idx,
            self.source_pad_idx,
            mask_future=False,
        )
        for encoder in self.encoders:
            encoder_z = encoder(encoder_z, encoder_attention_mask)

        # Decoder.
        decoder_z = self.decoder_token_encoding(decoder_x)

        self_attention_mask = get_attention_mask(
            decoder_x,
            decoder_x,
            self.target_pad_idx,
            self.target_pad_idx,
            mask_future=True,
        )
        cross_attention_mask = get_attention_mask(
            q_tokens=decoder_x,
            k_tokens=encoder_x,
            q_pad_idx=self.target_pad_idx,
            k_pad_idx=self.source_pad_idx,
            mask_future=False,
        )
        for decoder in self.decoders:
            decoder_z = decoder(
                decoder_z,
                encoder_z,
                encoder_z,
                self_attention_mask,
                cross_attention_mask,
            )

        z = self.layer_norm(decoder_z)
        logits = self.model_head(z)
        return logits
