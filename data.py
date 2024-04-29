import re
import torch

from tqdm import tqdm
from pathlib import Path


class BytePairEncoder:
    """Byte-pair encoder."""

    def __init__(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256, f"vocab_size {vocab_size} must be >= 256."

        split_pattern = re.compile(r"""[ ']?[a-zA-Z]+|\d{1,4}|\s+(?!\S)|.+?""")
        text_chunks = split_pattern.findall(text)
        token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        print("\nTraining byte pair encoder")
        for i in tqdm(range(vocab_size - 256)):
            bigram_freqs = {}
            for chunk in token_chunks:  # TODO: multiprocessing.
                bigram_freqs = self.calc_bigram_freqs(chunk, bigram_freqs)

            max_bigram = max(bigram_freqs, key=bigram_freqs.get)
            if bigram_freqs[max_bigram] == 1:
                break

            new_token = 256 + i
            token_chunks = [
                self.merge(chunk, max_bigram, new_token) for chunk in token_chunks
            ]

            self.merges[max_bigram] = new_token
            self.vocab[new_token] = (
                self.vocab[max_bigram[0]] + self.vocab[max_bigram[1]]
            )

        print("\nNew vocab")
        print([self.vocab[i] for i in range(256, len(self.vocab))])

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            bigram_freqs = self.calc_bigram_freqs(tokens)

            bigram = min(
                bigram_freqs, key=lambda bigram: self.merges.get(bigram, float("inf"))
            )
            if bigram not in self.merges:
                break

            tokens = self.merge(tokens, bigram, self.merges[bigram])

        return tokens

    def decode(self, tokens: list[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in tokens)
        return text_bytes.decode("utf-8", errors="replace")

    def calc_bigram_freqs(
        self, tokens: list[int], bigram_freqs: dict = None
    ) -> dict[tuple[int, int], int]:
        if not bigram_freqs:
            bigram_freqs = {}
        for bigram in zip(tokens[:-1], tokens[1:]):
            if bigram not in bigram_freqs:
                bigram_freqs[bigram] = 0
            bigram_freqs[bigram] += 1

        return bigram_freqs

    def merge(
        self, tokens: list[int], bigram: tuple[int, int], new_token: int
    ) -> list[int]:
        new_tokens = []

        i = 0
        while i < len(tokens):
            if (i == (len(tokens) - 1)) or ((tokens[i], tokens[i + 1]) != bigram):
                new_tokens.append(tokens[i])
            else:
                new_tokens.append(new_token)
                i += 1

            i += 1

        return new_tokens


def get_encoder_data(
    text_file: Path | str, val_split: float, vocab_size: int
) -> tuple[BytePairEncoder, torch.Tensor, torch.Tensor]:
    """Get encoded text data. Return encoder, train data, and val data."""
    with open(text_file, mode="r") as f:
        text = f.read()

    bpe = BytePairEncoder(text, vocab_size)
    data = torch.tensor(bpe.encode(text), dtype=torch.int64)

    train_data, val_data = train_val_split(data, val_split)
    return bpe, train_data, val_data


def train_val_split(
    data: torch.Tensor, val_split: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data into train and val."""
    n_val_samples = int(len(data) * val_split)

    val_data = data[:n_val_samples]
    train_data = data[n_val_samples:]
    return train_data, val_data
