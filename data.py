import requests
import torch

from pathlib import Path


def get_shakespeare_text() -> str:
    """Download and return shakespeare text."""
    shakespeare_path = Path("data/shakespeare.txt")
    
    if not shakespeare_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text_bytes = requests.get(url).content
        print("Shakespeare text downloaded successfully.")

        shakespeare_path.parent.mkdir()

        with open(shakespeare_path, mode="wb") as f:
            f.write(text_bytes)
    
    with open(shakespeare_path, mode="r") as f:
        text = f.read()
    return text


def encode(chars: str, vocab: list[str]) -> list[int]:
    """Encode string as list of ints."""
    char_to_int = {c: i for (i, c) in enumerate(vocab)}
    ints = [char_to_int[c] for c in chars]
    return ints


def decode(ints: list[int], vocab: list[str]) -> str:
    """Decode string from list of ints."""
    int_to_char = {i: c for (i, c) in enumerate(vocab)}
    string = "".join([int_to_char[i] for i in ints])
    return string


def get_shakespeare_vocab_data(val_split: float) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Get encoded shakespeare data. Return vocab, train data, and val data."""
    text = get_shakespeare_text()
    vocab = sorted(list(set(text)))
    data = torch.tensor(encode(text, vocab), dtype=torch.int64)

    train_data, val_data = train_val_split(data, val_split)
    return vocab, train_data, val_data


def train_val_split(data: torch.Tensor, val_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data into train and val."""
    n_val_samples = int(len(data) * val_split)

    val_data = data[:n_val_samples]
    train_data = data[n_val_samples:]
    return train_data, val_data
