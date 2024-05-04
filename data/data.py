import torch

from data.bpe import BytePairEncoder as BPE


def get_encoder_data(
    text_file: str, val_split: float, vocab_size: int
) -> tuple[BPE, torch.Tensor, torch.Tensor]:
    """Get encoded text data. Return encoder, train data, and val data."""
    with open(text_file, mode="r") as f:
        text = f.read()

    bpe = BPE(text, vocab_size)
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
