import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

from train_funcs import HyperParams, train_model, plot_loss
from models.gpt import GPT
from bpe import BPE, load_bpe


def get_encoder_dataloaders(
    data_dir: Path,
    block_size: int,
    val_split: float,
    batch_size: int,
) -> tuple[BPE, DataLoader, DataLoader]:
    """Get encoded text data. Return encoder, train and val dataloaders."""
    bpe = load_bpe(data_dir / "bpe.model")
    tokens = torch.load(data_dir / "tokens.pt")

    ds = GPTDataset(tokens, block_size)
    train_ds, val_ds = random_split(ds, [1 - val_split, val_split])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size * 2, shuffle=True)
    return bpe, train_dl, val_dl


class GPTDataset(Dataset):
    """Dataset for GPT training."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


@torch.no_grad()
def generate_text(
    model: GPT, block_size: int, bpe: BPE, prompt: str, n_tokens: int
) -> str:
    """Generate text."""
    model.eval()

    tokens = torch.tensor(
        bpe.encode(prompt), dtype=torch.int64, device=HyperParams.DEVICE
    ).unsqueeze(0)

    for i in range(n_tokens):
        logits = model(tokens[:, -block_size:])  # (B, T, C)
        logits = logits[:, -1, :]  # only use last pred col. (B, C)

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_tokens), dim=-1)  # append. (B, T+1)

    return bpe.decode(tokens[0].tolist())


if __name__ == "__main__":
    # Get data.
    BLOCK_SIZE = 256
    dataset = "shakespeare"
    bpe, train_dl, val_dl = get_encoder_dataloaders(
        Path("data") / dataset,
        block_size=BLOCK_SIZE,
        val_split=0.01,
        batch_size=32,
    )

    # Create model.
    gpt = GPT(
        vocab_size=len(bpe.vocab),
        d_model=256,
        d_ffwd=1024,
        block_size=BLOCK_SIZE,
        n_heads=8,
        n_layers=10,
        dropout=0.6,
    )
    gpt = gpt.to(HyperParams.DEVICE)

    # Train model.
    optimizer = torch.optim.Adam(gpt.parameters(), lr=3e-4)
    loss_steps, train_losses, val_losses = train_model(
        gpt,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        steps=30_000,
        eval_step_size=1_000,
        eval_steps=15,
    )

    # Generate new text.
    print("\nGenerating text")
    new_text = generate_text(gpt, BLOCK_SIZE, bpe, prompt="To be or ", n_tokens=1_000)
    print(new_text)

    # Save model outputs.
    save_dir = Path("outputs") / dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "new_text.txt", mode="w") as f:
        f.write(new_text)

    plot_loss(loss_steps, train_losses, val_losses, save_dir / "loss_plot.png")

    torch.save(gpt.state_dict(), save_dir / "weights.pt")
