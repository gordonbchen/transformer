import torch
import torch.nn.functional as F

from pathlib import Path

from train_funcs import HyperParams, train_model, plot_loss
from models.gpt import GPT
from bpe import BytePairEncoder as BPE


def train_val_split(
    data: torch.Tensor, val_split: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data into train and val."""
    n_val_samples = int(len(data) * val_split)

    val_data = data[:n_val_samples]
    train_data = data[n_val_samples:]
    return train_data, val_data


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


@torch.no_grad()
def generate_text(model: GPT, bpe: BPE, prompt: str, n_tokens: int) -> str:
    """Generate text."""
    model.eval()

    tokens = torch.tensor(
        bpe.encode(prompt), dtype=torch.int64, device=HyperParams.DEVICE
    ).unsqueeze(0)

    for i in range(n_tokens):
        logits = model(tokens[:, -model.block_size :])  # (B, T, C)
        logits = logits[:, -1, :]  # only use last pred col. (B, C)

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat((tokens, next_tokens), dim=-1)  # append. (B, T+1)

    return bpe.decode(tokens[0].tolist())


if __name__ == "__main__":
    bpe, train_data, val_data = get_encoder_data(
        "datasets/war_and_peace.txt", val_split=0.1, vocab_size=256 + 256
    )

    gpt = GPT(
        vocab_size=len(bpe.vocab),
        d_model=256,
        d_ffwd=1024,
        block_size=128,
        n_heads=8,
        n_layers=8,
        dropout=0.6,
    )
    gpt = gpt.to(HyperParams.DEVICE)

    optimizer = torch.optim.Adam(gpt.parameters(), lr=3e-4)

    loss_steps, train_losses, val_losses = train_model(
        gpt,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=32,
        steps=5_000,
        eval_step_size=250,
        eval_steps=10,
    )

    print("\nGenerating text")
    print(generate_text(gpt, bpe, prompt="To be or ", n_tokens=1_000))

    save_name = "war_and_peace"
    plot_loss(loss_steps, train_losses, val_losses, Path("loss_plots") / save_name)

    weights_path = Path("weights")
    weights_path.mkdir(exist_ok=True)
    torch.save(gpt.state_dict(), weights_path / f"{save_name}.pt")
