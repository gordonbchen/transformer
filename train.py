import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from data import get_encoder_data, BytePairEncoder
from model import Transformer


class HyperParams:
    """Stores hyperparams."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {DEVICE}")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> torch.Tensor:
    """Get random batch."""
    inds = torch.randint(len(data) - block_size - 1, size=(batch_size,))

    xb = torch.stack([data[i : i + block_size] for i in inds])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in inds])
    return xb, yb


def calc_batch_loss(
    model: torch.nn.Module, data: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Calculate model loss on a data batch."""
    xb, yb = get_batch(data, model.block_size, batch_size)
    xb, yb = xb.to(HyperParams.DEVICE), yb.to(HyperParams.DEVICE)

    logits, loss = model(xb, targets=yb)
    return loss


@torch.no_grad()
def eval_model(
    model: torch.nn.Module, data: torch.Tensor, batch_size: int, eval_steps: int
):
    """Evaluate model loss across many steps."""
    model.eval()

    total_loss = 0
    for step in range(eval_steps):
        loss = calc_batch_loss(model, data, batch_size)
        total_loss += loss.item()

    return total_loss / eval_steps


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    steps: int,
    eval_step_size: int,
    eval_steps: int,
) -> tuple[list[float], list[float]]:
    """Train a model. Return step number, train and val losses."""
    loss_steps = []
    train_losses = []
    val_losses = []

    print("\nTraining model")
    for step in tqdm(range(steps)):
        model.train()
        loss = calc_batch_loss(model, train_data, batch_size)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step % eval_step_size == 0) or (step == steps - 1):
            loss_steps.append(step)

            train_losses.append(loss.item())

            val_loss = eval_model(model, val_data, batch_size, eval_steps)
            val_losses.append(val_loss)

    return loss_steps, train_losses, val_losses


def plot_loss(
    loss_steps: list[int],
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
) -> None:
    """Plot train and val loss."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(loss_steps, train_losses, color="blue", label="train")
    ax.plot(loss_steps, val_losses, color="red", label="val")

    ax.legend(loc="best")

    ax.set_xticks(loss_steps)
    ax.tick_params(axis="x", which="both", labelrotation=45)

    min_y = int(min(val_losses + train_losses))
    max_y = int(max(val_losses + train_losses)) + 1
    ax.set_yticks(torch.arange(min_y, max_y, step=0.25))

    ax.grid(visible=True, which="both", axis="both")

    if not save_path.parent.exists():
        save_path.parent.mkdir()
    fig.savefig(save_path)


def generate_text(
    model: torch.nn.Module, bpe: BytePairEncoder, prompt: str, n_tokens: int
) -> str:
    """Generate text."""
    model.eval()

    input_prompt_tokens = torch.tensor(
        bpe.encode(prompt), dtype=torch.int64, device=HyperParams.DEVICE
    ).unsqueeze(0)
    new_tokens = model.generate(input_prompt_tokens, n_tokens)
    return bpe.decode(new_tokens[0].tolist())


if __name__ == "__main__":
    bpe, train_data, val_data = get_encoder_data(
        "data/war_and_peace.txt", val_split=0.1, vocab_size=256 + 256
    )

    transformer = Transformer(
        vocab_size=len(bpe.vocab),
        d_model=256,
        d_ffwd=1024,
        block_size=128,
        n_heads=8,
        n_layers=8,
        dropout=0.6,
    )
    transformer = transformer.to(HyperParams.DEVICE)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)

    loss_steps, train_losses, val_losses = train_model(
        transformer,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        batch_size=32,
        steps=10_000,
        eval_step_size=250,
        eval_steps=10,
    )
    print(f"\nMin val loss: {min(val_losses)}")

    print(f"\nGenerating text")
    print(generate_text(transformer, bpe, prompt="To be or ", n_tokens=1_000))

    save_name = "war_and_peace"
    plot_loss(loss_steps, train_losses, val_losses, Path("loss_plots") / save_name)

    weights_path = Path("weights")
    weights_path.mkdir(exist_ok=True)
    torch.save(transformer.state_dict(), weights_path / f"{save_name}.pt")
