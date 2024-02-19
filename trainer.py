import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from data import get_shakespeare_vocab_data, decode
from models import BigramModel


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int) -> torch.Tensor:
    """Get random batch."""
    inds = torch.randint(len(data)-block_size-1, size=(batch_size,))

    xb = torch.stack([data[i : i+block_size] for i in inds])
    yb = torch.stack([data[i+1 : i+block_size+1] for i in inds])
    return xb, yb


def calc_batch_loss(model: torch.nn.Module, data: torch.Tensor, block_size: int, batch_size: int) -> torch.Tensor:
    """Calculate model loss on a data batch."""
    xb, yb = get_batch(data, block_size, batch_size)
    xb, yb = xb.to(device), yb.to(device)
    
    logits, loss = model(xb, targets=yb)
    return loss


@torch.no_grad()
def eval_model(model: torch.nn.Module, data: torch.Tensor, batch_size: int, block_size: int, eval_steps: int):
    """Evaluate model loss across many steps."""
    model.eval()

    total_loss = 0
    for step in range(eval_steps):
        loss = calc_batch_loss(model, data, block_size, batch_size)
        total_loss += loss.item()

    return total_loss / eval_steps


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    steps: int,
    eval_step_size: int,
    eval_steps: int
) -> tuple[list[float], list[float]]:
    """Train a model. Return step number, train and val losses."""
    model = model.to(device)

    loss_steps = []
    train_losses = []
    val_losses = []
    for step in tqdm(range(steps)):
        model.train()
        loss = calc_batch_loss(model, train_data, block_size, batch_size)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step % eval_step_size == 0) or (step == steps - 1):
            train_loss = eval_model(model, train_data, batch_size, block_size, eval_steps)
            val_loss = eval_model(model, val_data, batch_size, block_size, eval_steps) 

            loss_steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    return loss_steps, train_losses, val_losses


def plot_loss(loss_steps: list[int], train_losses: list[float], val_losses: list[float], save_path: Path) -> None:
    """Plot train and val loss."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(loss_steps, train_losses, color="blue", label="train")
    ax.plot(loss_steps, val_losses, color="red", label="val")

    ax.legend(loc="best")

    if not save_path.parent.exists():
        save_path.parent.mkdir()
    fig.savefig(save_path)


if __name__ == "__main__":
    vocab, train_data, val_data = get_shakespeare_vocab_data(val_split=0.1)

    bigram_model = BigramModel(vocab_size=len(vocab), embed_size=32)

    loss_steps, train_losses, val_losses = train_model(
        bigram_model,
        optimizer=torch.optim.Adam(bigram_model.parameters(), lr=1e-3),
        train_data=train_data,
        val_data=val_data,
        batch_size=32,
        block_size=8,
        steps=10_000,
        eval_step_size=250,
        eval_steps=100
    )

    plot_loss(loss_steps, train_losses, val_losses, save_path=Path("loss_plots/bigram_with_linear_head"))

    input_prompt_tokens = torch.zeros((1, 1), dtype=torch.int64, device=device)
    new_tokens = bigram_model.generate(input_prompt_tokens, n_tokens=500)
    print(decode(new_tokens[0].tolist(), vocab))
