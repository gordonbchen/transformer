import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path


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
    model: nn.Module, data: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Calculate model loss on a data batch."""
    xb, yb = get_batch(data, model.block_size, batch_size)
    xb, yb = xb.to(HyperParams.DEVICE), yb.to(HyperParams.DEVICE)

    logits = model(xb)

    # Calc cross-entropy loss.
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    yb = yb.view(B * T)

    loss = F.cross_entropy(logits, yb)
    return loss


@torch.no_grad()
def eval_model(model: nn.Module, data: torch.Tensor, batch_size: int, eval_steps: int):
    """Evaluate model loss across many steps."""
    model.eval()

    total_loss = 0
    for step in range(eval_steps):
        loss = calc_batch_loss(model, data, batch_size)
        total_loss += loss.item()

    return total_loss / eval_steps


def train_model(
    model: nn.Module,
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
        optimizer.zero_grad(set_to_none=True)

        loss = calc_batch_loss(model, train_data, batch_size)

        loss.backward()
        optimizer.step()

        if (step % eval_step_size == 0) or (step == steps - 1):
            loss_steps.append(step)

            train_losses.append(loss.item())

            # Use 2 * batch_size b/c no grad storing.
            val_loss = eval_model(model, val_data, batch_size * 2, eval_steps)
            val_losses.append(val_loss)

    print(f"\nMin val loss: {min(val_losses)}")
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
