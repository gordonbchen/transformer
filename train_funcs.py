import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Iterator


class HyperParams:
    """Stores hyperparams."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {DEVICE}")


def calc_batch_loss(logits: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Calculate model loss on a batch of data."""
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    yb = yb.view(B * T)

    loss = F.cross_entropy(logits, yb)
    return loss


@torch.no_grad()
def eval_model(model: nn.Module, val_dl: DataLoader, eval_steps: int) -> float:
    """Evaluate model loss."""
    model.eval()

    val_dl_iterator = iter(val_dl)

    total_loss = 0
    for step in range(eval_steps):
        xb, yb, val_dl_iterator = get_next_batch(val_dl_iterator, val_dl)
        total_loss += calc_batch_loss(model(xb), yb).item()

    return total_loss / eval_steps


def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    steps: int,
    eval_step_size: int,
    eval_steps: int,
) -> tuple[list[float], list[float]]:
    """Train a model. Return step number, train and val losses."""
    loss_steps = []
    train_losses = []
    val_losses = []

    print("\nTraining model")
    train_dl_iterator = iter(train_dl)

    for step in tqdm(range(steps)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        xb, yb, train_dl_iterator = get_next_batch(train_dl_iterator, train_dl)

        loss = calc_batch_loss(model(xb), yb)
        loss.backward()
        optimizer.step()

        if (step % eval_step_size == 0) or (step == steps - 1):
            loss_steps.append(step)
            train_losses.append(loss.item())

            val_loss = eval_model(model, val_dl, eval_steps)
            val_losses.append(val_loss)

    print(f"\nMin val loss: {min(val_losses)}")
    return loss_steps, train_losses, val_losses


def get_next_batch(
    dl_iterator: Iterator, dl: DataLoader
) -> tuple[torch.Tensor, torch.Tensor, Iterator]:
    try:
        xb, yb = next(dl_iterator)
    except StopIteration:
        dl_iterator = iter(dl)
        xb, yb = next(dl_iterator)

    # Handle when xb is multiple tensors for nmt.
    if type(xb) is not torch.Tensor:
        xb = tuple(i.to(HyperParams.DEVICE) for i in xb)
    else:
        xb = xb.to(HyperParams.DEVICE)

    return xb, yb.to(HyperParams.DEVICE), dl_iterator


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
