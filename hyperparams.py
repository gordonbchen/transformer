import torch


class HyperParams:
    """Stores hyperparams."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
