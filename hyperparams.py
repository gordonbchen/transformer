import torch


class HyperParms():
    """Stores hyperparams."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    BLOCK_SIZE = 8