import torch

from pathlib import Path

from bpe import train_bpe


# Read text and create bpe.
dataset = "shakespeare"
with open(f"datasets/{dataset}.txt", mode="r", encoding="utf-8") as f:
    text = f.read()

bpe = train_bpe(text, vocab_size=256 + 256)

print("\nEncoding text")
tokens = torch.tensor(bpe.encode(text), dtype=torch.int64)

# Save tokens and bpe.
save_dir = Path(f"data/{dataset}")
save_dir.mkdir(parents=True, exist_ok=True)

torch.save(tokens, save_dir / "tokens.pt")
bpe.save(save_dir / "bpe.model")
