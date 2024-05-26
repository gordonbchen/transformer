import torch

from pathlib import Path

from bpe import train_bpe


# Read data.
dataset = "eng_spa"
with open(f"datasets/{dataset}.txt", mode="r", encoding="utf-8") as f:
    lines = f.read().split("\n")
lines.pop()  # Remove empty last line

# Split into english and spanish text.
eng_lines = []
spa_lines = []

for line in lines:
    eng, spa = line.split("\t")
    eng_lines.append(eng)
    spa_lines.append(spa)

# Train english and spanish bpe tokenizers.
eng_bpe = train_bpe(eng_lines, vocab_size=256 + 256, special_tokens=["PAD"])
spa_bpe = train_bpe(
    spa_lines, vocab_size=256 + 256, special_tokens=["START", "END", "PAD"]
)

# Tokenize lines.
print("\nTokenizing text")
eng_tokens = [torch.tensor(eng_bpe.encode(i), dtype=torch.int64) for i in eng_lines]
spa_tokens = [torch.tensor(spa_bpe.encode(i), dtype=torch.int64) for i in spa_lines]

# Save tokens and bpe.
save_dir = Path(f"data/{dataset}")
save_dir.mkdir(parents=True, exist_ok=True)

torch.save(eng_tokens, save_dir / "eng_tokens.pt")
torch.save(spa_tokens, save_dir / "spa_tokens.pt")

eng_bpe.save(save_dir / "eng_bpe.model")
spa_bpe.save(save_dir / "spa_bpe.model")
