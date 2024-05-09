import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from pathlib import Path

from bpe import BytePairEncoder as BPE
from models.translator import Translator
from train_funcs import HyperParams, train_model, plot_loss


def get_encoders_dataloaders(
    vocab_size: int,
    block_size: int,
    val_split: float,
    batch_size: int,
) -> tuple[BPE, BPE, DataLoader, DataLoader]:
    """Return eng and spa encoders, and train and val dataloaders"""
    # Read data.
    with open("datasets/eng_spa.txt") as f:
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
    eng_bpe = BPE(eng_lines, vocab_size, ["PAD"])
    spa_bpe = BPE(spa_lines, vocab_size, ["START", "END", "PAD"])

    # Tokenize lines.
    print("\nTokenizing text")
    eng_data = [eng_bpe.encode(i) for i in eng_lines]
    spa_data = [spa_bpe.encode(i) for i in spa_lines]

    # Create dataloaders.
    ds = TranslationDataset(eng_data, spa_data, eng_bpe, spa_bpe, block_size)
    train_ds, val_ds = random_split(ds, [1 - val_split, val_split])

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size * 2, shuffle=True)

    print(f"\nTrain batches: {len(train_dl)}")
    print(f"Val batches: {len(val_dl)}")

    return eng_bpe, spa_bpe, train_dl, val_dl


class TranslationDataset(Dataset):
    """Dataset for language translation."""

    def __init__(
        self,
        eng_data: list[torch.Tensor],
        spa_data: list[torch.Tensor],
        eng_bpe: BPE,
        spa_bpe: BPE,
        block_size: int,
    ) -> None:
        # Remove examples longer than nonspecial block_size.
        n_unfiltered_lines = len(eng_data)

        i = 0
        while i < len(eng_data):
            # Spanish examples will have START (decoder) and END (targets) added, so -1.
            if (len(eng_data[i]) > block_size) or (len(spa_data[i]) > block_size - 1):
                eng_data.pop(i)
                spa_data.pop(i)
            else:
                i += 1

        self.n_lines = len(eng_data)
        print(f"\nExamples lost: {n_unfiltered_lines} - {self.n_lines} ", end="")
        print(f"= {n_unfiltered_lines - self.n_lines}")

        # Save data and bpes.
        self.eng_data = eng_data
        self.spa_data = spa_data

        self.block_size = block_size

        self.eng_bpe = eng_bpe
        self.spa_bpe = spa_bpe

    def __len__(self) -> int:
        return self.n_lines

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        eng_tokens = pad_tokens(self.eng_data[idx], self.block_size, self.eng_bpe)
        spa_decoder_tokens = pad_tokens(
            [self.spa_bpe.special_tokens["START"]] + self.spa_data[idx],
            self.block_size,
            self.spa_bpe,
        )
        spa_target_tokens = pad_tokens(
            self.spa_data[idx] + [self.spa_bpe.special_tokens["END"]],
            self.block_size,
            self.spa_bpe,
        )
        return (eng_tokens, spa_decoder_tokens), spa_target_tokens


def pad_tokens(tokens: list[int], block_size: int, bpe: BPE) -> torch.Tensor:
    """Pad tokens to fit block size and return as tensor."""
    for i in range(block_size - len(tokens)):
        tokens.append(bpe.special_tokens["PAD"])

    return torch.tensor(tokens, dtype=torch.int64)


@torch.no_grad()
def translate(
    eng_text: str, model: Translator, eng_bpe: BPE, spa_bpe: BPE, block_size: int
) -> str:
    model.eval()

    eng_tokens = (
        pad_tokens(eng_bpe.encode(eng_text), block_size, eng_bpe)
        .unsqueeze(0)
        .to(device=HyperParams.DEVICE)
    )
    spa_tokens = (
        pad_tokens([spa_bpe.special_tokens["START"]], block_size, spa_bpe)
        .unsqueeze(0)
        .to(device=HyperParams.DEVICE)
    )

    for i in range(block_size - 1):
        logits = model((eng_tokens, spa_tokens))[:, i, :]
        probs = F.softmax(logits, dim=-1)
        next_spa_token = torch.multinomial(probs, num_samples=1)

        if next_spa_token == spa_bpe.special_tokens["END"]:
            break

        spa_tokens[0, i + 1] = next_spa_token

    return spa_bpe.decode(spa_tokens[0, 1:].tolist())


if __name__ == "__main__":
    BLOCK_SIZE = 48
    eng_bpe, spa_bpe, train_dl, val_dl = get_encoders_dataloaders(
        vocab_size=256 + 256,
        block_size=BLOCK_SIZE,
        val_split=0.15,
        batch_size=32,
    )

    translator = Translator(
        source_vocab_size=len(eng_bpe.vocab),
        target_vocab_size=len(spa_bpe.vocab),
        d_model=512,
        d_ffwd=2048,
        block_size=BLOCK_SIZE,
        n_heads=8,
        n_layers=8,
        dropout=0.6,
        source_pad_idx=eng_bpe.special_tokens["PAD"],
        target_pad_idx=spa_bpe.special_tokens["PAD"],
    )
    translator = translator.to(HyperParams.DEVICE)

    optimizer = torch.optim.Adam(translator.parameters(), lr=3e-4)

    loss_steps, train_losses, val_losses = train_model(
        model=translator,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        steps=5_000,
        eval_step_size=500,
        eval_steps=15,
        cross_entropy_ignore_index=spa_bpe.special_tokens["PAD"],
    )

    print("\nTranslating")
    eng_sentences = [
        "I like to eat apples and swim.",
        "Can you speak French?",
        "You can't blame her for not knowing what she hasn't been taught.",
        "Get off my lawn!",
        "He can speak ten languages.",
        "I met a girl I like very much.",
        "I am going to a wedding tomorrow.",
        "Do you want to have lunch with me tomorrow?",
    ]
    for eng_sentence in eng_sentences:
        spa_sentence = translate(eng_sentence, translator, eng_bpe, spa_bpe, BLOCK_SIZE)

        print(f"Eng: {eng_sentence}")
        print(f"Spa: {spa_sentence}\n")

    save_name = "eng_spa"
    plot_loss(loss_steps, train_losses, val_losses, Path("loss_plots") / save_name)

    weights_path = Path("weights")
    weights_path.mkdir(exist_ok=True)
    torch.save(translator.state_dict(), weights_path / f"{save_name}.pt")
