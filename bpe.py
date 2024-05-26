import re

from tqdm import tqdm
from pathlib import Path


def calc_bigram_freqs(
    tokens: list[int], bigram_freqs: dict = None
) -> dict[tuple[int, int], int]:
    bigram_freqs = bigram_freqs if bigram_freqs else {}
    for bigram in zip(tokens[:-1], tokens[1:]):
        bigram_freqs[bigram] = bigram_freqs.get(bigram, 0) + 1

    return bigram_freqs


def merge(tokens: list[int], bigram: tuple[int, int], new_token: int) -> list[int]:
    new_tokens = []

    i = 0
    while i < len(tokens):
        if (i == (len(tokens) - 1)) or ((tokens[i], tokens[i + 1]) != bigram):
            new_tokens.append(tokens[i])
        else:
            new_tokens.append(new_token)
            i += 1

        i += 1

    return new_tokens


class BPE:
    """Byte-pair encoder."""

    def __init__(
        self, merges: dict[tuple[int, int], int], special_tokens: list[str]
    ) -> None:
        self.merges = merges
        self.vocab, self.special_tokens = self._build_vocab(self.merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            bigram_freqs = calc_bigram_freqs(tokens)

            bigram = min(
                bigram_freqs, key=lambda bigram: self.merges.get(bigram, float("inf"))
            )
            if bigram not in self.merges:
                break

            tokens = merge(tokens, bigram, self.merges[bigram])

        return tokens

    def decode(self, tokens: list[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in tokens)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, save_path: Path) -> None:
        with open(save_path, mode="w") as f:
            # Write special tokens.
            f.write(f"{len(self.special_tokens)}\n")
            for special_token in self.special_tokens:
                f.write(f"{special_token}\n")

            # Write merges.
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")

    def _build_vocab(
        self, merges: dict[tuple[int, int], int], special_tokens: list[str]
    ) -> tuple[dict[int, bytes], dict[str, int]]:
        # Build vocab from merges.
        vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_token in merges.items():
            vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]

        # Add special tokens.
        special_token_dict = {}
        for token in special_tokens:
            i = len(vocab)
            vocab[i] = bytes(token.encode("utf-8"))
            special_token_dict[token] = i

        print(f"\nNew vocab: {[vocab[i] for i in range(256, len(vocab))]}")
        return vocab, special_token_dict


def train_bpe(
    text: str | list[str],
    vocab_size: int,
    special_tokens: list[str] = [],
) -> BPE:
    # Ensure enough vocab size is large enough to fit 256 bytes.
    nonspecial_vocab_size = vocab_size - len(special_tokens)
    assert (
        nonspecial_vocab_size >= 256
    ), f"Nonspecial vocab size {nonspecial_vocab_size} must be >= 256."

    # Convert text to chunks of ints.
    # Use list of text for translation tasks (no cross-sentence merging).
    text: list[str] = [text] if type(text) is str else text

    SPLIT_PATTERN = re.compile(r"""[ ']?[a-zA-Z]+|\d{1,4}|\s+(?!\S)|.+?""")

    text_chunks = []
    for sentence in text:
        text_chunks += SPLIT_PATTERN.findall(sentence)

    token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

    # Train BPE.
    merges = {}

    print("\nTraining BPE")
    for i in tqdm(range(nonspecial_vocab_size - 256)):
        bigram_freqs = {}
        for chunk in token_chunks:
            bigram_freqs = calc_bigram_freqs(chunk, bigram_freqs)

        max_bigram = max(bigram_freqs, key=bigram_freqs.get)
        if bigram_freqs[max_bigram] == 1:
            break

        new_token = 256 + i
        token_chunks = [merge(chunk, max_bigram, new_token) for chunk in token_chunks]

        merges[max_bigram] = new_token

    return BPE(merges, special_tokens)


def load_bpe(save_path: Path) -> BPE:
    with open(save_path, mode="r") as f:
        n_special_tokens = int(f.readline().strip())
        special_tokens = [f.readline().strip() for i in range(n_special_tokens)]

        merge_pairs = [tuple(map(int, line.strip().split())) for line in f.readlines()]

    merges = {}
    for i, merge_pair in enumerate(merge_pairs):
        merges[merge_pair] = i + 256

    return BPE(merges, special_tokens)
