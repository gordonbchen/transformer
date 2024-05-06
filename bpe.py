import re

from tqdm import tqdm


class BytePairEncoder:
    """Byte-pair encoder."""

    def __init__(
        self,
        text: str | list[str],
        vocab_size: int,
        special_tokens: list[str] = [],
    ) -> None:
        # Ensure enough vocab size is large enough to fit 256 bytes.
        nonspecial_vocab_size = vocab_size - len(special_tokens)
        assert (
            nonspecial_vocab_size >= 256
        ), f"Nonspecial vocab size {nonspecial_vocab_size} must be >= 256."

        # Convert text to chunks of ints.
        if type(text) is str:
            text = [text]

        SPLIT_PATTERN = re.compile(r"""[ ']?[a-zA-Z]+|\d{1,4}|\s+(?!\S)|.+?""")

        text_chunks = []
        for sentence in text:
            text_chunks += SPLIT_PATTERN.findall(sentence)

        token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        # Train BPE.
        self.merges, self.vocab = self._train_bpe(token_chunks, nonspecial_vocab_size)
        print(f"\nNew vocab: {[self.vocab[i] for i in range(256, len(self.vocab))]}")

        self.special_tokens = self._add_special_tokens(special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            bigram_freqs = self._calc_bigram_freqs(tokens)

            bigram = min(
                bigram_freqs, key=lambda bigram: self.merges.get(bigram, float("inf"))
            )
            if bigram not in self.merges:
                break

            tokens = self._merge(tokens, bigram, self.merges[bigram])

        return tokens

    def decode(self, tokens: list[int]) -> str:
        text_bytes = b"".join(self.vocab[i] for i in tokens)
        return text_bytes.decode("utf-8", errors="replace")

    def _train_bpe(
        self, token_chunks: list[list[int]], vocab_size: int
    ) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
        """Train bpe, return dict of merges and vocab."""
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        print("\nTraining BPE")
        for i in tqdm(range(vocab_size - 256)):
            bigram_freqs = {}
            for chunk in token_chunks:
                bigram_freqs = self._calc_bigram_freqs(chunk, bigram_freqs)

            max_bigram = max(bigram_freqs, key=bigram_freqs.get)
            if bigram_freqs[max_bigram] == 1:
                break

            new_token = 256 + i
            token_chunks = [
                self._merge(chunk, max_bigram, new_token) for chunk in token_chunks
            ]

            merges[max_bigram] = new_token
            vocab[new_token] = vocab[max_bigram[0]] + vocab[max_bigram[1]]

        return merges, vocab

    def _calc_bigram_freqs(
        self, tokens: list[int], bigram_freqs: dict = None
    ) -> dict[tuple[int, int], int]:
        if not bigram_freqs:
            bigram_freqs = {}
        for bigram in zip(tokens[:-1], tokens[1:]):
            if bigram not in bigram_freqs:
                bigram_freqs[bigram] = 0
            bigram_freqs[bigram] += 1

        return bigram_freqs

    def _merge(
        self, tokens: list[int], bigram: tuple[int, int], new_token: int
    ) -> list[int]:
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

    def _add_special_tokens(self, special_tokens: list[str]) -> dict[str, int]:
        special_token_dict = {}
        for token in special_tokens:
            i = len(self.vocab)
            self.vocab[i] = bytes()  # Placeholder just to increase vocab size.
            special_token_dict[token] = i

        return special_token_dict
