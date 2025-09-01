import pickle
from typing import Iterable, Iterator

import numpy as np
import torch

from cs336_basics.train_bpe import pretokenize

REPLACEMENT_CHAR = "\ufffd"


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Given a vocabulary, a list of merges, and a list of special tokens,
        return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        self.vocab = vocab
        # map from vocab to token id
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        # sort special tokens in terms of decreasing order of length
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens_list = pretokenize(text, self.special_tokens, keep_special_tokens=True)
        for tokens in tokens_list:
            for p in self.merges:
                new_token = p[0] + p[1]
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == p:
                        i += 2
                        new_tokens.append(new_token)
                    else:
                        i += 1
                        new_tokens.append(tokens[i])
                tokens = new_tokens
        # note that at this point all items in tokens have already being merged
        return [self.vocab_reversed[tokens] for tokens in tokens_list]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        tokens = bytes()
        for id in ids:
            if id < len(self.vocab):
                tokens += self.vocab(id)
            else:
                tokens += REPLACEMENT_CHAR.encode("utf-8")
        return tokens.decode("utf-8", errors="replace")
