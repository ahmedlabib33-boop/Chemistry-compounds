"""Atom-level SMILES tokenizer and vocabulary."""
from __future__ import annotations

import re

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

_SMILES_TOKEN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def tokenize(smiles: str) -> list[str]:
    tokens = _SMILES_TOKEN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"cannot tokenize SMILES: {smiles!r}")
    return tokens


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = [PAD, BOS, EOS] + sorted(set(tokens) - {PAD, BOS, EOS})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, smiles: list[str]) -> "Vocab":
        return cls([t for s in smiles for t in tokenize(s)])

    @classmethod
    def from_list(cls, itos: list[str]) -> "Vocab":
        vocab = cls([])
        vocab.itos = list(itos)
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        return vocab

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str) -> list[int]:
        return [self.stoi[t] for t in tokenize(smiles)]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)
