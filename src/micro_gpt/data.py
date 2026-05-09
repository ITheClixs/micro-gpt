"""Data utilities for next-token micro-GPT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for char, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(token_id)] for token_id in token_ids)


def load_text(path, text_field="text"):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line)[text_field])
        return "\n".join(rows)
    return path.read_text(encoding="utf-8")


def make_lm_batch(tokens, block_size, batch_size, seed=0):
    if tokens.numel() <= block_size:
        raise ValueError("Token sequence must be longer than block_size.")
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    max_start = tokens.numel() - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator, device=tokens.device)
    x = torch.stack([tokens[start:start + block_size] for start in starts])
    y = torch.stack([tokens[start + 1:start + block_size + 1] for start in starts])
    return x.long(), y.long()


def encode_text(text):
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokens, tokenizer


def train_validation_split(tokens, validation_fraction=0.1):
    split_index = max(1, int(tokens.numel() * (1.0 - validation_fraction)))
    split_index = min(split_index, tokens.numel() - 1)
    return tokens[:split_index], tokens[split_index:]
