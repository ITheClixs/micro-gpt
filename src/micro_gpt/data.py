"""Data utilities for next-token micro-GPT experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path

import torch


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    kind: str = "char"

    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        stoi = {char: index for index, char in enumerate(chars)}
        itos = {index: char for char, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_dict(cls, payload):
        return cls(
            stoi={str(char): int(index) for char, index in payload["stoi"].items()},
            itos={int(index): str(char) for index, char in payload["itos"].items()},
        )

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, token_ids):
        return "".join(self.itos[int(token_id)] for token_id in token_ids)

    def to_dict(self):
        return {
            "type": self.kind,
            "stoi": dict(self.stoi),
            "itos": dict(self.itos),
        }


@dataclass
class BPETokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]
    kind: str = "bpe"

    @classmethod
    def from_text(cls, text, target_vocab_size=None):
        if len(set(text)) < 2:
            text = text + "\n "
        base_tokens = sorted(set(text))
        stoi = {token: index for index, token in enumerate(base_tokens)}
        itos = {index: token for token, index in stoi.items()}
        merges: list[tuple[str, str]] = []
        sequence = list(text)
        target_vocab_size = max(target_vocab_size or len(stoi), len(stoi))
        while len(stoi) < target_vocab_size and len(sequence) > 1:
            pair_counts = Counter(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, _ = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            merged = pair[0] + pair[1]
            if merged in stoi:
                break
            merges.append(pair)
            stoi[merged] = len(stoi)
            itos[len(itos)] = merged
            sequence = _merge_pair(sequence, pair)
        return cls(stoi=stoi, itos=itos, merges=merges)

    @classmethod
    def from_dict(cls, payload):
        merges = [tuple(pair) for pair in payload.get("merges", [])]
        stoi = {str(token): int(index) for token, index in payload["stoi"].items()}
        itos = {int(index): str(token) for index, token in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos, merges=merges)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        symbols = list(text)
        if not symbols:
            return []
        rank = {pair: index for index, pair in enumerate(self.merges)}
        while len(symbols) > 1:
            best_index = None
            best_rank = None
            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                pair_rank = rank.get(pair)
                if pair_rank is None:
                    continue
                if best_rank is None or pair_rank < best_rank:
                    best_rank = pair_rank
                    best_index = index
            if best_index is None:
                break
            merged = symbols[best_index] + symbols[best_index + 1]
            symbols = symbols[:best_index] + [merged] + symbols[best_index + 2 :]
        return [self.stoi[symbol] for symbol in symbols]

    def decode(self, token_ids):
        return "".join(self.itos[int(token_id)] for token_id in token_ids)

    def to_dict(self):
        return {
            "type": self.kind,
            "stoi": dict(self.stoi),
            "itos": dict(self.itos),
            "merges": [list(pair) for pair in self.merges],
        }


def _merge_pair(sequence, pair):
    merged = []
    index = 0
    while index < len(sequence):
        if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
            merged.append(sequence[index] + sequence[index + 1])
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


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


def encode_text_with_tokenizer(text, tokenizer_kind="char", target_vocab_size=None):
    if tokenizer_kind == "bpe":
        tokenizer = BPETokenizer.from_text(text, target_vocab_size=target_vocab_size)
    else:
        tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokens, tokenizer


def train_validation_split(tokens, validation_fraction=0.1):
    split_index = max(1, int(tokens.numel() * (1.0 - validation_fraction)))
    split_index = min(split_index, tokens.numel() - 1)
    return tokens[:split_index], tokens[split_index:]
