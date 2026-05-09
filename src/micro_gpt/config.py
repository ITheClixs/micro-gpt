"""Configuration schema for micro-GPT experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class MicroGPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    bias: bool = False
    batch_size: int = 8
    max_steps: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    def __post_init__(self):
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.n_layer < 1:
            raise ValueError("n_layer must be positive.")
        if self.n_head < 1:
            raise ValueError("n_head must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return MicroGPTConfig(**data)


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
