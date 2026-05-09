"""Checkpoint helpers for terminal micro-GPT smoke runs."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import MicroGPTConfig
from .data import CharTokenizer


CHECKPOINT_VERSION = 1


def tokenizer_to_dict(tokenizer):
    return {
        "stoi": dict(tokenizer.stoi),
        "itos": {str(index): char for index, char in tokenizer.itos.items()},
    }


def tokenizer_from_dict(payload):
    return CharTokenizer(
        stoi={str(char): int(index) for char, index in payload["stoi"].items()},
        itos={int(index): str(char) for index, char in payload["itos"].items()},
    )


def save_micro_gpt_checkpoint(path, model, config, tokenizer, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer": tokenizer_to_dict(tokenizer),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_micro_gpt_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu")
    return {
        "version": payload.get("version", 0),
        "model": payload["model"],
        "config": MicroGPTConfig(**payload["config"]),
        "tokenizer": tokenizer_from_dict(payload["tokenizer"]),
        "metadata": payload.get("metadata", {}),
    }
