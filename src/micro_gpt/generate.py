"""Generation CLI for micro-GPT checkpoints."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_micro_gpt_checkpoint
from .cli import _generate_with_tokenizer_vocab, non_negative_int
from .config import load_config
from .data import CharTokenizer
from .model import MicroGPT


def build_parser():
    parser = argparse.ArgumentParser(description="Generate text from a micro-GPT checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=non_negative_int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        checkpoint = load_micro_gpt_checkpoint(args.checkpoint)
        config = checkpoint["config"]
        tokenizer = checkpoint["tokenizer"]
    except KeyError:
        config = load_config(args.config)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        tokenizer = CharTokenizer(
            stoi=checkpoint["tokenizer"]["stoi"],
            itos={int(key): value for key, value in checkpoint["tokenizer"]["itos"].items()},
        )
    model = MicroGPT(config)
    model.load_state_dict(checkpoint["model"])
    if not args.prompt:
        raise SystemExit("generate --prompt must not be empty.")
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    generated = _generate_with_tokenizer_vocab(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
