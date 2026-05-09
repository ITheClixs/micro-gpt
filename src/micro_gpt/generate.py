"""Generation CLI for micro-GPT checkpoints."""

from __future__ import annotations

import argparse

import torch

from .config import load_config
from .data import CharTokenizer
from .model import MicroGPT


def build_parser():
    parser = argparse.ArgumentParser(description="Generate text from a micro-GPT checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = CharTokenizer(
        stoi=checkpoint["tokenizer"]["stoi"],
        itos={int(key): value for key, value in checkpoint["tokenizer"]["itos"].items()},
    )
    model = MicroGPT(config)
    model.load_state_dict(checkpoint["model"])
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
