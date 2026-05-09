"""Dry-run and training harness for micro-GPT experiments."""

from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .data import encode_text, make_lm_batch
from .metrics import perplexity, tokens_per_second
from .model import MicroGPT


DEFAULT_DEMO_TEXT = (
    "micro gpt research lab studies gradients, attention, recurrence, "
    "convolutions, and reinforcement learning with transparent experiments. "
)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dry_training(config, text=DEFAULT_DEMO_TEXT):
    set_seed(config.seed)
    tokens, tokenizer = encode_text(text)
    effective_vocab = max(config.vocab_size, tokenizer.vocab_size)
    if effective_vocab != config.vocab_size:
        config = type(config)(**{**config.to_dict(), "vocab_size": effective_vocab})
    model = MicroGPT(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    started = time.perf_counter()
    final_loss = None
    for step in range(config.max_steps):
        x, y = make_lm_batch(
            tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            seed=config.seed + step,
        )
        output = model(x, labels=y)
        final_loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    elapsed = time.perf_counter() - started
    token_count = config.max_steps * config.batch_size * config.block_size
    return {
        "steps": config.max_steps,
        "loss": float(final_loss.detach()),
        "perplexity": perplexity(final_loss.detach()),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokens_per_second": tokens_per_second(token_count, elapsed),
        "dry_run": True,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Micro-GPT research training harness.")
    parser.add_argument("--config", required=True, help="Path to a micro-GPT JSON config.")
    parser.add_argument("--text", default=DEFAULT_DEMO_TEXT, help="Inline dry-run text corpus.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny deterministic smoke training loop without checkpointing.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if not args.dry_run:
        raise SystemExit("Only --dry-run is enabled in this research skeleton.")
    metrics = run_dry_training(config, text=args.text)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
