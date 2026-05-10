"""Dry-run and training harness for micro-GPT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .checkpoint import save_micro_gpt_checkpoint
from .config import load_config
from .data import encode_text, load_text, make_lm_batch
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
    return _run_training_loop(config, text=text, dry_run=True)


def resolve_training_text(text=DEFAULT_DEMO_TEXT, text_file=None, text_field="text"):
    if text_file is None:
        return text
    loaded_text = load_text(text_file, text_field=text_field)
    if not loaded_text.strip():
        raise ValueError("training text file must not be empty.")
    return loaded_text


def run_training(
    config,
    text=DEFAULT_DEMO_TEXT,
    checkpoint_path=None,
    metrics_path=None,
    run_name="local-train",
):
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for non-dry training.")
    metrics = _run_training_loop(config, text=text, dry_run=False, run_name=run_name)
    save_micro_gpt_checkpoint(
        checkpoint_path,
        metrics.pop("_model"),
        metrics.pop("_config"),
        metrics.pop("_tokenizer"),
        metadata={
            "run_name": run_name,
            "steps": metrics["steps"],
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
            "dry_run": False,
        },
    )
    metrics["checkpoint_path"] = str(checkpoint_path)
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return metrics


def _run_training_loop(config, text=DEFAULT_DEMO_TEXT, dry_run=True, run_name="dry-run"):
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
    metrics = {
        "run_name": run_name,
        "steps": config.max_steps,
        "loss": float(final_loss.detach()),
        "perplexity": perplexity(final_loss.detach()),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokens_per_second": tokens_per_second(token_count, elapsed),
        "tokens_processed": token_count,
        "dry_run": dry_run,
    }
    if not dry_run:
        metrics["_model"] = model
        metrics["_config"] = config
        metrics["_tokenizer"] = tokenizer
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Micro-GPT research training harness.")
    parser.add_argument("--config", required=True, help="Path to a micro-GPT JSON config.")
    parser.add_argument("--text", default=DEFAULT_DEMO_TEXT, help="Inline training text corpus.")
    parser.add_argument(
        "--text-file",
        help="Path to a UTF-8 text or JSONL corpus. JSONL reads the field selected by --text-field.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSONL field to read when --text-file points to a .jsonl corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a tiny deterministic smoke training loop without checkpointing.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run bounded local training and write a checkpoint.",
    )
    parser.add_argument("--checkpoint-out", help="Path for a local training checkpoint.")
    parser.add_argument("--metrics-out", help="Optional path for JSON metrics.")
    parser.add_argument("--run-name", default="local-train")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    try:
        text = resolve_training_text(args.text, args.text_file, text_field=args.text_field)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.train == args.dry_run:
        raise SystemExit("Choose exactly one of --dry-run or --train.")
    if args.dry_run:
        metrics = run_dry_training(config, text=text)
    else:
        if args.checkpoint_out is None:
            raise SystemExit("--checkpoint-out is required with --train.")
        metrics = run_training(
            config,
            text=text,
            checkpoint_path=args.checkpoint_out,
            metrics_path=args.metrics_out,
            run_name=args.run_name,
        )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
