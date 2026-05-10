"""Unified terminal CLI for micro-GPT inspection, smoke runs, and generation."""

from __future__ import annotations

import argparse
import json
import time

import torch

from .checkpoint import load_micro_gpt_checkpoint, save_micro_gpt_checkpoint
from .config import load_config
from .data import CharTokenizer, make_lm_batch
from .metrics import perplexity, tokens_per_second
from .model import MicroGPT
from .train import set_seed


ARCHITECTURE_FLAGS = {
    "decoder_only": True,
    "causal_attention": True,
    "rope": True,
    "rms_norm": True,
    "swiglu": True,
    "weight_tying": True,
}


def _config_with_vocab(config, vocab_size):
    effective_vocab = max(2, vocab_size)
    if config.vocab_size == effective_vocab:
        return config
    return type(config)(**{**config.to_dict(), "vocab_size": effective_vocab})


def _build_tokenizer(text):
    if len(set(text)) < 2:
        text = text + "\n "
    return CharTokenizer.from_text(text)


def _encode_known(tokenizer, text):
    missing = sorted({char for char in text if char not in tokenizer.stoi})
    if missing:
        display = "".join(missing)
        raise SystemExit(f"Prompt contains characters missing from tokenizer: {display!r}")
    return tokenizer.encode(text)


def _decode_known(tokenizer, token_ids):
    return "".join(tokenizer.itos.get(int(token_id), "") for token_id in token_ids)


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def mask_logits_to_vocab(logits, vocab_size):
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if vocab_size >= logits.size(-1):
        return logits
    masked = logits.clone()
    masked[..., vocab_size:] = -float("inf")
    return masked


@torch.no_grad()
def _generate_with_tokenizer_vocab(
    model,
    input_ids,
    max_new_tokens,
    tokenizer_vocab_size,
    temperature=1.0,
    top_k=None,
):
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.block_size:]
        logits = model(context).logits[:, -1, :] / max(temperature, 1e-8)
        logits = mask_logits_to_vocab(logits, tokenizer_vocab_size)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, tokenizer_vocab_size, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    if was_training:
        model.train()
    return input_ids


def _ensure_batchable_text(text, block_size):
    if not text:
        raise SystemExit("smoke --text must not be empty.")
    required_length = block_size + 1
    if len(text) > block_size:
        return text
    separator = " " if text and not text.endswith(" ") else ""
    expanded = text
    while len(expanded) < required_length:
        expanded = expanded + separator + text
    return expanded


def inspect_config(config_path):
    config = load_config(config_path)
    model = MicroGPT(config)
    parameter_count = model.parameter_count()
    return {
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "estimated_size_mb": round(parameter_count * 4 / (1024 * 1024), 6),
        "architecture": dict(ARCHITECTURE_FLAGS),
    }


def smoke(config_path, text, max_new_tokens, save_checkpoint=None):
    base_config = load_config(config_path)
    training_text = _ensure_batchable_text(text, base_config.block_size)
    tokenizer = _build_tokenizer(training_text)
    config = _config_with_vocab(base_config, tokenizer.vocab_size)
    set_seed(config.seed)
    tokens = torch.tensor(tokenizer.encode(training_text), dtype=torch.long)
    model = MicroGPT(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    started = time.perf_counter()
    x, y = make_lm_batch(
        tokens,
        block_size=config.block_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    output = model(x, labels=y)
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    elapsed = time.perf_counter() - started

    prompt_text = text[: config.block_size]
    prompt = torch.tensor([_encode_known(tokenizer, prompt_text)], dtype=torch.long)
    generated = _generate_with_tokenizer_vocab(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    generated_text = _decode_known(tokenizer, generated[0].tolist())

    checkpoint_saved = False
    if save_checkpoint is not None:
        save_micro_gpt_checkpoint(
            save_checkpoint,
            model,
            config,
            tokenizer,
            metadata={"source": "micro_gpt_cli_smoke", "steps": 1},
        )
        checkpoint_saved = True

    token_count = config.batch_size * config.block_size
    return {
        "dry_run": True,
        "steps": 1,
        "loss": float(output.loss.detach()),
        "perplexity": perplexity(output.loss.detach()),
        "parameter_count": model.parameter_count(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokens_per_second": tokens_per_second(token_count, elapsed),
        "prompt": prompt_text,
        "generated_text": generated_text,
        "checkpoint_saved": checkpoint_saved,
    }


def generate_text(config_path, prompt, max_new_tokens, random_init=False, checkpoint=None, vocab_text=""):
    if not prompt:
        raise SystemExit("generate --prompt must not be empty.")
    if checkpoint is not None:
        payload = load_micro_gpt_checkpoint(checkpoint)
        config = payload["config"]
        tokenizer = payload["tokenizer"]
        model = MicroGPT(config)
        model.load_state_dict(payload["model"])
    elif random_init:
        base_config = load_config(config_path)
        tokenizer = _build_tokenizer(prompt + vocab_text)
        config = _config_with_vocab(base_config, tokenizer.vocab_size)
        set_seed(config.seed)
        model = MicroGPT(config)
    else:
        raise SystemExit("generate requires --checkpoint or --random-init.")

    input_ids = torch.tensor([_encode_known(tokenizer, prompt)], dtype=torch.long)
    generated = _generate_with_tokenizer_vocab(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    return _decode_known(tokenizer, generated[0].tolist())


def build_parser():
    parser = argparse.ArgumentParser(description="Terminal-first micro-GPT research CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect model config and architecture.")
    inspect_parser.add_argument("--config", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run one tiny forward/backward/update step.")
    smoke_parser.add_argument("--config", required=True)
    smoke_parser.add_argument("--text", required=True)
    smoke_parser.add_argument("--max-new-tokens", type=non_negative_int, default=8)
    smoke_parser.add_argument("--save-checkpoint")

    generate_parser = subparsers.add_parser("generate", help="Generate from a checkpoint or random model.")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=non_negative_int, default=8)
    generate_parser.add_argument("--random-init", action="store_true")
    generate_parser.add_argument("--checkpoint")
    generate_parser.add_argument("--vocab-text", default="")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_config(args.config), indent=2, sort_keys=True))
    elif args.command == "smoke":
        print(
            json.dumps(
                smoke(args.config, args.text, args.max_new_tokens, args.save_checkpoint),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate":
        print(
            generate_text(
                args.config,
                args.prompt,
                args.max_new_tokens,
                random_init=args.random_init,
                checkpoint=args.checkpoint,
                vocab_text=args.vocab_text,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
