"""Metrics for micro-GPT dry runs and experiments."""

from __future__ import annotations

import math


def perplexity(loss):
    return float(math.exp(min(float(loss), 20.0)))


def tokens_per_second(token_count, seconds):
    if seconds <= 0:
        return 0.0
    return float(token_count / seconds)
