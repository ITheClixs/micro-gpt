"""Parameter-efficient adapter primitives for LLM research demos."""

from __future__ import annotations

import math

import torch


class LoRALinear(torch.nn.Module):
    """Low-rank adaptation wrapper for a frozen linear projection.

    The base projection is preserved and frozen while trainable A/B matrices
    add a low-rank residual. Initializing B to zero makes the wrapper start as
    an exact copy of the base layer, which is useful for controlled demos.
    """

    def __init__(self, base, rank, alpha=1.0):
        super().__init__()
        if not isinstance(base, torch.nn.Linear):
            raise TypeError("base must be a torch.nn.Linear module")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_a = torch.nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.base(x)
        low_rank_output = (x @ self.lora_a.T) @ self.lora_b.T
        return base_output + self.scaling * low_rank_output


def trainable_parameter_count(module):
    """Count only parameters still marked trainable after adapter wrapping."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
