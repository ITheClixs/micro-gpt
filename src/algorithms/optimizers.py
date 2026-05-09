"""Optimizer update rules written as inspectable tensor functions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdamWState:
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @classmethod
    def zeros_like(cls, parameter):
        return cls(
            step=0,
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )


def sgd_step(parameter, gradient, lr):
    return parameter - lr * gradient


def momentum_step(parameter, gradient, velocity, lr, momentum=0.9):
    velocity = momentum * velocity + gradient
    return parameter - lr * velocity, velocity


def rmsprop_step(parameter, gradient, square_avg, lr, alpha=0.99, eps=1e-8):
    square_avg = alpha * square_avg + (1.0 - alpha) * gradient.pow(2)
    return parameter - lr * gradient / (square_avg.sqrt() + eps), square_avg


def adamw_step(
    parameter,
    gradient,
    state,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
):
    beta1, beta2 = betas
    step = state.step + 1
    exp_avg = beta1 * state.exp_avg + (1.0 - beta1) * gradient
    exp_avg_sq = beta2 * state.exp_avg_sq + (1.0 - beta2) * gradient.pow(2)
    bias_corrected_avg = exp_avg / (1.0 - beta1 ** step)
    bias_corrected_sq = exp_avg_sq / (1.0 - beta2 ** step)
    decayed_parameter = parameter * (1.0 - lr * weight_decay)
    updated = decayed_parameter - lr * bias_corrected_avg / (bias_corrected_sq.sqrt() + eps)
    return updated, AdamWState(step=step, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)


def optimizer_payload():
    parameter = torch.linspace(-1, 1, 21)
    loss_surface = parameter.pow(2)
    gradient = 2 * parameter
    adam_state = AdamWState.zeros_like(parameter)
    updated, _ = adamw_step(parameter, gradient, adam_state, lr=0.05)
    return {
        "parameter": parameter.tolist(),
        "loss_surface": loss_surface.tolist(),
        "gradient": gradient.tolist(),
        "adamw_update": updated.tolist(),
    }
