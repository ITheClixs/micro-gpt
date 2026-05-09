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


def lion_step(
    parameter,
    gradient,
    momentum_buffer,
    lr=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.0,
):
    """Single Lion optimizer update as a small inspectable tensor primitive."""
    beta1, beta2 = betas
    decayed_parameter = parameter * (1.0 - lr * weight_decay)
    update_direction = (beta1 * momentum_buffer + (1.0 - beta1) * gradient).sign()
    updated = decayed_parameter - lr * update_direction
    next_momentum = beta2 * momentum_buffer + (1.0 - beta2) * gradient
    return updated, next_momentum


def orthogonalize_newton_schulz(matrix, steps=5, eps=1e-7):
    """Approximate the polar factor of a 2D update with Newton-Schulz steps.

    Muon-style optimizers use this kind of matrix update normalization for
    hidden-layer weight matrices. The implementation stays explicit so tests
    and visualizations can inspect the orthogonalized update directly.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    transposed = matrix.shape[0] < matrix.shape[1]
    x = matrix.T if transposed else matrix
    original_dtype = x.dtype
    x = x.to(torch.float32)
    x = x / (torch.linalg.matrix_norm(x, ord=2) + eps)

    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * identity - x.T @ x)

    x = x.to(original_dtype)
    return x.T if transposed else x


def muon_step(parameter, gradient, momentum_buffer, lr=0.02, momentum=0.95, steps=5):
    """Single Muon-style matrix update for 2D parameters.

    This is intentionally a primitive rather than a full optimizer class: it
    exposes the momentum buffer and orthogonalized update for research demos.
    """
    if parameter.ndim != 2 or gradient.ndim != 2 or momentum_buffer.ndim != 2:
        raise ValueError("muon_step expects 2D parameter, gradient, and momentum_buffer")
    if parameter.shape != gradient.shape or parameter.shape != momentum_buffer.shape:
        raise ValueError("parameter, gradient, and momentum_buffer must have matching shapes")

    next_momentum = momentum * momentum_buffer + gradient
    update = orthogonalize_newton_schulz(next_momentum, steps=steps)
    return parameter - lr * update, next_momentum, update


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
