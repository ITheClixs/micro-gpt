"""Backpropagation primitives with explicit gradients for inspection."""

from __future__ import annotations

import torch


def relu(x):
    return torch.clamp_min(x, 0.0)


def relu_grad(x):
    return (x > 0).to(x.dtype)


def two_layer_mlp_forward(x, w1, b1, w2, b2):
    hidden_pre = x @ w1 + b1
    hidden = relu(hidden_pre)
    prediction = hidden @ w2 + b2
    cache = {
        "x": x,
        "hidden_pre": hidden_pre,
        "hidden": hidden,
        "prediction": prediction,
    }
    return prediction, cache


def mse_loss(prediction, target):
    error = prediction - target
    return (error.pow(2)).mean()


def two_layer_mlp_backward(cache, target, w2):
    x = cache["x"]
    hidden_pre = cache["hidden_pre"]
    hidden = cache["hidden"]
    prediction = cache["prediction"]
    batch_size, output_size = target.shape

    d_prediction = 2.0 * (prediction - target) / (batch_size * output_size)
    d_w2 = hidden.T @ d_prediction
    d_b2 = d_prediction.sum(dim=0)
    d_hidden = d_prediction @ w2.T
    d_hidden_pre = d_hidden * relu_grad(hidden_pre)
    d_w1 = x.T @ d_hidden_pre
    d_b1 = d_hidden_pre.sum(dim=0)
    return {"w1": d_w1, "b1": d_b1, "w2": d_w2, "b2": d_b2}


def gradient_check(seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, generator=generator)
    target = torch.randn(4, 2, generator=generator)
    w1 = torch.randn(3, 5, generator=generator) * 0.2
    b1 = torch.randn(5, generator=generator) * 0.1
    w2 = torch.randn(5, 2, generator=generator) * 0.2
    b2 = torch.randn(2, generator=generator) * 0.1

    prediction, cache = two_layer_mlp_forward(x, w1, b1, w2, b2)
    manual_grads = two_layer_mlp_backward(cache, target, w2)

    autograd_params = {
        "w1": w1.detach().clone().requires_grad_(True),
        "b1": b1.detach().clone().requires_grad_(True),
        "w2": w2.detach().clone().requires_grad_(True),
        "b2": b2.detach().clone().requires_grad_(True),
    }
    autograd_prediction, _ = two_layer_mlp_forward(
        x,
        autograd_params["w1"],
        autograd_params["b1"],
        autograd_params["w2"],
        autograd_params["b2"],
    )
    loss = mse_loss(autograd_prediction, target)
    loss.backward()

    max_abs_error = max(
        (manual_grads[name] - autograd_params[name].grad).abs().max().item()
        for name in manual_grads
    )
    return {
        "loss": float(loss.detach()),
        "max_abs_error": max_abs_error,
        "layer_count": 2,
        "gradient_norms": {
            name: float(value.norm())
            for name, value in manual_grads.items()
        },
    }


def demo_loss_curve(steps=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 3, generator=generator)
    target = torch.sin(x[:, :1]) + 0.25 * x[:, 1:2]
    w1 = torch.randn(3, 8, generator=generator) * 0.2
    b1 = torch.zeros(8)
    w2 = torch.randn(8, 1, generator=generator) * 0.2
    b2 = torch.zeros(1)
    losses = []
    gradient_norms = []

    for _ in range(steps):
        prediction, cache = two_layer_mlp_forward(x, w1, b1, w2, b2)
        losses.append(float(mse_loss(prediction, target)))
        grads = two_layer_mlp_backward(cache, target, w2)
        gradient_norms.append(float(grads["w1"].norm() + grads["w2"].norm()))
        w1 = w1 - 0.1 * grads["w1"]
        b1 = b1 - 0.1 * grads["b1"]
        w2 = w2 - 0.1 * grads["w2"]
        b2 = b2 - 0.1 * grads["b2"]

    return {"loss_curve": losses, "gradient_norms": gradient_norms}


if __name__ == "__main__":
    print(gradient_check())
