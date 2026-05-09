"""Convolutional-network primitives for visual inspection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conv2d_manual(x, weight, bias=None, stride=1, padding=0):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if padding != (0, 0):
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))

    batch, in_channels, height, width = x.shape
    out_channels, kernel_channels, kernel_h, kernel_w = weight.shape
    if in_channels != kernel_channels:
        raise ValueError("Input channel count must match kernel channel count.")

    out_h = (height - kernel_h) // stride[0] + 1
    out_w = (width - kernel_w) // stride[1] + 1
    output = x.new_zeros((batch, out_channels, out_h, out_w))

    for i in range(out_h):
        row = i * stride[0]
        for j in range(out_w):
            col = j * stride[1]
            patch = x[:, :, row:row + kernel_h, col:col + kernel_w]
            output[:, :, i, j] = torch.einsum("bchw,ochw->bo", patch, weight)

    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


def max_pool2d_manual(x, kernel_size=2, stride=None):
    if stride is None:
        stride = kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    batch, channels, height, width = x.shape
    out_h = (height - kernel_size[0]) // stride[0] + 1
    out_w = (width - kernel_size[1]) // stride[1] + 1
    output = x.new_empty((batch, channels, out_h, out_w))

    for i in range(out_h):
        row = i * stride[0]
        for j in range(out_w):
            col = j * stride[1]
            patch = x[:, :, row:row + kernel_size[0], col:col + kernel_size[1]]
            output[:, :, i, j] = patch.amax(dim=(-1, -2))
    return output


def batch_norm_2d(x, eps=1e-5):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    variance = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(variance + eps)


def feature_map_payload(seed=0):
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 1, 8, 8, generator=generator)
    edge_kernel = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    activation = torch.relu(conv2d_manual(image, edge_kernel, padding=1))
    pooled = max_pool2d_manual(activation, kernel_size=2)
    return {
        "image": image.squeeze().tolist(),
        "kernel": edge_kernel.squeeze().tolist(),
        "activation": activation.squeeze().tolist(),
        "pooled": pooled.squeeze().tolist(),
    }
