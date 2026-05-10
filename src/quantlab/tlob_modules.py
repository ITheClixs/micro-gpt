"""TLOB-Q building blocks: normalization, dropout, attention, MLP mixers.

Phase 1 (Task 5): BilinearNormalization, RMSNorm, MCDropout.
Phase 2 (Task 6): Multi-head attention modules appended here.
Phase 3 (Task 7): MLP-mixer modules appended here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class BilinearNormalization(nn.Module):
    """Per-feature + per-timestep affine normalization with running statistics.

    Approximates the BiN layer described in TLOB / BINCTABL: applies a learned
    rescale/shift along each axis. Uses exponential moving averages of the
    batch mean and variance for inference stability.
    """

    def __init__(self, num_features: int, num_timesteps: int, momentum: float = 0.05):
        super().__init__()
        self.num_features = int(num_features)
        self.num_timesteps = int(num_timesteps)
        self.momentum = float(momentum)
        self.feature_gain = nn.Parameter(torch.ones(num_features))
        self.feature_bias = nn.Parameter(torch.zeros(num_features))
        self.temporal_gain = nn.Parameter(torch.ones(num_timesteps))
        self.temporal_bias = nn.Parameter(torch.zeros(num_timesteps))
        self.register_buffer("running_feat_mean", torch.zeros(num_features))
        self.register_buffer("running_feat_var", torch.ones(num_features))
        self.register_buffer("running_temp_mean", torch.zeros(num_timesteps))
        self.register_buffer("running_temp_var", torch.ones(num_timesteps))

    def _update(self, buffer_mean: torch.Tensor, buffer_var: torch.Tensor,
                batch_mean: torch.Tensor, batch_var: torch.Tensor) -> None:
        if not self.training:
            return
        buffer_mean.mul_(1 - self.momentum).add_(batch_mean.detach() * self.momentum)
        buffer_var.mul_(1 - self.momentum).add_(batch_var.detach() * self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("BiN expects (B, T, F).")
        feat_mean = x.mean(dim=(0, 1))
        feat_var = x.var(dim=(0, 1), unbiased=False) + 1e-6
        self._update(self.running_feat_mean, self.running_feat_var, feat_mean, feat_var)
        if self.training:
            mean = feat_mean
            var = feat_var
        else:
            mean = self.running_feat_mean
            var = self.running_feat_var
        x = (x - mean) / var.sqrt()
        x = x * self.feature_gain + self.feature_bias
        temp_mean = x.mean(dim=(0, 2))
        temp_var = x.var(dim=(0, 2), unbiased=False) + 1e-6
        self._update(self.running_temp_mean, self.running_temp_var, temp_mean, temp_var)
        if self.training:
            tm, tv = temp_mean, temp_var
        else:
            tm, tv = self.running_temp_mean, self.running_temp_var
        x = (x - tm.unsqueeze(-1)) / tv.sqrt().unsqueeze(-1)
        x = x * self.temporal_gain.unsqueeze(-1) + self.temporal_bias.unsqueeze(-1)
        return x


class MCDropout(nn.Module):
    """Dropout that stays active in eval mode when mc_active=True."""

    def __init__(self, p: float = 0.1, mc_active: bool = True):
        super().__init__()
        self.p = float(p)
        self.mc_active = bool(mc_active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.mc_active:
            return F.dropout(x, p=self.p, training=True)
        return x

    def set_mc_active(self, active: bool) -> None:
        self.mc_active = bool(active)
