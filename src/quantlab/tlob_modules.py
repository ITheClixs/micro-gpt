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


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.causal = bool(causal)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("attention expects (B, L, d_model).")
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.full(
                (length, length), float("-inf"), device=x.device, dtype=attn.dtype
            ).triu_(1)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(attn, dim=-1)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout, training=True)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(b, length, self.d_model)
        return self.out_proj(output)


class SpatialAttention(_MultiHeadAttention):
    """Multi-head attention across the feature axis (non-causal)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, n_heads=n_heads, causal=False, dropout=dropout)


class TemporalCausalAttention(_MultiHeadAttention):
    """Causal multi-head attention across the time axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, n_heads=n_heads, causal=True, dropout=dropout)


class MLPLOBFeatMix(nn.Module):
    """MLP-Mixer style feature-axis MLP applied row-wise (per timestep)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model) * int(expansion)
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc2(x)
        return residual + x


class MLPLOBTempMix(nn.Module):
    """MLP-Mixer style temporal-axis MLP applied column-wise (per feature channel)."""

    def __init__(self, sequence_length: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(sequence_length) * int(expansion)
        self.norm = RMSNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, hidden)
        self.fc2 = nn.Linear(hidden, sequence_length)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        x_t = self.norm(x_t)
        x_t = F.gelu(self.fc1(x_t))
        if self.dropout > 0.0 and self.training:
            x_t = F.dropout(x_t, p=self.dropout, training=True)
        x_t = self.fc2(x_t)
        return residual + x_t.transpose(1, 2)
