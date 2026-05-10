"""TLOB-Q model: dual-attention transformer with multi-horizon multi-task heads."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from .tlob_modules import (
    BilinearNormalization,
    MCDropout,
    MLPLOBFeatMix,
    MLPLOBTempMix,
    RMSNorm,
    SpatialAttention,
    TemporalCausalAttention,
)


@dataclass(frozen=True)
class TLOBQConfig:
    feature_keys: tuple[str, ...]
    sequence_length: int = 128
    d_model: int = 320
    n_heads: int = 10
    n_layers: int = 8
    ffn_expansion: int = 4
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 5, 20, 100)
    head_volatility: bool = True
    head_spread: bool = True
    seed: int = 2026
    alpha_dir: float = 1.0
    alpha_vol: float = 0.1
    alpha_spread: float = 0.05
    kappa: float = 1.0
    no_trade_threshold: float = 0.05
    mc_samples: int = 16
    ema_decay: float = 0.999

    def to_dict(self):
        payload = asdict(self)
        payload["feature_keys"] = list(self.feature_keys)
        payload["horizons"] = list(self.horizons)
        return payload

    @classmethod
    def from_dict(cls, payload):
        kwargs = dict(payload)
        kwargs["feature_keys"] = tuple(kwargs["feature_keys"])
        kwargs["horizons"] = tuple(int(h) for h in kwargs["horizons"])
        return cls(**kwargs)


class TLOBQBlock(nn.Module):
    def __init__(self, config: TLOBQConfig):
        super().__init__()
        self.bin = BilinearNormalization(
            num_features=config.d_model, num_timesteps=config.sequence_length
        )
        self.spatial_attn = SpatialAttention(d_model=config.d_model, n_heads=config.n_heads)
        self.spatial_norm = RMSNorm(config.d_model)
        self.spatial_dropout = MCDropout(p=config.dropout)
        self.feat_mix = MLPLOBFeatMix(d_model=config.d_model, expansion=config.ffn_expansion)
        self.temporal_attn = TemporalCausalAttention(d_model=config.d_model, n_heads=config.n_heads)
        self.temporal_norm = RMSNorm(config.d_model)
        self.temporal_dropout = MCDropout(p=config.dropout)
        self.temp_mix = MLPLOBTempMix(
            sequence_length=config.sequence_length, expansion=config.ffn_expansion
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bin(x)
        x = x + self.spatial_dropout(self.spatial_attn(self.spatial_norm(x)))
        x = self.feat_mix(x)
        x = x + self.temporal_dropout(self.temporal_attn(self.temporal_norm(x)))
        x = self.temp_mix(x)
        return x


class TLOBQModel(nn.Module):
    def __init__(self, config: TLOBQConfig):
        super().__init__()
        if config.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        self.config = config
        self.feature_proj = nn.Linear(len(config.feature_keys), config.d_model)
        self.blocks = nn.ModuleList([TLOBQBlock(config) for _ in range(config.n_layers)])
        self.trunk_norm = RMSNorm(config.d_model)
        self.direction_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 3) for h in config.horizons}
        )
        self.vol_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 1) for h in config.horizons}
        )
        self.spread_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 1) for h in config.horizons}
        )

    def set_mc_active(self, active: bool) -> None:
        for module in self.modules():
            if isinstance(module, MCDropout):
                module.set_mc_active(active)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError("TLOBQModel expects input (B, T, F).")
        z = self.trunk(x)
        outputs: dict[int, dict[str, torch.Tensor]] = {}
        for horizon in self.config.horizons:
            key = str(horizon)
            outputs[horizon] = {
                "direction": self.direction_heads[key](z),
                "future_vol": self.vol_heads[key](z),
                "future_spread": self.spread_heads[key](z),
            }
        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EMAWeightTracker:
    def __init__(self, model: TLOBQModel, decay: float = 0.999):
        self.decay = float(decay)
        self._shadow = {name: param.detach().clone() for name, param in model.state_dict().items()}

    def update(self, model: TLOBQModel) -> None:
        for name, param in model.state_dict().items():
            if name not in self._shadow:
                self._shadow[name] = param.detach().clone()
                continue
            self._shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def shadow_state(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._shadow.items()}

    def load_into(self, model: TLOBQModel) -> None:
        model.load_state_dict(self._shadow)


def save_tlob_q_artifact(
    path,
    model: TLOBQModel,
    ema: "EMAWeightTracker | None" = None,
    config: "TLOBQConfig | None" = None,
    standardizer: dict | None = None,
    metrics: dict | None = None,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow_state() if ema is not None else None,
        "config": (config or model.config).to_dict(),
        "standardizer": standardizer or {},
        "metrics": metrics or {},
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_tlob_q_artifact(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = TLOBQConfig.from_dict(payload["config"])
    return {
        "state_dict": payload["state_dict"],
        "ema_state_dict": payload.get("ema_state_dict"),
        "config": config,
        "standardizer": payload.get("standardizer", {}),
        "metrics": payload.get("metrics", {}),
        "meta": payload.get("meta", {}),
    }
