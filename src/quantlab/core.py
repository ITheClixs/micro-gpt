"""Canonical market event, feature, and label schemas."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class MarketEvent:
    timestamp_ms: int
    symbol: str
    event_type: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float | None = None
    last_size: float | None = None
    side: str | None = None
    source: str = "public"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        payload = asdict(self)
        payload.update(payload.pop("extras"))
        return payload


@dataclass(frozen=True)
class FeatureFrame:
    timestamp_ms: int
    symbol: str
    midprice: float
    spread: float
    microprice: float
    order_flow_imbalance: float
    book_imbalance: float
    depth_imbalance: float
    signed_volume: float
    realized_volatility: float
    funding: float = 0.0
    liquidation_intensity: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        payload = asdict(self)
        payload.update(payload.pop("extras"))
        return payload


@dataclass(frozen=True)
class LabelFrame:
    timestamp_ms: int
    symbol: str
    horizon_ms: int
    midprice_direction: int
    future_return: float
    realized_volatility: float
    triple_barrier: int
    action: str = "hold"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        payload = asdict(self)
        payload.update(payload.pop("extras"))
        return payload

