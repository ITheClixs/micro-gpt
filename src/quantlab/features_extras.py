"""Extended feature channels (32 total: 8 base + 24 derived) for TLOB-Q."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .core import MarketEvent
from .features import (
    book_imbalance,
    depth_imbalance,
    microprice,
    midprice,
    order_flow_imbalance,
    realized_volatility,
    signed_volume,
    spread,
)


BASE_FEATURE_KEYS: tuple[str, ...] = (
    "midprice_logret",
    "spread",
    "microprice_minus_mid",
    "ofi_level_1",
    "book_imbalance",
    "depth_imbalance",
    "signed_volume",
    "realized_volatility",
)

EXTRA_FEATURE_KEYS: tuple[str, ...] = (
    "ofi_level_2",
    "ofi_level_3",
    "ofi_level_5",
    "log_spread",
    "funding_rate",
    "liquidation_intensity",
    "signed_volume_ema_slow",
    "signed_volume_ema_fast",
    "directional_run_length",
    "trade_burst_rate",
    "queue_depth_l2_over_l1",
    "queue_depth_l3_over_l1",
    "cumulative_depth_near_mid",
    "relative_tick_imbalance",
    "midret_autocorr_lag1",
    "realized_vol_ratio",
    "vpin_proxy",
    "signed_volume_zscore",
    "ofi_zscore",
    "microprice_momentum",
    "depth_imbalance_ema_fast",
    "depth_imbalance_ema_slow",
    "spread_regime",
    "last_trade_aggressor_persistence",
)

EXTENDED_FEATURE_KEYS: tuple[str, ...] = BASE_FEATURE_KEYS + EXTRA_FEATURE_KEYS

_EPS = 1e-12
_CLIP = 50.0


def _clip(value: float) -> float:
    if value != value:
        return 0.0
    if value > _CLIP:
        return _CLIP
    if value < -_CLIP:
        return -_CLIP
    return value


def _ema_update(prev: float, sample: float, alpha: float) -> float:
    return alpha * sample + (1.0 - alpha) * prev


def _depths(event: MarketEvent, side: str) -> list[float]:
    key = f"{side}_depth"
    raw = event.extras.get(key)
    if raw is None:
        return [float(event.bid_size if side == "bid" else event.ask_size)]
    if isinstance(raw, (list, tuple)):
        return [float(value) for value in raw]
    return [float(raw)]


def _level_ofi(previous: MarketEvent, current: MarketEvent, level: int) -> float:
    prev_bid = _depths(previous, "bid")
    prev_ask = _depths(previous, "ask")
    curr_bid = _depths(current, "bid")
    curr_ask = _depths(current, "ask")
    bid_levels = min(level, len(prev_bid), len(curr_bid))
    ask_levels = min(level, len(prev_ask), len(curr_ask))
    bid_delta = sum(curr_bid[:bid_levels]) - sum(prev_bid[:bid_levels])
    ask_delta = sum(curr_ask[:ask_levels]) - sum(prev_ask[:ask_levels])
    return float(bid_delta - ask_delta)


@dataclass
class _State:
    sv_ema_slow: float = 0.0
    sv_ema_fast: float = 0.0
    di_ema_slow: float = 0.0
    di_ema_fast: float = 0.0
    sv_history: deque = None
    ofi_history: deque = None
    midret_history: deque = None
    spread_history: deque = None
    short_vol_window: deque = None
    long_vol_window: deque = None
    tick_signs: deque = None
    aggressor_run_length: float = 0.0
    last_aggressor_side: str | None = None
    directional_run_length: float = 0.0
    last_direction_sign: int = 0
    last_microprice: float | None = None
    last_timestamp_ms: int | None = None

    def __post_init__(self):
        self.sv_history = deque(maxlen=64)
        self.ofi_history = deque(maxlen=64)
        self.midret_history = deque(maxlen=32)
        self.spread_history = deque(maxlen=64)
        self.short_vol_window = deque(maxlen=8)
        self.long_vol_window = deque(maxlen=32)
        self.tick_signs = deque(maxlen=32)


def _zscore(history: deque, value: float) -> float:
    if len(history) < 4:
        return 0.0
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / max(1, len(history) - 1)
    std = math.sqrt(variance)
    if std < _EPS:
        return 0.0
    return _clip((value - mean) / std)


def _autocorr_lag1(values: deque) -> float:
    if len(values) < 3:
        return 0.0
    seq = list(values)
    mean = sum(seq) / len(seq)
    num = 0.0
    den = 0.0
    for prev, curr in zip(seq[:-1], seq[1:]):
        num += (prev - mean) * (curr - mean)
    for value in seq:
        den += (value - mean) ** 2
    if den < _EPS:
        return 0.0
    return _clip(num / den)


def _build_row(previous: MarketEvent, current: MarketEvent, state: _State) -> dict:
    prev_mid = midprice(previous.bid_price, previous.ask_price)
    curr_mid = midprice(current.bid_price, current.ask_price)
    curr_spread = spread(current.bid_price, current.ask_price)
    curr_micro = microprice(
        current.bid_price, current.ask_price, current.bid_size, current.ask_size
    )
    sv = signed_volume(current.last_size or 0.0, current.side)
    ofi_l1 = order_flow_imbalance(previous, current)
    log_ret = math.log(curr_mid / prev_mid) if prev_mid > 0 and curr_mid > 0 else 0.0

    state.sv_ema_slow = _ema_update(state.sv_ema_slow, sv, 0.1)
    state.sv_ema_fast = _ema_update(state.sv_ema_fast, sv, 0.5)
    bi = book_imbalance(current.bid_size, current.ask_size)
    state.di_ema_slow = _ema_update(state.di_ema_slow, bi, 0.1)
    state.di_ema_fast = _ema_update(state.di_ema_fast, bi, 0.5)

    state.sv_history.append(sv)
    state.ofi_history.append(ofi_l1)
    state.midret_history.append(log_ret)
    state.spread_history.append(curr_spread)
    state.short_vol_window.append(curr_mid)
    state.long_vol_window.append(curr_mid)

    side_sign = 1 if sv > 0 else (-1 if sv < 0 else 0)
    if side_sign != 0:
        state.tick_signs.append(side_sign)

    # Directional run-length: tracks consecutive same-sign signed-volume ticks.
    if side_sign != 0:
        if side_sign == state.last_direction_sign:
            state.directional_run_length += 1.0
        else:
            state.directional_run_length = 1.0
        state.last_direction_sign = side_sign
    # Zero-volume tick neither extends nor resets the directional run.

    # Aggressor run-length: tracks consecutive same-side explicit aggressor string.
    side = (current.side or "").lower()
    if side and side == (state.last_aggressor_side or ""):
        state.aggressor_run_length += 1.0
    else:
        state.aggressor_run_length = 1.0
    state.last_aggressor_side = side

    burst_dt = 1.0
    if state.last_timestamp_ms is not None:
        burst_dt = max(1.0, (current.timestamp_ms - state.last_timestamp_ms) / 1000.0)
    state.last_timestamp_ms = current.timestamp_ms

    bid_depth = _depths(current, "bid")
    ask_depth = _depths(current, "ask")
    l1 = (bid_depth[0] + ask_depth[0]) / 2.0 if bid_depth and ask_depth else 0.0
    l2 = (
        (bid_depth[1] + ask_depth[1]) / 2.0
        if len(bid_depth) > 1 and len(ask_depth) > 1
        else l1
    )
    l3 = (
        (bid_depth[2] + ask_depth[2]) / 2.0
        if len(bid_depth) > 2 and len(ask_depth) > 2
        else l1
    )

    short_vol = realized_volatility(list(state.short_vol_window))
    long_vol = realized_volatility(list(state.long_vol_window))
    realized_vol_ratio = short_vol / long_vol if long_vol > _EPS else 0.0

    median_spread = (
        sorted(state.spread_history)[len(state.spread_history) // 2]
        if state.spread_history
        else curr_spread
    )
    spread_regime = curr_spread / median_spread if median_spread > _EPS else 1.0

    up_ticks = sum(1 for s in state.tick_signs if s > 0)
    down_ticks = sum(1 for s in state.tick_signs if s < 0)
    if down_ticks > 0:
        rel_tick = math.log((up_ticks + 1) / (down_ticks + 1))
    else:
        rel_tick = float(up_ticks)

    micro_momentum = 0.0
    if state.last_microprice is not None and state.last_microprice > 0:
        micro_momentum = (curr_micro - state.last_microprice) / state.last_microprice
    state.last_microprice = curr_micro

    total_signed = sum(abs(value) for value in state.sv_history) or _EPS
    vpin_proxy = sum(state.sv_history) / total_signed

    cum_depth_near_mid = (sum(bid_depth[:3]) + sum(ask_depth[:3])) / max(
        1, min(3, len(bid_depth)) + min(3, len(ask_depth))
    )

    return {
        "timestamp_ms": int(current.timestamp_ms),
        "symbol": str(current.symbol),
        "midprice": float(curr_mid),
        "midprice_logret": _clip(log_ret),
        "spread": _clip(curr_spread),
        "microprice_minus_mid": _clip(curr_micro - curr_mid),
        "ofi_level_1": _clip(ofi_l1),
        "book_imbalance": _clip(bi),
        "depth_imbalance": _clip(depth_imbalance(bid_depth, ask_depth)),
        "signed_volume": _clip(sv),
        "realized_volatility": _clip(realized_volatility(list(state.short_vol_window))),
        "ofi_level_2": _clip(_level_ofi(previous, current, 2)),
        "ofi_level_3": _clip(_level_ofi(previous, current, 3)),
        "ofi_level_5": _clip(_level_ofi(previous, current, 5)),
        "log_spread": _clip(math.log(max(_EPS, curr_spread))),
        "funding_rate": _clip(float(current.extras.get("funding", 0.0))),
        "liquidation_intensity": _clip(
            float(current.extras.get("liquidation_intensity", 0.0))
        ),
        "signed_volume_ema_slow": _clip(state.sv_ema_slow),
        "signed_volume_ema_fast": _clip(state.sv_ema_fast),
        "directional_run_length": _clip(state.directional_run_length),
        "trade_burst_rate": _clip(1.0 / burst_dt),
        "queue_depth_l2_over_l1": _clip(l2 / l1) if l1 > _EPS else 0.0,
        "queue_depth_l3_over_l1": _clip(l3 / l1) if l1 > _EPS else 0.0,
        "cumulative_depth_near_mid": _clip(cum_depth_near_mid),
        "relative_tick_imbalance": _clip(rel_tick),
        "midret_autocorr_lag1": _clip(_autocorr_lag1(state.midret_history)),
        "realized_vol_ratio": _clip(realized_vol_ratio),
        "vpin_proxy": _clip(vpin_proxy),
        "signed_volume_zscore": _zscore(state.sv_history, sv),
        "ofi_zscore": _zscore(state.ofi_history, ofi_l1),
        "microprice_momentum": _clip(micro_momentum),
        "depth_imbalance_ema_fast": _clip(state.di_ema_fast),
        "depth_imbalance_ema_slow": _clip(state.di_ema_slow),
        "spread_regime": _clip(spread_regime),
        "last_trade_aggressor_persistence": _clip(state.aggressor_run_length),
    }


def build_extended_feature_rows(events: Iterable[MarketEvent]) -> list[dict]:
    """Build extended feature rows from a sequence of MarketEvent instances.

    Returns one row per consecutive event pair (len(events) - 1 rows total).
    Each row contains all 32 keys defined in EXTENDED_FEATURE_KEYS plus
    metadata keys (timestamp_ms, symbol, midprice).
    """
    events = list(events)
    if len(events) < 2:
        return []
    state = _State()
    rows: list[dict] = []
    previous = events[0]
    for current in events[1:]:
        rows.append(_build_row(previous, current, state))
        previous = current
    return rows
