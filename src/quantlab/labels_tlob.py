"""TLOB-style decoupled (h, k) labelling and multi-horizon label generation."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .core import MarketEvent, MultiHorizonLabelFrame
from .features import midprice, realized_volatility, spread


def tlob_smoothed_label(
    prices: Sequence[float],
    t: int,
    horizon: int,
    k_smooth: int,
    theta: float,
) -> int:
    """Compute the TLOB ternary trend label using decoupled (h, k).

    l(t, h, k) = (w_plus - w_minus) / w_minus
    where w_plus is the mean of p(t+h-i) for i in 0..k
    and w_minus is the mean of p(t-i) for i in 0..k.

    Returns:
        1  if relative change > +theta
        -1 if relative change < -theta
        0  otherwise (stationary or insufficient data)
    """
    prices = [float(value) for value in prices]
    n = len(prices)
    k_smooth = max(0, int(k_smooth))
    horizon = int(horizon)

    if t < k_smooth or t + horizon >= n:
        return 0

    forward = [
        prices[t + horizon - i]
        for i in range(k_smooth + 1)
        if 0 <= t + horizon - i < n
    ]
    backward = [
        prices[t - i]
        for i in range(k_smooth + 1)
        if 0 <= t - i < n
    ]

    if not forward or not backward:
        return 0

    w_plus = sum(forward) / len(forward)
    w_minus = sum(backward) / len(backward)

    if abs(w_minus) < 1e-12:
        return 0

    rel = (w_plus - w_minus) / w_minus
    theta = abs(float(theta))

    if rel > theta:
        return 1
    if rel < -theta:
        return -1
    return 0


def _resolve_theta(theta, spread_value: float, mid_value: float) -> float:
    """Resolve theta to a numeric value.

    If theta is the string "spread", returns spread/mid as the threshold.
    Otherwise casts to float directly.
    """
    if isinstance(theta, str) and theta.lower() == "spread":
        if mid_value <= 0:
            return 0.0
        return float(spread_value) / float(mid_value)
    return float(theta)


def build_multihorizon_labels(
    events: Iterable[MarketEvent],
    horizons: Iterable[int],
    k_smooth: Callable[[int], int] | int = lambda h: max(1, h // 2),
    theta: float | str = "spread",
) -> list[MultiHorizonLabelFrame]:
    """Build multi-horizon TLOB labels for a sequence of market events.

    For each event at index t, computes direction, future_return, future_vol,
    and future_spread for every horizon h in `horizons`.

    Args:
        events: Ordered sequence of MarketEvent objects.
        horizons: Tuple/list of integer look-ahead horizons.
        k_smooth: Either a callable mapping horizon -> k, or a fixed int k.
        theta: Threshold for ternary labelling. Use "spread" to auto-scale
               per-event by spread/mid ratio.

    Returns:
        List of MultiHorizonLabelFrame (one per event that has data for at
        least one horizon). Returns [] when the event sequence is too short.

    Raises:
        ValueError: If horizons is empty.
    """
    events = list(events)
    horizons = tuple(int(h) for h in horizons)

    if not horizons:
        raise ValueError("horizons must not be empty.")

    if len(events) < 2 + max(horizons):
        return []

    prices = [midprice(event.bid_price, event.ask_price) for event in events]
    spreads = [spread(event.bid_price, event.ask_price) for event in events]

    if callable(k_smooth):
        smooth_for = k_smooth
    else:
        smooth_const = int(k_smooth)

        def smooth_for(_h: int, _v: int = smooth_const) -> int:
            return _v

    rows: list[MultiHorizonLabelFrame] = []
    for t in range(len(events) - 1):
        horizon_payload: dict[int, dict[str, float]] = {}

        for h in horizons:
            if t + h >= len(events):
                continue

            k = max(1, int(smooth_for(h)))
            theta_value = _resolve_theta(theta, spreads[t], prices[t])
            direction = tlob_smoothed_label(
                prices, t=t, horizon=h, k_smooth=k, theta=theta_value
            )

            future_return = (
                (prices[t + h] - prices[t]) / prices[t] if prices[t] > 0 else 0.0
            )

            window_end = min(len(prices), t + h + 1)
            window_prices = prices[t:window_end]
            future_vol = realized_volatility(window_prices)
            future_spread = sum(spreads[t:window_end]) / max(
                1, len(spreads[t:window_end])
            )

            horizon_payload[h] = {
                "direction": float(direction),
                "future_return": float(future_return),
                "future_vol": float(future_vol),
                "future_spread": float(future_spread),
            }

        if not horizon_payload:
            continue

        rows.append(
            MultiHorizonLabelFrame(
                timestamp_ms=events[t].timestamp_ms,
                symbol=events[t].symbol,
                horizons=horizon_payload,
            )
        )

    return rows
