"""Deterministic market microstructure feature formulas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .core import FeatureFrame, MarketEvent
from .io import parse_float, parse_int, read_table, write_jsonl


def midprice(bid_price, ask_price):
    return (float(bid_price) + float(ask_price)) / 2.0


def spread(bid_price, ask_price):
    return float(ask_price) - float(bid_price)


def microprice(bid_price, ask_price, bid_size, ask_size):
    total_size = float(bid_size) + float(ask_size)
    if total_size <= 0:
        return midprice(bid_price, ask_price)
    return (float(ask_price) * float(bid_size) + float(bid_price) * float(ask_size)) / total_size


def book_imbalance(bid_size, ask_size):
    total_size = float(bid_size) + float(ask_size)
    if total_size <= 0:
        return 0.0
    return (float(bid_size) - float(ask_size)) / total_size


def depth_imbalance(bid_depth, ask_depth):
    bid_sum = sum(float(value) for value in bid_depth)
    ask_sum = sum(float(value) for value in ask_depth)
    return book_imbalance(bid_sum, ask_sum)


def signed_volume(size, side):
    if side is None:
        return float(size)
    side = side.lower()
    if side in {"buy", "bid", "b"}:
        return float(size)
    if side in {"sell", "ask", "s"}:
        return -float(size)
    return float(size)


def order_flow_imbalance(previous: MarketEvent, current: MarketEvent):
    bid_delta = _level_change(
        previous_price=previous.bid_price,
        current_price=current.bid_price,
        previous_size=previous.bid_size,
        current_size=current.bid_size,
        higher_is_better=True,
    )
    ask_delta = _level_change(
        previous_price=previous.ask_price,
        current_price=current.ask_price,
        previous_size=previous.ask_size,
        current_size=current.ask_size,
        higher_is_better=False,
    )
    return bid_delta - ask_delta


def _level_change(previous_price, current_price, previous_size, current_size, higher_is_better):
    if current_price > previous_price and higher_is_better:
        return float(current_size)
    if current_price < previous_price and higher_is_better:
        return -float(previous_size)
    if current_price < previous_price and not higher_is_better:
        return float(current_size)
    if current_price > previous_price and not higher_is_better:
        return -float(previous_size)
    return float(current_size) - float(previous_size)


def realized_volatility(prices):
    prices = [float(price) for price in prices]
    if len(prices) < 2:
        return 0.0
    log_returns = []
    for prev_price, curr_price in zip(prices, prices[1:]):
        if prev_price <= 0 or curr_price <= 0:
            continue
        log_returns.append(math.log(curr_price / prev_price))
    return math.sqrt(sum(ret * ret for ret in log_returns))


def feature_frame_from_events(previous: MarketEvent, current: MarketEvent, recent_prices=None):
    recent_prices = list(recent_prices or [previous.midprice if hasattr(previous, "midprice") else midprice(previous.bid_price, previous.ask_price), midprice(current.bid_price, current.ask_price)])
    current_mid = midprice(current.bid_price, current.ask_price)
    return FeatureFrame(
        timestamp_ms=current.timestamp_ms,
        symbol=current.symbol,
        midprice=current_mid,
        spread=spread(current.bid_price, current.ask_price),
        microprice=microprice(current.bid_price, current.ask_price, current.bid_size, current.ask_size),
        order_flow_imbalance=order_flow_imbalance(previous, current),
        book_imbalance=book_imbalance(current.bid_size, current.ask_size),
        depth_imbalance=depth_imbalance(
            current.extras.get("bid_depth", [current.bid_size]),
            current.extras.get("ask_depth", [current.ask_size]),
        ),
        signed_volume=signed_volume(current.last_size or 0.0, current.side),
        realized_volatility=realized_volatility(recent_prices),
        funding=parse_float(current.extras.get("funding", 0.0)),
        liquidation_intensity=parse_float(current.extras.get("liquidation_intensity", 0.0)),
    )


def parse_market_event(row):
    core_keys = {
        "timestamp_ms",
        "symbol",
        "event_type",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "last_price",
        "last_size",
        "side",
        "source",
    }
    extras = {key: value for key, value in row.items() if key not in core_keys}
    if "bid_depth" in extras:
        extras["bid_depth"] = _parse_depth(extras["bid_depth"], row["bid_size"])
    if "ask_depth" in extras:
        extras["ask_depth"] = _parse_depth(extras["ask_depth"], row["ask_size"])
    return MarketEvent(
        timestamp_ms=parse_int(row["timestamp_ms"]),
        symbol=str(row["symbol"]),
        event_type=str(row.get("event_type", "book")),
        bid_price=parse_float(row["bid_price"]),
        ask_price=parse_float(row["ask_price"]),
        bid_size=parse_float(row["bid_size"]),
        ask_size=parse_float(row["ask_size"]),
        last_price=parse_float(row.get("last_price"), default=0.0) if row.get("last_price") not in {None, ""} else None,
        last_size=parse_float(row.get("last_size"), default=0.0) if row.get("last_size") not in {None, ""} else None,
        side=row.get("side") or None,
        source=str(row.get("source", "public")),
        extras=extras,
    )


def _parse_depth(value, fallback):
    if value is None or value == "":
        return [parse_float(fallback)]
    if isinstance(value, (list, tuple)):
        return [parse_float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            return [parse_float(item) for item in json.loads(stripped)]
        return [parse_float(item.strip()) for item in stripped.split(",") if item.strip()]
    return [parse_float(value)]


def build_feature_rows(events):
    events = list(events)
    if len(events) < 2:
        return []
    rows = []
    recent_prices = [midprice(events[0].bid_price, events[0].ask_price)]
    previous = events[0]
    for current in events[1:]:
        recent_prices.append(midprice(current.bid_price, current.ask_price))
        rows.append(feature_frame_from_events(previous, current, recent_prices=recent_prices))
        previous = current
    return rows


def _feature_rows_to_dicts(rows):
    return [row.to_dict() if hasattr(row, "to_dict") else row for row in rows]


def build_from_path(input_path, output_path):
    events = [parse_market_event(row) for row in read_table(input_path)]
    rows = _feature_rows_to_dicts(build_feature_rows(events))
    write_jsonl(output_path, rows)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Build local quantlab feature artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Convert market events to feature rows.")
    build_parser.add_argument("--input", required=True)
    build_parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "build":
        rows = build_from_path(args.input, args.output)
        print(json.dumps({"rows": len(rows), "output": str(Path(args.output))}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
