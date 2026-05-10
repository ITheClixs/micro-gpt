"""Quantlab target generation for direction and triple-barrier labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .core import LabelFrame
from .features import midprice, realized_volatility, spread
from .io import parse_float, parse_int, read_table, write_jsonl


def label_direction(current_midprice, future_midprice, cost_threshold=0.0):
    edge = float(future_midprice) - float(current_midprice)
    threshold = abs(float(cost_threshold))
    if edge > threshold:
        return 1
    if edge < -threshold:
        return -1
    return 0


def triple_barrier_label(prices, upper_barrier, lower_barrier, max_horizon):
    prices = [float(price) for price in prices]
    if len(prices) < 2:
        return 0
    start_price = prices[0]
    upper_price = start_price * (1.0 + float(upper_barrier))
    lower_price = start_price * (1.0 - float(lower_barrier))
    horizon = min(int(max_horizon), len(prices) - 1)
    for price in prices[1 : horizon + 1]:
        if price >= upper_price:
            return 1
        if price <= lower_price:
            return -1
    return 0


def _resolve_cost_threshold(cost_threshold, current_spread):
    if isinstance(cost_threshold, str) and cost_threshold.lower() == "spread":
        return float(current_spread)
    return float(cost_threshold)


def build_label_frame(events, horizon_ms, cost_threshold=0.0, upper_barrier=0.002, lower_barrier=0.002):
    events = list(events)
    if len(events) < 2:
        return []
    rows = []
    for index, current in enumerate(events[:-1]):
        start_mid = midprice(current.bid_price, current.ask_price)
        future = [midprice(event.bid_price, event.ask_price) for event in events[index:]]
        future_mid = future[min(len(future) - 1, 1)]
        current_cost_threshold = _resolve_cost_threshold(cost_threshold, spread(current.bid_price, current.ask_price))
        rows.append(
            LabelFrame(
                timestamp_ms=current.timestamp_ms,
                symbol=current.symbol,
                horizon_ms=int(horizon_ms),
                midprice_direction=label_direction(start_mid, future_mid, cost_threshold=current_cost_threshold),
                future_return=(future_mid - start_mid) / start_mid if start_mid else 0.0,
                realized_volatility=realized_volatility(future[: min(len(future), 8)]),
                triple_barrier=triple_barrier_label(
                    future,
                    upper_barrier=upper_barrier,
                    lower_barrier=lower_barrier,
                    max_horizon=min(len(future) - 1, 8),
                ),
                action="buy" if future_mid > start_mid else "sell" if future_mid < start_mid else "hold",
            )
        )
    return rows


def parse_horizons(raw):
    horizons = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        if value.endswith("ms"):
            horizons.append(int(value[:-2]))
        elif value.endswith("s"):
            horizons.append(int(float(value[:-1]) * 1000))
        else:
            horizons.append(int(value))
    return horizons


def parse_event_rows(path):
    return [row for row in read_table(path)]


def build_labels_from_path(input_path, output_path, horizons, cost_threshold=0.0):
    rows = [row for row in read_table(input_path)]
    events = [
        {
            "timestamp_ms": parse_int(row["timestamp_ms"]),
            "symbol": str(row["symbol"]),
            "bid_price": parse_float(row["bid_price"]),
            "ask_price": parse_float(row["ask_price"]),
            "bid_size": parse_float(row["bid_size"]),
            "ask_size": parse_float(row["ask_size"]),
        }
        for row in rows
    ]
    label_rows = []
    for horizon in horizons:
        for index, current in enumerate(events[:-1]):
            future = events[index:]
            start_mid = midprice(current["bid_price"], current["ask_price"])
            future_mid = midprice(future[min(len(future) - 1, 1)]["bid_price"], future[min(len(future) - 1, 1)]["ask_price"])
            future_path = [midprice(item["bid_price"], item["ask_price"]) for item in future]
            current_cost_threshold = _resolve_cost_threshold(cost_threshold, spread(current["bid_price"], current["ask_price"]))
            label_rows.append(
                LabelFrame(
                    timestamp_ms=current["timestamp_ms"],
                    symbol=current["symbol"],
                    horizon_ms=int(horizon),
                    midprice_direction=label_direction(start_mid, future_mid, cost_threshold=current_cost_threshold),
                    future_return=(future_mid - start_mid) / start_mid if start_mid else 0.0,
                    realized_volatility=realized_volatility(future_path[: min(len(future_path), 8)]),
                    triple_barrier=triple_barrier_label(future_path, upper_barrier=0.002, lower_barrier=0.002, max_horizon=min(len(future_path) - 1, 8)),
                    action="buy" if future_mid > start_mid else "sell" if future_mid < start_mid else "hold",
                ).to_dict()
            )
    write_jsonl(output_path, label_rows)
    return label_rows


def build_parser():
    parser = argparse.ArgumentParser(description="Build local quantlab label artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Convert market events to label rows.")
    build_parser.add_argument("--input", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--horizons", default="1000")
    build_parser.add_argument("--cost-threshold", default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "build":
        label_rows = build_labels_from_path(
            args.input,
            args.output,
            horizons=parse_horizons(args.horizons),
            cost_threshold=args.cost_threshold,
        )
        print(json.dumps({"rows": len(label_rows), "output": str(Path(args.output))}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
