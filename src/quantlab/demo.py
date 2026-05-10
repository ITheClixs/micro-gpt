"""End-to-end synthetic quantlab pipeline for CPU smoke research runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .backtest import build_backtest_from_paths
from .core import MarketEvent
from .features import build_from_path as build_features_from_path
from .io import write_jsonl
from .labels import build_labels_from_path
from .models import train_from_paths


PAPER_REFERENCES = (
    {
        "title": "FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance",
        "url": "https://arxiv.org/abs/2111.09395",
        "local_scope": "Pipeline shape: data, model predictions, transaction-cost-aware backtest, and reproducible artifacts.",
    },
    {
        "title": "JANET: Joint Adaptive predictioN-region Estimation for Time-series",
        "url": "https://arxiv.org/abs/2407.06390",
        "local_scope": "Future target: sequential uncertainty calibration; this demo preserves aligned timestamp artifacts for that step.",
    },
    {
        "title": "DOME: Recommendations for supervised machine learning validation in biology",
        "url": "https://arxiv.org/abs/2006.16189",
        "local_scope": "Validation discipline: keep data, optimization, model, and evaluation outputs explicit.",
    },
)


def generate_btcusdt_like_events(
    rows=96,
    symbol="BTCUSDT",
    start_timestamp_ms=1_700_000_000_000,
    interval_ms=1000,
    start_price=60_000.0,
    seed=2026,
):
    """Generate deterministic regime-switching market events.

    The latent order pressure at event t influences the next midprice move. That
    makes the resulting feature rows useful for a fast supervised smoke test
    without pretending to represent live exchange data.
    """

    rng = random.Random(int(seed))
    rows = int(rows)
    if rows < 8:
        raise ValueError("rows must be at least 8 for train/backtest smoke coverage.")

    price = float(start_price)
    previous_pressure = 0.0
    events = []
    for index in range(rows):
        regime = _regime_name(index, rows)
        drift = _regime_drift(regime)
        shock = rng.gauss(0.0, 0.00006)
        if index > 0:
            price *= 1.0 + drift + 0.00034 * previous_pressure + shock
            price = max(price, 1.0)

        pressure = _latent_pressure(index, rows, regime, rng)
        spread_width = price * (0.00008 + 0.00002 * abs(pressure))
        bid_price = price - spread_width / 2.0
        ask_price = price + spread_width / 2.0
        base_depth = 7.5 + 1.5 * math.cos(index / 5.0)
        bid_size = max(0.1, base_depth * (1.0 + 0.45 * pressure))
        ask_size = max(0.1, base_depth * (1.0 - 0.45 * pressure))
        side = "buy" if pressure >= 0 else "sell"
        last_size = abs(pressure) * 2.0 + 0.25 + rng.random() * 0.1
        timestamp_ms = int(start_timestamp_ms) + index * int(interval_ms)
        events.append(
            MarketEvent(
                timestamp_ms=timestamp_ms,
                symbol=symbol,
                event_type="book",
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                last_price=price,
                last_size=last_size,
                side=side,
                source="synthetic_demo",
                extras={
                    "bid_depth": [bid_size, bid_size * 0.65, bid_size * 0.35],
                    "ask_depth": [ask_size, ask_size * 0.65, ask_size * 0.35],
                    "funding": 0.00001 * math.sin(index / 12.0),
                    "liquidation_intensity": max(0.0, abs(pressure) - 0.7),
                    "latent_pressure": pressure,
                    "regime": regime,
                },
            )
        )
        previous_pressure = pressure
    return events


def align_feature_label_rows(feature_rows, label_rows):
    feature_by_timestamp = {int(row["timestamp_ms"]): row for row in feature_rows}
    label_by_timestamp = {int(row["timestamp_ms"]): row for row in label_rows}
    timestamps = sorted(set(feature_by_timestamp) & set(label_by_timestamp))
    if not timestamps:
        raise ValueError("feature and label rows have no overlapping timestamps.")
    return [feature_by_timestamp[timestamp] for timestamp in timestamps], [label_by_timestamp[timestamp] for timestamp in timestamps]


def run_demo_pipeline(
    output_dir,
    rows=96,
    symbol="BTCUSDT",
    seed=2026,
    interval_ms=1000,
    horizon_ms=1000,
    start_price=60_000.0,
    hidden_dim=16,
    max_epochs=50,
    learning_rate=0.01,
    no_trade_threshold=0.05,
    cost_threshold="spread",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "btcusdt_synthetic_events.jsonl"
    raw_features_path = output_dir / "btcusdt_features_raw.jsonl"
    raw_labels_path = output_dir / "btcusdt_labels_raw.jsonl"
    features_path = output_dir / "btcusdt_features.jsonl"
    labels_path = output_dir / "btcusdt_labels.jsonl"
    model_path = output_dir / "quantlab_mlp_direction.pt"
    predictions_path = output_dir / "btcusdt_mlp_predictions.jsonl"
    metrics_path = output_dir / "quantlab_mlp_direction_metrics.json"
    backtest_path = output_dir / "btcusdt_mlp_backtest.json"
    summary_path = output_dir / "quantlab_demo_summary.json"

    events = generate_btcusdt_like_events(
        rows=rows,
        symbol=symbol,
        interval_ms=interval_ms,
        start_price=start_price,
        seed=seed,
    )
    write_jsonl(events_path, [event.to_dict() for event in events])

    raw_feature_rows = build_features_from_path(events_path, raw_features_path)
    raw_label_rows = build_labels_from_path(
        events_path,
        raw_labels_path,
        horizons=[int(horizon_ms)],
        cost_threshold=cost_threshold,
    )
    feature_rows, label_rows = align_feature_label_rows(raw_feature_rows, raw_label_rows)
    write_jsonl(features_path, feature_rows)
    write_jsonl(labels_path, label_rows)

    artifact = train_from_paths(
        features_path,
        labels_path,
        model_path,
        predictions_out=predictions_path,
        metrics_out=metrics_path,
        hidden_dim=hidden_dim,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        no_trade_threshold=no_trade_threshold,
        seed=seed,
    )
    backtest_report = build_backtest_from_paths(
        predictions_path,
        labels_path,
        backtest_path,
        no_trade_threshold=no_trade_threshold,
    )

    summary = {
        "artifacts": {
            "events": str(events_path),
            "features": str(features_path),
            "labels": str(labels_path),
            "model": str(model_path),
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "backtest": str(backtest_path),
        },
        "counts": {
            "events": len(events),
            "features": len(feature_rows),
            "labels": len(label_rows),
        },
        "config": {
            "symbol": symbol,
            "seed": int(seed),
            "interval_ms": int(interval_ms),
            "horizon_ms": int(horizon_ms),
            "hidden_dim": int(hidden_dim),
            "max_epochs": int(max_epochs),
            "learning_rate": float(learning_rate),
            "no_trade_threshold": float(no_trade_threshold),
            "cost_threshold": cost_threshold,
        },
        "metrics": artifact.metrics,
        "backtest": backtest_report.to_dict(),
        "paper_references": PAPER_REFERENCES,
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _regime_name(index, rows):
    fraction = index / max(1, rows - 1)
    if fraction < 0.34:
        return "accumulation"
    if fraction < 0.67:
        return "distribution"
    return "mean_reversion"


def _regime_drift(regime):
    if regime == "accumulation":
        return 0.00002
    if regime == "distribution":
        return -0.00002
    return 0.0


def _latent_pressure(index, rows, regime, rng):
    cyclical = 0.55 * math.sin(index / 3.0) + 0.25 * math.sin(index / 9.0)
    if regime == "accumulation":
        bias = 0.35
    elif regime == "distribution":
        bias = -0.35
    else:
        midpoint = rows * 0.83
        bias = -0.25 if index < midpoint else 0.25
    pressure = bias + cyclical + rng.gauss(0.0, 0.08)
    return max(-0.98, min(0.98, pressure))


def build_parser():
    parser = argparse.ArgumentParser(description="Run an end-to-end synthetic quantlab smoke pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate synthetic events, train MLP, and run a backtest.")
    run_parser.add_argument("--output-dir", default="/tmp/quantlab_demo")
    run_parser.add_argument("--rows", type=int, default=96)
    run_parser.add_argument("--symbol", default="BTCUSDT")
    run_parser.add_argument("--seed", type=int, default=2026)
    run_parser.add_argument("--interval-ms", type=int, default=1000)
    run_parser.add_argument("--horizon-ms", type=int, default=1000)
    run_parser.add_argument("--start-price", type=float, default=60_000.0)
    run_parser.add_argument("--hidden-dim", type=int, default=16)
    run_parser.add_argument("--max-epochs", type=int, default=50)
    run_parser.add_argument("--learning-rate", type=float, default=0.01)
    run_parser.add_argument("--no-trade-threshold", type=float, default=0.05)
    run_parser.add_argument("--cost-threshold", default="spread")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        summary = run_demo_pipeline(
            output_dir=args.output_dir,
            rows=args.rows,
            symbol=args.symbol,
            seed=args.seed,
            interval_ms=args.interval_ms,
            horizon_ms=args.horizon_ms,
            start_price=args.start_price,
            hidden_dim=args.hidden_dim,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            no_trade_threshold=args.no_trade_threshold,
            cost_threshold=args.cost_threshold,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
