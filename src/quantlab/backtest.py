"""Cost-aware walk-forward backtesting for quantlab signals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .baselines import predict_trade_action
from .io import read_table


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    max_position: int = 1
    no_trade_threshold: float = 0.0
    max_drawdown: float = 0.2
    initial_cash: float = 1.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BacktestReport:
    equity_curve: list[float]
    actions: list[str]
    trades: int
    total_pnl: float
    max_drawdown: float
    stopped: bool

    def to_dict(self):
        return asdict(self)


def walk_forward_splits(n_observations, train_window, validation_window, test_window, embargo=0):
    n_observations = int(n_observations)
    train_window = int(train_window)
    validation_window = int(validation_window)
    test_window = int(test_window)
    embargo = int(embargo)
    start = 0
    while True:
        train_end = start + train_window
        validation_end = train_end + validation_window
        test_end = validation_end + test_window
        if test_end > n_observations:
            break
        yield {
            "train": (start, train_end),
            "validation": (train_end + embargo, validation_end),
            "test": (validation_end + embargo, test_end),
        }
        start = test_end


def _drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _position_from_action(action, current_position, max_position):
    if action == "buy":
        return min(current_position + 1, max_position)
    if action == "sell":
        return max(current_position - 1, -max_position)
    return current_position


def _future_return(label):
    if hasattr(label, "future_return"):
        return float(label.future_return)
    if isinstance(label, dict):
        return float(label.get("future_return", 0.0))
    raise KeyError("future_return")


def run_backtest(feature_rows, label_rows, expected_edges, config=None):
    config = config or BacktestConfig()
    feature_rows = list(feature_rows)
    label_rows = list(label_rows)
    expected_edges = list(expected_edges)
    if not feature_rows or not label_rows or not expected_edges:
        raise ValueError("feature_rows, label_rows, and expected_edges must not be empty.")
    if not (len(feature_rows) == len(label_rows) == len(expected_edges)):
        raise ValueError("feature_rows, label_rows, and expected_edges must have the same length.")

    equity_curve = [float(config.initial_cash)]
    actions = []
    position = 0
    trades = 0
    stopped = False
    for feature, label, expected_edge in zip(feature_rows, label_rows, expected_edges):
        action = predict_trade_action(expected_edge, threshold=config.no_trade_threshold)
        next_position = _position_from_action(action, position, config.max_position)
        if next_position != position:
            trades += 1
        turnover = abs(next_position - position)
        cost_bps = config.fee_bps + config.slippage_bps
        gross_pnl = next_position * _future_return(label)
        net_pnl = gross_pnl - turnover * cost_bps / 10000.0
        equity_curve.append(equity_curve[-1] + net_pnl)
        actions.append(action)
        position = next_position
        if _drawdown(equity_curve) > config.max_drawdown:
            stopped = True
            break
    max_drawdown = _drawdown(equity_curve)
    return BacktestReport(
        equity_curve=equity_curve,
        actions=actions,
        trades=trades,
        total_pnl=equity_curve[-1] - equity_curve[0],
        max_drawdown=max_drawdown,
        stopped=stopped,
    )


def build_backtest_from_paths(predictions_path, labels_path, output_path, no_trade_threshold=0.0):
    prediction_rows = [row for row in read_table(predictions_path)]
    label_rows_raw = [row for row in read_table(labels_path)]
    if len(prediction_rows) != len(label_rows_raw):
        raise ValueError("predictions and labels must have the same length.")
    feature_rows = []
    label_rows = []
    expected_edges = []
    for prediction_row, label_row in zip(prediction_rows, label_rows_raw):
        feature_rows.append(
            {
                "timestamp_ms": int(prediction_row["timestamp_ms"]),
                "symbol": prediction_row["symbol"],
                "midprice": float(prediction_row.get("midprice", 0.0)),
                "spread": float(prediction_row.get("spread", 0.0)),
                "microprice": float(prediction_row.get("microprice", 0.0)),
                "order_flow_imbalance": float(prediction_row.get("order_flow_imbalance", 0.0)),
                "book_imbalance": float(prediction_row.get("book_imbalance", 0.0)),
                "depth_imbalance": float(prediction_row.get("depth_imbalance", 0.0)),
                "signed_volume": float(prediction_row.get("signed_volume", 0.0)),
                "realized_volatility": float(prediction_row.get("realized_volatility", 0.0)),
            }
        )
        label_rows.append(
            {
                "future_return": float(label_row.get("future_return", 0.0)),
            }
        )
        expected_edges.append(
            float(
                prediction_row.get(
                    "expected_edge",
                    float(prediction_row.get("microprice", 0.0)) - float(prediction_row.get("midprice", 0.0)),
                )
            )
        )
    report = run_backtest(feature_rows, label_rows, expected_edges, config=BacktestConfig(no_trade_threshold=no_trade_threshold))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_backtest_from_path(input_path, output_path, no_trade_threshold=0.0):
    return build_backtest_from_paths(input_path, input_path, output_path, no_trade_threshold=no_trade_threshold)


def build_parser():
    parser = argparse.ArgumentParser(description="Run a local quantlab backtest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Execute a cost-aware backtest.")
    run_parser.add_argument("--predictions", required=True)
    run_parser.add_argument("--labels", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--no-trade-threshold", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        report = build_backtest_from_paths(
            args.predictions,
            args.labels,
            args.output,
            no_trade_threshold=args.no_trade_threshold,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
