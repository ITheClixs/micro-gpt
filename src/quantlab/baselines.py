"""Deterministic quant baselines for direction and volatility."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .core import FeatureFrame, LabelFrame
from .io import read_table, write_jsonl


@dataclass(frozen=True)
class LinearDirectionModel:
    feature_keys: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    ridge: float = 1.0

    def to_dict(self):
        payload = asdict(self)
        payload["feature_keys"] = list(self.feature_keys)
        payload["weights"] = list(self.weights)
        return payload


@dataclass(frozen=True)
class EWMAVolatilityModel:
    alpha: float
    variance: float

    def to_dict(self):
        return asdict(self)


def _feature_value(frame, key):
    if hasattr(frame, key):
        return float(getattr(frame, key))
    if isinstance(frame, dict):
        return float(frame[key])
    raise KeyError(key)


def fit_ridge_direction_model(feature_frames, label_frames, feature_keys=("order_flow_imbalance", "book_imbalance", "microprice")):
    feature_frames = list(feature_frames)
    label_frames = list(label_frames)
    if not feature_frames:
        raise ValueError("feature_frames must not be empty.")
    if len(feature_frames) != len(label_frames):
        raise ValueError("feature_frames and label_frames must have the same length.")
    x = torch.tensor([[ _feature_value(frame, key) for key in feature_keys ] for frame in feature_frames], dtype=torch.float32)
    y = torch.tensor([float(label.midprice_direction) for label in label_frames], dtype=torch.float32).unsqueeze(1)
    ones = torch.ones((x.size(0), 1), dtype=torch.float32)
    design = torch.cat([x, ones], dim=1)
    ridge = 1.0
    eye = torch.eye(design.size(1), dtype=torch.float32)
    eye[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y).squeeze(1)
    return LinearDirectionModel(
        feature_keys=tuple(feature_keys),
        weights=tuple(float(value) for value in weights[:-1]),
        bias=float(weights[-1]),
        ridge=ridge,
    )


def predict_direction_score(model, feature_frame):
    score = model.bias
    for weight, key in zip(model.weights, model.feature_keys):
        score += float(weight) * _feature_value(feature_frame, key)
    return score


def predict_trade_action(score, threshold=0.0):
    score = float(score)
    threshold = abs(float(threshold))
    if score > threshold:
        return "buy"
    if score < -threshold:
        return "sell"
    return "hold"


def predict_ewma_volatility(prices, alpha=0.94):
    prices = [float(price) for price in prices]
    if len(prices) < 2:
        return EWMAVolatilityModel(alpha=float(alpha), variance=0.0)
    variance = 0.0
    for previous, current in zip(prices, prices[1:]):
        if previous <= 0 or current <= 0:
            continue
        log_return = torch.log(torch.tensor(current / previous, dtype=torch.float32)).item()
        variance = float(alpha) * variance + (1.0 - float(alpha)) * (log_return * log_return)
    return EWMAVolatilityModel(alpha=float(alpha), variance=variance)


def _load_feature_rows(path):
    rows = [row for row in read_table(path)]
    return [
        FeatureFrame(
            timestamp_ms=int(row["timestamp_ms"]),
            symbol=str(row["symbol"]),
            midprice=float(row["midprice"]),
            spread=float(row["spread"]),
            microprice=float(row["microprice"]),
            order_flow_imbalance=float(row["order_flow_imbalance"]),
            book_imbalance=float(row["book_imbalance"]),
            depth_imbalance=float(row["depth_imbalance"]),
            signed_volume=float(row["signed_volume"]),
            realized_volatility=float(row["realized_volatility"]),
        )
        for row in rows
    ]


def _load_label_rows(path):
    rows = [row for row in read_table(path)]
    return [
        LabelFrame(
            timestamp_ms=int(row["timestamp_ms"]),
            symbol=str(row["symbol"]),
            horizon_ms=int(row.get("horizon_ms", 1000)),
            midprice_direction=int(row["midprice_direction"]),
            future_return=float(row.get("future_return", 0.0)),
            realized_volatility=float(row.get("realized_volatility", 0.0)),
            triple_barrier=int(row.get("triple_barrier", 0)),
            action=str(row.get("action", "hold")),
        )
        for row in rows
    ]


def build_training_artifacts(features_path, output_path, labels_path=None):
    feature_rows = _load_feature_rows(features_path)
    label_rows = _load_label_rows(labels_path or features_path)
    if len(feature_rows) != len(label_rows):
        raise ValueError("feature and label rows must have the same length.")
    model = fit_ridge_direction_model(feature_rows, label_rows)
    payload = {
        "direction_model": model.to_dict(),
        "volatility_model": predict_ewma_volatility([frame.midprice for frame in feature_rows]).to_dict(),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Train local quantlab baselines.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Fit deterministic quant baselines.")
    train_parser.add_argument("--features", required=True)
    train_parser.add_argument("--labels")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--model", default="ofi_logistic")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        payload = build_training_artifacts(args.features, args.output, labels_path=args.labels)
        print(json.dumps({"output": str(Path(args.output)), "model": args.model, "keys": sorted(payload)}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
