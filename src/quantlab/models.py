"""Small supervised ML models for quantlab direction prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .baselines import _load_feature_rows, _load_label_rows, _feature_value, predict_trade_action
from .core import FeatureFrame
from .io import write_jsonl


DEFAULT_FEATURE_KEYS = (
    "order_flow_imbalance",
    "book_imbalance",
    "depth_imbalance",
    "signed_volume",
    "realized_volatility",
    "spread",
)

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


@dataclass(frozen=True)
class Standardizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, matrix):
        mean = matrix.mean(dim=0)
        scale = matrix.std(dim=0, unbiased=False)
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        return cls(
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
        )

    def transform(self, matrix):
        mean = torch.tensor(self.mean, dtype=torch.float32, device=matrix.device)
        scale = torch.tensor(self.scale, dtype=torch.float32, device=matrix.device)
        return (matrix - mean) / scale

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(mean=tuple(payload["mean"]), scale=tuple(payload["scale"]))


class DirectionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, features):
        return self.network(features)


@dataclass
class MLPDirectionArtifact:
    feature_keys: tuple[str, ...]
    hidden_dim: int
    standardizer: Standardizer
    model: DirectionMLP
    metrics: dict[str, float]

    def predict_probabilities(self, feature_frame):
        matrix = feature_matrix([feature_frame], self.feature_keys)
        logits = self.model(self.standardizer.transform(matrix))
        probabilities = torch.softmax(logits, dim=-1)[0]
        return {
            "prob_down": float(probabilities[0]),
            "prob_hold": float(probabilities[1]),
            "prob_up": float(probabilities[2]),
        }

    def predict_edge(self, feature_frame):
        probabilities = self.predict_probabilities(feature_frame)
        return probabilities["prob_up"] - probabilities["prob_down"]

    def to_payload(self):
        return {
            "feature_keys": list(self.feature_keys),
            "hidden_dim": self.hidden_dim,
            "standardizer": self.standardizer.to_dict(),
            "state_dict": self.model.state_dict(),
            "metrics": dict(self.metrics),
        }


def feature_matrix(feature_frames, feature_keys=DEFAULT_FEATURE_KEYS):
    rows = [[_feature_value(frame, key) for key in feature_keys] for frame in feature_frames]
    if not rows:
        raise ValueError("feature_frames must not be empty.")
    return torch.tensor(rows, dtype=torch.float32)


def label_tensor(label_frames):
    labels = []
    for label in label_frames:
        direction = int(label.midprice_direction)
        if direction not in LABEL_TO_CLASS:
            raise ValueError("midprice_direction must be -1, 0, or 1.")
        labels.append(LABEL_TO_CLASS[direction])
    if not labels:
        raise ValueError("label_frames must not be empty.")
    return torch.tensor(labels, dtype=torch.long)


def train_mlp_direction_model(
    feature_frames,
    label_frames,
    feature_keys=DEFAULT_FEATURE_KEYS,
    hidden_dim=16,
    max_epochs=50,
    learning_rate=0.01,
    seed=2026,
):
    feature_frames = list(feature_frames)
    label_frames = list(label_frames)
    if len(feature_frames) != len(label_frames):
        raise ValueError("feature_frames and label_frames must have the same length.")
    torch.manual_seed(int(seed))
    x = feature_matrix(feature_frames, feature_keys)
    y = label_tensor(label_frames)
    standardizer = Standardizer.fit(x)
    x_scaled = standardizer.transform(x)
    model = DirectionMLP(input_dim=x_scaled.size(1), hidden_dim=int(hidden_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)

    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(x_scaled), y))

    final_loss = initial_loss
    for _ in range(int(max_epochs)):
        logits = model(x_scaled)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    with torch.no_grad():
        logits = model(x_scaled)
        predictions = logits.argmax(dim=-1)
        accuracy = float((predictions == y).float().mean())
        final_loss = float(F.cross_entropy(logits, y))

    return MLPDirectionArtifact(
        feature_keys=tuple(feature_keys),
        hidden_dim=int(hidden_dim),
        standardizer=standardizer,
        model=model,
        metrics={
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "accuracy": accuracy,
            "rows": float(len(feature_frames)),
            "epochs": float(max_epochs),
        },
    )


def build_prediction_rows(artifact, feature_frames, no_trade_threshold=0.0):
    rows = []
    for frame in feature_frames:
        probabilities = artifact.predict_probabilities(frame)
        expected_edge = probabilities["prob_up"] - probabilities["prob_down"]
        rows.append(
            {
                "timestamp_ms": frame.timestamp_ms,
                "symbol": frame.symbol,
                "midprice": frame.midprice,
                "spread": frame.spread,
                "microprice": frame.microprice,
                "order_flow_imbalance": frame.order_flow_imbalance,
                "book_imbalance": frame.book_imbalance,
                "depth_imbalance": frame.depth_imbalance,
                "signed_volume": frame.signed_volume,
                "realized_volatility": frame.realized_volatility,
                "prob_down": probabilities["prob_down"],
                "prob_hold": probabilities["prob_hold"],
                "prob_up": probabilities["prob_up"],
                "expected_edge": expected_edge,
                "action": predict_trade_action(expected_edge, threshold=no_trade_threshold),
            }
        )
    return rows


def save_mlp_direction_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_payload(), path)


def load_mlp_direction_artifact(path):
    payload = torch.load(Path(path), map_location="cpu")
    feature_keys = tuple(payload["feature_keys"])
    model = DirectionMLP(input_dim=len(feature_keys), hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return MLPDirectionArtifact(
        feature_keys=feature_keys,
        hidden_dim=int(payload["hidden_dim"]),
        standardizer=Standardizer.from_dict(payload["standardizer"]),
        model=model,
        metrics=dict(payload.get("metrics", {})),
    )


def train_from_paths(
    features_path,
    labels_path,
    model_out,
    predictions_out=None,
    metrics_out=None,
    hidden_dim=16,
    max_epochs=50,
    learning_rate=0.01,
    no_trade_threshold=0.0,
    seed=2026,
):
    feature_rows = _load_feature_rows(features_path)
    label_rows = _load_label_rows(labels_path)
    artifact = train_mlp_direction_model(
        feature_rows,
        label_rows,
        hidden_dim=hidden_dim,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    save_mlp_direction_artifact(artifact, model_out)
    if predictions_out is not None:
        write_jsonl(predictions_out, build_prediction_rows(artifact, feature_rows, no_trade_threshold=no_trade_threshold))
    if metrics_out is not None:
        metrics_path = Path(metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(artifact.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def predict_from_paths(model_path, features_path, predictions_out, no_trade_threshold=0.0):
    artifact = load_mlp_direction_artifact(model_path)
    feature_rows = _load_feature_rows(features_path)
    prediction_rows = build_prediction_rows(artifact, feature_rows, no_trade_threshold=no_trade_threshold)
    write_jsonl(predictions_out, prediction_rows)
    return prediction_rows


def build_parser():
    parser = argparse.ArgumentParser(description="Train and run small quantlab ML direction models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a CPU-safe MLP direction model.")
    train_parser.add_argument("--features", required=True)
    train_parser.add_argument("--labels", required=True)
    train_parser.add_argument("--model-out", required=True)
    train_parser.add_argument("--predictions-out")
    train_parser.add_argument("--metrics-out")
    train_parser.add_argument("--hidden-dim", type=int, default=16)
    train_parser.add_argument("--max-epochs", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--no-trade-threshold", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=2026)

    predict_parser = subparsers.add_parser("predict", help="Write prediction rows from a saved MLP model.")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--features", required=True)
    predict_parser.add_argument("--predictions-out", required=True)
    predict_parser.add_argument("--no-trade-threshold", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        artifact = train_from_paths(
            args.features,
            args.labels,
            args.model_out,
            predictions_out=args.predictions_out,
            metrics_out=args.metrics_out,
            hidden_dim=args.hidden_dim,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            no_trade_threshold=args.no_trade_threshold,
            seed=args.seed,
        )
        print(json.dumps({"model": str(Path(args.model_out)), "metrics": artifact.metrics}, indent=2, sort_keys=True))
        return 0
    if args.command == "predict":
        rows = predict_from_paths(
            args.model,
            args.features,
            args.predictions_out,
            no_trade_threshold=args.no_trade_threshold,
        )
        print(json.dumps({"predictions": str(Path(args.predictions_out)), "rows": len(rows)}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
