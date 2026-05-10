import json
import tempfile
import unittest
from pathlib import Path

from src.quantlab.backtest import BacktestConfig, build_backtest_from_paths, run_backtest, walk_forward_splits
from src.quantlab.baselines import fit_ridge_direction_model, predict_direction_score, predict_trade_action, predict_ewma_volatility
from src.quantlab.core import FeatureFrame, LabelFrame, MarketEvent, MultiHorizonLabelFrame, SequenceWindow
from src.quantlab.demo import generate_btcusdt_like_events, run_demo_pipeline
from src.quantlab.features import book_imbalance, depth_imbalance, microprice, midprice, order_flow_imbalance, realized_volatility, signed_volume
from src.quantlab.labels import label_direction, triple_barrier_label
from src.quantlab.datasets import build_dataset_manifest
from src.quantlab.io import write_jsonl
from src.quantlab.models import build_prediction_rows, predict_from_paths, train_from_paths, train_mlp_direction_model


class QuantlabTest(unittest.TestCase):
    def test_feature_formulas_are_deterministic(self):
        previous = MarketEvent(
            timestamp_ms=1,
            symbol="BTCUSDT",
            event_type="book",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=4.0,
            ask_size=1.0,
        )
        current = MarketEvent(
            timestamp_ms=2,
            symbol="BTCUSDT",
            event_type="book",
            bid_price=100.5,
            ask_price=101.0,
            bid_size=6.0,
            ask_size=1.0,
            last_size=2.0,
            side="buy",
            extras={"bid_depth": [6.0, 2.0], "ask_depth": [1.0, 1.0]},
        )

        self.assertEqual(midprice(100.0, 101.0), 100.5)
        self.assertEqual(book_imbalance(4.0, 1.0), 0.6)
        self.assertEqual(depth_imbalance([6.0, 2.0], [1.0, 1.0]), 0.6)
        self.assertEqual(microprice(100.0, 101.0, 4.0, 1.0), 100.8)
        self.assertEqual(signed_volume(2.0, "buy"), 2.0)
        self.assertEqual(order_flow_imbalance(previous, current), 6.0)
        self.assertGreater(realized_volatility([100.0, 101.0, 102.0]), 0.0)

    def test_label_helpers_and_triple_barrier(self):
        self.assertEqual(label_direction(100.0, 100.1, cost_threshold=0.05), 1)
        self.assertEqual(label_direction(100.0, 99.9, cost_threshold=0.05), -1)
        self.assertEqual(label_direction(100.0, 100.01, cost_threshold=0.05), 0)
        self.assertEqual(triple_barrier_label([100.0, 100.5, 99.0], upper_barrier=0.004, lower_barrier=0.02, max_horizon=2), 1)

    def test_ridge_baseline_learns_known_signal(self):
        feature_frames = [
            FeatureFrame(1, "BTCUSDT", 100.0, 1.0, 100.1, 2.0, 0.2, 0.2, 1.0, 0.01),
            FeatureFrame(2, "BTCUSDT", 100.2, 1.0, 100.3, -2.0, -0.2, -0.2, -1.0, 0.02),
            FeatureFrame(3, "BTCUSDT", 100.4, 1.0, 100.5, 3.0, 0.3, 0.3, 1.0, 0.03),
        ]
        label_frames = [
            LabelFrame(1, "BTCUSDT", 1000, 1, 0.01, 0.0, 1, "buy"),
            LabelFrame(2, "BTCUSDT", 1000, -1, -0.01, 0.0, -1, "sell"),
            LabelFrame(3, "BTCUSDT", 1000, 1, 0.02, 0.0, 1, "buy"),
        ]

        model = fit_ridge_direction_model(feature_frames, label_frames, feature_keys=("order_flow_imbalance", "book_imbalance"))
        score = predict_direction_score(model, feature_frames[0])

        self.assertGreater(score, 0.0)
        self.assertEqual(predict_trade_action(score, threshold=0.0), "buy")

    def test_ewma_volatility_model_is_positive(self):
        model = predict_ewma_volatility([100.0, 101.0, 100.5, 101.5])

        self.assertGreater(model.variance, 0.0)

    def test_walk_forward_splits_are_disjoint(self):
        splits = list(walk_forward_splits(18, train_window=6, validation_window=3, test_window=3, embargo=1))

        self.assertTrue(splits)
        for split in splits:
            train = split["train"]
            validation = split["validation"]
            test = split["test"]
            self.assertLessEqual(train[1], validation[0])
            self.assertLessEqual(validation[1], test[0])

    def test_backtest_applies_costs_and_halts_on_drawdown(self):
        feature_rows = [
            FeatureFrame(1, "BTCUSDT", 100.0, 1.0, 100.1, 2.0, 0.2, 0.2, 1.0, 0.01),
            FeatureFrame(2, "BTCUSDT", 100.2, 1.0, 100.3, 2.0, 0.2, 0.2, 1.0, 0.01),
            FeatureFrame(3, "BTCUSDT", 100.4, 1.0, 100.5, 2.0, 0.2, 0.2, 1.0, 0.01),
        ]
        label_rows = [
            LabelFrame(1, "BTCUSDT", 1000, 1, 0.01, 0.0, 1, "buy"),
            LabelFrame(2, "BTCUSDT", 1000, 1, -0.02, 0.0, -1, "buy"),
            LabelFrame(3, "BTCUSDT", 1000, 1, 0.01, 0.0, 1, "buy"),
        ]
        report = run_backtest(feature_rows, label_rows, [0.1, 0.1, 0.1], config=BacktestConfig(fee_bps=5.0, slippage_bps=5.0, max_drawdown=0.5))

        self.assertIn("buy", report.actions)
        self.assertEqual(len(report.equity_curve), 4)
        self.assertGreaterEqual(report.total_pnl, -1.0)

    def test_dataset_manifest_includes_registry_and_adapter_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "quantlab.json"
            config_path.write_text(json.dumps({"name": "btc", "symbol": "BTCUSDT"}), encoding="utf-8")

            manifest = build_dataset_manifest(config_path)

        self.assertIn("btcusdt_microbar_v2", manifest["registry"])
        self.assertIn("cme_futures", manifest["adapter_targets"])

    def test_mlp_direction_model_learns_synthetic_orderflow_signal(self):
        feature_rows, label_rows = self._synthetic_orderflow_rows()

        artifact = train_mlp_direction_model(
            feature_rows,
            label_rows,
            hidden_dim=8,
            max_epochs=80,
            learning_rate=0.05,
            seed=11,
        )
        positive_score = artifact.predict_edge(feature_rows[-1])
        negative_score = artifact.predict_edge(feature_rows[0])

        self.assertLess(artifact.metrics["final_loss"], artifact.metrics["initial_loss"])
        self.assertGreater(positive_score, 0.0)
        self.assertLess(negative_score, 0.0)

    def test_mlp_prediction_rows_feed_backtest_file_builder(self):
        feature_rows, label_rows = self._synthetic_orderflow_rows()
        artifact = train_mlp_direction_model(
            feature_rows,
            label_rows,
            hidden_dim=8,
            max_epochs=80,
            learning_rate=0.05,
            seed=11,
        )
        prediction_rows = build_prediction_rows(artifact, feature_rows, no_trade_threshold=0.05)
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions_path = Path(temp_dir) / "predictions.jsonl"
            labels_path = Path(temp_dir) / "labels.jsonl"
            report_path = Path(temp_dir) / "report.json"
            write_jsonl(predictions_path, prediction_rows)
            write_jsonl(labels_path, [row.to_dict() for row in label_rows])

            report = build_backtest_from_paths(predictions_path, labels_path, report_path, no_trade_threshold=0.05)
            report_exists = report_path.exists()

        self.assertTrue(report_exists)
        self.assertIn("expected_edge", prediction_rows[0])
        self.assertEqual(len(report.actions), len(label_rows))

    def test_mlp_file_artifacts_round_trip_predictions(self):
        feature_rows, label_rows = self._synthetic_orderflow_rows()
        with tempfile.TemporaryDirectory() as temp_dir:
            features_path = Path(temp_dir) / "features.jsonl"
            labels_path = Path(temp_dir) / "labels.jsonl"
            model_path = Path(temp_dir) / "model.pt"
            metrics_path = Path(temp_dir) / "metrics.json"
            first_predictions_path = Path(temp_dir) / "train_predictions.jsonl"
            second_predictions_path = Path(temp_dir) / "predict_predictions.jsonl"
            write_jsonl(features_path, [row.to_dict() for row in feature_rows])
            write_jsonl(labels_path, [row.to_dict() for row in label_rows])

            artifact = train_from_paths(
                features_path,
                labels_path,
                model_path,
                predictions_out=first_predictions_path,
                metrics_out=metrics_path,
                hidden_dim=8,
                max_epochs=80,
                learning_rate=0.05,
                no_trade_threshold=0.05,
                seed=11,
            )
            prediction_rows = predict_from_paths(model_path, features_path, second_predictions_path, no_trade_threshold=0.05)

        self.assertLess(artifact.metrics["final_loss"], artifact.metrics["initial_loss"])
        self.assertEqual(len(prediction_rows), len(feature_rows))
        self.assertIn("action", prediction_rows[0])

    def test_demo_pipeline_writes_end_to_end_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_demo_pipeline(
                temp_dir,
                rows=36,
                hidden_dim=8,
                max_epochs=30,
                learning_rate=0.03,
                no_trade_threshold=0.05,
                seed=17,
            )

            artifact_paths = [Path(path) for path in summary["artifacts"].values()]
            summary_path = Path(summary["summary"])

            self.assertEqual(summary["counts"]["events"], 36)
            self.assertEqual(summary["counts"]["features"], summary["counts"]["labels"])
            self.assertGreater(summary["counts"]["features"], 0)
            self.assertLess(summary["metrics"]["final_loss"], summary["metrics"]["initial_loss"])
            self.assertIn("equity_curve", summary["backtest"])
            self.assertEqual(len(summary["paper_references"]), 3)
            self.assertTrue(all(path.exists() for path in artifact_paths))
            self.assertTrue(summary_path.exists())

    def test_synthetic_events_include_order_book_depth_metadata(self):
        events = generate_btcusdt_like_events(rows=8, seed=5)

        self.assertEqual(len(events), 8)
        self.assertEqual(events[0].source, "synthetic_demo")
        self.assertIn("bid_depth", events[0].extras)
        self.assertIn("latent_pressure", events[0].extras)

    def test_multihorizon_label_frame_round_trip(self):
        frame = MultiHorizonLabelFrame(
            timestamp_ms=1_700_000_000_000,
            symbol="BTCUSDT",
            horizons={
                1: {"direction": 1, "future_return": 0.0001, "future_vol": 0.0002, "future_spread": 0.5},
                5: {"direction": 0, "future_return": 0.0000, "future_vol": 0.0003, "future_spread": 0.4},
            },
        )
        payload = frame.to_dict()
        self.assertEqual(payload["timestamp_ms"], 1_700_000_000_000)
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["horizons"]["1"]["direction"], 1)
        self.assertEqual(payload["horizons"]["5"]["future_return"], 0.0)

    def test_sequence_window_round_trip(self):
        window = SequenceWindow(
            timestamp_ms=1_700_000_000_000,
            symbol="BTCUSDT",
            sequence_length=128,
            feature_keys=("a", "b"),
        )
        payload = window.to_dict()
        self.assertEqual(payload["sequence_length"], 128)
        self.assertEqual(payload["feature_keys"], ["a", "b"])

    def _synthetic_orderflow_rows(self):
        feature_rows = []
        label_rows = []
        for index, ofi in enumerate([-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0], start=1):
            direction = 1 if ofi > 0 else -1
            feature_rows.append(
                FeatureFrame(
                    index,
                    "BTCUSDT",
                    100.0 + index,
                    0.5,
                    100.0 + index + 0.01 * ofi,
                    ofi,
                    ofi / 4.0,
                    ofi / 5.0,
                    ofi,
                    0.01,
                )
            )
            label_rows.append(
                LabelFrame(
                    index,
                    "BTCUSDT",
                    1000,
                    direction,
                    0.002 * direction,
                    0.01,
                    direction,
                    "buy" if direction > 0 else "sell",
                )
            )
        return feature_rows, label_rows


if __name__ == "__main__":
    unittest.main()
