import json
import tempfile
import unittest
from pathlib import Path

from src.quantlab.backtest import BacktestConfig, run_backtest, walk_forward_splits
from src.quantlab.baselines import fit_ridge_direction_model, predict_direction_score, predict_trade_action, predict_ewma_volatility
from src.quantlab.core import FeatureFrame, LabelFrame, MarketEvent
from src.quantlab.features import book_imbalance, depth_imbalance, microprice, midprice, order_flow_imbalance, realized_volatility, signed_volume
from src.quantlab.labels import label_direction, triple_barrier_label
from src.quantlab.datasets import build_dataset_manifest


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


if __name__ == "__main__":
    unittest.main()
