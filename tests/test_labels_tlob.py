import unittest

from src.quantlab.core import MarketEvent
from src.quantlab.labels_tlob import (
    build_multihorizon_labels,
    tlob_smoothed_label,
)


def _events(prices, start_ts=1_700_000_000_000, dt=1000, spread=0.4):
    return [
        MarketEvent(
            timestamp_ms=start_ts + i * dt,
            symbol="X",
            event_type="book",
            bid_price=price - spread / 2,
            ask_price=price + spread / 2,
            bid_size=1.0,
            ask_size=1.0,
            last_price=price,
            last_size=0.1,
            side="buy",
            source="test",
        )
        for i, price in enumerate(prices)
    ]


class TestTlobLabels(unittest.TestCase):
    def test_smoothed_label_monotonic_up(self):
        prices = [100.0 + i * 0.5 for i in range(20)]
        label = tlob_smoothed_label(prices, t=10, horizon=5, k_smooth=2, theta=0.0)
        self.assertEqual(label, 1)

    def test_smoothed_label_monotonic_down(self):
        prices = [100.0 - i * 0.5 for i in range(20)]
        label = tlob_smoothed_label(prices, t=10, horizon=5, k_smooth=2, theta=0.0)
        self.assertEqual(label, -1)

    def test_smoothed_label_stable_within_theta(self):
        prices = [100.0] * 20
        label = tlob_smoothed_label(prices, t=10, horizon=5, k_smooth=2, theta=0.001)
        self.assertEqual(label, 0)

    def test_horizon_bias_removed(self):
        prices = [100.0 + (i % 3) * 0.1 for i in range(40)]
        narrow = tlob_smoothed_label(prices, t=20, horizon=2, k_smooth=2, theta=0.0)
        wide = tlob_smoothed_label(prices, t=20, horizon=2, k_smooth=10, theta=0.0)
        self.assertIsInstance(narrow, int)
        self.assertIsInstance(wide, int)

    def test_multihorizon_keys_match_input(self):
        events = _events([100.0 + 0.1 * i for i in range(200)])
        rows = build_multihorizon_labels(
            events, horizons=(1, 5, 20), k_smooth=lambda h: max(1, h // 2), theta="spread"
        )
        self.assertGreater(len(rows), 0)
        sample = rows[0].horizons
        self.assertEqual(sorted(sample.keys()), [1, 5, 20])
        for payload in sample.values():
            self.assertIn("direction", payload)
            self.assertIn("future_return", payload)
            self.assertIn("future_vol", payload)
            self.assertIn("future_spread", payload)


if __name__ == "__main__":
    unittest.main()
