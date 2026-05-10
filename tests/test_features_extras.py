import unittest

from src.quantlab.core import MarketEvent
from src.quantlab.features_extras import (
    EXTENDED_FEATURE_KEYS,
    build_extended_feature_rows,
)


def _make_event(t, mid=100.0, half_spread=0.5, bid_size=10.0, ask_size=10.0,
                last_size=1.0, side="buy", depths=None, funding=0.0, liq=0.0):
    extras = {"funding": funding, "liquidation_intensity": liq}
    if depths is not None:
        extras["bid_depth"] = depths["bid"]
        extras["ask_depth"] = depths["ask"]
    return MarketEvent(
        timestamp_ms=t,
        symbol="X",
        event_type="book",
        bid_price=mid - half_spread,
        ask_price=mid + half_spread,
        bid_size=bid_size,
        ask_size=ask_size,
        last_price=mid,
        last_size=last_size,
        side=side,
        source="test",
        extras=extras,
    )


class TestFeaturesExtras(unittest.TestCase):
    def test_feature_key_count_is_32(self):
        self.assertEqual(len(EXTENDED_FEATURE_KEYS), 32)

    def test_build_empty_returns_empty(self):
        self.assertEqual(build_extended_feature_rows([]), [])

    def test_row_count_is_n_minus_one(self):
        events = [_make_event(t * 1000) for t in range(10)]
        rows = build_extended_feature_rows(events)
        self.assertEqual(len(rows), 9)

    def test_each_row_has_all_keys(self):
        events = [_make_event(t * 1000) for t in range(5)]
        rows = build_extended_feature_rows(events)
        for row in rows:
            for key in EXTENDED_FEATURE_KEYS:
                self.assertIn(key, row, f"missing key {key}")
                self.assertTrue(
                    isinstance(row[key], (int, float)),
                    f"key {key} value type {type(row[key])}",
                )
                value = row[key]
                self.assertTrue(
                    abs(value) < 1e6 and value == value,
                    f"key {key} value {value} not finite-reasonable",
                )

    def test_symmetric_book_zero_imbalance(self):
        events = [_make_event(t * 1000, bid_size=5.0, ask_size=5.0) for t in range(5)]
        rows = build_extended_feature_rows(events)
        for row in rows:
            self.assertAlmostEqual(row["book_imbalance"], 0.0, places=6)

    def test_buy_side_run_length(self):
        events = [_make_event(t * 1000, side="buy") for t in range(6)]
        rows = build_extended_feature_rows(events)
        self.assertGreater(rows[-1]["last_trade_aggressor_persistence"], 1.0)

    def test_multilevel_ofi_keys(self):
        depths = {"bid": [10.0, 5.0, 3.0, 2.0, 1.0], "ask": [10.0, 5.0, 3.0, 2.0, 1.0]}
        events = [_make_event(t * 1000, depths=depths) for t in range(5)]
        rows = build_extended_feature_rows(events)
        for key in ("ofi_level_2", "ofi_level_3", "ofi_level_5"):
            self.assertIn(key, rows[0])

    def test_zscore_warmup_returns_zero(self):
        # With only 2-3 prior samples, _zscore should return 0.0 (warm-up gate).
        events = [_make_event(t * 1000, last_size=float(t + 1)) for t in range(3)]
        rows = build_extended_feature_rows(events)
        self.assertEqual(rows[0]["signed_volume_zscore"], 0.0)

    def test_directional_run_length_differs_from_aggressor_persistence(self):
        # last_size=0 → signed_volume=0 (no directional tick), but side="buy" still extends aggressor run.
        events = [_make_event(t * 1000, last_size=0.0, side="buy") for t in range(6)]
        rows = build_extended_feature_rows(events)
        last_row = rows[-1]
        self.assertEqual(last_row["directional_run_length"], 0.0)
        self.assertGreater(last_row["last_trade_aggressor_persistence"], 1.0)


if __name__ == "__main__":
    unittest.main()
