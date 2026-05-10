import unittest

import torch

from src.quantlab.features_extras import EXTENDED_FEATURE_KEYS
from src.quantlab.sequence import (
    align_feature_label_rows,
    build_sequence_windows,
    walk_forward_index_ranges,
)


def _fake_feature_rows(n, ts_start=1_700_000_000_000, dt_ms=1000):
    rows = []
    for i in range(n):
        row = {key: float(i) for key in EXTENDED_FEATURE_KEYS}
        row["timestamp_ms"] = ts_start + i * dt_ms
        row["symbol"] = "X"
        row["midprice"] = 100.0 + i
        rows.append(row)
    return rows


def _fake_label_rows(n, horizons=(1, 5, 20), ts_start=1_700_000_000_000, dt_ms=1000):
    return [
        {
            "timestamp_ms": ts_start + i * dt_ms,
            "symbol": "X",
            "horizons": {
                str(h): {
                    "direction": (i % 3) - 1,
                    "future_return": 0.0001 * i,
                    "future_vol": 0.0002,
                    "future_spread": 0.5,
                }
                for h in horizons
            },
        }
        for i in range(n)
    ]


class TestSequence(unittest.TestCase):
    def test_align_intersects_by_timestamp(self):
        features = _fake_feature_rows(10)
        labels = _fake_label_rows(8)
        aligned_features, aligned_labels = align_feature_label_rows(features, labels)
        self.assertEqual(len(aligned_features), 8)
        self.assertEqual(len(aligned_labels), 8)

    def test_align_raises_on_no_overlap(self):
        features = _fake_feature_rows(5, ts_start=1)
        labels = _fake_label_rows(5, ts_start=10_000_000)
        with self.assertRaises(ValueError):
            align_feature_label_rows(features, labels)

    def test_build_windows_shape(self):
        features = _fake_feature_rows(20)
        labels = _fake_label_rows(20)
        x, y_by_horizon = build_sequence_windows(features, labels, sequence_length=4, horizons=(1, 5))
        self.assertEqual(x.shape, (17, 4, len(EXTENDED_FEATURE_KEYS)))
        self.assertEqual(set(y_by_horizon.keys()), {1, 5})
        for horizon_key in (1, 5):
            payload = y_by_horizon[horizon_key]
            self.assertEqual(payload["direction"].shape, (17,))
            self.assertEqual(payload["future_vol"].shape, (17,))

    def test_walk_forward_ranges_dont_overlap(self):
        ranges = list(walk_forward_index_ranges(n=300, n_folds=3, embargo=4))
        self.assertEqual(len(ranges), 3)
        for fold in ranges:
            for split in ("train", "val", "test"):
                start, end = fold[split]
                self.assertLess(start, end)
            self.assertGreaterEqual(fold["val"][0], fold["train"][1])
            self.assertGreaterEqual(fold["test"][0], fold["val"][1])


if __name__ == "__main__":
    unittest.main()
