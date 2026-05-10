"""Sliding-window builder and walk-forward index helpers for TLOB-Q."""

from __future__ import annotations

from typing import Iterable

import torch

from .features_extras import EXTENDED_FEATURE_KEYS


def align_feature_label_rows(feature_rows, label_rows):
    features_by_ts = {int(row["timestamp_ms"]): row for row in feature_rows}
    labels_by_ts = {int(row["timestamp_ms"]): row for row in label_rows}
    common = sorted(set(features_by_ts) & set(labels_by_ts))
    if not common:
        raise ValueError("feature and label rows have no overlapping timestamps.")
    return (
        [features_by_ts[ts] for ts in common],
        [labels_by_ts[ts] for ts in common],
    )


def _feature_matrix(feature_rows, feature_keys: Iterable[str]) -> torch.Tensor:
    feature_keys = list(feature_keys)
    matrix = [[float(row.get(key, 0.0)) for key in feature_keys] for row in feature_rows]
    if not matrix:
        raise ValueError("feature_rows must not be empty.")
    return torch.tensor(matrix, dtype=torch.float32)


def _label_payload(label_row, horizon: int) -> dict[str, float]:
    raw = label_row.get("horizons", {})
    if isinstance(raw, dict) and str(horizon) in raw:
        return raw[str(horizon)]
    if isinstance(raw, dict) and horizon in raw:
        return raw[horizon]
    raise KeyError(f"label row missing horizon {horizon}")


def build_sequence_windows(
    feature_rows,
    label_rows,
    sequence_length: int,
    horizons: Iterable[int],
    feature_keys: Iterable[str] = EXTENDED_FEATURE_KEYS,
):
    sequence_length = int(sequence_length)
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")
    feature_keys = tuple(feature_keys)
    horizons = tuple(int(h) for h in horizons)
    matrix = _feature_matrix(feature_rows, feature_keys)
    if matrix.size(0) < sequence_length:
        raise ValueError("not enough rows for the requested sequence_length.")
    if not torch.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values.")
    n_windows = matrix.size(0) - sequence_length + 1
    windows = torch.stack(
        [matrix[start : start + sequence_length] for start in range(n_windows)],
        dim=0,
    )
    labels_aligned = label_rows[sequence_length - 1 : sequence_length - 1 + n_windows]
    y_by_horizon: dict[int, dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        dir_values: list[int] = []
        vol_values: list[float] = []
        spread_values: list[float] = []
        ret_values: list[float] = []
        for label_row in labels_aligned:
            payload = _label_payload(label_row, horizon)
            dir_values.append(int(payload["direction"]))
            vol_values.append(float(payload["future_vol"]))
            spread_values.append(float(payload["future_spread"]))
            ret_values.append(float(payload["future_return"]))
        y_by_horizon[horizon] = {
            "direction": torch.tensor(dir_values, dtype=torch.long),
            "future_vol": torch.tensor(vol_values, dtype=torch.float32),
            "future_spread": torch.tensor(spread_values, dtype=torch.float32),
            "future_return": torch.tensor(ret_values, dtype=torch.float32),
        }
    return windows, y_by_horizon


def walk_forward_index_ranges(n: int, n_folds: int, embargo: int):
    n = int(n)
    n_folds = max(1, int(n_folds))
    embargo = max(0, int(embargo))
    fold_size = n // n_folds
    if fold_size < 10:
        raise ValueError("not enough rows for the requested fold count.")
    train_frac, val_frac = 0.6, 0.1
    for fold_index in range(n_folds):
        start = fold_index * fold_size
        end = start + fold_size if fold_index < n_folds - 1 else n
        size = end - start
        train_end = start + int(size * train_frac)
        val_start = min(end, train_end + embargo)
        val_end = min(end, val_start + int(size * val_frac))
        test_start = min(end, val_end + embargo)
        test_end = end
        if val_start >= val_end or test_start >= test_end:
            raise ValueError("walk-forward fold collapsed; reduce embargo or n_folds.")
        yield {
            "fold": fold_index,
            "train": (start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }
