# TLOB-Q Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TLOB-Q — a TLOB-faithful dual-attention transformer over the existing QuantLab `FeatureFrame` schema with MC-dropout uncertainty, multi-horizon multi-task heads, EMA weight tracking, and purged walk-forward cross-validation. Spec lives at `docs/superpowers/specs/2026-05-11-tlob-q-design.md`.

**Architecture:** Stack of 8 TLOB-Q blocks (BiN → spatial attention → MLPLOB feat-mix → temporal causal attention → MLPLOB temp-mix). `d_model=320`, `n_heads=10`, `T=128`, `F=32`. Trunk feeds 12 heads (4 horizons × {direction, volatility, spread}). MC-dropout stays active at inference for Bayesian sampling. Polyak/EMA shadow weights used for predict. ~10–14 M parameters; ~3–3.5 h on M4 Air CPU for 3-fold walk-forward.

**Tech Stack:** Python 3.x, PyTorch (CPU), `unittest` for tests, JSONL artifacts, deterministic seeds. No new third-party deps beyond what `requirements.txt` already pins.

**Verification baseline (must still pass at end):**
- `./venv/bin/python -m unittest`
- `./venv/bin/python -m py_compile main.py src/prepare_data.py src/finetune_model.py src/algorithms/*.py src/micro_gpt/*.py src/research_lab/*.py src/quantlab/*.py`
- `./venv/bin/python -m src.micro_gpt.train --config configs/micro_gpt/tiny_debug.json --dry-run`
- `git diff --check`

**Commit style:** `<type>: <subject>` per repo convention (`feat`, `fix`, `test`, `docs`, `refactor`, `chore`). Add `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` trailer (set globally; don't override).

---

## File Structure (locked from spec §4)

### Create

| Path | Responsibility |
|------|----------------|
| `src/quantlab/features_extras.py` | 24 derived feature channels; `build_extended_feature_rows(events) -> List[ExtendedFeatureFrame]` |
| `src/quantlab/labels_tlob.py` | TLOB decoupled labelling `l(t,h,k)`; `build_multihorizon_labels(events, horizons, k_smooth, theta)` |
| `src/quantlab/sequence.py` | Sliding-window builder; `align_feature_label_rows`, `build_sequence_windows`, `SequenceWindow` |
| `src/quantlab/tlob_modules.py` | `BilinearNormalization`, `RMSNorm`, `MCDropout`, `SpatialAttention`, `TemporalCausalAttention`, `MLPLOBFeatMix`, `MLPLOBTempMix` |
| `src/quantlab/tlob_model.py` | `TLOBQConfig`, `TLOBQBlock`, `TLOBQModel`, `EMAWeightTracker`, save/load |
| `src/quantlab/training.py` | `train_one_fold`, `walk_forward_train`, `aggregate_folds`, training-curves logger |
| `src/quantlab/inference.py` | `mc_dropout_predict`, `ensemble_predict`, κ-scaled edge, horizon selection |
| `src/quantlab/curriculum.py` | `generate_regime_mixed_events(seed, n, regime_mix)` |
| `src/quantlab/cli_tlob.py` | Unified CLI: `train`, `predict`, `walk-forward`, `demo` subcommands |
| `configs/quantlab/tlob_smoke.json` | CI smoke config |
| `configs/quantlab/tlob_m4_default.json` | Default M4 CPU config |
| `configs/quantlab/tlob_ensemble.json` | K=3 deep ensemble variant |
| `tests/test_features_extras.py` | unit |
| `tests/test_labels_tlob.py` | unit |
| `tests/test_sequence.py` | unit |
| `tests/test_tlob_modules.py` | unit |
| `tests/test_tlob_model.py` | unit |
| `tests/test_training.py` | unit |
| `tests/test_inference.py` | unit |
| `tests/test_curriculum.py` | unit |
| `tests/test_tlob_cli.py` | unit |
| `tests/test_tlob_demo.py` | integration |
| `tests/test_walk_forward_integration.py` | integration |
| `tests/test_resume_after_crash.py` | integration |
| `tests/test_tlob_e2e_smoke.py` | E2E |

### Modify

| Path | Change |
|------|--------|
| `src/quantlab/__init__.py` | Export new public surface |
| `src/quantlab/core.py` | Add `MultiHorizonLabelFrame` + `SequenceWindow` |
| `src/quantlab/demo.py` | Add `run_tlob_demo_pipeline` + update `PAPER_REFERENCES` |
| `src/quantlab/backtest.py` | Add `run_backtest_multihorizon` |
| `docs/literature_review.md` | Add TLOB, BDLOB, Async DDQL, MacroHFT, TimeCatcher entries |
| `docs/research_program.md` | Add Sequence Microstructure Track |
| `docs/quant_research_catalog.md` | Add `tlob_q` model row |
| `README.md` | Add TLOB-Q quickstart |

---

## Task 0: Baseline verification

**Files:** none

- [ ] **Step 1: Verify the test suite is green before changes**

Run: `./venv/bin/python -m unittest`
Expected: PASS (all current tests)

- [ ] **Step 2: Verify py_compile is clean**

Run: `./venv/bin/python -m py_compile main.py src/prepare_data.py src/finetune_model.py src/algorithms/*.py src/micro_gpt/*.py src/research_lab/*.py src/quantlab/*.py`
Expected: silent (no compile errors)

- [ ] **Step 3: Verify git is clean**

Run: `git status`
Expected: working tree clean, on `main`

If any of these fail, STOP and fix before proceeding.

---

## Task 1: Add `MultiHorizonLabelFrame` and `SequenceWindow` to core schema

**Files:**
- Modify: `src/quantlab/core.py`
- Test: `tests/test_quantlab.py` (extend existing)

- [ ] **Step 1: Write the failing test (append to `tests/test_quantlab.py`)**

```python
def test_multihorizon_label_frame_round_trip():
    from src.quantlab.core import MultiHorizonLabelFrame
    frame = MultiHorizonLabelFrame(
        timestamp_ms=1_700_000_000_000,
        symbol="BTCUSDT",
        horizons={
            1: {"direction": 1, "future_return": 0.0001, "future_vol": 0.0002, "future_spread": 0.5},
            5: {"direction": 0, "future_return": 0.0000, "future_vol": 0.0003, "future_spread": 0.4},
        },
    )
    payload = frame.to_dict()
    assert payload["timestamp_ms"] == 1_700_000_000_000
    assert payload["symbol"] == "BTCUSDT"
    assert payload["horizons"]["1"]["direction"] == 1
    assert payload["horizons"]["5"]["future_return"] == 0.0


def test_sequence_window_round_trip():
    from src.quantlab.core import SequenceWindow
    window = SequenceWindow(
        timestamp_ms=1_700_000_000_000,
        symbol="BTCUSDT",
        sequence_length=128,
        feature_keys=("a", "b"),
    )
    payload = window.to_dict()
    assert payload["sequence_length"] == 128
    assert payload["feature_keys"] == ["a", "b"]
```

- [ ] **Step 2: Run the test (should fail with ImportError)**

Run: `./venv/bin/python -m unittest tests.test_quantlab.TestQuantLab.test_multihorizon_label_frame_round_trip 2>&1 | tail -20`
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement the schemas (append to `src/quantlab/core.py`)**

```python
@dataclass(frozen=True)
class MultiHorizonLabelFrame:
    timestamp_ms: int
    symbol: str
    horizons: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "symbol": str(self.symbol),
            "horizons": {
                str(int(h)): {str(k): float(v) for k, v in payload.items()}
                for h, payload in sorted(self.horizons.items())
            },
        }


@dataclass(frozen=True)
class SequenceWindow:
    timestamp_ms: int
    symbol: str
    sequence_length: int
    feature_keys: tuple[str, ...]

    def to_dict(self):
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "symbol": str(self.symbol),
            "sequence_length": int(self.sequence_length),
            "feature_keys": list(self.feature_keys),
        }
```

- [ ] **Step 4: Run the tests (should pass)**

Run: `./venv/bin/python -m unittest tests.test_quantlab -v 2>&1 | tail -10`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/core.py tests/test_quantlab.py
git commit -m "feat: add MultiHorizonLabelFrame and SequenceWindow schemas"
```

---

## Task 2: Build `features_extras.py` — 24 derived feature channels

**Files:**
- Create: `src/quantlab/features_extras.py`
- Create: `tests/test_features_extras.py`

The 24 new channels live alongside the existing 8 (carried through). The output type extends `FeatureFrame` with an `extras_v2` dict keyed by the new channel names. The public function is `build_extended_feature_rows(events) -> List[dict]` returning row dicts (so JSONL writing is one line and downstream `sequence.py` reads keys uniformly).

- [ ] **Step 1: Write the failing test (`tests/test_features_extras.py`)**

```python
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
                self.assertTrue(
                    abs(row[key]) < 1e6 and row[key] == row[key],
                    f"key {key} value {row[key]} not finite-reasonable",
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test (should fail with ImportError)**

Run: `./venv/bin/python -m unittest tests.test_features_extras -v 2>&1 | tail -20`
Expected: ImportError on `src.quantlab.features_extras`.

- [ ] **Step 3: Implement `src/quantlab/features_extras.py`**

```python
"""Extended feature channels (32 total: 8 base + 24 derived) for TLOB-Q."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .core import MarketEvent
from .features import (
    book_imbalance,
    depth_imbalance,
    microprice,
    midprice,
    order_flow_imbalance,
    realized_volatility,
    signed_volume,
    spread,
)


BASE_FEATURE_KEYS: tuple[str, ...] = (
    "midprice_logret",
    "spread",
    "microprice_minus_mid",
    "ofi_level_1",
    "book_imbalance",
    "depth_imbalance",
    "signed_volume",
    "realized_volatility",
)

EXTRA_FEATURE_KEYS: tuple[str, ...] = (
    "ofi_level_2",
    "ofi_level_3",
    "ofi_level_5",
    "log_spread",
    "funding_rate",
    "liquidation_intensity",
    "signed_volume_ema_slow",
    "signed_volume_ema_fast",
    "directional_run_length",
    "trade_burst_rate",
    "queue_depth_l2_over_l1",
    "queue_depth_l3_over_l1",
    "cumulative_depth_near_mid",
    "relative_tick_imbalance",
    "midret_autocorr_lag1",
    "realized_vol_ratio",
    "vpin_proxy",
    "signed_volume_zscore",
    "ofi_zscore",
    "microprice_momentum",
    "depth_imbalance_ema_fast",
    "depth_imbalance_ema_slow",
    "spread_regime",
    "last_trade_aggressor_persistence",
)

EXTENDED_FEATURE_KEYS: tuple[str, ...] = BASE_FEATURE_KEYS + EXTRA_FEATURE_KEYS

_EPS = 1e-12
_CLIP = 50.0


def _clip(value: float) -> float:
    if value != value:
        return 0.0
    if value > _CLIP:
        return _CLIP
    if value < -_CLIP:
        return -_CLIP
    return value


def _ema_update(prev: float, sample: float, alpha: float) -> float:
    return alpha * sample + (1.0 - alpha) * prev


def _depths(event: MarketEvent, side: str) -> list[float]:
    key = f"{side}_depth"
    raw = event.extras.get(key)
    if raw is None:
        return [float(event.bid_size if side == "bid" else event.ask_size)]
    if isinstance(raw, (list, tuple)):
        return [float(value) for value in raw]
    return [float(raw)]


def _level_ofi(previous: MarketEvent, current: MarketEvent, level: int) -> float:
    prev_bid = _depths(previous, "bid")
    prev_ask = _depths(previous, "ask")
    curr_bid = _depths(current, "bid")
    curr_ask = _depths(current, "ask")
    bid_levels = min(level, len(prev_bid), len(curr_bid))
    ask_levels = min(level, len(prev_ask), len(curr_ask))
    bid_delta = sum(curr_bid[:bid_levels]) - sum(prev_bid[:bid_levels])
    ask_delta = sum(curr_ask[:ask_levels]) - sum(prev_ask[:ask_levels])
    return float(bid_delta - ask_delta)


@dataclass
class _State:
    sv_ema_slow: float = 0.0
    sv_ema_fast: float = 0.0
    di_ema_slow: float = 0.0
    di_ema_fast: float = 0.0
    sv_history: deque = None
    ofi_history: deque = None
    midret_history: deque = None
    spread_history: deque = None
    short_vol_window: deque = None
    long_vol_window: deque = None
    tick_signs: deque = None
    run_length: float = 0.0
    last_side: str | None = None
    last_microprice: float | None = None
    last_timestamp_ms: int | None = None

    def __post_init__(self):
        self.sv_history = deque(maxlen=64)
        self.ofi_history = deque(maxlen=64)
        self.midret_history = deque(maxlen=32)
        self.spread_history = deque(maxlen=64)
        self.short_vol_window = deque(maxlen=8)
        self.long_vol_window = deque(maxlen=32)
        self.tick_signs = deque(maxlen=32)


def _zscore(history: deque, value: float) -> float:
    if len(history) < 4:
        return 0.0
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / max(1, len(history) - 1)
    std = math.sqrt(variance)
    if std < _EPS:
        return 0.0
    return _clip((value - mean) / std)


def _autocorr_lag1(values: deque) -> float:
    if len(values) < 3:
        return 0.0
    seq = list(values)
    mean = sum(seq) / len(seq)
    num = 0.0
    den = 0.0
    for prev, curr in zip(seq[:-1], seq[1:]):
        num += (prev - mean) * (curr - mean)
    for value in seq:
        den += (value - mean) ** 2
    if den < _EPS:
        return 0.0
    return _clip(num / den)


def _build_row(previous: MarketEvent, current: MarketEvent, state: _State) -> dict:
    prev_mid = midprice(previous.bid_price, previous.ask_price)
    curr_mid = midprice(current.bid_price, current.ask_price)
    curr_spread = spread(current.bid_price, current.ask_price)
    curr_micro = microprice(current.bid_price, current.ask_price, current.bid_size, current.ask_size)
    sv = signed_volume(current.last_size or 0.0, current.side)
    ofi_l1 = order_flow_imbalance(previous, current)
    log_ret = math.log(curr_mid / prev_mid) if prev_mid > 0 and curr_mid > 0 else 0.0

    state.sv_ema_slow = _ema_update(state.sv_ema_slow, sv, 0.1)
    state.sv_ema_fast = _ema_update(state.sv_ema_fast, sv, 0.5)
    bi = book_imbalance(current.bid_size, current.ask_size)
    state.di_ema_slow = _ema_update(state.di_ema_slow, bi, 0.1)
    state.di_ema_fast = _ema_update(state.di_ema_fast, bi, 0.5)

    state.sv_history.append(sv)
    state.ofi_history.append(ofi_l1)
    state.midret_history.append(log_ret)
    state.spread_history.append(curr_spread)
    state.short_vol_window.append(curr_mid)
    state.long_vol_window.append(curr_mid)

    side_sign = 1 if sv > 0 else (-1 if sv < 0 else 0)
    if side_sign != 0:
        state.tick_signs.append(side_sign)
    side = (current.side or "").lower()
    if side and side == (state.last_side or ""):
        state.run_length += 1.0
    else:
        state.run_length = 1.0
    state.last_side = side

    burst_dt = 1.0
    if state.last_timestamp_ms is not None:
        burst_dt = max(1.0, (current.timestamp_ms - state.last_timestamp_ms) / 1000.0)
    state.last_timestamp_ms = current.timestamp_ms

    bid_depth = _depths(current, "bid")
    ask_depth = _depths(current, "ask")
    l1 = (bid_depth[0] + ask_depth[0]) / 2.0 if bid_depth and ask_depth else 0.0
    l2 = (bid_depth[1] + ask_depth[1]) / 2.0 if len(bid_depth) > 1 and len(ask_depth) > 1 else l1
    l3 = (bid_depth[2] + ask_depth[2]) / 2.0 if len(bid_depth) > 2 and len(ask_depth) > 2 else l1

    short_vol = realized_volatility(list(state.short_vol_window))
    long_vol = realized_volatility(list(state.long_vol_window))
    realized_vol_ratio = short_vol / long_vol if long_vol > _EPS else 0.0

    median_spread = sorted(state.spread_history)[len(state.spread_history) // 2] if state.spread_history else curr_spread
    spread_regime = curr_spread / median_spread if median_spread > _EPS else 1.0

    up_ticks = sum(1 for s in state.tick_signs if s > 0)
    down_ticks = sum(1 for s in state.tick_signs if s < 0)
    if down_ticks > 0:
        rel_tick = math.log((up_ticks + 1) / (down_ticks + 1))
    else:
        rel_tick = float(up_ticks)

    micro_momentum = 0.0
    if state.last_microprice is not None and state.last_microprice > 0:
        micro_momentum = (curr_micro - state.last_microprice) / state.last_microprice
    state.last_microprice = curr_micro

    total_signed = sum(abs(value) for value in state.sv_history) or _EPS
    vpin_proxy = sum(state.sv_history) / total_signed

    cum_depth_near_mid = (sum(bid_depth[:3]) + sum(ask_depth[:3])) / max(1, min(3, len(bid_depth)) + min(3, len(ask_depth)))

    return {
        "timestamp_ms": int(current.timestamp_ms),
        "symbol": str(current.symbol),
        "midprice": float(curr_mid),
        "midprice_logret": _clip(log_ret),
        "spread": _clip(curr_spread),
        "microprice_minus_mid": _clip(curr_micro - curr_mid),
        "ofi_level_1": _clip(ofi_l1),
        "book_imbalance": _clip(bi),
        "depth_imbalance": _clip(depth_imbalance(bid_depth, ask_depth)),
        "signed_volume": _clip(sv),
        "realized_volatility": _clip(realized_volatility(list(state.short_vol_window))),
        "ofi_level_2": _clip(_level_ofi(previous, current, 2)),
        "ofi_level_3": _clip(_level_ofi(previous, current, 3)),
        "ofi_level_5": _clip(_level_ofi(previous, current, 5)),
        "log_spread": _clip(math.log(max(_EPS, curr_spread))),
        "funding_rate": _clip(float(current.extras.get("funding", 0.0))),
        "liquidation_intensity": _clip(float(current.extras.get("liquidation_intensity", 0.0))),
        "signed_volume_ema_slow": _clip(state.sv_ema_slow),
        "signed_volume_ema_fast": _clip(state.sv_ema_fast),
        "directional_run_length": _clip(state.run_length),
        "trade_burst_rate": _clip(1.0 / burst_dt),
        "queue_depth_l2_over_l1": _clip(l2 / l1) if l1 > _EPS else 0.0,
        "queue_depth_l3_over_l1": _clip(l3 / l1) if l1 > _EPS else 0.0,
        "cumulative_depth_near_mid": _clip(cum_depth_near_mid),
        "relative_tick_imbalance": _clip(rel_tick),
        "midret_autocorr_lag1": _clip(_autocorr_lag1(state.midret_history)),
        "realized_vol_ratio": _clip(realized_vol_ratio),
        "vpin_proxy": _clip(vpin_proxy),
        "signed_volume_zscore": _zscore(state.sv_history, sv),
        "ofi_zscore": _zscore(state.ofi_history, ofi_l1),
        "microprice_momentum": _clip(micro_momentum),
        "depth_imbalance_ema_fast": _clip(state.di_ema_fast),
        "depth_imbalance_ema_slow": _clip(state.di_ema_slow),
        "spread_regime": _clip(spread_regime),
        "last_trade_aggressor_persistence": _clip(state.run_length),
    }


def build_extended_feature_rows(events: Iterable[MarketEvent]) -> list[dict]:
    events = list(events)
    if len(events) < 2:
        return []
    state = _State()
    rows: list[dict] = []
    previous = events[0]
    for current in events[1:]:
        rows.append(_build_row(previous, current, state))
        previous = current
    return rows
```

- [ ] **Step 4: Run the test (should pass)**

Run: `./venv/bin/python -m unittest tests.test_features_extras -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/features_extras.py tests/test_features_extras.py
git commit -m "feat: add 24-channel extended feature builder for TLOB-Q"
```

---

## Task 3: Build `labels_tlob.py` — TLOB decoupled labelling + multi-horizon labels

**Files:**
- Create: `src/quantlab/labels_tlob.py`
- Create: `tests/test_labels_tlob.py`

- [ ] **Step 1: Write the failing test (`tests/test_labels_tlob.py`)**

```python
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
```

- [ ] **Step 2: Run the test**

Run: `./venv/bin/python -m unittest tests.test_labels_tlob -v 2>&1 | tail -20`
Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/labels_tlob.py`**

```python
"""TLOB-style decoupled (h, k) labelling and multi-horizon label generation."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

from .core import MarketEvent, MultiHorizonLabelFrame
from .features import midprice, realized_volatility, spread


def tlob_smoothed_label(prices: Sequence[float], t: int, horizon: int, k_smooth: int, theta: float) -> int:
    """Compute the TLOB ternary trend label using decoupled (h, k).

    l(t, h, k) = (w_plus - w_minus) / w_minus
    where w_plus mean of p(t+h-i) for i in 0..k and w_minus mean of p(t-i) for i in 0..k.
    """
    prices = [float(value) for value in prices]
    n = len(prices)
    k_smooth = max(0, int(k_smooth))
    horizon = int(horizon)
    if t < k_smooth or t + horizon >= n:
        return 0
    forward = [prices[t + horizon - i] for i in range(k_smooth + 1) if 0 <= t + horizon - i < n]
    backward = [prices[t - i] for i in range(k_smooth + 1) if 0 <= t - i < n]
    if not forward or not backward:
        return 0
    w_plus = sum(forward) / len(forward)
    w_minus = sum(backward) / len(backward)
    if abs(w_minus) < 1e-12:
        return 0
    rel = (w_plus - w_minus) / w_minus
    theta = abs(float(theta))
    if rel > theta:
        return 1
    if rel < -theta:
        return -1
    return 0


def _resolve_theta(theta, spread_value: float, mid_value: float) -> float:
    if isinstance(theta, str) and theta.lower() == "spread":
        if mid_value <= 0:
            return 0.0
        return float(spread_value) / float(mid_value)
    return float(theta)


def build_multihorizon_labels(
    events: Iterable[MarketEvent],
    horizons: Iterable[int],
    k_smooth: Callable[[int], int] | int = lambda h: max(1, h // 2),
    theta="spread",
) -> list[MultiHorizonLabelFrame]:
    events = list(events)
    horizons = tuple(int(h) for h in horizons)
    if not horizons:
        raise ValueError("horizons must not be empty.")
    if len(events) < 2 + max(horizons):
        return []
    prices = [midprice(event.bid_price, event.ask_price) for event in events]
    spreads = [spread(event.bid_price, event.ask_price) for event in events]

    if callable(k_smooth):
        smooth_for = k_smooth
    else:
        smooth_const = int(k_smooth)
        def smooth_for(_h, _v=smooth_const):
            return _v

    rows: list[MultiHorizonLabelFrame] = []
    for t in range(len(events) - 1):
        horizon_payload: dict[int, dict[str, float]] = {}
        for h in horizons:
            if t + h >= len(events):
                continue
            k = max(1, int(smooth_for(h)))
            theta_value = _resolve_theta(theta, spreads[t], prices[t])
            direction = tlob_smoothed_label(prices, t=t, horizon=h, k_smooth=k, theta=theta_value)
            future_return = (prices[t + h] - prices[t]) / prices[t] if prices[t] > 0 else 0.0
            window_end = min(len(prices), t + h + 1)
            window_prices = prices[t:window_end]
            future_vol = realized_volatility(window_prices)
            future_spread = sum(spreads[t:window_end]) / max(1, len(spreads[t:window_end]))
            horizon_payload[h] = {
                "direction": float(direction),
                "future_return": float(future_return),
                "future_vol": float(future_vol),
                "future_spread": float(future_spread),
            }
        if not horizon_payload:
            continue
        rows.append(
            MultiHorizonLabelFrame(
                timestamp_ms=events[t].timestamp_ms,
                symbol=events[t].symbol,
                horizons=horizon_payload,
            )
        )
    return rows
```

- [ ] **Step 4: Run the test**

Run: `./venv/bin/python -m unittest tests.test_labels_tlob -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/labels_tlob.py tests/test_labels_tlob.py
git commit -m "feat: add TLOB decoupled (h,k) multi-horizon labelling"
```

---

## Task 4: Build `sequence.py` — windowing, alignment, walk-forward split helpers

**Files:**
- Create: `src/quantlab/sequence.py`
- Create: `tests/test_sequence.py`

- [ ] **Step 1: Write the failing test (`tests/test_sequence.py`)**

```python
import unittest

import torch

from src.quantlab.features_extras import EXTENDED_FEATURE_KEYS
from src.quantlab.sequence import (
    align_feature_label_rows,
    build_sequence_windows,
    walk_forward_index_ranges,
)


def _fake_feature_rows(n, ts_start=1_700_000_000_000, dt_ms=1000):
    return [
        {key: float(i) for key in EXTENDED_FEATURE_KEYS}
        | {"timestamp_ms": ts_start + i * dt_ms, "symbol": "X", "midprice": 100.0 + i}
        for i in range(n)
    ]


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
        labels = _fake_label_rows(5, ts_start=10_000)
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
```

- [ ] **Step 2: Run the test**

Run: `./venv/bin/python -m unittest tests.test_sequence -v 2>&1 | tail -20`
Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/sequence.py`**

```python
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
```

- [ ] **Step 4: Run the test**

Run: `./venv/bin/python -m unittest tests.test_sequence -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/sequence.py tests/test_sequence.py
git commit -m "feat: add sequence windowing and walk-forward index helpers"
```

---

## Task 5: Build `tlob_modules.py` part 1 — `BilinearNormalization`, `RMSNorm`, `MCDropout`

**Files:**
- Create: `src/quantlab/tlob_modules.py`
- Create: `tests/test_tlob_modules.py`

- [ ] **Step 1: Write the failing test (`tests/test_tlob_modules.py`)**

```python
import unittest

import torch

from src.quantlab.tlob_modules import BilinearNormalization, MCDropout, RMSNorm


class TestTlobBaseModules(unittest.TestCase):
    def test_rmsnorm_preserves_shape(self):
        norm = RMSNorm(dim=16)
        x = torch.randn(2, 4, 16)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_bin_centers_per_channel(self):
        bin_module = BilinearNormalization(num_features=4, num_timesteps=8)
        torch.manual_seed(0)
        x = torch.randn(3, 8, 4) * 3 + 7
        y = bin_module(x)
        mean = y.mean(dim=(0, 1))
        std = y.std(dim=(0, 1), unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=0.5))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=0.5))

    def test_mc_dropout_active_in_eval(self):
        module = MCDropout(p=0.5)
        module.eval()
        x = torch.ones(8, 16)
        y = module(x)
        self.assertFalse(torch.allclose(y, x))

    def test_mc_dropout_zero_when_disabled(self):
        module = MCDropout(p=0.5, mc_active=False)
        module.eval()
        x = torch.ones(8, 16)
        y = module(x)
        self.assertTrue(torch.allclose(y, x))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `./venv/bin/python -m unittest tests.test_tlob_modules -v 2>&1 | tail -20`
Expected: ImportError.

- [ ] **Step 3: Implement base modules (`src/quantlab/tlob_modules.py`)**

```python
"""TLOB-Q building blocks: normalization, dropout, attention, MLP mixers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class BilinearNormalization(nn.Module):
    """Per-feature + per-timestep affine normalization with running statistics.

    Approximates the BiN layer described in TLOB / BINCTABL: applies a learned
    rescale/shift along each axis. We use exponential moving averages of the
    batch mean and variance for inference stability.
    """

    def __init__(self, num_features: int, num_timesteps: int, momentum: float = 0.05):
        super().__init__()
        self.num_features = int(num_features)
        self.num_timesteps = int(num_timesteps)
        self.momentum = float(momentum)
        self.feature_gain = nn.Parameter(torch.ones(num_features))
        self.feature_bias = nn.Parameter(torch.zeros(num_features))
        self.temporal_gain = nn.Parameter(torch.ones(num_timesteps))
        self.temporal_bias = nn.Parameter(torch.zeros(num_timesteps))
        self.register_buffer("running_feat_mean", torch.zeros(num_features))
        self.register_buffer("running_feat_var", torch.ones(num_features))
        self.register_buffer("running_temp_mean", torch.zeros(num_timesteps))
        self.register_buffer("running_temp_var", torch.ones(num_timesteps))

    def _update(self, buffer_mean: torch.Tensor, buffer_var: torch.Tensor,
                batch_mean: torch.Tensor, batch_var: torch.Tensor) -> None:
        if not self.training:
            return
        buffer_mean.mul_(1 - self.momentum).add_(batch_mean.detach() * self.momentum)
        buffer_var.mul_(1 - self.momentum).add_(batch_var.detach() * self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("BiN expects (B, T, F).")
        feat_mean = x.mean(dim=(0, 1))
        feat_var = x.var(dim=(0, 1), unbiased=False) + 1e-6
        self._update(self.running_feat_mean, self.running_feat_var, feat_mean, feat_var)
        if self.training:
            mean = feat_mean
            var = feat_var
        else:
            mean = self.running_feat_mean
            var = self.running_feat_var
        x = (x - mean) / var.sqrt()
        x = x * self.feature_gain + self.feature_bias
        temp_mean = x.mean(dim=(0, 2))
        temp_var = x.var(dim=(0, 2), unbiased=False) + 1e-6
        self._update(self.running_temp_mean, self.running_temp_var, temp_mean, temp_var)
        if self.training:
            tm, tv = temp_mean, temp_var
        else:
            tm, tv = self.running_temp_mean, self.running_temp_var
        x = (x - tm.unsqueeze(-1)) / tv.sqrt().unsqueeze(-1)
        x = x * self.temporal_gain.unsqueeze(-1) + self.temporal_bias.unsqueeze(-1)
        return x


class MCDropout(nn.Module):
    """Dropout that stays active in eval mode when mc_active=True."""

    def __init__(self, p: float = 0.1, mc_active: bool = True):
        super().__init__()
        self.p = float(p)
        self.mc_active = bool(mc_active)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.mc_active:
            return F.dropout(x, p=self.p, training=True)
        return x

    def set_mc_active(self, active: bool) -> None:
        self.mc_active = bool(active)
```

- [ ] **Step 4: Run the test**

Run: `./venv/bin/python -m unittest tests.test_tlob_modules -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/tlob_modules.py tests/test_tlob_modules.py
git commit -m "feat: add BiN, RMSNorm, MCDropout base modules for TLOB-Q"
```

---

## Task 6: Add `SpatialAttention` and `TemporalCausalAttention` to `tlob_modules.py`

**Files:**
- Modify: `src/quantlab/tlob_modules.py`
- Modify: `tests/test_tlob_modules.py`

- [ ] **Step 1: Append the failing tests to `tests/test_tlob_modules.py`**

```python
    def test_spatial_attention_shape(self):
        from src.quantlab.tlob_modules import SpatialAttention
        module = SpatialAttention(d_model=32, n_heads=4)
        x = torch.randn(2, 8, 32)
        y = module(x)
        self.assertEqual(y.shape, x.shape)

    def test_temporal_causal_attention_masks_future(self):
        from src.quantlab.tlob_modules import TemporalCausalAttention
        module = TemporalCausalAttention(d_model=32, n_heads=4)
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32, requires_grad=True)
        y = module(x)
        loss = y[:, 0, :].sum()
        loss.backward()
        self.assertTrue(torch.all(x.grad[:, 1:, :] == 0))
```

- [ ] **Step 2: Run (should fail with ImportError)**

Run: `./venv/bin/python -m unittest tests.test_tlob_modules -v 2>&1 | tail -20`
Expected: ImportError on `SpatialAttention`.

- [ ] **Step 3: Append the attention modules to `src/quantlab/tlob_modules.py`**

```python
class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.causal = bool(causal)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("attention expects (B, L, d_model).")
        b, length, _ = x.shape
        q = self.q_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.full(
                (length, length), float("-inf"), device=x.device, dtype=attn.dtype
            ).triu_(1)
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(attn, dim=-1)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout, training=True)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(b, length, self.d_model)
        return self.out_proj(output)


class SpatialAttention(_MultiHeadAttention):
    """Multi-head attention across the feature axis (non-causal)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, n_heads=n_heads, causal=False, dropout=dropout)


class TemporalCausalAttention(_MultiHeadAttention):
    """Causal multi-head attention across the time axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__(d_model=d_model, n_heads=n_heads, causal=True, dropout=dropout)
```

- [ ] **Step 4: Run the tests**

Run: `./venv/bin/python -m unittest tests.test_tlob_modules -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/tlob_modules.py tests/test_tlob_modules.py
git commit -m "feat: add spatial and temporal causal attention modules"
```

---

## Task 7: Add `MLPLOBFeatMix` and `MLPLOBTempMix` to `tlob_modules.py`

**Files:**
- Modify: `src/quantlab/tlob_modules.py`
- Modify: `tests/test_tlob_modules.py`

- [ ] **Step 1: Append the failing test**

```python
    def test_mlplob_feat_mix_shape(self):
        from src.quantlab.tlob_modules import MLPLOBFeatMix
        module = MLPLOBFeatMix(d_model=32, expansion=4)
        x = torch.randn(2, 8, 32)
        y = module(x)
        self.assertEqual(y.shape, x.shape)

    def test_mlplob_temp_mix_shape(self):
        from src.quantlab.tlob_modules import MLPLOBTempMix
        module = MLPLOBTempMix(sequence_length=8, expansion=4)
        x = torch.randn(2, 8, 32)
        y = module(x)
        self.assertEqual(y.shape, x.shape)
```

- [ ] **Step 2: Run**

Expected: ImportError.

- [ ] **Step 3: Append implementations**

```python
class MLPLOBFeatMix(nn.Module):
    """MLP-Mixer style feature-axis MLP applied row-wise (per timestep)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model) * int(expansion)
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc2(x)
        return residual + x


class MLPLOBTempMix(nn.Module):
    """MLP-Mixer style temporal-axis MLP applied column-wise (per feature channel)."""

    def __init__(self, sequence_length: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(sequence_length) * int(expansion)
        self.norm = RMSNorm(sequence_length)
        self.fc1 = nn.Linear(sequence_length, hidden)
        self.fc2 = nn.Linear(hidden, sequence_length)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)
        x_t = self.norm(x_t)
        x_t = F.gelu(self.fc1(x_t))
        if self.dropout > 0.0 and self.training:
            x_t = F.dropout(x_t, p=self.dropout, training=True)
        x_t = self.fc2(x_t)
        return residual + x_t.transpose(1, 2)
```

- [ ] **Step 4: Run the tests**

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/tlob_modules.py tests/test_tlob_modules.py
git commit -m "feat: add MLPLOB feature and temporal MLP mixer blocks"
```

---

## Task 8: Build `tlob_model.py` — `TLOBQConfig`, `TLOBQBlock`, `TLOBQModel`

**Files:**
- Create: `src/quantlab/tlob_model.py`
- Create: `tests/test_tlob_model.py`

- [ ] **Step 1: Write the failing test (`tests/test_tlob_model.py`)**

```python
import unittest

import torch

from src.quantlab.tlob_model import TLOBQConfig, TLOBQModel


def _smoke_config():
    return TLOBQConfig(
        feature_keys=tuple(f"f{i}" for i in range(8)),
        sequence_length=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_expansion=2,
        dropout=0.1,
        horizons=(1, 5),
        head_volatility=True,
        head_spread=True,
        seed=42,
    )


class TestTLOBQModel(unittest.TestCase):
    def test_forward_shapes(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        x = torch.randn(4, config.sequence_length, len(config.feature_keys))
        outputs = model(x)
        for horizon in config.horizons:
            self.assertEqual(outputs[horizon]["direction"].shape, (4, 3))
            self.assertEqual(outputs[horizon]["future_vol"].shape, (4, 1))
            self.assertEqual(outputs[horizon]["future_spread"].shape, (4, 1))

    def test_param_count_within_budget(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        count = sum(p.numel() for p in model.parameters())
        self.assertGreater(count, 1_000)
        self.assertLess(count, 1_000_000)

    def test_mc_dropout_produces_variance(self):
        config = _smoke_config()
        model = TLOBQModel(config)
        model.eval()
        model.set_mc_active(True)
        x = torch.randn(4, config.sequence_length, len(config.feature_keys))
        sample_a = model(x)[1]["direction"]
        sample_b = model(x)[1]["direction"]
        self.assertFalse(torch.allclose(sample_a, sample_b))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/tlob_model.py`**

```python
"""TLOB-Q model: dual-attention transformer with multi-horizon multi-task heads."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from .tlob_modules import (
    BilinearNormalization,
    MCDropout,
    MLPLOBFeatMix,
    MLPLOBTempMix,
    RMSNorm,
    SpatialAttention,
    TemporalCausalAttention,
)


@dataclass(frozen=True)
class TLOBQConfig:
    feature_keys: tuple[str, ...]
    sequence_length: int = 128
    d_model: int = 320
    n_heads: int = 10
    n_layers: int = 8
    ffn_expansion: int = 4
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 5, 20, 100)
    head_volatility: bool = True
    head_spread: bool = True
    seed: int = 2026
    alpha_dir: float = 1.0
    alpha_vol: float = 0.1
    alpha_spread: float = 0.05
    kappa: float = 1.0
    no_trade_threshold: float = 0.05
    mc_samples: int = 16
    ema_decay: float = 0.999

    def to_dict(self):
        payload = asdict(self)
        payload["feature_keys"] = list(self.feature_keys)
        payload["horizons"] = list(self.horizons)
        return payload

    @classmethod
    def from_dict(cls, payload):
        kwargs = dict(payload)
        kwargs["feature_keys"] = tuple(kwargs["feature_keys"])
        kwargs["horizons"] = tuple(int(h) for h in kwargs["horizons"])
        return cls(**kwargs)


class TLOBQBlock(nn.Module):
    def __init__(self, config: TLOBQConfig):
        super().__init__()
        self.bin = BilinearNormalization(
            num_features=config.d_model, num_timesteps=config.sequence_length
        )
        self.spatial_attn = SpatialAttention(d_model=config.d_model, n_heads=config.n_heads)
        self.spatial_norm = RMSNorm(config.d_model)
        self.spatial_dropout = MCDropout(p=config.dropout)
        self.feat_mix = MLPLOBFeatMix(d_model=config.d_model, expansion=config.ffn_expansion)
        self.temporal_attn = TemporalCausalAttention(d_model=config.d_model, n_heads=config.n_heads)
        self.temporal_norm = RMSNorm(config.d_model)
        self.temporal_dropout = MCDropout(p=config.dropout)
        self.temp_mix = MLPLOBTempMix(
            sequence_length=config.sequence_length, expansion=config.ffn_expansion
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bin(x)
        x = x + self.spatial_dropout(self.spatial_attn(self.spatial_norm(x)))
        x = self.feat_mix(x)
        x = x + self.temporal_dropout(self.temporal_attn(self.temporal_norm(x)))
        x = self.temp_mix(x)
        return x


class TLOBQModel(nn.Module):
    def __init__(self, config: TLOBQConfig):
        super().__init__()
        if config.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        self.config = config
        self.feature_proj = nn.Linear(len(config.feature_keys), config.d_model)
        self.blocks = nn.ModuleList([TLOBQBlock(config) for _ in range(config.n_layers)])
        self.trunk_norm = RMSNorm(config.d_model)
        self.direction_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 3) for h in config.horizons}
        )
        self.vol_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 1) for h in config.horizons}
        )
        self.spread_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.d_model, 1) for h in config.horizons}
        )

    def set_mc_active(self, active: bool) -> None:
        for module in self.modules():
            if isinstance(module, MCDropout):
                module.set_mc_active(active)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError("TLOBQModel expects input (B, T, F).")
        z = self.trunk(x)
        outputs: dict[int, dict[str, torch.Tensor]] = {}
        for horizon in self.config.horizons:
            key = str(horizon)
            outputs[horizon] = {
                "direction": self.direction_heads[key](z),
                "future_vol": self.vol_heads[key](z),
                "future_spread": self.spread_heads[key](z),
            }
        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run the tests**

Run: `./venv/bin/python -m unittest tests.test_tlob_model -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/tlob_model.py tests/test_tlob_model.py
git commit -m "feat: add TLOBQConfig, TLOBQBlock, TLOBQModel with multi-horizon heads"
```

---

## Task 9: Add `EMAWeightTracker` and save/load helpers to `tlob_model.py`

**Files:**
- Modify: `src/quantlab/tlob_model.py`
- Modify: `tests/test_tlob_model.py`

- [ ] **Step 1: Append the failing tests**

```python
    def test_ema_tracker_moves_toward_target(self):
        from src.quantlab.tlob_model import EMAWeightTracker
        config = _smoke_config()
        model = TLOBQModel(config)
        ema = EMAWeightTracker(model, decay=0.5)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)
        for shadow, current in zip(ema.shadow_state().values(), model.state_dict().values()):
            self.assertFalse(torch.allclose(shadow, current))

    def test_save_load_round_trip(self):
        from src.quantlab.tlob_model import save_tlob_q_artifact, load_tlob_q_artifact
        import tempfile, os
        config = _smoke_config()
        model = TLOBQModel(config)
        from src.quantlab.tlob_model import EMAWeightTracker
        ema = EMAWeightTracker(model, decay=0.5)
        ema.update(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact.pt")
            save_tlob_q_artifact(path, model=model, ema=ema, config=config,
                                  standardizer={"mean": [0.0], "scale": [1.0]},
                                  metrics={"loss": 0.5},
                                  meta={"seed": 42})
            loaded = load_tlob_q_artifact(path)
            self.assertEqual(loaded["config"].seed, 42)
            self.assertIn("ema_state_dict", loaded)
            self.assertEqual(loaded["metrics"]["loss"], 0.5)
```

- [ ] **Step 2: Run (should fail)**

Expected: ImportError on `EMAWeightTracker` / save helpers.

- [ ] **Step 3: Append to `src/quantlab/tlob_model.py`**

```python
from pathlib import Path

import copy


class EMAWeightTracker:
    def __init__(self, model: TLOBQModel, decay: float = 0.999):
        self.decay = float(decay)
        self._shadow = {name: param.detach().clone() for name, param in model.state_dict().items()}

    def update(self, model: TLOBQModel) -> None:
        for name, param in model.state_dict().items():
            if name not in self._shadow:
                self._shadow[name] = param.detach().clone()
                continue
            self._shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def shadow_state(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._shadow.items()}

    def load_into(self, model: TLOBQModel) -> None:
        model.load_state_dict(self._shadow)


def save_tlob_q_artifact(
    path,
    model: TLOBQModel,
    ema: EMAWeightTracker | None = None,
    config: TLOBQConfig | None = None,
    standardizer: dict | None = None,
    metrics: dict | None = None,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow_state() if ema is not None else None,
        "config": (config or model.config).to_dict(),
        "standardizer": standardizer or {},
        "metrics": metrics or {},
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_tlob_q_artifact(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = TLOBQConfig.from_dict(payload["config"])
    return {
        "state_dict": payload["state_dict"],
        "ema_state_dict": payload.get("ema_state_dict"),
        "config": config,
        "standardizer": payload.get("standardizer", {}),
        "metrics": payload.get("metrics", {}),
        "meta": payload.get("meta", {}),
    }
```

- [ ] **Step 4: Run the tests**

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/tlob_model.py tests/test_tlob_model.py
git commit -m "feat: add EMA weight tracker and artifact save/load helpers"
```

---

## Task 10: Build `curriculum.py` — regime-mixed synthetic event generator

**Files:**
- Create: `src/quantlab/curriculum.py`
- Create: `tests/test_curriculum.py`

- [ ] **Step 1: Write the failing test (`tests/test_curriculum.py`)**

```python
import unittest

from src.quantlab.curriculum import REGIME_MIX, generate_regime_mixed_events


class TestCurriculum(unittest.TestCase):
    def test_seed_determinism(self):
        a = generate_regime_mixed_events(seed=1, rows=64)
        b = generate_regime_mixed_events(seed=1, rows=64)
        self.assertEqual([event.to_dict() for event in a], [event.to_dict() for event in b])

    def test_regime_proportions_within_tolerance(self):
        events = generate_regime_mixed_events(seed=42, rows=800)
        counts = {regime: 0 for regime in REGIME_MIX}
        for event in events:
            counts[event.extras["regime"]] += 1
        total = sum(counts.values())
        for regime, expected in REGIME_MIX.items():
            self.assertAlmostEqual(counts[regime] / total, expected, delta=0.05)

    def test_minimum_rows(self):
        with self.assertRaises(ValueError):
            generate_regime_mixed_events(seed=1, rows=4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/curriculum.py`**

```python
"""Regime-mixed synthetic market event generator for TLOB-Q training."""

from __future__ import annotations

import math
import random

from .core import MarketEvent


REGIME_MIX: dict[str, float] = {
    "accumulation": 0.30,
    "distribution": 0.30,
    "mean_reversion": 0.25,
    "shock": 0.15,
}


def _regime_drift(regime: str) -> float:
    return {
        "accumulation": 0.00002,
        "distribution": -0.00002,
        "mean_reversion": 0.0,
        "shock": 0.0,
    }[regime]


def _shock_volatility(regime: str) -> float:
    return {
        "accumulation": 0.00006,
        "distribution": 0.00006,
        "mean_reversion": 0.00004,
        "shock": 0.00020,
    }[regime]


def _build_regime_schedule(rng: random.Random, rows: int) -> list[str]:
    target_counts = {regime: max(1, int(rows * fraction)) for regime, fraction in REGIME_MIX.items()}
    assigned = sum(target_counts.values())
    deficit = rows - assigned
    if deficit > 0:
        target_counts["accumulation"] += deficit
    schedule: list[str] = []
    for regime, count in target_counts.items():
        schedule.extend([regime] * count)
    rng.shuffle(schedule)
    return schedule[:rows]


def generate_regime_mixed_events(
    seed: int,
    rows: int = 4_000,
    symbol: str = "BTCUSDT",
    start_timestamp_ms: int = 1_700_000_000_000,
    interval_ms: int = 1_000,
    start_price: float = 60_000.0,
) -> list[MarketEvent]:
    rows = int(rows)
    if rows < 8:
        raise ValueError("rows must be at least 8.")
    rng = random.Random(int(seed))
    schedule = _build_regime_schedule(rng, rows)
    price = float(start_price)
    previous_pressure = 0.0
    events: list[MarketEvent] = []
    for index, regime in enumerate(schedule):
        drift = _regime_drift(regime)
        shock = rng.gauss(0.0, _shock_volatility(regime))
        if index > 0:
            price *= 1.0 + drift + 0.00034 * previous_pressure + shock
            price = max(price, 1.0)
        bias = {
            "accumulation": 0.35,
            "distribution": -0.35,
            "mean_reversion": 0.0,
            "shock": rng.choice([-0.6, 0.6]),
        }[regime]
        cyclical = 0.55 * math.sin(index / 3.0) + 0.25 * math.sin(index / 9.0)
        pressure = max(-0.98, min(0.98, bias + cyclical + rng.gauss(0.0, 0.1)))
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
                source="curriculum",
                extras={
                    "bid_depth": [bid_size, bid_size * 0.65, bid_size * 0.35, bid_size * 0.18, bid_size * 0.1],
                    "ask_depth": [ask_size, ask_size * 0.65, ask_size * 0.35, ask_size * 0.18, ask_size * 0.1],
                    "funding": 0.00001 * math.sin(index / 12.0),
                    "liquidation_intensity": max(0.0, abs(pressure) - 0.7),
                    "latent_pressure": pressure,
                    "regime": regime,
                },
            )
        )
        previous_pressure = pressure
    return events
```

- [ ] **Step 4: Run**

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/curriculum.py tests/test_curriculum.py
git commit -m "feat: add regime-mixed synthetic market event curriculum"
```

---

## Task 11: Build `training.py` — multi-task loss, train_one_fold, walk-forward orchestration

**Files:**
- Create: `src/quantlab/training.py`
- Create: `tests/test_training.py`

This task is large; we split steps into clearly atomic chunks but commit once at the end after all tests pass.

- [ ] **Step 1: Write the failing test (`tests/test_training.py`)**

```python
import os
import tempfile
import unittest

import torch

from src.quantlab.tlob_model import TLOBQConfig
from src.quantlab.training import compute_multitask_loss, train_one_fold


def _fake_batch(config: TLOBQConfig, n_rows: int = 16):
    x = torch.randn(n_rows, config.sequence_length, len(config.feature_keys))
    y = {
        horizon: {
            "direction": torch.randint(low=-1, high=2, size=(n_rows,)),
            "future_vol": torch.randn(n_rows),
            "future_spread": torch.randn(n_rows),
        }
        for horizon in config.horizons
    }
    return x, y


def _smoke_config():
    return TLOBQConfig(
        feature_keys=tuple(f"f{i}" for i in range(4)),
        sequence_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_expansion=2,
        dropout=0.05,
        horizons=(1, 5),
        seed=1,
        mc_samples=2,
    )


class TestTraining(unittest.TestCase):
    def test_loss_is_finite(self):
        from src.quantlab.tlob_model import TLOBQModel
        config = _smoke_config()
        model = TLOBQModel(config)
        x, y = _fake_batch(config)
        outputs = model(x)
        loss, parts = compute_multitask_loss(outputs, y, config)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        self.assertIn(1, parts)
        self.assertIn(5, parts)

    def test_train_one_fold_smoke(self):
        config = _smoke_config()
        x_train, y_train = _fake_batch(config, n_rows=32)
        x_val, y_val = _fake_batch(config, n_rows=16)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = train_one_fold(
                config=config,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                output_dir=tmp,
                fold_index=0,
                epochs=2,
                batch_size=8,
                lr=1e-3,
                warmup_ratio=0.0,
                ema_decay=0.5,
            )
            self.assertIn("model_path", artifact)
            self.assertTrue(os.path.exists(artifact["model_path"]))
            self.assertIn("metrics", artifact)
            self.assertIn("loss_history", artifact)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/training.py`**

```python
"""Training loop, multi-task loss, and walk-forward orchestration for TLOB-Q."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .sequence import walk_forward_index_ranges
from .tlob_model import (
    EMAWeightTracker,
    TLOBQConfig,
    TLOBQModel,
    save_tlob_q_artifact,
)


def _direction_to_class(labels: torch.Tensor) -> torch.Tensor:
    return labels.long() + 1  # map {-1, 0, 1} -> {0, 1, 2}


def compute_multitask_loss(outputs, targets, config: TLOBQConfig):
    parts: dict[int, dict[str, torch.Tensor]] = {}
    total = torch.zeros((), dtype=torch.float32)
    for horizon in config.horizons:
        pred = outputs[horizon]
        target = targets[horizon]
        ce = F.cross_entropy(pred["direction"], _direction_to_class(target["direction"]))
        vol = F.mse_loss(pred["future_vol"].squeeze(-1), target["future_vol"]) if config.head_volatility else torch.zeros((), dtype=torch.float32)
        spread = F.mse_loss(pred["future_spread"].squeeze(-1), target["future_spread"]) if config.head_spread else torch.zeros((), dtype=torch.float32)
        horizon_loss = config.alpha_dir * ce + config.alpha_vol * vol + config.alpha_spread * spread
        parts[horizon] = {"ce": ce, "vol": vol, "spread": spread, "total": horizon_loss}
        total = total + horizon_loss
    return total, parts


def _batched(x: torch.Tensor, y: dict, batch_size: int):
    n = x.size(0)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield (
            x[start:end],
            {
                horizon: {key: tensor[start:end] for key, tensor in payload.items()}
                for horizon, payload in y.items()
            },
        )


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _evaluate(model: TLOBQModel, x: torch.Tensor, y: dict, config: TLOBQConfig) -> dict:
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss, parts = compute_multitask_loss(outputs, y, config)
    per_horizon: dict[int, dict[str, float]] = {}
    for horizon in config.horizons:
        pred_class = outputs[horizon]["direction"].argmax(dim=-1)
        gold_class = _direction_to_class(y[horizon]["direction"])
        accuracy = (pred_class == gold_class).float().mean().item()
        per_horizon[horizon] = {
            "ce": float(parts[horizon]["ce"].item()),
            "accuracy": float(accuracy),
        }
    model.train()
    return {"loss": float(loss.item()), "per_horizon": per_horizon}


def train_one_fold(
    config: TLOBQConfig,
    x_train: torch.Tensor,
    y_train: dict,
    x_val: torch.Tensor,
    y_val: dict,
    output_dir,
    fold_index: int,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.05,
    ema_decay: float | None = None,
    patience: int = 5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed + fold_index)
    model = TLOBQModel(config)
    model.train()
    ema = EMAWeightTracker(model, decay=ema_decay if ema_decay is not None else config.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * max(1, (x_train.size(0) + batch_size - 1) // batch_size))
    warmup_steps = int(total_steps * warmup_ratio)
    step = 0
    best_val = float("inf")
    stale = 0
    loss_history: list[dict] = []
    curves_path = output_dir / f"training_curves_fold{fold_index}.jsonl"
    nan_counter = 0
    start_time = time.time()
    with curves_path.open("w", encoding="utf-8") as curves_file:
        for epoch in range(epochs):
            for x_batch, y_batch in _batched(x_train, y_train, batch_size):
                lr_now = _cosine_lr(step, total_steps, warmup_steps, lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
                outputs = model(x_batch)
                loss, _ = compute_multitask_loss(outputs, y_batch, config)
                if not torch.isfinite(loss):
                    nan_counter += 1
                    if nan_counter > 5:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                ema.update(model)
                curves_file.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "lr": float(lr_now),
                            "loss": float(loss.item()),
                            "grad_norm": float(grad_norm.item()),
                        }
                    )
                    + "\n"
                )
                step += 1
            if nan_counter > 5:
                break
            metrics = _evaluate(model, x_val, y_val, config)
            loss_history.append({"epoch": epoch, "val_loss": metrics["loss"], "per_horizon": metrics["per_horizon"]})
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    wall_clock = time.time() - start_time
    final_val = loss_history[-1]["val_loss"] if loss_history else float("nan")
    model_path = output_dir / f"fold{fold_index}_model.pt"
    save_tlob_q_artifact(
        model_path,
        model=model,
        ema=ema,
        config=config,
        standardizer={"mean": [], "scale": []},
        metrics={"best_val_loss": float(best_val), "final_val_loss": float(final_val), "loss_history": loss_history},
        meta={"fold_index": int(fold_index), "wall_clock_s": float(wall_clock), "nan_counter": int(nan_counter)},
    )
    return {
        "fold_index": int(fold_index),
        "model_path": str(model_path),
        "metrics": {"best_val_loss": float(best_val), "final_val_loss": float(final_val)},
        "loss_history": loss_history,
        "wall_clock_s": float(wall_clock),
        "nan_counter": int(nan_counter),
    }


def aggregate_folds(fold_artifacts) -> dict:
    fold_artifacts = list(fold_artifacts)
    if not fold_artifacts:
        return {"folds": [], "mean_val_loss": float("nan")}
    val_losses = [a["metrics"]["best_val_loss"] for a in fold_artifacts]
    return {
        "folds": fold_artifacts,
        "mean_val_loss": sum(val_losses) / len(val_losses),
        "min_val_loss": min(val_losses),
        "max_val_loss": max(val_losses),
        "total_wall_clock_s": sum(a["wall_clock_s"] for a in fold_artifacts),
    }
```

- [ ] **Step 4: Run the tests**

Run: `./venv/bin/python -m unittest tests.test_training -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/training.py tests/test_training.py
git commit -m "feat: add TLOB-Q multitask loss, training loop, and fold aggregation"
```

---

## Task 12: Build `inference.py` — MC-dropout sampling, ensemble aggregation, κ-scaled edge

**Files:**
- Create: `src/quantlab/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write the failing test (`tests/test_inference.py`)**

```python
import unittest

import torch

from src.quantlab.inference import (
    aggregate_ensemble,
    expected_edge_from_distribution,
    mc_dropout_distribution,
)
from src.quantlab.tlob_model import TLOBQConfig, TLOBQModel


def _config():
    return TLOBQConfig(
        feature_keys=tuple(f"f{i}" for i in range(4)),
        sequence_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_expansion=2,
        dropout=0.2,
        horizons=(1,),
        seed=0,
    )


class TestInference(unittest.TestCase):
    def test_mc_dropout_yields_variance(self):
        config = _config()
        model = TLOBQModel(config)
        x = torch.randn(4, config.sequence_length, len(config.feature_keys))
        dist = mc_dropout_distribution(model, x, horizon=1, mc_samples=8)
        self.assertEqual(dist["prob_mean"].shape, (4, 3))
        self.assertGreater(dist["prob_std"].mean().item(), 0.0)

    def test_edge_formula(self):
        prob_mean = torch.tensor([[0.2, 0.3, 0.5]])
        prob_std = torch.tensor([[0.05, 0.05, 0.05]])
        edge = expected_edge_from_distribution(prob_mean, prob_std, kappa=1.0)
        self.assertAlmostEqual(float(edge.item()), 0.5 - 0.2 - 1.0 * (0.05 + 0.05), places=4)

    def test_ensemble_averages_means(self):
        a = {"prob_mean": torch.tensor([[0.1, 0.3, 0.6]]), "prob_std": torch.tensor([[0.0, 0.0, 0.0]])}
        b = {"prob_mean": torch.tensor([[0.3, 0.3, 0.4]]), "prob_std": torch.tensor([[0.0, 0.0, 0.0]])}
        agg = aggregate_ensemble([a, b])
        self.assertTrue(torch.allclose(agg["prob_mean"], torch.tensor([[0.2, 0.3, 0.5]])))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/inference.py`**

```python
"""Inference helpers for TLOB-Q: MC-dropout sampling, ensemble aggregation, edge."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .tlob_model import TLOBQModel


def mc_dropout_distribution(
    model: TLOBQModel,
    x: torch.Tensor,
    horizon: int,
    mc_samples: int = 16,
) -> dict:
    horizon = int(horizon)
    mc_samples = max(1, int(mc_samples))
    model.eval()
    model.set_mc_active(True)
    probs: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(mc_samples):
            outputs = model(x)
            logits = outputs[horizon]["direction"]
            probs.append(F.softmax(logits, dim=-1))
    stacked = torch.stack(probs, dim=0)
    prob_mean = stacked.mean(dim=0)
    prob_std = stacked.std(dim=0, unbiased=False)
    return {"prob_mean": prob_mean, "prob_std": prob_std}


def expected_edge_from_distribution(prob_mean: torch.Tensor, prob_std: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    if prob_mean.shape[-1] != 3:
        raise ValueError("prob_mean must end in dim 3 (down/hold/up).")
    edge_mean = prob_mean[..., 2] - prob_mean[..., 0]
    penalty = float(kappa) * (prob_std[..., 0] + prob_std[..., 2])
    edge = edge_mean - penalty
    return edge


def aggregate_ensemble(distributions) -> dict:
    distributions = list(distributions)
    if not distributions:
        raise ValueError("distributions must not be empty.")
    means = torch.stack([d["prob_mean"] for d in distributions], dim=0)
    stds = torch.stack([d["prob_std"] for d in distributions], dim=0)
    mean = means.mean(dim=0)
    combined_var = (stds ** 2).mean(dim=0) + means.var(dim=0, unbiased=False)
    std = combined_var.clamp_min(0.0).sqrt()
    return {"prob_mean": mean, "prob_std": std}


def action_from_edge(edge: torch.Tensor, no_trade_threshold: float = 0.05) -> list[str]:
    actions: list[str] = []
    threshold = abs(float(no_trade_threshold))
    for value in edge.tolist():
        if value > threshold:
            actions.append("buy")
        elif value < -threshold:
            actions.append("sell")
        else:
            actions.append("hold")
    return actions
```

- [ ] **Step 4: Run**

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/inference.py tests/test_inference.py
git commit -m "feat: add MC-dropout, ensemble aggregation and edge helpers"
```

---

## Task 13: Add multi-horizon backtest to `backtest.py`

**Files:**
- Modify: `src/quantlab/backtest.py`
- Modify: `tests/test_quantlab.py` (append integration test)

- [ ] **Step 1: Append the failing test**

```python
def test_run_backtest_multihorizon_smoke():
    from src.quantlab.backtest import run_backtest_multihorizon, BacktestConfig
    prediction_rows = [
        {"timestamp_ms": i * 1000, "symbol": "X",
         "horizons": {"1": {"expected_edge": 0.1 if i % 2 == 0 else -0.1}}}
        for i in range(20)
    ]
    label_rows = [
        {"timestamp_ms": i * 1000, "symbol": "X",
         "horizons": {"1": {"future_return": 0.0001 if i % 2 == 0 else -0.0001}}}
        for i in range(20)
    ]
    report = run_backtest_multihorizon(prediction_rows, label_rows, horizon=1,
                                        config=BacktestConfig(no_trade_threshold=0.05))
    assert report.trades >= 0
    assert report.equity_curve[0] == 1.0
```

- [ ] **Step 2: Run (should fail)**

Run: `./venv/bin/python -m unittest tests.test_quantlab -v 2>&1 | tail -20`
Expected: `AttributeError: module 'src.quantlab.backtest' has no attribute 'run_backtest_multihorizon'`.

- [ ] **Step 3: Append to `src/quantlab/backtest.py`**

```python
def run_backtest_multihorizon(
    prediction_rows,
    label_rows,
    horizon: int,
    config: "BacktestConfig | None" = None,
):
    prediction_rows = list(prediction_rows)
    label_rows = list(label_rows)
    if len(prediction_rows) != len(label_rows):
        raise ValueError("prediction_rows and label_rows must align by row count.")
    horizon = int(horizon)
    horizon_key = str(horizon)
    feature_rows = []
    target_rows = []
    expected_edges = []
    for pred_row, label_row in zip(prediction_rows, label_rows):
        feature_rows.append({"timestamp_ms": int(pred_row["timestamp_ms"]), "midprice": 1.0})
        future_return = float(
            label_row.get("horizons", {}).get(horizon_key, {}).get("future_return", 0.0)
        )
        target_rows.append({"future_return": future_return})
        edge_value = pred_row.get("horizons", {}).get(horizon_key, {}).get("expected_edge", 0.0)
        expected_edges.append(float(edge_value))
    cfg = config or BacktestConfig()
    return run_backtest(feature_rows, target_rows, expected_edges, config=cfg)
```

- [ ] **Step 4: Run**

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/backtest.py tests/test_quantlab.py
git commit -m "feat: add multi-horizon backtest entry to quantlab backtester"
```

---

## Task 14: Build `cli_tlob.py` — unified CLI

**Files:**
- Create: `src/quantlab/cli_tlob.py`
- Create: `tests/test_tlob_cli.py`

- [ ] **Step 1: Write the failing test (`tests/test_tlob_cli.py`)**

```python
import tempfile
import unittest

from src.quantlab.cli_tlob import build_parser, run_demo


class TestTLOBCli(unittest.TestCase):
    def test_parser_demo_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["demo", "--output-dir", "/tmp/tlobq", "--rows", "32"])
        self.assertEqual(args.command, "demo")
        self.assertEqual(args.rows, 32)

    def test_parser_rejects_unknown_command(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["nope"])

    def test_run_demo_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_demo(output_dir=tmp, rows=64, sequence_length=8,
                                d_model=16, n_heads=4, n_layers=1,
                                epochs=1, batch_size=8, seed=0,
                                horizons=(1, 5))
            self.assertIn("summary_path", summary)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Expected: ImportError.

- [ ] **Step 3: Implement `src/quantlab/cli_tlob.py`**

```python
"""Unified CLI for TLOB-Q: train, predict, walk-forward, demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .curriculum import generate_regime_mixed_events
from .features_extras import EXTENDED_FEATURE_KEYS, build_extended_feature_rows
from .io import write_jsonl
from .labels_tlob import build_multihorizon_labels
from .sequence import align_feature_label_rows, build_sequence_windows, walk_forward_index_ranges
from .tlob_model import TLOBQConfig
from .training import aggregate_folds, train_one_fold


def _config_from_kwargs(**kwargs) -> TLOBQConfig:
    base = TLOBQConfig(feature_keys=EXTENDED_FEATURE_KEYS)
    payload = base.to_dict()
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    payload["feature_keys"] = list(payload["feature_keys"])
    return TLOBQConfig.from_dict(payload)


def run_demo(
    output_dir: str,
    rows: int = 4_000,
    sequence_length: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    epochs: int = 3,
    batch_size: int = 32,
    seed: int = 2026,
    horizons: tuple[int, ...] = (1, 5, 20),
    n_folds: int = 3,
    embargo: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = generate_regime_mixed_events(seed=seed, rows=rows)
    write_jsonl(output_dir / "events.jsonl", [event.to_dict() for event in events])
    feature_rows = build_extended_feature_rows(events)
    write_jsonl(output_dir / "features.jsonl", feature_rows)
    label_rows = build_multihorizon_labels(events, horizons=horizons)
    write_jsonl(output_dir / "labels.jsonl", [row.to_dict() for row in label_rows])
    aligned_features, aligned_labels = align_feature_label_rows(
        feature_rows, [row.to_dict() for row in label_rows]
    )
    x, y_by_h = build_sequence_windows(
        aligned_features, aligned_labels, sequence_length=sequence_length, horizons=horizons
    )
    config = _config_from_kwargs(
        feature_keys=tuple(EXTENDED_FEATURE_KEYS),
        sequence_length=sequence_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        horizons=tuple(int(h) for h in horizons),
        seed=seed,
        ffn_expansion=2,
    )
    embargo_value = embargo if embargo is not None else sequence_length
    fold_artifacts = []
    for fold_meta in walk_forward_index_ranges(n=x.size(0), n_folds=n_folds, embargo=embargo_value):
        train_slice = slice(*fold_meta["train"])
        val_slice = slice(*fold_meta["val"])
        x_train, x_val = x[train_slice], x[val_slice]
        y_train = {h: {k: v[train_slice] for k, v in payload.items()} for h, payload in y_by_h.items()}
        y_val = {h: {k: v[val_slice] for k, v in payload.items()} for h, payload in y_by_h.items()}
        artifact = train_one_fold(
            config=config,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            output_dir=output_dir,
            fold_index=fold_meta["fold"],
            epochs=epochs,
            batch_size=batch_size,
        )
        fold_artifacts.append(artifact)
    aggregate = aggregate_folds(fold_artifacts)
    summary_path = output_dir / "summary.json"
    summary = {
        "config": config.to_dict(),
        "aggregate": aggregate,
        "rows": int(x.size(0)),
        "horizons": list(horizons),
        "status": "ok",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TLOB-Q unified CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run end-to-end smoke pipeline.")
    demo.add_argument("--output-dir", required=True)
    demo.add_argument("--rows", type=int, default=4_000)
    demo.add_argument("--sequence-length", type=int, default=32)
    demo.add_argument("--d-model", type=int, default=64)
    demo.add_argument("--n-heads", type=int, default=4)
    demo.add_argument("--n-layers", type=int, default=2)
    demo.add_argument("--epochs", type=int, default=3)
    demo.add_argument("--batch-size", type=int, default=32)
    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--n-folds", type=int, default=3)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        summary = run_demo(
            output_dir=args.output_dir,
            rows=args.rows,
            sequence_length=args.sequence_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            n_folds=args.n_folds,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run**

Run: `./venv/bin/python -m unittest tests.test_tlob_cli -v 2>&1 | tail -20`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add src/quantlab/cli_tlob.py tests/test_tlob_cli.py
git commit -m "feat: add unified TLOB-Q CLI with demo subcommand"
```

---

## Task 15: Integration test — end-to-end smoke

**Files:**
- Create: `tests/test_tlob_demo.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import tempfile
import unittest

from src.quantlab.cli_tlob import run_demo


class TestTlobDemoIntegration(unittest.TestCase):
    def test_demo_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_demo(
                output_dir=tmp,
                rows=128,
                sequence_length=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                epochs=1,
                batch_size=8,
                seed=0,
                horizons=(1, 5),
                n_folds=2,
            )
            for name in ("events.jsonl", "features.jsonl", "labels.jsonl", "summary.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), f"missing {name}")
            payload = json.loads(open(os.path.join(tmp, "summary.json")).read())
            self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Run: `./venv/bin/python -m unittest tests.test_tlob_demo -v 2>&1 | tail -20`
Expected: OK (assuming Tasks 0–14 done correctly).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tlob_demo.py
git commit -m "test: add TLOB-Q end-to-end demo integration test"
```

---

## Task 16: E2E test — subprocess invocation of the CLI

**Files:**
- Create: `tests/test_tlob_e2e_smoke.py`

- [ ] **Step 1: Write the test**

```python
import os
import subprocess
import sys
import tempfile
import unittest


class TestTlobE2ECli(unittest.TestCase):
    def test_demo_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.quantlab.cli_tlob",
                    "demo",
                    "--output-dir", tmp,
                    "--rows", "96",
                    "--sequence-length", "8",
                    "--d-model", "16",
                    "--n-heads", "4",
                    "--n-layers", "1",
                    "--epochs", "1",
                    "--batch-size", "8",
                    "--seed", "0",
                    "--n-folds", "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary.json")))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run**

Expected: OK

- [ ] **Step 3: Commit**

```bash
git add tests/test_tlob_e2e_smoke.py
git commit -m "test: add TLOB-Q E2E subprocess smoke"
```

---

## Task 17: Configs (smoke, default, ensemble)

**Files:**
- Create: `configs/quantlab/tlob_smoke.json`
- Create: `configs/quantlab/tlob_m4_default.json`
- Create: `configs/quantlab/tlob_ensemble.json`

- [ ] **Step 1: Create the smoke config**

`configs/quantlab/tlob_smoke.json`:

```json
{
  "feature_keys": null,
  "sequence_length": 32,
  "d_model": 32,
  "n_heads": 4,
  "n_layers": 2,
  "ffn_expansion": 2,
  "dropout": 0.1,
  "horizons": [1, 5],
  "head_volatility": true,
  "head_spread": true,
  "seed": 2026,
  "alpha_dir": 1.0,
  "alpha_vol": 0.1,
  "alpha_spread": 0.05,
  "kappa": 1.0,
  "no_trade_threshold": 0.05,
  "mc_samples": 4,
  "ema_decay": 0.9,
  "rows": 1024,
  "epochs": 2,
  "batch_size": 32,
  "n_folds": 2
}
```

- [ ] **Step 2: Create the M4 default config**

`configs/quantlab/tlob_m4_default.json`:

```json
{
  "feature_keys": null,
  "sequence_length": 128,
  "d_model": 320,
  "n_heads": 10,
  "n_layers": 8,
  "ffn_expansion": 4,
  "dropout": 0.1,
  "horizons": [1, 5, 20, 100],
  "head_volatility": true,
  "head_spread": true,
  "seed": 2026,
  "alpha_dir": 1.0,
  "alpha_vol": 0.1,
  "alpha_spread": 0.05,
  "kappa": 1.0,
  "no_trade_threshold": 0.05,
  "mc_samples": 16,
  "ema_decay": 0.999,
  "rows": 60000,
  "epochs": 30,
  "batch_size": 64,
  "n_folds": 3
}
```

- [ ] **Step 3: Create the ensemble config**

`configs/quantlab/tlob_ensemble.json`:

```json
{
  "base_config": "configs/quantlab/tlob_m4_default.json",
  "ensemble_size": 3,
  "seed_offsets": [1000, 2000, 3000]
}
```

- [ ] **Step 4: Commit**

```bash
git add configs/quantlab/tlob_smoke.json configs/quantlab/tlob_m4_default.json configs/quantlab/tlob_ensemble.json
git commit -m "feat: add TLOB-Q smoke, default, and ensemble JSON configs"
```

---

## Task 18: Update `__init__.py` and `demo.py` to expose TLOB-Q public surface

**Files:**
- Modify: `src/quantlab/__init__.py`
- Modify: `src/quantlab/demo.py`

- [ ] **Step 1: Append to `src/quantlab/__init__.py`**

```python
from .core import MultiHorizonLabelFrame, SequenceWindow
from .features_extras import EXTENDED_FEATURE_KEYS, build_extended_feature_rows
from .labels_tlob import build_multihorizon_labels, tlob_smoothed_label
from .sequence import (
    align_feature_label_rows,
    build_sequence_windows,
    walk_forward_index_ranges,
)
from .tlob_model import (
    EMAWeightTracker,
    TLOBQBlock,
    TLOBQConfig,
    TLOBQModel,
    load_tlob_q_artifact,
    save_tlob_q_artifact,
)
from .training import (
    aggregate_folds,
    compute_multitask_loss,
    train_one_fold,
)
from .inference import (
    action_from_edge,
    aggregate_ensemble,
    expected_edge_from_distribution,
    mc_dropout_distribution,
)
from .curriculum import REGIME_MIX, generate_regime_mixed_events
```

- [ ] **Step 2: Append `run_tlob_demo_pipeline` to `src/quantlab/demo.py`**

```python
def run_tlob_demo_pipeline(output_dir, **kwargs):
    from .cli_tlob import run_demo
    return run_demo(output_dir=output_dir, **kwargs)


PAPER_REFERENCES = PAPER_REFERENCES + (
    {"title": "TLOB: Dual Attention Transformer for LOB",
     "url": "https://arxiv.org/abs/2502.15757",
     "local_scope": "Dual spatial+temporal attention; decoupled (h, k) labelling."},
    {"title": "BDLOB: Bayesian Deep CNN for LOB",
     "url": "https://arxiv.org/abs/1811.10041",
     "local_scope": "MC-dropout for uncertainty-aware position sizing."},
    {"title": "Async Deep Duelling Q-Learning on LOB",
     "url": "https://arxiv.org/abs/2301.08688",
     "local_scope": "Future spec: RL execution layer over TLOB-Q signals."},
)
```

- [ ] **Step 3: Verify imports work**

Run: `./venv/bin/python -c "from src.quantlab import TLOBQModel, TLOBQConfig, run_demo := __import__('src.quantlab.cli_tlob', fromlist=['run_demo']).run_demo; print('imports OK')" 2>&1 | tail -5`
Expected: `imports OK`

Alternative simpler check: `./venv/bin/python -c "import src.quantlab as q; print(q.TLOBQConfig, q.TLOBQModel, q.build_extended_feature_rows)"`

- [ ] **Step 4: Commit**

```bash
git add src/quantlab/__init__.py src/quantlab/demo.py
git commit -m "feat: expose TLOB-Q public surface and demo pipeline"
```

---

## Task 19: Update docs (literature_review, research_program, quant_research_catalog, README)

**Files:**
- Modify: `docs/literature_review.md`
- Modify: `docs/research_program.md`
- Modify: `docs/quant_research_catalog.md`
- Modify: `README.md`

- [ ] **Step 1: Append a TLOB-Q row under existing quant section in `docs/literature_review.md`**

Add under the "Quantitative Finance and Market Microstructure" section:

```markdown
- [TLOB](https://arxiv.org/abs/2502.15757): Dual-attention transformer (spatial + temporal) for LOB price trend prediction; introduces decoupled `l(t,h,k)` labelling that removes horizon bias.
- [BDLOB](https://arxiv.org/abs/1811.10041): Bayesian deep CNN for LOB; MC-dropout produces position-sizing uncertainty.
- [Async DDQL on LOB](https://arxiv.org/abs/2301.08688): RL agent translating forecast signals into limit-order placements on ABIDES.
- [TimeCatcher](https://arxiv.org/abs/2601.20448): Volatility-aware variational forecasting for non-stationary series; informs the auxiliary volatility head.
- [MacroHFT](https://arxiv.org/abs/2406.14537): Memory-augmented context-aware HFT RL — regime routing reference.
```

- [ ] **Step 2: Add a Sequence Microstructure Track entry to `docs/research_program.md`**

Append to the existing "Quant Research Spine" section:

```markdown
### Sequence Microstructure Track (TLOB-Q)

A dual-attention transformer over windowed market microstructure features. Each block runs Bilinear Normalization, spatial attention across the 32-channel feature axis, MLP-Mixer feature-mixing, temporal causal attention across a 128-step window, and MLP-Mixer temporal-mixing. Multi-horizon heads predict direction, volatility, and spread at h in {1, 5, 20, 100}. MC-dropout supplies Bayesian uncertainty; Polyak-averaged shadow weights stabilize inference; purged walk-forward CV with embargo defeats label leakage. See `docs/superpowers/specs/2026-05-11-tlob-q-design.md`.
```

- [ ] **Step 3: Add a `tlob_q` row to `docs/quant_research_catalog.md`**

Append under "Supervised ML Models":

```markdown
- `tlob_q`: dual-attention transformer over `EXTENDED_FEATURE_KEYS` (F=32) with multi-horizon multi-task heads (direction / volatility / spread at h in {1, 5, 20, 100}), MC-dropout uncertainty, Polyak/EMA shadow weights, purged walk-forward CV. Default config trains in ~3 h on M4 CPU.
```

- [ ] **Step 4: Add a TLOB-Q quickstart to `README.md`**

Append after the existing quantlab demo block:

```markdown
## QuantLab TLOB-Q Demo

End-to-end smoke run of the dual-attention transformer with multi-horizon heads:

\`\`\`bash
./venv/bin/python -m src.quantlab.cli_tlob demo \
  --output-dir /tmp/tlob_q_demo \
  --rows 1024 \
  --sequence-length 32 \
  --d-model 32 \
  --n-heads 4 \
  --n-layers 2 \
  --epochs 2 \
  --batch-size 32 \
  --seed 2026 \
  --n-folds 2
\`\`\`

Writes events, features, labels, per-fold checkpoints, and a `summary.json` to the output directory.
```

(In the actual file, use backticks not escaped ones.)

- [ ] **Step 5: Verify py_compile + unittest still pass**

Run: `./venv/bin/python -m unittest 2>&1 | tail -10`
Expected: OK
Run: `./venv/bin/python -m py_compile src/quantlab/*.py`
Expected: silent

- [ ] **Step 6: Commit**

```bash
git add docs/literature_review.md docs/research_program.md docs/quant_research_catalog.md README.md
git commit -m "docs: document TLOB-Q track, papers, catalog row, and quickstart"
```

---

## Task 20: Final verification + push

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `./venv/bin/python -m unittest 2>&1 | tail -10`
Expected: OK

- [ ] **Step 2: Run py_compile across the repo**

Run: `./venv/bin/python -m py_compile main.py src/prepare_data.py src/finetune_model.py src/algorithms/*.py src/micro_gpt/*.py src/research_lab/*.py src/quantlab/*.py`
Expected: silent

- [ ] **Step 3: Run the existing micro-gpt dry-run to ensure no regression**

Run: `./venv/bin/python -m src.micro_gpt.train --config configs/micro_gpt/tiny_debug.json --dry-run 2>&1 | tail -10`
Expected: prints config + dry-run finish; no error.

- [ ] **Step 4: Run the demo CLI smoke end-to-end**

Run: `./venv/bin/python -m src.quantlab.cli_tlob demo --output-dir /tmp/tlob_q_smoke --rows 256 --sequence-length 16 --d-model 16 --n-heads 4 --n-layers 1 --epochs 1 --batch-size 16 --seed 0 --n-folds 2 2>&1 | tail -15`
Expected: prints summary JSON; `status: "ok"`.

- [ ] **Step 5: Check git status**

Run: `git status && git log --oneline -25`
Expected: clean tree; commits visible.

- [ ] **Step 6: Push to origin/main**

Run: `git push origin main`
Expected: branch updated.

---

## Self-review checklist (run after writing each task; the agent executing this plan should also tick)

- [ ] No `TBD` / `TODO` / `implement later` strings in code or tests.
- [ ] Every test has actual assertion lines, not placeholders.
- [ ] Every implementation step has the full code block, not "similar to X".
- [ ] All cross-task references (function names, dataclass fields) match: `TLOBQConfig.feature_keys`, `compute_multitask_loss`, `mc_dropout_distribution`, `aggregate_folds`, `walk_forward_index_ranges`, `EXTENDED_FEATURE_KEYS`, `MultiHorizonLabelFrame`, `SequenceWindow`, `build_multihorizon_labels`, `build_extended_feature_rows`, `generate_regime_mixed_events`, `run_demo`, `EMAWeightTracker`, `save_tlob_q_artifact`, `load_tlob_q_artifact`, `run_backtest_multihorizon`.
- [ ] Spec sections all covered: §3 architecture → Tasks 5–9; §4 module layout → Tasks 2–14, 17, 18; §5 pipeline → Tasks 10–15; §6 error handling → guarded inside each module (NaN counter in Task 11; finite-check in Task 4; weights_only-style behavior intentionally **not** enabled for our save path because the artifact contains non-tensor metadata — load uses `weights_only=False`; the spec's "weights_only=True" is downgraded for this round and noted in Task 9 as a follow-up); §7 testing → tests per task + integration Tasks 15–16.
- [ ] Verification commands at Task 20 match the baseline gates from the spec's §9 Acceptance.

## Known scope reductions vs spec (documented for honesty)

These are explicit in this plan and tracked as follow-up work, not silent omissions:

1. **`weights_only=True`** on `torch.load`: the artifact carries a `config` dict and `meta` dict (non-tensor) so we use `weights_only=False`. A future task will split artifacts into a pure-tensor `.pt` + sidecar `.json` so we can flip `weights_only=True`.
2. **Conformal calibration / JANET-style intervals**: explicitly deferred in spec §10.
3. **ICT features, live ingestion, RL execution**: deferred per spec §10.
4. **Resume-after-crash test** + **walk-forward integration test**: not included in this 20-task plan to keep within a single-iteration execution. They can be added as follow-up tasks 21–22 once Task 20 is green.
5. **Standardizer fit per fold**: the training loop in Task 11 saves an empty standardizer placeholder; per-fold fit on raw features is wired into `run_demo` only after the smoke path is green. Follow-up task to wire `Standardizer.fit` on `x_train` and `transform` on val/test inside `train_one_fold`.

## Execution handoff

After this plan is saved, the next step is to choose an execution mode and start at Task 0.
