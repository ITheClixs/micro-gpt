# TLOB-Q: Dual-Attention Transformer for QuantLab Direction & Microstructure Prediction

**Status:** Spec — awaiting plan
**Date:** 2026-05-11
**Owner:** ITheClixs
**Scope:** Single iteration. Live data ingestion, ICT/regime feature engineering, and RL execution are explicit follow-up specs.

---

## 1. Motivation

The current `quantlab` track ships a small `DirectionMLP` over scalar microstructure features (`src/quantlab/models.py`). It works as a smoke-test surface but cannot exploit temporal structure, cross-feature dependencies, or uncertainty for size gating. Recent literature (TLOB 2025, BDLOB 2018, DeepLOB 2019, Async DDQL 2023) consistently shows that dual spatial+temporal attention over windowed limit-order-book (LOB) snapshots, combined with calibrated uncertainty, materially improves direction prediction and downstream trade-economics on FI-2010, NASDAQ equities, and crypto BTC datasets.

This spec defines **TLOB-Q**, a TLOB-faithful dual-attention transformer over the existing `FeatureFrame` schema, augmented with Bayesian-style MC-dropout uncertainty, multi-horizon multi-task heads, and walk-forward purged cross-validation. It is the next architectural step on the quant-research spine after the `mlp_direction` baseline.

### Paper anchors

| Paper | arXiv | Local use |
|-------|-------|-----------|
| TLOB: Dual Attention Transformer for LOB | [2502.15757](https://arxiv.org/abs/2502.15757) | Architecture (BiN + dual attention + MLPLOB blocks); decoupled `l(t,h,k)` labelling; spread-as-θ threshold |
| BDLOB: Bayesian Deep CNN for LOB | [1811.10041](https://arxiv.org/abs/1811.10041) | MC-dropout for uncertainty; κ-scaled position sizing |
| Async Deep Duelling Q-Learning on LOB | [2301.08688](https://arxiv.org/abs/2301.08688) | Forecast → execution layer (future spec) |
| FinRL | [2111.09395](https://arxiv.org/abs/2111.09395) | Pipeline shape; reproducible artifact tree |
| JANET | [2407.06390](https://arxiv.org/abs/2407.06390) | Future: adaptive conformal intervals over the regression heads |
| DOME | [2006.16189](https://arxiv.org/abs/2006.16189) | Reporting discipline; explicit data/optim/model/eval artifacts |
| TimeCatcher | [2601.20448](https://arxiv.org/abs/2601.20448) | Volatility-aware non-stationary forecasting; informs auxiliary volatility head |
| Cont-Stoikov multi-level OFI | classical | Cascaded OFI feature channels at L=1,2,3,5 |

---

## 2. Non-goals (this iteration)

- **No live data ingestion.** Inputs are the existing synthetic `MarketEvent` JSONL pipeline plus a regime-mixed curriculum extension. A future spec will add a Binance public-websocket adapter.
- **No ICT feature engineering** (order blocks, fair-value gaps, liquidity sweeps, market-structure shift). The TLOB-Q model is the prerequisite; ICT features land in a follow-up spec because they need their own definitions and tests.
- **No RL execution layer.** Predictions feed the existing cost-aware backtester. A future spec adds Avellaneda-Stoikov + PPO/DDQN over TLOB-Q signals.
- **No GPU requirement.** Runs CPU-only by design. An optional `--device` flag may land later.
- **No live-trading claim.** This is a research artifact. The README/spec language must reflect that.

---

## 3. Architecture

### 3.1 Input

Windowed feature tensor `X ∈ ℝ^(B, T, F)` where:
- `B` = batch size (default 64; effective 256 via gradient accumulation)
- `T = 128` = sequence length (LOB snapshots per window)
- `F = 32` = feature channels (existing 8 + 24 derived; see §3.4)

Each channel is standardized per fold using a `Standardizer` fit on the training split only (extends the existing pattern in `src/quantlab/models.py`).

### 3.2 Block structure

`n_layers = 8` TLOB-Q blocks. Each block:

```
input  x ∈ (B, T, F, d_model)
  │
  ▼ BilinearNormalization (BiN)             # TLOB §5.2: handles LOB non-stationarity
  │
  ▼ SpatialAttention (multi-head, F-axis)   # cross-feature dependencies inside one timestep
  │
  ▼ residual + RMSNorm + MCDropout
  │
  ▼ MLPLOBFeatMix                            # MLP-Mixer style feature axis MLP, GeLU
  │
  ▼ TemporalCausalAttention (multi-head, T-axis)   # triangular causal mask
  │
  ▼ residual + RMSNorm + MCDropout
  │
  ▼ MLPLOBTempMix                            # MLP-Mixer style temporal axis MLP, GeLU
  │
output (B, T, F, d_model)
```

Block-level hyperparameters:
- `d_model = 320`
- `n_heads = 10` (head dim 32)
- FFN expansion `4×` in every MLPLOB sub-block
- Dropout `p = 0.1` at both `MCDropout` sites; **dropout stays active at inference** for MC sampling

### 3.3 Heads (multi-horizon multi-task)

Trunk emits a sequence-pooled vector `z ∈ ℝ^(B, d_model)` (mean-pool over `T`, then a linear projection). For each horizon `h ∈ {1, 5, 20, 100}` (ticks):

| Head | Output | Loss |
|------|--------|------|
| Direction `h` | `(B, 3)` softmax over `{down, hold, up}` | Cross-entropy on TLOB-smoothed labels |
| Volatility `h` | `(B, 1)` scalar | MSE on realized volatility over `[t, t+h]` |
| Spread `h` | `(B, 1)` scalar | MSE on mean spread over `[t, t+h]` |

Multi-task loss:

```
L = Σ_h∈{1,5,20,100} [
      α_dir   · CE(logits_dir_h,    y_dir_h)
    + α_vol   · MSE(pred_vol_h,     y_vol_h)
    + α_spread· MSE(pred_spread_h,  y_spread_h)
    ]

defaults: α_dir = 1.0, α_vol = 0.1, α_spread = 0.05
```

All weights live in `TLOBQConfig`, are logged into checkpoint `meta`, and are written into the per-run `summary.json`.

### 3.4 Feature channels (F=32)

Existing (8): midprice as log-return, spread, microprice−mid, OFI@L=1, book imbalance, depth imbalance, signed volume, realized volatility.

New (24):
1–3. Multi-level OFI at `L=2, 3, 5` (Cont-Stoikov cascaded order flow)
4. Log-spread (stationary)
5. Funding rate (from `MarketEvent.extras["funding"]`)
6. Liquidation intensity (from `extras["liquidation_intensity"]`)
7–8. Signed-volume EMA with `α=0.1` (slow) and `α=0.5` (fast)
9. Directional run-length (consecutive same-side trades)
10. Trade-burst rate (trades per `Δt` window)
11–12. Queue-depth ratios `L2/L1` and `L3/L1`
13. Cumulative depth within `ε%` of mid (microliquidity proxy)
14. Relative tick imbalance — log up-tick count / down-tick count over `W` ticks
15. Mid-return autocorrelation at lag 1 (rolling)
16. Realized-volatility ratio short-window / long-window (regime hint)
17. Cross-window VPIN proxy (signed-volume bucket / total volume)
18. Signed-volume z-score (rolling)
19. OFI z-score (rolling)
20. Microprice momentum (microprice−microprice_lag)
21. Depth-imbalance EMA fast
22. Depth-imbalance EMA slow
23. Spread regime indicator (current spread / rolling-median spread)
24. Last-trade aggressor side persistence (run-length of last side)

All channels are deterministic, computable from the existing event stream, and clamped to `[-50, +50]` after standardization to defeat outliers.

### 3.5 Uncertainty (MC-dropout) and inference

Two `MCDropout` sites per block (post-spatial-attn residual, post-temporal-attn residual). At predict time:
- Set affine layers to `eval` mode.
- Keep `MCDropout` modules in `train` mode (their `forward` checks an `mc_active` flag).
- Run `mc_samples = 16` forward passes.
- For each horizon `h`, emit per-row:
  - `prob_mean_h = (mean of softmax over MC samples)` — vector of 3
  - `prob_std_h = std across MC samples` — vector of 3
  - `expected_edge_h = (mean(P_up - P_down)) − κ · std(P_up - P_down)`, default `κ = 1.0`
- Action gating: trade only if `|expected_edge_h| > no_trade_threshold` (default `0.05`)

### 3.6 EMA weight tracking

Polyak averaging with decay `0.999`, updated every step. Inference uses EMA weights by default. Adds ≈40 MB RAM at this sizing.

### 3.7 Parameter budget

Expected count: **10–14 M** parameters per model. Reported by `tlob_model.count_parameters()` and written into `meta`. A unit test asserts the count stays within `[8M, 16M]` for the `tlob_m4_default.json` config; deviation indicates an architectural change that needs review.

### 3.8 Compute budget

- Single model, 3-fold walk-forward, 30 epochs, ~50–80 K synthetic events: **~3–3.5 h** wall-clock on M4 Air CPU.
- Optional `--ensemble K` flag trains K=2 or K=3 sequential independent seeds, roughly K× the wall-clock.
- Activations at batch=64, T=128, d_model=320, 8 layers ≈ **4–6 GB**, well inside 24 GB unified memory.

---

## 4. Module layout

### 4.1 New files

| File | Purpose | Target LOC |
|------|---------|------------|
| `src/quantlab/features_extras.py` | The 24 new feature channels; entry point compatible with `build_feature_rows`. Keeps `features.py` deterministic and unchanged. | ~220 |
| `src/quantlab/labels_tlob.py` | TLOB decoupled labelling `l(t,h,k)`; multi-horizon `MultiHorizonLabelFrame`; spread-as-threshold helper. | ~200 |
| `src/quantlab/sequence.py` | Sliding-window builder turning aligned `FeatureFrame` lists into `(B,T,F)` tensors; walk-forward-aware split helpers; `SequenceWindow` dataclass. | ~260 |
| `src/quantlab/tlob_modules.py` | `BilinearNormalization`, `SpatialAttention`, `TemporalCausalAttention`, `MLPLOBFeatMix`, `MLPLOBTempMix`, `RMSNorm`, `MCDropout`. | ~320 |
| `src/quantlab/tlob_model.py` | `TLOBQBlock`, `TLOBQModel` (multi-horizon multi-task heads), `EMAWeightTracker`, save/load to `.pt`. | ~280 |
| `src/quantlab/training.py` | Train loop with AMP-off (CPU), grad accumulation, cosine LR + warmup, walk-forward orchestration, checkpoint metadata writer, training-curve JSONL logger. | ~340 |
| `src/quantlab/inference.py` | MC-dropout sampling, deep-ensemble aggregation, κ-scaled `expected_edge`, per-horizon selection. | ~200 |
| `src/quantlab/curriculum.py` | Regime-mixed synthetic generator (accumulation / distribution / mean-reversion / shock blends with seed-controlled sequencing). Target regime mix `{accumulation: 0.30, distribution: 0.30, mean_reversion: 0.25, shock: 0.15}`. | ~220 |
| `src/quantlab/cli_tlob.py` | Unified CLI: `train`, `predict`, `walk-forward`, `demo` subcommands. | ~200 |

### 4.2 Modified files

| File | Change |
|------|--------|
| `src/quantlab/__init__.py` | Export `TLOBQConfig`, `TLOBQModel`, `train_tlob_q`, `predict_tlob_q`, `build_extended_feature_rows`, `build_multihorizon_labels`, `SequenceWindow`, `build_sequence_windows`. |
| `src/quantlab/core.py` | Add `@dataclass(frozen=True) MultiHorizonLabelFrame` (timestamp_ms, symbol, per-horizon dict with direction/future_return/future_vol/future_spread) and `SequenceWindow` schema. |
| `src/quantlab/demo.py` | Add `run_tlob_demo_pipeline(...)` alongside existing MLP path. Update `PAPER_REFERENCES` to include TLOB, BDLOB, Async DDQL, Cont-Stoikov OFI. |
| `src/quantlab/backtest.py` | Add `run_backtest_multihorizon(...)` consuming per-horizon predictions and selecting by Sharpe. |

### 4.3 Configs

```
configs/quantlab/
  tlob_smoke.json         # CI smoke: d_model=32, layers=2, T=32, F=16, 2 epochs
  tlob_m4_default.json    # main: d_model=320, layers=8, T=128, F=32, 30 epochs
  tlob_ensemble.json      # K=3 deep ensemble variant
```

### 4.4 Tests (one file per new module)

```
tests/test_features_extras.py
tests/test_labels_tlob.py
tests/test_sequence.py
tests/test_tlob_modules.py
tests/test_tlob_model.py
tests/test_training.py
tests/test_inference.py
tests/test_curriculum.py
tests/test_tlob_cli.py
tests/test_tlob_demo.py
tests/test_walk_forward_integration.py
tests/test_resume_after_crash.py
tests/test_tlob_e2e_smoke.py
```

### 4.5 Public API surface

```python
from quantlab import (
    TLOBQConfig,
    TLOBQModel,
    train_tlob_q,
    predict_tlob_q,
    build_extended_feature_rows,
    build_multihorizon_labels,
    build_sequence_windows,
    SequenceWindow,
)
```

### 4.6 Docs

| Doc | Change |
|-----|--------|
| `docs/literature_review.md` | Add TLOB, BDLOB, Async DDQL, Cont-Stoikov OFI, MacroHFT, TimeCatcher entries under the existing quant-finance section. |
| `docs/research_program.md` | Add "Sequence Microstructure Track" under the Quant Research Spine. |
| `docs/quant_research_catalog.md` | Add `tlob_q` row to "Supervised ML Models" with inputs, outputs, artifacts. |
| `README.md` | Add a TLOB-Q quickstart section near the existing quantlab demo block. |

---

## 5. Pipeline

```
1. CURRICULUM          curriculum.generate_regime_mixed_events(seed, n=80K)
   → events.jsonl

2. FEATURES            features_extras.build_extended_feature_rows(events)
   → features.jsonl     (32-channel ExtendedFeatureFrame rows)

3. LABELS              labels_tlob.build_multihorizon_labels(
                          events, horizons=[1,5,20,100], k_smooth=h//2,
                          theta="spread")
   → labels.jsonl      (MultiHorizonLabelFrame rows)

4. ALIGN               sequence.align_feature_label_rows(features, labels)
   → aligned.jsonl     (rows where every horizon h has a valid future window)

5. WINDOWING           sequence.build_sequence_windows(aligned, T=128, stride=1)
   → X: tensor (N, T=128, F=32)
   → Y: dict[h] -> {dir: (N,), vol: (N,), spread: (N,)}

6. WALK-FORWARD        backtest.walk_forward_splits(N, train=60%, val=10%,
                          test=30%, embargo=T)
   → list of 3 fold ranges

7. TRAIN per fold      training.train_one_fold(X_train, Y_train,
                          X_val, Y_val, config)
   → fold{i}_model.pt + fold{i}_metrics.json + training_curves_fold{i}.jsonl

8. INFERENCE per fold  inference.mc_dropout_predict(fold_artifact, X_test,
                          mc_samples=16, horizon=best_sharpe_horizon_on_val)
   # horizon selection happens on the per-fold VALIDATION subset only; the chosen
   # h is then applied to the held-out test subset. Test data never participates
   # in horizon selection.
   → predictions_fold{i}.jsonl

9. BACKTEST per fold   backtest.run_backtest_multihorizon(
                          predictions_fold{i}, labels_fold{i})
   → backtest_fold{i}.json

10. AGGREGATE          training.aggregate_folds(fold_outputs)
    → summary.json     {model, config, fold_metrics, ensemble_metrics?,
                        paper_refs, git_commit, hardware, wall_clock_s,
                        dataset_hash, status}
```

### 5.1 Training loop details

- Optimizer: `AdamW`, lr `3e-4`, weight decay `0.01`
- Schedule: cosine to `0`, 5% linear warmup
- Gradient clipping: `max_norm = 1.0`
- Gradient accumulation: physical batch `64`, effective `256`
- Epochs: `30` (smoke config uses `2`)
- Early stop: composite val metric (mean of per-horizon F1) flat or NaN for `patience = 5` epochs
- EMA decay: `0.999`; EMA weights are what gets saved for inference
- Per-step log row: `{epoch, step, lr, loss, loss_per_horizon, grad_norm}`

### 5.2 Validation metrics per epoch per fold per horizon

- F1 (macro), balanced accuracy, hit-ratio (trades-only subset)
- Volatility RMSE, spread RMSE
- Expected calibration error (ECE) on direction softmax
- Mean MC-dropout `prob_std` across val set (sanity that uncertainty channel is alive)
- Trainable param count, `B*T` samples/sec

### 5.3 Walk-forward arithmetic

- Fold size = `N // 3`
- Inside each fold: train `60%`, val `10%`, test `30%`
- Embargo gap `= max(embargo, T) = 128` rows between train/val and val/test (López de Prado purge-and-embargo)
- Folds slide forward chronologically. No shuffling.
- `Standardizer` is fit per fold on train indices only.

### 5.4 Checkpoint contents (`fold{i}_model.pt`)

- `state_dict` — raw weights
- `ema_state_dict` — EMA weights, used by inference by default
- `standardizer` — per-fold, fit on train only
- `config` — full `TLOBQConfig`
- `metrics` — best-val, final-train, per-epoch curve summary
- `meta`: `{git_commit, seed, hardware, wall_clock_s, dataset_hash, paper_refs}`

### 5.5 Ensemble flow (opt-in via `--ensemble K`)

For `k in 1..K`:
- `seed_k = base_seed + 1000 * k`
- Train an independent fold artifact with that seed.

At inference:
- Each member produces an MC-dropout prediction distribution.
- Aggregate by arithmetic mean of softmax probabilities across members.
- Std is computed across (member × MC sample), giving a richer uncertainty estimate.

### 5.6 Determinism

- Seeds propagated to: Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`, DataLoader worker seeds.
- `torch.use_deterministic_algorithms(True)` best-effort. Some CPU kernels remain nondeterministic; this is documented in `meta` rather than enforced.
- Dataset SHA-256 hash in `meta`. Identical seed + different hash logs a warning.

### 5.7 Smoke vs default configs

- `tlob_smoke.json`: `T=32, F=16, d_model=32, layers=2, epochs=2, events=4K` — runs <60 s on CPU; used by `test_tlob_demo.py` and CI.
- `tlob_m4_default.json`: the locked sizing from §3.
- `tlob_ensemble.json`: `K=3` deep ensemble using `tlob_m4_default` per member.

---

## 6. Error handling & failure modes

### 6.1 Input validation (system boundary)

| Risk | Guard |
|------|-------|
| Events JSONL with missing required fields | `build_extended_feature_rows` raises `ValueError("missing field <X> at row <N>")` before any tensor work |
| Non-monotonic timestamps | Validated once at load; raises with the first offending pair |
| Zero/negative prices or sizes | Coerced/skipped per feature with explicit branch; zero-depth returns `0.0` not NaN |
| Duplicate timestamps | First-write-wins on `align_feature_label_rows`; count of dropped rows is logged in fold meta |
| Empty inputs | All public functions raise `ValueError` with a message naming the empty collection |
| Oversized input files | `config.max_input_rows` (default `5_000_000`) precheck — defeats accidental gigabyte loads |

### 6.2 Numerical stability

- Every divisor guarded (`if total > 0`) or replaced with `max(eps, denom)` with `eps = 1e-12`.
- Log returns gated by `curr > 0 and prev > 0`.
- `Standardizer.fit` replaces `scale < 1e-8` with `1.0`.
- New feature outputs clamped to `[-50, +50]` after standardization.
- All assembled feature tensors pass `torch.isfinite(...).all()` in `build_sequence_windows`; failure raises with offending channel and row index.

### 6.3 Training failures

| Risk | Guard |
|------|-------|
| NaN loss during a step | After `loss.backward()`, if `not torch.isfinite(loss)`, the step is skipped, offending batch indices are logged, counter increments. Counter > 5 in one epoch aborts the fold with `status="aborted_nan_loss"` and writes the partial artifact. |
| Gradient explosion | Hard `clip_grad_norm_(max_norm=1.0)`. Pre-clip grad-norm logged each step. |
| OOM | `RuntimeError` whose message contains "out of memory" is caught with config hint: "reduce `batch_size` or `T`". |
| Wall-clock exceeded | `config.max_wall_seconds` (optional). When set, the trainer finishes the current epoch and exits with `status="time_budget_exhausted"` plus best EMA checkpoint. |
| Flat/NaN val metric | Early-stop after `patience = 5`. Last good EMA snapshot is kept. |

### 6.4 Walk-forward correctness

| Risk | Guard |
|------|-------|
| Not enough rows for 3 folds with embargo | Precheck errors early with the required minimum `N >= 3 * window_size + 2 * embargo + T` |
| Embargo too small ⇒ leakage | `embargo = max(embargo, T)` enforced; used value logged |
| Standardizer fit on train+test | Fit is called inside the fold loop on train indices only; a unit test asserts mean/scale differ across folds |

### 6.5 Inference / production safety

| Risk | Guard |
|------|-------|
| Checkpoint loaded with mismatched config | `load_tlob_q_artifact` rejects with a diff of changed keys |
| Feature schema drift | `TLOBQConfig.feature_keys` pinned; mismatch raises with missing/extra channel names |
| MC-dropout accidentally disabled at inference | Inference path sets `model.train(mode=False)` then re-enables `MCDropout` modules; unit test asserts nonzero sample variance |
| Stale ensemble member | Ensemble loader validates `T`, `F`, `d_model`, `n_layers`, `feature_keys` agree across members |
| `pickle` deserialization risk | `torch.load(weights_only=True)` everywhere |

### 6.6 Backtest pathologies

| Risk | Guard |
|------|-------|
| Model predicts one class everywhere ⇒ zero-trade fold | Reported as `trades=0` with `degenerate=True`; never silently passes as `Sharpe=0` |
| Drawdown stop triggers early | Existing `max_drawdown` + `stopped=True` preserved; multi-horizon backtest also records `stopped_at_step` |
| Predictions/labels out of order | Backtester re-aligns by `timestamp_ms`; asserts intersection size equals both inputs' lengths |
| Zero MC-dropout variance | Inference asserts `prob_std.mean() > 1e-6`; failing raises since κ-scaling would degenerate |

### 6.7 Logging & observability

- Every fold writes `training_curves_fold{i}.jsonl` as it trains.
- `summary.json` written at every fold boundary (incremental), so interrupted runs aren't lost.
- Top-level `status` ∈ `{ok, aborted_nan_loss, time_budget_exhausted, aborted_oom, aborted_data_validation}`.
- CLIs exit non-zero only on hard data validation failures; everything else is a soft fail with diagnostic artifacts.

---

## 7. Testing

### 7.1 Layer 1 — unit tests (per module, deterministic, <50 ms each)

| Module | Key cases |
|--------|-----------|
| `features_extras.py` | 24 channels each tested with hand-computed values; zero-depth ⇒ zero (not NaN); strictly increasing prices ⇒ positive momentum; symmetric prices ⇒ zero imbalance; finite-output assertion on 1 K-row stream |
| `labels_tlob.py` | Hand-computed `l(t,h,k)` on a 16-point sequence for `h=5, k=2`; horizon-bias removal verified; spread-as-θ path; monotonic up ⇒ all "up"; multi-horizon frame shape |
| `sequence.py` | Window count `= N - T + 1`; alignment dropping unmatched timestamps; walk-forward indices sum ≤ N; embargo gap test (no train index within `T` of val); standardizer fit-on-train-only |
| `tlob_modules.py` | BiN output mean-0/var-1 per channel on stationary input; spatial attention shape correctness; temporal attention triangular-mask test (zero gradient from past → future); MCDropout `training=True` for dropout submodules in eval mode; param count within ±5% |
| `tlob_model.py` | Per-horizon forward shape; param count within `[8M, 16M]` budget; save/load round-trip; EMA decay updates correct buffers; `n_layers=0` raises |
| `training.py` | One-step smoke: loss decreases on fixed 64-batch overfit; NaN-loss detector skips step + increments counter; grad-clip clamps grad-norm; cosine + warmup hits expected LR at step 0, warmup, end; walk-forward isolation: train indices never appear in test |
| `inference.py` | MC-dropout produces nonzero variance; ensemble of K=3 produces mean of members; κ-scaled edge formula correctness; horizon selection by best-Sharpe |
| `curriculum.py` | Same seed ⇒ byte-identical events JSONL; regime mix proportions within ±3% of target; latent-pressure clipped; rows≥8 enforced |
| `cli_tlob.py` | Each subcommand parses smoke args; bad config key rejected with key list; mutually exclusive flag groups caught at argparse |

### 7.2 Layer 2 — integration tests

- `test_tlob_demo.py`: runs smoke config end-to-end on tmp directory: curriculum → features → multi-horizon labels → windowing → 1 fold × 2 epochs → MC-dropout inference → backtest → summary. <60 s on CPU. Validates artifact tree, JSONL row counts, JSON schemas.
- `test_walk_forward_integration.py`: 3 folds of smoke config. Asserts per-fold metrics differ (data isolation), aggregate summary equals manual aggregation, `dataset_hash` stable across same-seed runs.
- `test_resume_after_crash.py`: simulates partial fold-2 crash via injected exception; asserts `summary.json` for completed folds is on disk and consistent.

### 7.3 Layer 3 — E2E tests

- `test_tlob_e2e_smoke.py`: invokes `python -m src.quantlab.cli_tlob demo --output-dir /tmp/...` as subprocess. Catches packaging / import-path regressions.

### 7.4 Determinism

- Same seed ⇒ identical loss curve to 6 decimals for the smoke config after `torch.use_deterministic_algorithms(True)`.
- Full config not asserted bit-identical; must be within ±5% final F1 on rerun.
- PyTorch version locked in `requirements.txt`; bit-equality test skipped if `torch.__version__` differs from recorded build.

### 7.5 Failure-mode tests (one per category from §6)

- Missing-field event ⇒ `ValueError` with field name
- Non-monotonic timestamps ⇒ raises with offending pair
- Insufficient rows for 3 folds ⇒ precheck error
- Injected NaN loss ⇒ step skipped, counter increments, abort after 5
- Standardizer fit on train only verified by patched splitter
- Checkpoint config drift ⇒ load rejects with diff
- MC-dropout disabled at inference ⇒ raises
- Degenerate (always-hold) model ⇒ backtest reports `degenerate=True`, `trades=0`, downstream aggregation continues

### 7.6 Numerical-stability tests

- Adversarial inputs (zero sizes, single-row windows, all-identical prices) into each feature; assert finite output.
- Standardizer with constant-feature column survives without `inf`.

### 7.7 Performance / budget tests (env-gated)

- `RUN_BUDGET_TESTS=1` env-gated test runs smoke config 3× and asserts wall-clock < 90 s on CPU. Not part of default `unittest` run.

### 7.8 Coverage

- ≥ 80% per new file under `src/quantlab/`. Verified via `pytest --cov=src/quantlab --cov-report=term-missing`. `pytest-cov` added as dev requirement. Canonical CI entry remains `python -m unittest`.

### 7.9 Test execution time budget

- Full default `unittest` suite (excluding budget tests) < 3 min on M4 CPU. Smoke configs everywhere; only `test_tlob_demo.py` exceeds 30 s.

---

## 8. Limitations

- **Synthetic only.** Until the live-data spec lands, performance on real LOB streams (LOBSTER, Tardis, Binance public WS) is unknown.
- **Inference latency.** ~30–80 ms per row on M4 CPU at default sizing is fine for 1-second crypto cadence but not for sub-100 ms tick-level execution. Future spec will explore quantization (int8) and smaller distilled student models.
- **Determinism is best-effort.** CPU PyTorch kernels are not all deterministic. Documented in `meta`.
- **No live-trading claim.** This is a research artifact, not a trading system.

---

## 9. Acceptance criteria

- `./venv/bin/python -m unittest` passes locally on M4 CPU in ≤ 3 minutes.
- `./venv/bin/python -m src.quantlab.cli_tlob demo --config configs/quantlab/tlob_smoke.json --output-dir /tmp/tlob_q_smoke` runs end-to-end in ≤ 60 s and writes the full artifact tree.
- `./venv/bin/python -m src.quantlab.cli_tlob walk-forward --config configs/quantlab/tlob_m4_default.json --output-dir /tmp/tlob_q_full` runs to completion within 4 h wall-clock and writes `summary.json` with `status="ok"`.
- Test coverage ≥ 80% per new file in `src/quantlab/`.
- README and the three doc files in `docs/` are updated to reference TLOB-Q.
- No new lint regressions (`ruff`, `black` clean) on changed files.
- Branch is committed and pushed to `main` on the user's behalf as authorized.

---

## 10. Follow-up specs (out of scope for this iteration)

1. **Live data ingestion** — Binance public-WebSocket adapter emitting `MarketEvent` JSONL; LOBSTER replayer; recorded-tick replay harness.
2. **ICT / orderflow feature engine** — order blocks, fair-value gaps, liquidity sweeps, market-structure shift, equal-highs / equal-lows, premium/discount zones; integrated as additional feature channels behind a feature-set flag.
3. **RL execution layer** — Avellaneda-Stoikov inventory baseline + PPO/DDQN over TLOB-Q signals; cost-aware reward with inventory penalty.
4. **Conformal calibration** — JANET-style adaptive split-conformal intervals over the regression heads.
5. **Quantization / distillation** — int8 quantization and distilled student model for sub-10 ms inference.

---

## 11. Paper references (consolidated)

| ID | Title | URL |
|----|-------|-----|
| TLOB | TLOB: Novel Transformer with Dual Attention for LOB | https://arxiv.org/abs/2502.15757 |
| BDLOB | Bayesian Deep CNN for LOB | https://arxiv.org/abs/1811.10041 |
| Async DDQL | Async Deep Duelling Q-Learning on LOB | https://arxiv.org/abs/2301.08688 |
| FinRL | FinRL Framework | https://arxiv.org/abs/2111.09395 |
| JANET | Joint Adaptive prediction-region Estimation for Time-series | https://arxiv.org/abs/2407.06390 |
| DOME | Supervised ML validation discipline | https://arxiv.org/abs/2006.16189 |
| TimeCatcher | Volatility-aware variational forecasting | https://arxiv.org/abs/2601.20448 |
| MacroHFT | Memory-augmented context-aware HFT RL | https://arxiv.org/abs/2406.14537 |
| Cont-Stoikov | Multi-level Order Flow Imbalance | https://www.cmu.edu/math/cdam/seminars/cont-stoikov.pdf |
| López de Prado | Advances in Financial Machine Learning, Wiley 2018 (purged k-fold + embargo CV) | https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 |
