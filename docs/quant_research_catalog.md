# Quant Research Catalog

This catalog defines the public-source market microstructure and quant reasoning surfaces that the repository can study without claiming access to proprietary hedge-fund alpha.

## MVP Data Surfaces

- `btcusdt_microbar_v2`: public BTCUSDT microbar-style market data for order-flow and short-horizon direction experiments.
- `diffquant_btcusdt_1m`: public BTCUSDT 1-minute time series for slower signal baselines and regime checks.
- `alpha_instruct`: public quant reasoning instruction data for formula grounding and explanation generation.
- `quantqa`: public quantitative reasoning QA data for terminology, derivation, and protocol language.

## Feature Registry

The `src.quantlab.features` module exposes deterministic primitives:

- Midprice: $m_t = (b_t + a_t)/2$
- Spread: $s_t = a_t - b_t$
- Microprice: weighted by best-bid and best-ask queue size
- Order-flow imbalance: best-quote changes translated into signed queue pressure
- Book imbalance and depth imbalance: normalized bid-vs-ask pressure
- Signed volume: buy volume positive, sell volume negative
- Realized volatility: root sum of squared log returns

## Label Registry

The `src.quantlab.labels` module defines research targets:

- Direction label: short-horizon sign of the future midprice move, optionally cost-thresholded.
- Realized volatility label: short-horizon volatility estimate on the future path.
- Triple-barrier label: first-touch classification over upper/lower return barriers.
- Action label: a derived buy/sell/hold tag for backtest inspection.

## Baselines

- Ridge-style linear direction model over order-flow features.
- EWMA volatility baseline.
- Cost-aware action gating that suppresses trades when the expected edge does not exceed fees, slippage, and a no-trade threshold.

## Supervised ML Models

- `mlp_direction`: a compact CPU-safe PyTorch MLP over stationary order-flow features.
- Inputs: order-flow imbalance, book imbalance, depth imbalance, signed volume, realized volatility, and spread.
- Outputs: down/hold/up probabilities, an expected-edge score defined as `P(up) - P(down)`, and a thresholded buy/sell/hold action.
- Artifacts: local `.pt` model files, JSON metrics, and JSONL prediction rows under `/tmp` by default.

## Backtest Rules

- Walk-forward splits must keep train, validation, and test windows disjoint.
- Transaction costs are modeled explicitly in basis points.
- Position limits and drawdown stops are mandatory.
- No live-trading claim should be made unless a separate execution stack exists and is tested.

## Adapter Targets

Future venue adapters can map the same schema to:

- CME and ICE futures
- FX spot feeds
- LOBSTER-style historical limit order book data
- Databento, Tardis, Polygon, broker webhooks, and exchange websockets

## Related References

- `docs/literature_review.md`
- `docs/research_program.md`
- `src/quantlab/features.py`
- `src/quantlab/labels.py`
- `src/quantlab/baselines.py`
- `src/quantlab/backtest.py`
- `src/quantlab/models.py`
