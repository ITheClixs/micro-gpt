---
title: Micro-GPT Domain Tuning for Quantitative Trading Research
authors: Micro-GPT Research Lab
date: 2026-05-10
tags: [quantitative-finance, algorithmic-trading, market-microstructure, language-models]
---

# Abstract

This note defines a source-backed domain-tuning corpus for a repository-native micro-GPT model oriented toward quantitative trading research. The goal is not to produce a live trading system or disclose proprietary hedge-fund alpha. The goal is to teach the micro-GPT vocabulary, formulas, and research protocols that recur in public quantitative finance: factor premia, statistical arbitrage, limit order book signals, inventory-aware market making, deep learning on order books, reinforcement learning for execution, and leakage-resistant validation.

# 1. Research Boundary

The model must answer as a research assistant, not as a signal seller. It should distinguish public strategy families from unverified proprietary claims. It should explicitly treat manipulative patterns such as spoofing, layering, wash trading, and marking the close as prohibited behaviors.

The training corpus therefore encodes three constraints:

- cite public sources when discussing fund-like strategy families;
- express strategies as mathematical research objects with assumptions and failure modes;
- require transaction-cost, market-impact, and leakage checks before any alpha claim.

# 2. Public Strategy Families

## 2.1 Style Premia

AQR publicly describes value, momentum, carry, and defensive styles as pervasive market-neutral premia used across asset classes. In corpus form:

```text
score_i = (x_i - mean(x)) / std(x)
position_i = clip(score_i, -c, c)
```

These are not HFT strategies. They are slower, scalable quant research patterns and provide a hedge-fund-style foundation for cross-sectional modeling.

## 2.2 Statistical Arbitrage

Pairs and basket arbitrage model relative value:

```text
s_t = y_t - beta x_t
z_t = (s_t - rolling_mean(s_t)) / rolling_std(s_t)
```

The research target is mean reversion after costs and financing, not a raw z-score. Cointegration tests, Kalman hedge ratios, PCA residuals, and risk-neutralization can be added later.

## 2.3 Market Making

Inventory-aware market making follows the public Avellaneda-Stoikov line of work. A simplified reservation price is:

```text
r_t = S_t - q_t * gamma * sigma^2 * (T - t)
```

This formula trains the model to explain why a dealer lowers or raises quotes when inventory becomes too long or too short.

## 2.4 Order-Flow and LOB Signals

Order-flow imbalance is a core short-horizon microstructure feature:

```text
delta_mid_t = beta * OFI_t / depth_t + epsilon_t
```

Limit-order-book learning also includes book imbalance, queue imbalance, spread, depth, cancel intensity, trade sign, and event-time sampling.

## 2.5 Deep LOB Models

DeepLOB combines convolutional layers over price-level structure with recurrent temporal layers. Newer attention and survival-analysis papers model fill probability:

```text
P(fill before tau | order features, queue state, market state)
```

This distinction matters because execution models often care more about fill and adverse-selection probabilities than about mid-price direction alone.

## 2.6 RL and Hierarchical Execution

Reinforcement learning strategies should be framed with explicit state, action, reward, and cost:

```text
state_t = [features_t, inventory_t, cash_t, risk_state_t]
reward_t = delta_PnL_t - cost_t - lambda_risk * risk_t
```

MacroHFT-style work motivates regime-aware or hierarchical policies, but the corpus should also teach overfitting risk, non-stationarity, and compliance concerns.

# 3. Validation Protocol

The tuned micro-GPT should repeatedly surface the following validation rules:

- use point-in-time data;
- apply triple-barrier labels for event outcomes;
- use meta-labeling to filter a base signal;
- run purged and embargoed cross-validation;
- include transaction costs, spread, slippage, borrow, fees, and impact;
- measure turnover, drawdown, tail loss, capacity, and stability;
- compare to naive baselines such as equal-weight, lagged return, and no-trade policies.

# 4. Training Configuration

The first quant-domain tuning pass uses a small CPU configuration:

```bash
./venv/bin/python -m src.micro_gpt.train \
  --config configs/micro_gpt/quant_cpu_smoke.json \
  --train \
  --text-file data/quant_research_corpus.md \
  --checkpoint-out /tmp/micro_gpt_quant_research.pt \
  --metrics-out /tmp/micro_gpt_quant_research_metrics.json \
  --run-name quant-research-cpu-smoke
```

The run is intentionally smoke-scale. It is a terminal verification that the micro-GPT can ingest a quant research corpus and produce a checkpoint, not evidence of financial skill.

# 5. References

- Avellaneda, M. and Stoikov, S. "High-frequency trading in a limit order book." Quantitative Finance, 2008. https://www.tandfonline.com/doi/abs/10.1080/14697680701381228
- Cont, R., Kukanov, A., and Stoikov, S. "The Price Impact of Order Book Events." Journal of Financial Econometrics, 2014. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822
- Zhang, Z., Zohren, S., and Roberts, S. "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing, 2019. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3519855
- Arroyo, A., Cartea, A., Moreno-Pino, F., and Zohren, S. "Deep attentive survival analysis in limit order books." Quantitative Finance, 2024. https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2286351
- Zong, C., Wang, C., Qin, M., Feng, L., Wang, X., and An, B. "MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading." KDD, 2024. https://huggingface.co/papers/2406.14537
- Easley, D., Lopez de Prado, M., and O'Hara, M. "The Volume Clock: Insights into the High Frequency Paradigm." Journal of Portfolio Management, 2012. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2034858
- Lopez de Prado, M. "Advances in Financial Machine Learning." Wiley, 2018. https://www.oreilly.com/library/view/advances-in-financial/9781119482086/
- AQR. "Investing with Style." https://www.aqr.com/Insights/Research/Journal-Article/Investing-With-Style
- SEC. "SEC Charges New York-Based High Frequency Trading Firm With Fraudulent Trading to Manipulate Closing Prices." https://www.sec.gov/newsroom/press-releases/2014-229
