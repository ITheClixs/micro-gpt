# Quantitative ML Trading Corpus for Micro-GPT

This corpus is for local language-model domain tuning only. It is not investment advice and it does not disclose proprietary trading systems. It summarizes public research patterns used in quantitative finance, algorithmic trading, and market microstructure.

## Research Rules

Quantitative trading research must separate alpha discovery, execution, risk sizing, and backtest validation. A signal is not a strategy until it survives transaction costs, slippage, borrow constraints, capacity limits, latency assumptions, data revisions, survivorship bias, and out-of-sample validation.

Never state that a private hedge fund uses a strategy unless the claim is public and cited. Use the phrase "publicly documented strategy family" for methods such as factor premia, statistical arbitrage, market making, trend following, order-flow models, and reinforcement learning execution.

## Strategy Families

Factor premia: a cross-sectional long-short portfolio ranks assets by a characteristic score. Public factor families include value, momentum, carry, and defensive or low-risk styles. A simple z-scored factor signal is:

score_i = (x_i - mean(x)) / std(x)
position_i = clip(score_i, -c, c)

Cross-sectional momentum goes long recent relative winners and short recent relative losers. Time-series momentum estimates each asset independently and trades in the direction of its own trend:

r_t = log(P_t / P_{t-L})
signal_t = sign(r_t)

Statistical arbitrage: pairs trading models a stationary spread between two related assets. Estimate beta by regression, form spread s_t = y_t - beta x_t, then trade deviations from equilibrium:

z_t = (s_t - rolling_mean(s_t)) / rolling_std(s_t)
enter short spread when z_t > z_entry
enter long spread when z_t < -z_entry
exit when abs(z_t) < z_exit

Cointegration-based stat arb tests whether a linear combination of prices is stationary. A Kalman filter can adapt hedge ratio beta_t through time.

Volatility arbitrage: compare implied volatility with forecast realized volatility. The edge is not the volatility forecast alone; it depends on options Greeks, transaction costs, hedging frequency, jumps, and volatility risk premium.

Event-driven and news sentiment: transform textual events into timestamped features. Prevent lookahead by using the publication timestamp available to the trader, not the later database timestamp.

## Market Microstructure Signals

Limit order book imbalance:

mid_t = (best_bid_t + best_ask_t) / 2
spread_t = best_ask_t - best_bid_t
book_imbalance_t = (bid_size_t - ask_size_t) / (bid_size_t + ask_size_t)

Order-flow imbalance at the best quotes summarizes signed pressure from limit orders, market orders, and cancellations. A short-horizon linear model often starts with:

delta_mid_t = beta * OFI_t / depth_t + epsilon_t

Queue imbalance:

QI_t = queue_ahead_bid_t / (queue_ahead_bid_t + queue_ahead_ask_t)

VPIN-style flow toxicity uses volume-time buckets rather than clock-time buckets. The research use case is detecting toxic flow or regime stress; it is not a standalone profit claim.

## Market Making

Inventory-aware market making balances spread capture against adverse selection and inventory risk. A simplified Avellaneda-Stoikov reservation price is:

r_t = S_t - q_t * gamma * sigma^2 * (T - t)

where S_t is mid-price, q_t is inventory, gamma is risk aversion, sigma is volatility, and T - t is remaining horizon. A simplified optimal spread is:

delta_bid + delta_ask = gamma * sigma^2 * (T - t) + (2 / gamma) * log(1 + gamma / k)

where k parameterizes order-arrival decay with quote distance. More aggressive inventory reduction shifts quotes to encourage trades that reduce q_t.

Do not train a model to perform spoofing, layering, marking the close, wash trading, or momentum ignition designed to manipulate prices. These are compliance failures, not alpha research.

## Deep Learning for LOB Data

DeepLOB-style models combine convolutional filters for spatial structure across price levels with recurrent modules for temporal dependence. The target is often a short-horizon class:

y_t = {-1, 0, +1}
y_t = sign(mid_{t+h} - mid_t) after thresholding by tick or volatility.

Transformer and attention models can model multi-level LOB states, fill probabilities, and execution timing. A survival-style fill model estimates:

P(fill before tau | order features, queue state, market state)

This probability supports passive-versus-aggressive routing decisions.

## Reinforcement Learning and Execution

RL trading should model state, action, reward, transaction costs, and market impact explicitly:

state_t = [features_t, inventory_t, cash_t, risk_state_t]
action_t = target_position or order_type
reward_t = delta_PnL_t - cost_t - lambda_risk * risk_t

Hierarchical or regime-aware RL can route between specialist sub-policies for trend, volatility, or liquidity regimes. This is useful for research but must be tested against overfitting, non-stationarity, and market-impact assumptions.

Optimal execution research separates alpha from execution. Almgren-Chriss style objectives penalize both expected cost and variance:

min E[cost] + lambda * Var(cost)

## Financial Machine Learning Hygiene

Labels should respect trading horizons. Triple-barrier labeling assigns an event outcome according to which barrier is touched first:

upper barrier: +theta
lower barrier: -theta
vertical barrier: max holding time

Meta-labeling predicts whether a primary signal should be acted upon. It learns trade filtering or sizing conditional on a base side prediction.

Purged cross-validation removes training samples whose label windows overlap the test fold. Embargo keeps a temporal gap after the test fold to reduce leakage.

Features should be timestamp-valid. Never use future-adjusted corporate actions, restated fundamentals, revised macro data, or complete-day aggregates before they would have been known.

## Risk, Costs, and Sizing

Expected return is not sufficient. Track Sharpe, Sortino, drawdown, turnover, hit rate, tail risk, capacity, latency sensitivity, and market impact.

Kelly sizing maximizes log growth under idealized assumptions:

f_star = edge / variance

For a binary bet with win probability W and win/loss ratio R:

K = W - (1 - W) / R

Use fractional Kelly in noisy strategies. Position caps, volatility targeting, drawdown limits, and stop-loss logic are risk controls, not proof of edge.

Transaction cost model:

net_return_t = gross_return_t - spread_cost_t - fee_t - slippage_t - borrow_t - impact_t

Market impact often scales nonlinearly with participation rate and volume. Capacity analysis asks whether expected alpha survives after size increases.

## Research Prompt Pack

Explain how an order-flow imbalance signal can be transformed into a leakage-aware supervised learning dataset.

Derive the inventory adjustment in a market-making reservation price and explain why inventory risk shifts quotes.

Compare cross-sectional momentum, time-series momentum, and statistical arbitrage.

Design a purged walk-forward backtest for an LOB mid-price classifier.

Explain why fill-probability forecasting is different from price forecasting.

Describe how a hierarchical RL trading system can overfit market regimes and how to test against that failure.

List compliance constraints for HFT research and identify manipulative patterns that must be excluded.

Convert a raw alpha score into a position using volatility targeting, transaction-cost filtering, and fractional Kelly sizing.

## Source Anchors

Public sources used to construct this corpus include Avellaneda and Stoikov on inventory-aware market making, Cont, Kukanov, and Stoikov on order-flow imbalance, Zhang, Zohren, and Roberts on DeepLOB, Arroyo, Cartea, Moreno-Pino, and Zohren on fill-probability forecasting with convolutional-transformers, Zong et al. on MacroHFT, Easley, Lopez de Prado, and O'Hara on volume-clock HFT, Lopez de Prado on financial machine learning validation, AQR on value, momentum, carry, and defensive style premia, and SEC enforcement material on manipulative HFT conduct.
