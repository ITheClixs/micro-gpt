# 2026-05-10 Smarter Quant Micro-GPT Implementation Checklist

## Objectives

- Add a reproducible `quantlab` research spine for public market microstructure work.
- Keep the existing char-level micro-GPT path intact while adding BPE-capable training metadata.
- Ground the new research vocabulary in documentation and unit tests before claiming any capability.

## Implementation Checklist

- [ ] Create `src/quantlab/` with canonical market-event, feature, label, baseline, and backtest primitives.
- [ ] Add deterministic formulas for midprice, spread, microprice, order-flow imbalance, realized volatility, and triple-barrier labels.
- [ ] Add walk-forward backtest accounting with fees, slippage, max position, and a no-trade guardrail.
- [ ] Add CLI entrypoints for dataset, feature, label, baseline, and backtest commands.
- [ ] Add a compact local BPE tokenizer implementation and quant BPE configs for smoke-scale tuning.
- [ ] Preserve the current char tokenizer and existing checkpoint format for backwards compatibility.
- [ ] Add `docs/quant_research_catalog.md` and connect the new work to the repo research docs.
- [ ] Add unit tests for feature math, label generation, baseline decisions, backtest costs, and BPE tokenization.
- [ ] Run repository verification commands and fix any regressions.

## Verification Targets

- `./venv/bin/python -m unittest`
- `./venv/bin/python -m py_compile main.py src/prepare_data.py src/finetune_model.py src/algorithms/*.py src/micro_gpt/*.py src/research_lab/*.py src/quantlab/*.py`
- `./venv/bin/python -m src.micro_gpt.train --config configs/micro_gpt/tiny_debug.json --dry-run`
- `git diff --check`

## Notes

- Keep outputs local by default under `/tmp`.
- Do not claim live trading capability.
- Treat public BTCUSDT orderflow as the MVP source shape; all other venues remain adapter targets.
