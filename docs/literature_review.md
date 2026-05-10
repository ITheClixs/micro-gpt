# Literature Review

This document tracks the research references that shape the repository. It is not a claim that this code reproduces each paper; it records design pressure and future comparison targets.

## Language Modeling and Small LMs

- [Attention Is All You Need](https://huggingface.co/papers/1706.03762): Transformer architecture and scaled dot-product attention. The micro-GPT track uses causal self-attention as the core sequence mixer.
- [TinyStories](https://huggingface.co/papers/2305.07759): Small language models trained on simple stories can produce coherent English below conventional parameter scales. This is the primary data regime for future micro-GPT experiments.
- [SmolLM2](https://huggingface.co/papers/2502.02737): Data-centric training of small language models; useful as a Hugging Face baseline family for constrained-compute comparison.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): A current compact open model reference with a 0.6B causal LM variant and published technical report linkage.
- [FlashAttention](https://huggingface.co/papers/2205.14135): IO-aware exact attention. This repo uses simple attention first and treats PyTorch SDPA/flash-compatible paths as a future efficiency track.

## Sequence Models and Memory

- [Mamba](https://huggingface.co/papers/2312.00752): Selective state-space sequence modeling and a modern alternative to quadratic attention.
- [Mamba-2 / State Space Duality](https://huggingface.co/papers/2405.21060): Connects Transformers and SSMs through structured state-space duality and motivates future sequence-model ablations.
- [Titans](https://huggingface.co/papers/2501.00663): Neural long-term memory for attention and recurrence. This is a future reference for memory-augmented micro-models.

## Vision and Convolution

- [ConvNeXt](https://huggingface.co/papers/2201.03545): Modernized ConvNets remain competitive by incorporating Transformer-era design choices while preserving convolutional inductive bias.

## Reinforcement Learning

- [DQN](https://www.nature.com/articles/nature14236): Deep Q-learning from high-dimensional inputs; a reference for value-based RL demos.
- [PPO](https://huggingface.co/papers/1707.06347): Clipped policy-gradient optimization; a reference for the PPO objective implemented in the RL track.
- [TRL](https://huggingface.co/docs/trl): Hugging Face post-training library covering SFT, DPO, GRPO, reward modeling, PPO, and related methods.
- [LeRobot](https://huggingface.co/docs/lerobot/index): Hugging Face robotics and RL ecosystem; useful for future embodied RL or imitation-learning extensions.

## Quantitative Finance and Market Microstructure

- [Avellaneda-Stoikov market making](https://www.math.nyu.edu/~avellane/AvellanedaStoikov.pdf): Inventory-aware market making with explicit spread and risk controls.
- [DeepLOB](https://huggingface.co/papers?q=Limit+Order+Book): Deep learning on limit order books; a reference point for order-book feature pipelines.
- [TLOB](https://huggingface.co/papers/2502.15757): Transformer-style limit order book modeling for future sequence-model experiments.
- [FinRL](https://arxiv.org/abs/2111.09395): Full-stack trading research framework with explicit data, environment, reward, and backtest layers. The quantlab demo borrows this reproducible pipeline shape without claiming DRL parity.
- [FinRL-Meta](https://huggingface.co/papers/2112.06753): Dataset and environment infrastructure for financial RL research.
- [DRL crypto market making](https://huggingface.co/papers/1911.08647): Reinforcement learning in crypto market making, useful as a public analogue for inventory-sensitive strategies.
- [MacroHFT](https://huggingface.co/papers/2406.14537): Regime routing and high-frequency inference ideas for future market-state gating.

## Time-Series Uncertainty and Validation

- [JANET](https://arxiv.org/abs/2407.06390): Adaptive conformal prediction regions for time series; a future target for calibrated quantlab prediction intervals over sequential market states.
- [DOME](https://arxiv.org/abs/2006.16189): Structured supervised-ML reporting discipline around data, optimization, model, and evaluation. The quantlab demo follows this by writing explicit artifacts for each stage.

## Research Hygiene

Use these references as anchors for design and evaluation, not as decoration. A new feature that claims to implement a paper-derived method should include:

- A source link.
- A local docstring or documentation note explaining the simplified scope.
- A unit test for the core mathematical behavior.
- A visualization or metric that helps inspect failure modes.
