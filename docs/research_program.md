# Micro-GPT Research Program

## Abstract

This repository studies how core deep-learning mechanisms behave at small scale. The project combines from-scratch PyTorch-primitive implementations, controlled visualizations, and Hugging Face baselines to support research-paper-style experimentation without requiring long training runs by default.

The central model track is a micro-GPT: a compact decoder-only Transformer for next-token language modeling. The surrounding laboratory implements and visualizes the foundations that make such a model work: backpropagation, optimizers, CNNs, RNNs, reinforcement learning objectives, attention, tokenization, sampling, and evaluation.

## Research Questions

1. Which architectural choices matter most for small decoder-only language models: positional encoding, normalization, MLP type, depth, width, context length, or optimizer schedule?
2. How do gradient flow, initialization, residual connections, and normalization affect trainability across MLPs, CNNs, RNNs, and Transformers?
3. Which visual probes best reveal whether a learning algorithm is behaving correctly?
4. How does a from-scratch micro-GPT compare with Hugging Face small-LM baselines such as SmolLM2 and Qwen3-0.6B under constrained compute?
5. Which post-training methods from TRL, including SFT, DPO, and GRPO, are appropriate future extensions after a pretraining baseline is established?

## System Design

The repository has three layers:

- Algorithm layer: manual, inspectable implementations of backpropagation, optimizers, convolution, recurrence, RL returns/objectives, and sequence modeling.
- Research model layer: a micro-GPT implementation with RoPE, RMSNorm, SwiGLU, causal attention, tied embeddings, generation, metrics, configs, and dry-run training.
- Visualization layer: a local Streamlit app that displays tensors produced by the implementation rather than static educational figures.

The existing GPT-2 summarization path remains as a Hugging Face baseline and historical comparison point. It is not the primary research target.

## Experimental Tracks

### Backpropagation and Optimization

The backpropagation track uses explicit derivatives and finite-difference/autograd agreement tests. It supports gradient-norm and loss-surface visualizations for learning-rate, activation, and initialization studies.

The optimizer track compares SGD, momentum, RMSProp, Adam, and AdamW updates on controlled surfaces. Later work should add warmup, cosine decay, gradient clipping, and schedule ablations.

### CNNs

The CNN track implements convolution and pooling directly with tensor operations. It visualizes kernels, feature maps, activation distributions, and gradient responses. ConvNeXt is the modern reference point for how convolutional priors remain competitive with Transformer-era design.

### RNNs and Sequence Models

The RNN track implements vanilla RNN, GRU, and LSTM cells. It visualizes hidden-state trajectories and gradient-flow pathologies under truncated BPTT. Mamba, Mamba-2, and Titans are treated as modern sequence-modeling references for future recurrence and memory modules.

### Reinforcement Learning

The RL track starts with GridWorld and implements returns, advantages, DQN-style value learning, REINFORCE, actor-critic, and PPO-style clipped objectives. It should later connect to TRL methods for language-model post-training, especially DPO, GRPO, and reward modeling.

### Micro-GPT

The micro-GPT track implements a small causal language model with modern decoder-only components. Initial experiments use deterministic dry runs and TinyStories-style corpora. Future training should report config, seed, parameter count, token budget, validation perplexity, generated samples, hardware, and wall-clock time.

## Evaluation

Minimum metrics:

- Train/eval loss and perplexity for language modeling.
- Parameter count, token count, and tokens/sec.
- Gradient norms and activation statistics.
- Attention entropy and token-probability concentration.
- RL episode return and value/policy maps.

No result should be included in README claims unless produced by a committed script and config.

## Future Work

- Add byte-pair or unigram tokenization and compare against character tokenization.
- Add ablations for learned positions vs RoPE, LayerNorm vs RMSNorm, GELU vs SwiGLU, and SDPA vs manual attention.
- Add a Hugging Face dataset-inspection command before any managed training job.
- Add TRL recipes for SFT, DPO, GRPO, and reward modeling after the pretraining baseline is validated.
- Add mechanistic interpretability probes: activation patching, attention-head entropy, residual stream norms, and logit-lens views.
