# Experiment Protocol

## Default Rule

Do not run long training by default. Use dry runs, small deterministic demos, and unit tests until the user explicitly approves a real experiment.

## Required Metadata

Every real experiment must record:

- Git commit.
- Config file path and config hash.
- Dataset name, split, local path, or Hugging Face identifier.
- Tokenizer type and vocabulary size.
- Parameter count.
- Seed.
- Hardware and PyTorch device.
- Number of tokens processed.
- Wall-clock time.
- Train loss, eval loss, perplexity, and sample outputs where applicable.

## Micro-GPT Protocol

1. Validate config.
2. Validate dataset and tokenizer.
3. Run `python -m src.micro_gpt.cli inspect --config <config>` and record parameter count.
4. Run a dry run.
5. Run a one-step terminal smoke pass with `python -m src.micro_gpt.cli smoke`.
6. If checkpoint plumbing changed, save a smoke checkpoint and generate from it.
7. Run the planned training pass only after explicit approval.
8. Save checkpoint metadata and metrics.
9. Generate deterministic samples from fixed prompts.
10. Add results to a metrics artifact before making README claims.

## Visualization Protocol

Visualizations must be generated from executable code. Each visualization should include the input assumptions and the code path that produced it.

Required visualization classes:

- Backprop: loss curve and gradient norms.
- CNN: input, kernel, activation, pooling output.
- RNN: hidden-state trajectory and gradient-flow proxy.
- RL: value map or policy map.
- Micro-GPT: attention map and top token probabilities.

## Hugging Face Protocol

Before any HF Jobs or TRL training:

- Validate dataset format.
- Estimate hardware and timeout.
- Enable Hub persistence.
- Enable Trackio or equivalent metric logging.
- Record base model and license.
- Prefer known baseline models such as SmolLM2 or Qwen3-0.6B for comparison, not as replacements for the from-scratch track.

No HF job should be launched without explicit user approval.

## Alignment and Adapter Protocol

Alignment and adapter primitives are local math utilities until a real model-training request is made.

- DPO tests must assert the reference-corrected margin, not only preference ordering.
- GRPO-style grouped advantages must be normalized within each prompt group.
- LoRA tests must verify frozen base weights, trainable low-rank parameters, initial equivalence, and dtype/device compatibility.
- Lion and Muon-style optimizer tests must assert the exact update equations on deterministic tensors.
