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
3. Run a dry run.
4. Run a short smoke training pass.
5. Run the planned training pass only after explicit approval.
6. Save checkpoint metadata and metrics.
7. Generate deterministic samples from fixed prompts.
8. Add results to a metrics artifact before making README claims.

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
