# Hugging Face CPU Training Runbook

This runbook keeps micro-GPT training small enough for a MacBook M4 with 24 GB RAM while still using Hugging Face dataset infrastructure.

## Scope

- Target model: repository-native `src.micro_gpt.model.MicroGPT`.
- Target device: CPU. Do not select MPS/GPU unless a separate experiment explicitly changes the protocol.
- Target corpus: a small text slice from a public Hugging Face Dataset Viewer endpoint.
- Target duration: smoke-scale training, not convergence.

## Dataset Viewer Corpus Fetch

Fetch a bounded TinyStories slice through the Hugging Face Dataset Viewer `/rows` API:

```bash
./venv/bin/python scripts/fetch_hf_text_sample.py \
  --dataset roneneldan/TinyStories \
  --config default \
  --split train \
  --text-field text \
  --rows 32 \
  --output /tmp/tinystories_cpu_sample.txt
```

The script validates the dataset with `/is-valid`, reads the first page through `/rows`, extracts the selected text field, and writes a temporary local corpus. The API returns at most 100 row-like records per request, which is appropriate for a CPU smoke run.

## Local CPU Micro-GPT Training

Run bounded training from the fetched corpus:

```bash
./venv/bin/python -m src.micro_gpt.train \
  --config configs/micro_gpt/cpu_m4_smoke.json \
  --train \
  --text-file /tmp/tinystories_cpu_sample.txt \
  --checkpoint-out /tmp/micro_gpt_cpu_tinystories.pt \
  --metrics-out /tmp/micro_gpt_cpu_tinystories_metrics.json \
  --run-name macbook-m4-cpu-tinystories-smoke
```

Generate from the resulting checkpoint:

```bash
./venv/bin/python -m src.micro_gpt.cli generate \
  --config configs/micro_gpt/cpu_m4_smoke.json \
  --checkpoint /tmp/micro_gpt_cpu_tinystories.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 48
```

## Hugging Face Jobs Boundary

Hugging Face Jobs are useful for managed remote experiments, but they require authenticated Hugging Face access and a Jobs-capable account plan. For this repository, remote Jobs should stay smoke-scale unless a paper experiment explicitly justifies a larger run.

Recommended remote CPU smoke profile:

- flavor: `cpu-basic`
- timeout: 10 to 20 minutes
- dependencies: `torch`, `trackio` only when the job is intended to report externally
- persistence: push meaningful checkpoints to the Hub; otherwise treat the job as ephemeral telemetry

Local paths cannot be passed to a remote Job container. Submit either inline code through the Hugging Face Jobs MCP tool or a public/private Hub-hosted script URL.
