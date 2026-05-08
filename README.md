# GPT-2 Fine-Tuning For Summarization

This project fine-tunes a GPT-2 style causal language model for summarization. By default it uses a very small subset of the Hugging Face `cnn_dailymail` dataset, and it can also train from local JSON/JSONL summarization files.

## What It Does

Running `python3 main.py`:

1. Loads `cnn_dailymail` version `3.0.0`.
2. Selects 100 training articles and 10 validation articles.
3. Formats each example as a causal language modeling prompt:

   ```text
   Summarize the following article:

   <article>

   Summary:
   <highlights>
   ```

4. Masks the prompt portion in the labels so loss is computed only on the summary tokens.
5. Fine-tunes GPT-2 for 1 epoch.
6. Saves the trained model to `./fine_tuned_gpt2_summarizer`.

For local data, provide separate training and validation files. Each record must include an article/text column and a summary column:

```json
{"text": "Long source document...", "summary": "Short target summary."}
```

## Requirements

- Python 3.8+
- `accelerate`
- `transformers`
- `datasets`
- `torch`
- `requests`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Default demo run:

```bash
cd /path/to/gpt2-finetune
python3 main.py
```

Local JSON/JSONL run:

```bash
python3 main.py \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --article-column text \
  --summary-column summary \
  --model-checkpoint gpt2 \
  --train-size 100 \
  --eval-size 10 \
  --num-train-epochs 1
```

See all options:

```bash
python3 main.py --help
```

Run the lightweight unit tests:

```bash
python3 -m unittest
```

## Notes

- The code expects the dataset and model to be available through Hugging Face, unless they are already cached locally.
- Training is intentionally small and fast, so summary quality will be limited.
- The preprocessing is designed for GPT-2 style causal LM training, not encoder-decoder summarization.
- Local files are loaded through Hugging Face `datasets` with `dataset_name="json"` automatically when `--train-file` and `--validation-file` are provided.
