"""Fetch a tiny text corpus through the Hugging Face Dataset Viewer API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASETS_SERVER = "https://datasets-server.huggingface.co"


def dataset_viewer_url(endpoint, **params):
    return f"{DATASETS_SERVER}/{endpoint}?{urlencode(params)}"


def fetch_json(endpoint, **params):
    with urlopen(dataset_viewer_url(endpoint, **params), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_row_text(row, text_field):
    payload = row.get("row", row)
    if text_field not in payload:
        raise KeyError(f"field {text_field!r} not found in row keys: {sorted(payload)}")
    text = str(payload[text_field]).strip()
    return text


def fetch_text_sample(dataset, config, split, text_field, rows):
    validation = fetch_json("is-valid", dataset=dataset)
    if validation.get("valid") is False:
        raise ValueError(f"{dataset} is not a valid Dataset Viewer dataset: {validation}")

    payload = fetch_json(
        "rows",
        dataset=dataset,
        config=config,
        split=split,
        offset=0,
        length=rows,
    )
    texts = [extract_row_text(row, text_field) for row in payload["rows"]]
    return "\n\n".join(text for text in texts if text)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rows < 1 or args.rows > 100:
        raise SystemExit("--rows must be between 1 and 100 for the Dataset Viewer rows API.")
    text = fetch_text_sample(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        text_field=args.text_field,
        rows=args.rows,
    )
    if not text.strip():
        raise SystemExit("Dataset Viewer returned no usable text.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {len(text)} characters to {output}")


if __name__ == "__main__":
    main()
