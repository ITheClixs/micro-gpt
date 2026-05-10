"""Build a temporary quant-finance reasoning corpus from Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


DATASETS_SERVER = "https://datasets-server.huggingface.co"

DATASET_SPECS = [
    {
        "dataset": "VedantPadwal/quantitative-finance-reasoning",
        "config": "default",
        "split": "train",
        "formatter": "vedantpadwal",
    },
    {
        "dataset": "Neil0930/quantitative_finance_dataset",
        "config": "default",
        "split": "train",
        "formatter": "neil0930",
    },
]


def dataset_viewer_url(endpoint, **params):
    return f"{DATASETS_SERVER}/{endpoint}?{urlencode(params)}"


def fetch_json(endpoint, **params):
    with urlopen(dataset_viewer_url(endpoint, **params), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_dataset(dataset):
    payload = fetch_json("is-valid", dataset=dataset)
    if payload.get("valid") is False:
        raise ValueError(f"{dataset} is not valid in Dataset Viewer: {payload}")


def fetch_rows(dataset, config, split, max_rows=None, page_size=100):
    validate_dataset(dataset)
    rows = []
    offset = 0
    while True:
        remaining = None if max_rows is None else max_rows - len(rows)
        if remaining is not None and remaining <= 0:
            break
        length = page_size if remaining is None else min(page_size, remaining)
        payload = fetch_json(
            "rows",
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=length,
        )
        page = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page)
        offset += len(page)
        total = payload.get("num_rows_total")
        if not page or (total is not None and offset >= total):
            break
    return rows


def stringify_validation(validation):
    if not isinstance(validation, dict):
        return ""
    parts = []
    if "score" in validation:
        parts.append(f"validation_score: {validation['score']}")
    if validation.get("reasoning"):
        parts.append(f"validation_reasoning: {validation['reasoning']}")
    if validation.get("key_differences"):
        parts.append("key_differences: " + "; ".join(map(str, validation["key_differences"])))
    if validation.get("error_analysis"):
        parts.append("error_analysis: " + "; ".join(map(str, validation["error_analysis"])))
    return "\n".join(parts)


def format_vedantpadwal(row):
    return "\n".join(
        [
            "### Quantitative Finance Reasoning Example",
            f"id: {row.get('id', '')}",
            f"question: {row.get('question', '')}",
            f"ground_truth: {row.get('ground_truth', '')}",
            f"reasoning_trace: {row.get('reasoning_trace', '')}",
            stringify_validation(row.get("validation", {})),
        ]
    ).strip()


def format_neil0930(row):
    return "\n".join(
        [
            "### Quantitative Finance QA Example",
            f"type: {row.get('type', '')}",
            f"question: {row.get('question', '')}",
            f"solution: {row.get('solution', '')}",
            f"rationale: {row.get('rationale', '')}",
        ]
    ).strip()


FORMATTERS = {
    "vedantpadwal": format_vedantpadwal,
    "neil0930": format_neil0930,
}


def build_corpus(max_rows_per_dataset=None):
    sections = [
        "# Hugging Face Quantitative Finance Reasoning Corpus",
        "",
        "This corpus is assembled from public Hugging Face datasets for local micro-GPT domain tuning.",
        "It is for quantitative-finance reasoning experiments only and is not investment advice.",
        "",
    ]
    metadata = []
    for spec in DATASET_SPECS:
        rows = fetch_rows(
            spec["dataset"],
            spec["config"],
            spec["split"],
            max_rows=max_rows_per_dataset,
        )
        formatter = FORMATTERS[spec["formatter"]]
        sections.append(f"## Dataset: {spec['dataset']}")
        sections.extend(formatter(row) for row in rows)
        sections.append("")
        metadata.append({"dataset": spec["dataset"], "rows": len(rows)})
    return "\n\n".join(section for section in sections if section.strip()) + "\n", metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-out")
    parser.add_argument("--max-rows-per-dataset", type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_rows_per_dataset is not None and args.max_rows_per_dataset < 1:
        raise SystemExit("--max-rows-per-dataset must be positive.")
    corpus, metadata = build_corpus(max_rows_per_dataset=args.max_rows_per_dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    if args.metadata_out:
        metadata_path = Path(args.metadata_out)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(corpus)} characters from {sum(item['rows'] for item in metadata)} rows to {output}")


if __name__ == "__main__":
    main()
