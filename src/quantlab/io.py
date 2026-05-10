"""Simple file IO helpers for local quant research artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def read_table(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value, default=0.0):
    if value is None or value == "":
        return float(default)
    return float(value)


def parse_int(value, default=0):
    if value is None or value == "":
        return int(default)
    return int(value)

