"""Dataset registry and manifest building for quant research corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import write_jsonl


DATASET_REGISTRY = {
    "btcusdt_microbar_v2": {
        "source": "Mindbyte-89/btcusdt-microbar-v2",
        "kind": "market_microbars",
        "venue": "crypto",
        "status": "public_mvp",
    },
    "diffquant_btcusdt_1m": {
        "source": "ResearchRL/diffquant-data",
        "kind": "time_series",
        "venue": "crypto",
        "status": "public_mvp",
    },
    "alpha_instruct": {
        "source": "VladHong/Alpha-Instruct",
        "kind": "reasoning",
        "domain": "quant",
        "status": "public_mvp",
    },
    "quantqa": {
        "source": "ReinforceNow/quantqa",
        "kind": "reasoning",
        "domain": "quant",
        "status": "public_mvp",
    },
}


ADAPTER_TARGETS = [
    "cme_futures",
    "ice_futures",
    "fx_spot",
    "lobster",
    "databento",
    "tardis",
    "polygon",
    "broker_websocket",
    "exchange_websocket",
]


def build_dataset_manifest(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = {
        "config_path": str(config_path),
        "dataset": config,
        "registry": DATASET_REGISTRY,
        "adapter_targets": ADAPTER_TARGETS,
    }
    return manifest


def build_manifest_file(config_path, output_path):
    manifest = build_dataset_manifest(config_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Build local dataset manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Write a dataset manifest.")
    build_parser.add_argument("--config", required=True)
    build_parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "build":
        manifest = build_manifest_file(args.config, args.output)
        print(json.dumps({"output": str(Path(args.output)), "datasets": sorted(manifest["registry"])}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

