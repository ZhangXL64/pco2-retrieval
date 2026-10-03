#!/usr/bin/env python3
"""Train the joint-cruise random-forest retrieval chain."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco2_retrieval.config import load_config
from pco2_retrieval.training import train_product


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--products", nargs="+", help="Subset of configured products")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    args = parser.parse_args()
    config = load_config(args.config)
    products = args.products or config.products
    unknown = sorted(set(products) - set(config.products))
    if unknown:
        parser.error(f"unknown products: {unknown}")
    for product in products:
        print(f"Training {product}...", flush=True)
        print(train_product(config, product, args.seed), flush=True)


if __name__ == "__main__":
    main()

