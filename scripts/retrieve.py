#!/usr/bin/env python3
"""Run full-scene retrieval for configured products and dates."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco2_retrieval.config import load_config
from pco2_retrieval.data import read_rrs, read_sst
from pco2_retrieval.retrieval import load_artifacts, retrieve_scene, write_retrieval_tif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--products", nargs="+", help="Subset of configured products")
    parser.add_argument("--dates", nargs="+", help="Subset of configured dates")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    args = parser.parse_args()
    config = load_config(args.config)
    products, dates = args.products or config.products, args.dates or config.dates
    if set(products) - set(config.products) or set(dates) - set(config.dates):
        parser.error("products and dates must be present in config.yaml")
    seed = int(config.training["random_seed"] if args.seed is None else args.seed)
    for product in products:
        artifacts = load_artifacts(config.models_dir / product / f"seed_{seed}")
        for date in dates:
            rrs = read_rrs(config.data_dir / "rrs" / product / f"{date}.tif")
            sst, profile = read_sst(config.data_dir / "sst" / f"{date}.tif")
            arrays, _ = retrieve_scene(rrs, sst, artifacts)
            output = config.outputs_dir / "retrieval" / product / f"{date}_all_parameters.tif"
            write_retrieval_tif(output, arrays, profile)
            print(output, flush=True)


if __name__ == "__main__":
    main()

