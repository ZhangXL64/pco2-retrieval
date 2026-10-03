#!/usr/bin/env python3
"""Run perturbation sensitivity analysis for trained retrieval models."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from pco2_retrieval.config import load_config
from pco2_retrieval.data import read_rrs, read_sst
from pco2_retrieval.retrieval import load_artifacts
from pco2_retrieval.sensitivity import run_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--products", nargs="+", help="Subset of configured products")
    parser.add_argument("--dates", nargs="+", help="Subset of configured dates")
    parser.add_argument("--seed", type=int, help="Override the model seed")
    parser.add_argument("--repeats", type=int, help="Override sensitivity repeats")
    parser.add_argument("--noise-std", type=float, help="Override relative noise standard deviation")
    args = parser.parse_args()
    config = load_config(args.config)
    products, dates = args.products or config.products, args.dates or config.dates
    seed = int(config.training["random_seed"] if args.seed is None else args.seed)
    repeats = int(config.sensitivity["repeats"] if args.repeats is None else args.repeats)
    noise = float(config.sensitivity["noise_std"] if args.noise_std is None else args.noise_std)
    rng_seed = int(config.sensitivity["random_seed"])
    if repeats <= 0 or noise <= 0:
        parser.error("repeats and noise-std must be positive")
    records = []
    for product in products:
        artifacts = load_artifacts(config.models_dir / product / f"seed_{seed}")
        for date in dates:
            rrs = read_rrs(config.data_dir / "rrs" / product / f"{date}.tif")
            sst, _ = read_sst(config.data_dir / "sst" / f"{date}.tif")
            result = run_sensitivity(rrs, sst, artifacts, repeats, noise, rng_seed)
            result.insert(0, "date", date)
            result.insert(0, "product", product)
            records.append(result)
    detail = pd.concat(records, ignore_index=True)
    output = config.outputs_dir / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "sensitivity_detail.csv", index=False)
    summary = detail.groupby(["product", "date", "source", "output"], as_index=False)["sensitivity"].agg(
        sensitivity_mean="mean", sensitivity_std="std", repeats="count"
    )
    summary.to_csv(output / "sensitivity_summary.csv", index=False)
    print(output, flush=True)


if __name__ == "__main__":
    main()

