"""Configuration loading for the portable retrieval package."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    products: tuple[str, ...]
    dates: tuple[str, ...]
    bands: tuple[int, ...]
    data_dir: Path
    models_dir: Path
    outputs_dir: Path
    training: dict
    sensitivity: dict


def load_config(path: str | Path) -> ExperimentConfig:
    """Load the supplied JSON-compatible YAML file and resolve its paths."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"products", "dates", "bands", "paths", "training", "sensitivity"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    bands = tuple(int(value) for value in raw["bands"])
    if bands != (443, 482, 561, 655):
        raise ValueError("bands must be ordered as 443, 482, 561, 655")
    base = path.parent
    paths = raw["paths"]
    for key in ("data", "models", "outputs"):
        if key not in paths:
            raise ValueError(f"Configuration paths is missing key: {key}")
    resolve = lambda value: (base / value).resolve()
    return ExperimentConfig(
        path=path,
        products=tuple(raw["products"]),
        dates=tuple(str(value) for value in raw["dates"]),
        bands=bands,
        data_dir=resolve(paths["data"]),
        models_dir=resolve(paths["models"]),
        outputs_dir=resolve(paths["outputs"]),
        training=dict(raw["training"]),
        sensitivity=dict(raw["sensitivity"]),
    )

