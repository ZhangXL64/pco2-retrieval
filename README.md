# Estuarine pCO2 retrieval workflow

This repository contains the minimum data and Python code needed to reproduce the paper's joint-cruise random-forest retrieval chain and its perturbation sensitivity analysis. It deliberately excludes plotting, paper-table generation, image-comparison metrics, and historical experiment variants.

## Included data

- Four-band Rrs GeoTIFFs (443, 482, 561, and 655 nm) for Landsat, MODIS, ESTARFM, USTFM, FSDAF, STFDCNN, STFGAN, SwinSTFM, and STFDiff.
- Scenes dated 2023-11-02, 2024-03-09, and 2024-11-27.
- Landsat SST for the same dates.
- Pixel-matched records from the November 2023, March 2024, and November 2024 cruises.

The Rrs files store the bands in the fixed order `443, 482, 561, 655`. They share the SST pixel dimensions; the retrieval outputs inherit georeferencing from the SST GeoTIFFs. `data/MANIFEST.sha256` records every published input file.

## Retrieval chain

For each Rrs product, the code trains a Chl RF from four bands and eight derived spectral features, and an SSS RF from the four bands. Predicted SSS and underway SST are used to fit the five-coefficient TA relationship. A linear relationship maps TA to DIC. PyCO2SYS computes the mixing component, while a third RF predicts the biological component from Rrs, Chl, SST, and SSS. Total pCO2 is the sum of mixing and biological components.

The seeded 70/30 split is used only for leakage-safe model selection and coefficient fitting. After hyperparameters are selected, the released models and coefficients are fitted using all valid cruise records, matching the final whole-scene inference setup.

## Installation

Create a Python 3.10 or newer environment, then run:

```powershell
python -m pip install -r requirements.txt
```

All paths in `config.yaml` are relative to that configuration file. The file uses JSON syntax, which is also valid YAML, and is parsed without an additional YAML dependency.

## Commands

Train all nine product-specific chains:

```powershell
python scripts/train.py --config config.yaml
```

Run all full-scene retrievals:

```powershell
python scripts/retrieve.py --config config.yaml
```

Run sensitivity analysis:

```powershell
python scripts/run_sensitivity.py --config config.yaml
```

Each command accepts `--products`; retrieval and sensitivity also accept `--dates`. Use `--help` for details.

Retrieval GeoTIFF bands are: total pCO2, mixing pCO2, biological pCO2, TA, DIC, SSS, SST, and Chl.

## Sensitivity defaults

The default experiment applies multiplicative Gaussian noise with relative standard deviation 0.15 for 20 repetitions using random seed 42. It evaluates Rrs443, Rrs482, Rrs561, Rrs655, Chl, SSS, SST, and TA against the eight retrieval outputs. Only CSV results are generated.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Full training is computationally intensive and is not part of the unit-test suite. Tests cover configuration, spectral features, TA/DIC fits, raster contracts, deterministic perturbation, and command-line availability.

## Release note

The repository owner must select and add an appropriate software license and confirm redistribution rights for the included satellite and cruise datasets before public release.
