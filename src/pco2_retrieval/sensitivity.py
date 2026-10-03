"""Reproducible perturbation sensitivity analysis without plotting."""

import numpy as np
import pandas as pd
from .retrieval import retrieve_scene

SOURCES = ("Rrs443", "Rrs482", "Rrs561", "Rrs655", "Chl", "SSS", "SST", "TA")
OUTPUTS = ("pCO2", "pCO2_mix", "pCO2_bio", "TA", "DIC", "SSS", "SST", "Chl")


def perturb_valid_values(values, rng, relative_std):
    result = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(result) & (result >= 0)
    result[valid] *= 1.0 + rng.normal(0, relative_std, valid.sum())
    result[valid] = np.maximum(result[valid], 0)
    return result


def relative_sensitivity(baseline, changed, source_before, source_after):
    valid = np.isfinite(baseline) & np.isfinite(changed)
    valid &= np.isfinite(source_before) & np.isfinite(source_after)
    if not valid.any():
        return np.nan
    output_scale = np.mean(np.abs(baseline[valid]))
    input_scale = np.mean(np.abs(source_before[valid]))
    output_change = np.mean(np.abs(changed[valid] - baseline[valid]))
    input_change = np.mean(np.abs(source_after[valid] - source_before[valid]))
    if min(output_scale, input_scale, input_change) == 0:
        return np.nan
    return float((output_change / output_scale) / (input_change / input_scale))


def run_sensitivity(
    rrs, sst, artifacts, repeats=20, noise_std=0.15, random_seed=42,
    mix_solver=None,
):
    retrieval_kwargs = {} if mix_solver is None else {"mix_solver": mix_solver}
    baseline, baseline_internal = retrieve_scene(rrs, sst, artifacts, **retrieval_kwargs)
    baseline_sources = {f"Rrs{band}": rrs[i] for i, band in enumerate((443, 482, 561, 655))}
    baseline_sources.update(baseline_internal)
    records = []
    for source_index, source in enumerate(SOURCES):
        for repeat in range(repeats):
            rng = np.random.default_rng(random_seed + source_index * 100000 + repeat)
            changed_source = None
            def perturb(stage, *values):
                nonlocal changed_source
                if stage == "inputs":
                    changed_rrs, changed_sst = values[0].copy(), values[1].copy()
                    if source.startswith("Rrs"):
                        index = SOURCES.index(source)
                        changed_rrs[index] = perturb_valid_values(changed_rrs[index], rng, noise_std)
                        changed_source = changed_rrs[index]
                    elif source == "SST":
                        changed_sst = perturb_valid_values(changed_sst, rng, noise_std)
                        changed_source = changed_sst
                    return changed_rrs, changed_sst
                if stage == "upstream":
                    changed = list(values)
                    mapping = {"Chl": 0, "SSS": 1, "SST": 2}
                    if source in mapping and not (source == "SST" and changed_source is not None):
                        index = mapping[source]
                        changed[index] = perturb_valid_values(changed[index], rng, noise_std)
                        changed_source = changed[index]
                    return tuple(changed)
                if stage == "ta":
                    value = values[0]
                    if source == "TA":
                        value = perturb_valid_values(value, rng, noise_std)
                        changed_source = value
                    return value
            changed, changed_internal = retrieve_scene(
                rrs, sst, artifacts, perturb=perturb, **retrieval_kwargs
            )
            source_after = changed_source if source.startswith("Rrs") else changed_internal[source]
            for output_index, output in enumerate(OUTPUTS):
                records.append({
                    "source": source, "repeat": repeat, "output": output,
                    "sensitivity": relative_sensitivity(
                        baseline[output_index], changed[output_index],
                        baseline_sources[source], source_after,
                    ),
                })
    return pd.DataFrame(records)
