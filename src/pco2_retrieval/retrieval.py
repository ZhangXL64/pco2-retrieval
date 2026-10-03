"""Full-scene inference using saved random-forest artifacts."""

import json
from pathlib import Path
import numpy as np

from .carbon import compute_pco2_mix, predict_dic, predict_ta
from .features import build_chl_features, build_pco2_bio_features, build_sss_features

OUTPUT_NAMES = ("pCO2", "pCO2_mix", "pCO2_bio", "TA", "DIC", "SSS", "SST", "Chl")


def load_artifacts(model_dir):
    import joblib
    root = Path(model_dir)
    coefficients = json.loads((root / "carbon_coefficients.json").read_text(encoding="utf-8"))
    return {
        "chl": joblib.load(root / "chl_rf.joblib"),
        "sss": joblib.load(root / "sss_rf.joblib"),
        "bio": joblib.load(root / "pco2_bio_rf.joblib"),
        "ta": np.asarray(coefficients["ta"], dtype=float),
        "dic": np.asarray(coefficients["dic"], dtype=float),
    }


def _predict(model, features, chunk_size=250_000):
    if features.shape[1] != model.n_features_in_:
        raise ValueError(f"Model expects {model.n_features_in_} features, got {features.shape[1]}")
    result = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        result[start:start + chunk_size] = model.predict(features[start:start + chunk_size])
    return result


def retrieve_scene(rrs, sst, artifacts, mix_solver=compute_pco2_mix, perturb=None):
    """Return eight retrieval layers and the internal source maps."""
    rrs = np.asarray(rrs, dtype=float).copy()
    sst = np.asarray(sst, dtype=float).copy()
    if rrs.ndim != 3 or rrs.shape[0] != 4 or rrs.shape[1:] != sst.shape:
        raise ValueError("Rrs must have shape (4, height, width) matching SST")
    if perturb:
        rrs, sst = perturb("inputs", rrs, sst)
    height, width = sst.shape
    flat_rrs = rrs.reshape(4, -1).T
    flat_sst = sst.ravel()
    valid = np.isfinite(flat_rrs).all(axis=1) & (flat_rrs > 0).all(axis=1)
    valid &= np.isfinite(flat_sst) & (flat_sst >= -2) & (flat_sst <= 45)
    output = np.full((8, height * width), np.nan, dtype=np.float32)
    sources = {name: np.full(height * width, np.nan) for name in ("Chl", "SSS", "SST", "TA")}
    if not valid.any():
        return output.reshape(8, height, width), {k: v.reshape(height, width) for k, v in sources.items()}
    pixels = np.flatnonzero(valid)
    values = flat_rrs[valid]
    chl = _predict(artifacts["chl"], build_chl_features(values))
    sss = _predict(artifacts["sss"], build_sss_features(values))
    temperature = flat_sst[valid]
    if perturb:
        chl, sss, temperature = perturb("upstream", chl, sss, temperature)
    ta = predict_ta(sss, temperature, artifacts["ta"])
    if perturb:
        ta = perturb("ta", ta)
    dic = predict_dic(ta, artifacts["dic"])
    good = np.isfinite(chl) & (chl >= 0) & np.isfinite(sss) & (sss >= 0)
    good &= np.isfinite(ta) & (ta > 0) & np.isfinite(dic) & (dic > 0)
    pixels, values = pixels[good], values[good]
    chl, sss, temperature, ta, dic = [item[good] for item in (chl, sss, temperature, ta, dic)]
    mix = mix_solver(ta, dic, sss, temperature)
    bio = _predict(artifacts["bio"], build_pco2_bio_features(values, chl, temperature, sss))
    total = mix + bio
    solved = np.isfinite(total) & np.isfinite(mix) & np.isfinite(bio) & (total >= 0)
    pixels = pixels[solved]
    layers = (total, mix, bio, ta, dic, sss, temperature, chl)
    for index, layer in enumerate(layers):
        output[index, pixels] = np.asarray(layer)[solved]
    for name, layer in zip(("Chl", "SSS", "SST", "TA"), (chl, sss, temperature, ta)):
        sources[name][pixels] = np.asarray(layer)[solved]
    return output.reshape(8, height, width), {k: v.reshape(height, width) for k, v in sources.items()}


def write_retrieval_tif(path, arrays, reference_profile):
    import rasterio
    profile = dict(reference_profile)
    profile.update(driver="GTiff", count=8, dtype="float32", nodata=np.nan, compress="deflate")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.asarray(arrays, dtype=np.float32))
        for index, name in enumerate(OUTPUT_NAMES, 1):
            destination.set_band_description(index, name)

