"""Joint-cruise random-forest training for the complete retrieval chain."""

import json
from pathlib import Path
import numpy as np

from .carbon import compute_pco2_mix, fit_dic_coefficients, fit_ta_coefficients
from .data import load_product_training_data
from .features import build_chl_features, build_pco2_bio_features, build_sss_features


RRS_COLUMNS = ["Rrs443", "Rrs482", "Rrs561", "Rrs655"]
RF_PARAMETERS = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 5],
    "max_features": ["sqrt", "log2", 1.0],
}


def _sklearn():
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV, train_test_split
    except ImportError as exc:
        raise ImportError("scikit-learn is required for model training") from exc
    return RandomForestRegressor, RandomizedSearchCV, train_test_split


def _fit_rf(features, target, seed, inner_splits, iterations):
    RF, Search, _ = _sklearn()
    search = Search(
        RF(random_state=seed, n_jobs=-1), RF_PARAMETERS, n_iter=iterations,
        scoring="neg_root_mean_squared_error", cv=inner_splits,
        random_state=seed, n_jobs=-1,
    )
    search.fit(features, target)
    return search.best_estimator_, search.best_params_


def train_product(config, product: str, seed: int | None = None) -> Path:
    """Train one product on all three cruises and save compact artifacts."""
    import joblib
    data = load_product_training_data(config, product)
    numeric_columns = RRS_COLUMNS + [
        "Chla", "Sal", "temperature", "pCO2water", "TA_mixed", "DIC_mixed"
    ]
    numeric = data[numeric_columns].apply(lambda value: __import__("pandas").to_numeric(value, errors="coerce"))
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    valid &= (numeric[RRS_COLUMNS].to_numpy(dtype=float) > 0).all(axis=1)
    work = data.loc[valid].reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(f"Too few valid training records for {product}: {len(work)}")
    rrs = work[RRS_COLUMNS].to_numpy(dtype=float)
    seed = int(config.training["random_seed"] if seed is None else seed)
    _, _, split = _sklearn()
    train_idx, validation_idx = split(
        np.arange(len(work)), test_size=float(config.training["test_size"]),
        random_state=seed, stratify=work["cruise"].astype(str),
    )
    cv = int(config.training["inner_splits"])
    iterations = int(config.training["search_iterations"])

    stage_chl, chl_params = _fit_rf(build_chl_features(rrs[train_idx]), work.loc[train_idx, "Chla"], seed, cv, iterations)
    stage_sss, sss_params = _fit_rf(build_sss_features(rrs[train_idx]), work.loc[train_idx, "Sal"], seed, cv, iterations)
    predicted_chl = stage_chl.predict(build_chl_features(rrs))
    predicted_sss = stage_sss.predict(build_sss_features(rrs))
    sst = work["temperature"].to_numpy(dtype=float)
    ta_coeff = fit_ta_coefficients(predicted_sss[train_idx], sst[train_idx], work.loc[train_idx, "TA_mixed"])
    dic_coeff = fit_dic_coefficients(work.loc[train_idx, "TA_mixed"], work.loc[train_idx, "DIC_mixed"])
    observed_mix = compute_pco2_mix(
        work["TA_mixed"], work["DIC_mixed"], work["Sal"], sst
    )
    bio_target = work["pCO2water"].to_numpy(dtype=float) - observed_mix
    stage_bio, bio_params = _fit_rf(
        build_pco2_bio_features(rrs[train_idx], predicted_chl[train_idx], sst[train_idx], predicted_sss[train_idx]),
        bio_target[train_idx], seed, cv, iterations,
    )
    # Refit every component on all valid records using the selected hyperparameters.
    RF, _, _ = _sklearn()
    chl_model = RF(random_state=seed, n_jobs=-1, **chl_params).fit(build_chl_features(rrs), work["Chla"])
    sss_model = RF(random_state=seed, n_jobs=-1, **sss_params).fit(build_sss_features(rrs), work["Sal"])
    final_chl = chl_model.predict(build_chl_features(rrs))
    final_sss = sss_model.predict(build_sss_features(rrs))
    ta_coeff = fit_ta_coefficients(final_sss, sst, work["TA_mixed"])
    dic_coeff = fit_dic_coefficients(work["TA_mixed"], work["DIC_mixed"])
    bio_model = RF(random_state=seed, n_jobs=-1, **bio_params).fit(
        build_pco2_bio_features(rrs, final_chl, sst, final_sss), bio_target
    )

    output = config.models_dir / product / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(chl_model, output / "chl_rf.joblib")
    joblib.dump(sss_model, output / "sss_rf.joblib")
    joblib.dump(bio_model, output / "pco2_bio_rf.joblib")
    (output / "carbon_coefficients.json").write_text(json.dumps({
        "ta": ta_coeff.tolist(), "dic": dic_coeff.tolist(),
        "band_order_nm": [443, 482, 561, 655], "random_seed": seed,
    }, indent=2), encoding="utf-8")
    return output

