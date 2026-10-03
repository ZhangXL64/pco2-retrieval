"""Spectral features used by the Chl and SSS random forests."""

import numpy as np


def _samples_by_band(rrs) -> np.ndarray:
    values = np.asarray(rrs, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Rrs must contain four columns ordered 443, 482, 561, 655")
    return values


def build_chl_features(rrs) -> np.ndarray:
    """Return four Rrs bands and the eight derived Chl features."""
    values = _samples_by_band(rrs)
    r443, r482, r561, r655 = values.T
    with np.errstate(divide="ignore", invalid="ignore"):
        ci2 = r561 - (r443 + (561 - 443) / (655 - 443) * (r655 - r443))
        oc4_x2 = np.maximum(r443, r482) / r561
        derived = np.column_stack((
            ci2, oc4_x2, r482 / r443, r561 / r443,
            r443 / r655, r482 / r561, r482 / r655, r561 / r655,
        ))
    return np.column_stack((values, derived))


def build_sss_features(rrs) -> np.ndarray:
    """Return the four Rrs bands used by the SSS model."""
    return _samples_by_band(rrs).copy()


def build_pco2_bio_features(rrs, chl, sst, sss) -> np.ndarray:
    values = _samples_by_band(rrs)
    extra = np.column_stack((chl, sst, sss)).astype(float)
    if len(extra) != len(values):
        raise ValueError("Rrs and upstream predictions must have the same length")
    return np.column_stack((values, extra))

