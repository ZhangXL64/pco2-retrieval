"""TA, DIC, and carbonate-system functions used by retrieval."""

import numpy as np


def _ta_design(sss, sst):
    sss = np.asarray(sss, dtype=float)
    sst = np.asarray(sst, dtype=float)
    if sss.shape != sst.shape:
        raise ValueError("SSS and SST must have the same shape")
    return np.column_stack((
        np.ones(sss.size), sss.ravel() - 35.0, (sss.ravel() - 35.0) ** 2,
        sst.ravel() - 20.0, (sst.ravel() - 20.0) ** 2,
    ))


def fit_ta_coefficients(sss, sst, ta) -> np.ndarray:
    design = _ta_design(sss, sst)
    target = np.asarray(ta, dtype=float).ravel()
    valid = np.isfinite(design).all(axis=1) & np.isfinite(target)
    if valid.sum() < 5:
        raise ValueError("At least five finite records are required to fit TA")
    return np.linalg.lstsq(design[valid], target[valid], rcond=None)[0]


def predict_ta(sss, sst, coefficients) -> np.ndarray:
    shape = np.asarray(sss).shape
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (5,):
        raise ValueError("TA coefficients must contain five values")
    return (_ta_design(sss, sst) @ coefficients).reshape(shape)


def fit_dic_coefficients(ta, dic) -> np.ndarray:
    ta = np.asarray(ta, dtype=float).ravel()
    dic = np.asarray(dic, dtype=float).ravel()
    valid = np.isfinite(ta) & np.isfinite(dic)
    if valid.sum() < 2:
        raise ValueError("At least two finite records are required to fit DIC")
    return np.linalg.lstsq(
        np.column_stack((ta[valid], np.ones(valid.sum()))), dic[valid], rcond=None
    )[0]


def predict_dic(ta, coefficients) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2,):
        raise ValueError("DIC coefficients must contain slope and intercept")
    return np.asarray(ta, dtype=float) * coefficients[0] + coefficients[1]


def compute_pco2_mix(ta, dic, salinity, temperature) -> np.ndarray:
    """Solve the carbonate system, leaving invalid inputs as NaN."""
    arrays = np.broadcast_arrays(*[
        np.asarray(value, dtype=float) for value in (ta, dic, salinity, temperature)
    ])
    output = np.full(arrays[0].shape, np.nan, dtype=float)
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    valid &= (arrays[0] > 0) & (arrays[1] > 0) & (arrays[2] >= 0)
    if not valid.any():
        return output
    try:
        import PyCO2SYS as pyco2
    except ImportError as exc:
        raise ImportError("PyCO2SYS is required for carbonate-system retrieval") from exc
    result = pyco2.sys(
        par1=arrays[0][valid], par2=arrays[1][valid], par1_type=1, par2_type=2,
        salinity=arrays[2][valid], temperature=arrays[3][valid],
        temperature_out=arrays[3][valid], pressure_out=0,
    )
    output[valid] = np.asarray(result["pCO2_out"], dtype=float)
    return output

