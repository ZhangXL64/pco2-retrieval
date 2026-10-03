"""Tabular and raster input handling."""

from pathlib import Path
import numpy as np
import pandas as pd


CRUISE_FILES = {
    "202311": "processed_data_202311_pixel_mean.csv",
    "202403": "processed_data_202403_pixel_mean.csv",
    "202411": "processed_data_202411_pixel_mean.csv",
}
CRUISE_DATES = {"202311": "20231102", "202403": "20240309", "202411": "20241127"}
REQUIRED_COLUMNS = {
    "source", "pixel_row", "pixel_col", "temperature", "Sal", "Chla",
    "pCO2water", "TA_mixed", "DIC_mixed",
}


def read_rrs(path: str | Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as dataset:
        if dataset.count != 4:
            raise ValueError(f"Rrs raster must contain four bands: {path}")
        return dataset.read(masked=True).filled(np.nan).astype(np.float32)


def read_sst(path: str | Path):
    import rasterio
    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            raise ValueError(f"SST raster must contain one band: {path}")
        values = dataset.read(1, masked=True).filled(np.nan).astype(np.float32)
        return values, dataset.profile.copy()


def load_product_training_data(config, product: str) -> pd.DataFrame:
    """Sample one product's Rrs scenes at every matched cruise pixel."""
    frames = []
    for cruise, filename in CRUISE_FILES.items():
        csv_path = config.data_dir / "cruise" / filename
        data = pd.read_csv(csv_path, low_memory=False)
        missing = sorted(REQUIRED_COLUMNS - set(data.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        rrs = read_rrs(config.data_dir / "rrs" / product / f"{CRUISE_DATES[cruise]}.tif")
        rows = pd.to_numeric(data["pixel_row"], errors="coerce").to_numpy()
        cols = pd.to_numeric(data["pixel_col"], errors="coerce").to_numpy()
        inside = np.isfinite(rows) & np.isfinite(cols)
        inside &= (rows >= 0) & (rows < rrs.shape[1]) & (cols >= 0) & (cols < rrs.shape[2])
        data = data.loc[inside].copy()
        rows, cols = rows[inside].astype(int), cols[inside].astype(int)
        for index, band in enumerate((443, 482, 561, 655)):
            data[f"Rrs{band}"] = rrs[index, rows, cols]
        data["cruise"] = cruise
        frames.append(data)
    return pd.concat(frames, ignore_index=True)

