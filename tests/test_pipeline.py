import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RasterDataTests(unittest.TestCase):
    def test_reads_four_band_rrs_and_rejects_wrong_count(self):
        import rasterio
        from rasterio.transform import from_origin
        from pco2_retrieval.data import read_rrs

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.tif"
            bad = Path(tmp) / "bad.tif"
            profile = {"driver": "GTiff", "height": 2, "width": 3,
                       "dtype": "float32", "transform": from_origin(0, 2, 1, 1)}
            with rasterio.open(good, "w", count=4, **profile) as dst:
                dst.write(np.ones((4, 2, 3), dtype="float32"))
            with rasterio.open(bad, "w", count=3, **profile) as dst:
                dst.write(np.ones((3, 2, 3), dtype="float32"))
            self.assertEqual(read_rrs(good).shape, (4, 2, 3))
            with self.assertRaisesRegex(ValueError, "four bands"):
                read_rrs(bad)


class SensitivityTests(unittest.TestCase):
    def test_perturbation_is_reproducible_and_preserves_nan(self):
        from pco2_retrieval.sensitivity import perturb_valid_values

        values = np.array([1.0, np.nan, 2.0])
        first = perturb_valid_values(values, np.random.default_rng(42), 0.15)
        second = perturb_valid_values(values, np.random.default_rng(42), 0.15)
        self.assertTrue(np.allclose(first, second, equal_nan=True))
        self.assertTrue(np.isnan(first[1]))

    def test_complete_sensitivity_table_has_all_source_output_pairs(self):
        from pco2_retrieval.sensitivity import OUTPUTS, SOURCES, run_sensitivity

        class MeanModel:
            def __init__(self, features, offset):
                self.n_features_in_ = features
                self.offset = offset
            def predict(self, values):
                return np.mean(values, axis=1) + self.offset

        artifacts = {
            "chl": MeanModel(12, 1), "sss": MeanModel(4, 25),
            "bio": MeanModel(7, 10),
            "ta": np.array([2200, 1, 0, 1, 0], dtype=float),
            "dic": np.array([0.9, 50], dtype=float),
        }
        rrs = np.full((4, 2, 2), 0.01)
        sst = np.full((2, 2), 25.0)
        solver = lambda ta, dic, sal, temp: ta - dic + sal + temp
        result = run_sensitivity(
            rrs, sst, artifacts, repeats=1, noise_std=0.15,
            random_seed=42, mix_solver=solver,
        )
        self.assertEqual(len(result), len(SOURCES) * len(OUTPUTS))
        self.assertEqual(set(result["source"]), set(SOURCES))


class RetrievalTests(unittest.TestCase):
    def test_no_valid_carbon_pixels_returns_nodata(self):
        from pco2_retrieval.retrieval import retrieve_scene

        class ConstantModel:
            def __init__(self, features, value):
                self.n_features_in_ = features
                self.value = value
            def predict(self, values):
                if len(values) == 0:
                    raise ValueError("empty model input")
                return np.full(len(values), self.value)

        artifacts = {
            "chl": ConstantModel(12, -1), "sss": ConstantModel(4, -1),
            "bio": ConstantModel(7, 10),
            "ta": np.array([2200, 1, 0, 1, 0]),
            "dic": np.array([0.9, 50]),
        }
        arrays, _ = retrieve_scene(
            np.full((4, 1, 1), 0.01), np.full((1, 1), 25.0), artifacts,
            mix_solver=lambda *args: np.array([]),
        )
        self.assertTrue(np.isnan(arrays).all())


if __name__ == "__main__":
    unittest.main()
