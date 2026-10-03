import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ConfigTests(unittest.TestCase):
    def test_loads_paths_relative_to_config(self):
        from pco2_retrieval.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps({
                "products": ["landsat", "modis"],
                "dates": ["20231102"],
                "bands": [443, 482, 561, 655],
                "paths": {"data": "data", "models": "models", "outputs": "outputs"},
                "training": {"random_seed": 0, "test_size": 0.3, "inner_splits": 2, "search_iterations": 1},
                "sensitivity": {"noise_std": 0.15, "repeats": 20, "random_seed": 42},
            }), encoding="utf-8")
            cfg = load_config(config_path)
            self.assertEqual(cfg.bands, (443, 482, 561, 655))
            self.assertEqual(cfg.data_dir, root / "data")


class FeatureTests(unittest.TestCase):
    def test_feature_values_and_order(self):
        from pco2_retrieval.features import build_chl_features, build_sss_features

        rrs = np.array([[1.0, 2.0, 4.0, 8.0]])
        chl = build_chl_features(rrs)
        self.assertEqual(chl.shape, (1, 12))
        self.assertTrue(np.allclose(chl[0, :4], [1, 2, 4, 8]))
        expected_ci = 4 - (1 + (561 - 443) / (655 - 443) * (8 - 1))
        self.assertAlmostEqual(chl[0, 4], expected_ci)
        self.assertTrue(np.allclose(build_sss_features(rrs), rrs))

    def test_rejects_wrong_band_count(self):
        from pco2_retrieval.features import build_chl_features
        with self.assertRaisesRegex(ValueError, "four"):
            build_chl_features(np.ones((2, 3)))


class CarbonTests(unittest.TestCase):
    def test_ta_and_dic_fit_recover_known_relations(self):
        from pco2_retrieval.carbon import fit_dic_coefficients, fit_ta_coefficients, predict_dic, predict_ta

        sss = np.array([28, 30, 32, 34, 36, 38, 40], dtype=float)
        sst = np.array([15, 18, 21, 24, 27, 30, 33], dtype=float)
        coeff = np.array([2200, 3, 0.2, -2, 0.1], dtype=float)
        ta = predict_ta(sss, sst, coeff)
        fitted = fit_ta_coefficients(sss, sst, ta)
        self.assertTrue(np.allclose(predict_ta(sss, sst, fitted), ta))
        dic = 0.9 * ta + 50
        dic_coeff = fit_dic_coefficients(ta, dic)
        self.assertTrue(np.allclose(predict_dic(ta, dic_coeff), dic))

    def test_invalid_carbonate_inputs_return_nan_without_solver(self):
        from pco2_retrieval.carbon import compute_pco2_mix
        result = compute_pco2_mix([-1.0, np.nan], [2000, 2000], [30, 30], [25, 25])
        self.assertTrue(np.isnan(result).all())


if __name__ == "__main__":
    unittest.main()
