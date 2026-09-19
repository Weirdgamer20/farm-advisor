"""
Unit tests for src/data_loader.py.
Tests model loading validation, profile parsing, and missing artifact fallbacks.
"""

import pathlib
import tempfile
import unittest

from config import SOIL_CROPS, SOIL_FEATURES, SOIL_PROFILES
from src.data_loader import (
    check_artifact_availability,
    get_sample_images,
    load_crop_data,
    load_soil_profiles,
)


class TestDataLoader(unittest.TestCase):
    def test_load_soil_profiles_existing(self):
        """Verify empirical quantile profiles load properly from models/soil_condition_profiles.json."""
        if SOIL_PROFILES.exists():
            profiles = load_soil_profiles(SOIL_PROFILES)
            self.assertIsInstance(profiles, dict)
            for crop in SOIL_CROPS:
                self.assertIn(crop, profiles)
                for feat in SOIL_FEATURES:
                    self.assertIn(feat, profiles[crop])
                    stats = profiles[crop][feat]
                    self.assertIn("p10", stats)
                    self.assertIn("p25", stats)
                    self.assertIn("p75", stats)
                    self.assertIn("p90", stats)
                    self.assertLessEqual(stats["p10"], stats["p25"])
                    self.assertLessEqual(stats["p25"], stats["p75"])
                    self.assertLessEqual(stats["p75"], stats["p90"])

    def test_load_soil_profiles_nonexistent(self):
        """Verify load_soil_profiles returns empty dict when path is nonexistent."""
        profiles = load_soil_profiles(pathlib.Path("/nonexistent/path/profiles.json"))
        self.assertEqual(profiles, {})

    def test_load_crop_data_custom_df(self):
        """Verify load_crop_data loads and normalizes CSV data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("N,P,K,temperature,humidity,ph,rainfall,crop\n")
            f.write("90,42,43,20.8,82.0,6.5,202.9,Corn (Maize)\n")
            f.write("60,50,40,25.0,70.0,6.0,100.0,Tomato\n")
            temp_path = pathlib.Path(f.name)

        try:
            df = load_crop_data(temp_path)
            self.assertIsNotNone(df)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]["crop"], "maize")
            self.assertEqual(df.iloc[1]["crop"], "tomato")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_check_artifact_availability(self):
        """Verify check_artifact_availability returns structured telemetry dictionary."""
        telemetry = check_artifact_availability()
        self.assertIsInstance(telemetry, dict)
        self.assertIn("disease_model", telemetry)
        self.assertIn("soil_model", telemetry)
        self.assertIn("disease_classes", telemetry)
        self.assertIn("soil_profiles", telemetry)
        self.assertIn("dataset_csv", telemetry)

    def test_get_sample_images(self):
        """Verify get_sample_images handles non-existent or empty directory gracefully."""
        res = get_sample_images(pathlib.Path("/nonexistent/test_dir"))
        self.assertEqual(res, [])


if __name__ == "__main__":
    unittest.main()
