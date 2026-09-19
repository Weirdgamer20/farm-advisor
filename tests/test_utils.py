"""
Unit tests for src/utils.py.
Tests canonical crop normalization, pretty formatting, disease treatment lookups, and device detection.
"""

import unittest
from config import SOIL_CROPS
from src.utils import (
    CROP_ALIASES,
    PRETTY_CROP_NAMES,
    canonical_crop,
    get_disease_treatment,
    pretty_crop,
    setup_device,
)


class TestUtils(unittest.TestCase):
    def test_canonical_crop_standard_names(self):
        """Verify standard canonical crop names pass through correctly."""
        for crop in SOIL_CROPS:
            self.assertEqual(canonical_crop(crop), crop)

    def test_canonical_crop_aliases_and_case(self):
        """Verify case insensitivity and alias normalization."""
        test_cases = [
            ("Corn", "maize"),
            ("corn (maize)", "maize"),
            ("Corn (Maize)", "maize"),
            ("corn maize", "maize"),
            ("Bell Pepper", "bell_pepper"),
            ("bell pepper", "bell_pepper"),
            ("Grape", "grapes"),
            ("grapes", "grapes"),
            ("TOMATO", "tomato"),
            ("Apple", "apple"),
            ("potato", "potato"),
        ]
        for raw, expected in test_cases:
            self.assertEqual(canonical_crop(raw), expected)

    def test_canonical_crop_strict_mode(self):
        """Strict mode should raise ValueError on unsupported crops."""
        with self.assertRaises(ValueError):
            canonical_crop("pineapple", strict=True)
        with self.assertRaises(ValueError):
            canonical_crop("dragonfruit", strict=True)

    def test_canonical_crop_non_strict_mode(self):
        """Non-strict mode should return None on unrecognized crops."""
        self.assertIsNone(canonical_crop("dragonfruit", strict=False))
        self.assertIsNone(canonical_crop("", strict=False))

    def test_pretty_crop(self):
        """Verify human-readable display names."""
        self.assertEqual(pretty_crop("tomato"), "Tomato")
        self.assertEqual(pretty_crop("bell_pepper"), "Bell Pepper")
        self.assertEqual(pretty_crop("maize"), "Corn (Maize)")
        self.assertEqual(pretty_crop("grapes"), "Grape")

    def test_get_disease_treatment_known(self):
        """Verify treatment lookups for known pathology classes."""
        treatments = [
            ("Early Blight", "Chlorothalonil"),
            ("Late Blight", "Mancozeb"),
            ("Apple Scab", "Captan"),
            ("Common Rust", "Pyraclostrobin"),
            ("Bacterial Spot", "copper"),
        ]
        for disease, key_phrase in treatments:
            t = get_disease_treatment(disease)
            self.assertIn("symptoms", t)
            self.assertIn("cultural", t)
            self.assertIn("chemical", t)
            self.assertTrue(
                key_phrase.lower() in t["chemical"].lower()
                or key_phrase.lower() in t["cultural"].lower()
            )

    def test_get_disease_treatment_unknown_fallback(self):
        """Verify fallback treatment recommendations for uncataloged pathologies."""
        t = get_disease_treatment("Mysterious Foliar Condition")
        self.assertIn("symptoms", t)
        self.assertIn("cultural", t)
        self.assertIn("chemical", t)

    def test_setup_device(self):
        """Verify setup_device returns a boolean and non-empty status string."""
        gpu_active, msg = setup_device()
        self.assertIsInstance(gpu_active, bool)
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)


if __name__ == "__main__":
    unittest.main()
