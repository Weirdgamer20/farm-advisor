"""
Tests for Farmer Crop Advisory Engine (src.engine)
"""

import unittest
from src.engine import (
    canonical_crop,
    pretty_crop,
    get_disease_treatment,
    generate_advisory,
    plot_soil_bars,
    SOIL_CROPS,
    SOIL_FEATURES,
    CROP_ALIASES,
)


class TestEngine(unittest.TestCase):
    def test_canonical_crop(self):
        self.assertEqual(canonical_crop("corn"), "maize")
        self.assertEqual(canonical_crop("Corn (Maize)"), "maize")
        self.assertEqual(canonical_crop("corn maize"), "maize")
        self.assertEqual(canonical_crop("Bell Pepper"), "bell_pepper")
        self.assertEqual(canonical_crop("grape"), "grapes")
        self.assertEqual(canonical_crop("Tomato"), "tomato")
        self.assertEqual(canonical_crop("unknown_crop"), "unknown crop")

    def test_pretty_crop(self):
        self.assertEqual(pretty_crop("apple"), "Apple")
        self.assertEqual(pretty_crop("bell_pepper"), "Bell Pepper")
        self.assertEqual(pretty_crop("maize"), "Corn (Maize)")
        self.assertEqual(pretty_crop("custom_crop"), "Custom Crop")

    def test_get_disease_treatment(self):
        apple_scab = get_disease_treatment("Apple Scab")
        self.assertIn("symptoms", apple_scab)
        self.assertIn("cultural", apple_scab)
        self.assertIn("chemical", apple_scab)

        # Fallback for unknown disease
        unknown = get_disease_treatment("Nonexistent Disease")
        self.assertIn("symptoms", unknown)
        self.assertIn("Consult local agricultural extension", unknown["chemical"])

    def test_generate_advisory_healthy(self):
        leaf_res = {
            "crop": "tomato",
            "crop_display": "Tomato",
            "disease": "Healthy",
            "confidence": 0.95,
            "treatment": {},
        }
        soil_res = {
            "crop": "tomato",
            "score": 85.0,
            "status": "GOOD",
        }
        diagnostics = [
            {
                "Parameter": "N",
                "Value": 80.0,
                "Status": "NORMAL",
                "Analysis": "N is within optimal range.",
                "Optimal_Range": "60.0 - 90.0",
            }
        ]
        top_crops = [
            {"crop": "tomato", "crop_display": "Tomato", "score": 85.0, "status": "GOOD"}
        ]
        readings = {"N": 80.0, "humidity": 65.0, "rainfall": 80.0, "ph": 6.5, "K": 50.0}

        advisory = generate_advisory(leaf_res, soil_res, diagnostics, top_crops, readings)
        self.assertEqual(advisory["crop_display"], "Tomato")
        self.assertEqual(advisory["disease"], "Healthy")
        self.assertEqual(len(advisory["disease_causes"]), 0)
        self.assertEqual(len(advisory["prescriptions"]), 0)
        self.assertTrue(any("well-aligned" in r for r in advisory["recommendations"]))

    def test_generate_advisory_disease_and_imbalance(self):
        leaf_res = {
            "crop": "apple",
            "crop_display": "Apple",
            "disease": "Apple Scab",
            "confidence": 0.92,
            "treatment": get_disease_treatment("Apple Scab"),
        }
        soil_res = {
            "crop": "apple",
            "score": 35.0,
            "status": "NOT SUITABLE",
        }
        diagnostics = [
            {
                "Parameter": "N",
                "Value": 20.0,
                "Status": "LOW",
                "Analysis": "N is critically low.",
                "Optimal_Range": "50.0 - 80.0",
            },
            {
                "Parameter": "ph",
                "Value": 5.0,
                "Status": "LOW",
                "Analysis": "ph is critically low.",
                "Optimal_Range": "6.0 - 7.0",
            },
        ]
        top_crops = [
            {"crop": "grapes", "crop_display": "Grape", "score": 88.0, "status": "GOOD"},
            {"crop": "apple", "crop_display": "Apple", "score": 35.0, "status": "NOT SUITABLE"},
        ]
        readings = {
            "N": 20.0,
            "humidity": 85.0,
            "rainfall": 150.0,
            "ph": 5.0,
            "K": 20.0,
        }

        advisory = generate_advisory(leaf_res, soil_res, diagnostics, top_crops, readings)
        self.assertEqual(advisory["crop_display"], "Apple")
        self.assertEqual(advisory["disease"], "Apple Scab")
        self.assertTrue(len(advisory["disease_causes"]) > 0)
        self.assertEqual(len(advisory["prescriptions"]), 2)
        self.assertTrue(any("Crop rotation insight" in r for r in advisory["recommendations"]))

    def test_plot_soil_bars(self):
        self.assertIsNone(plot_soil_bars([]))
        diagnostics = [
            {"Parameter": "N", "Value": 50.0, "Status": "NORMAL"},
            {"Parameter": "P", "Value": 20.0, "Status": "LOW"},
        ]
        fig = plot_soil_bars(diagnostics)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
