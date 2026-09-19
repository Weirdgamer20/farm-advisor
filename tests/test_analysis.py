"""
Unit tests for src/analysis.py.
Tests soil diagnostics, advisory generation, root cause correlation, and prescription logic.
"""

import unittest

from config import SOIL_FEATURES
from src.analysis import diagnose_soil, generate_advisory_summary


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.mock_profiles = {
            "tomato": {
                "N": {"p10": 40.0, "p25": 60.0, "p75": 100.0, "p90": 120.0},
                "P": {"p10": 30.0, "p25": 40.0, "p75": 60.0, "p90": 70.0},
                "K": {"p10": 35.0, "p25": 50.0, "p75": 80.0, "p90": 95.0},
                "temperature": {"p10": 18.0, "p25": 21.0, "p75": 27.0, "p90": 30.0},
                "humidity": {"p10": 55.0, "p25": 65.0, "p75": 80.0, "p90": 85.0},
                "ph": {"p10": 5.5, "p25": 6.0, "p75": 6.8, "p90": 7.2},
                "rainfall": {"p10": 60.0, "p25": 80.0, "p75": 120.0, "p90": 140.0},
            }
        }

    def test_diagnose_soil_deviations(self):
        """Verify empirical quantile diagnostic statuses."""
        readings = {
            "N": 30.0,            # below p10 (40.0) -> LOW
            "P": 50.0,            # between p25 & p75 -> NORMAL
            "K": 20.0,            # below p10 (35.0) -> LOW
            "temperature": 24.0,  # NORMAL
            "humidity": 90.0,     # above p90 (85.0) -> HIGH
            "ph": 5.0,            # below p10 (5.5) -> LOW
            "rainfall": 150.0,    # above p90 (140.0) -> HIGH
        }

        diag = diagnose_soil(readings, "tomato", profiles_dict=self.mock_profiles)
        self.assertEqual(len(diag), len(SOIL_FEATURES))

        status_map = {d["Parameter"]: d["Status"] for d in diag}
        self.assertEqual(status_map["N"], "LOW")
        self.assertEqual(status_map["P"], "NORMAL")
        self.assertEqual(status_map["K"], "LOW")
        self.assertEqual(status_map["humidity"], "HIGH")
        self.assertEqual(status_map["ph"], "LOW")
        self.assertEqual(status_map["rainfall"], "HIGH")

    def test_generate_advisory_summary_healthy(self):
        """Verify advisory output when crop is healthy."""
        leaf_res = {
            "crop": "tomato",
            "crop_display": "Tomato",
            "disease": "Healthy",
            "confidence": 0.98,
            "reliable": True,
            "treatment": {},
        }
        soil_res = {
            "crop": "tomato",
            "crop_display": "Tomato",
            "score": 85.0,
            "status": "GOOD",
        }
        advisory = generate_advisory_summary(
            leaf_result=leaf_res,
            soil_result=soil_res,
            diagnostics=[],
            top_crops=[],
            values={"humidity": 65.0, "rainfall": 80.0, "K": 60.0, "N": 70.0, "ph": 6.5},
        )

        self.assertEqual(advisory["crop_display"], "Tomato")
        self.assertEqual(advisory["disease"], "Healthy")
        self.assertEqual(len(advisory["disease_causes"]), 0)

    def test_generate_advisory_summary_disease_root_causes(self):
        """Verify that fungal disease triggers root cause explanations when conditions are adverse."""
        leaf_res = {
            "crop": "tomato",
            "crop_display": "Tomato",
            "disease": "Early Blight",
            "confidence": 0.94,
            "reliable": True,
            "treatment": {"cultural": "Mulch soil", "chemical": "Apply Mancozeb"},
        }
        soil_res = {
            "crop": "tomato",
            "crop_display": "Tomato",
            "score": 42.0,
            "status": "NEEDS ATTENTION",
        }
        adverse_values = {
            "humidity": 85.0,
            "rainfall": 140.0,
            "K": 25.0,
            "N": 120.0,
            "ph": 5.2,
        }

        diag = [
            {"Parameter": "N", "Value": 120.0, "Optimal_Range": "60.0 - 100.0", "Status": "HIGH", "Analysis": "high"},
            {"Parameter": "K", "Value": 25.0, "Optimal_Range": "50.0 - 80.0", "Status": "LOW", "Analysis": "low"},
            {"Parameter": "ph", "Value": 5.2, "Optimal_Range": "6.0 - 6.8", "Status": "LOW", "Analysis": "acidic"},
        ]

        advisory = generate_advisory_summary(
            leaf_result=leaf_res,
            soil_result=soil_res,
            diagnostics=diag,
            top_crops=[],
            values=adverse_values,
            extra_dummy_kwarg="safe",
        )

        self.assertEqual(advisory["disease"], "Early Blight")
        self.assertGreaterEqual(len(advisory["disease_causes"]), 3)
        self.assertTrue(any("humidity" in c.lower() for c in advisory["disease_causes"]))
        self.assertTrue(any("potassium" in c.lower() for c in advisory["disease_causes"]))

        self.assertEqual(len(advisory["prescriptions"]), 3)
        prescription_params = [p["parameter"] for p in advisory["prescriptions"]]
        self.assertIn("N", prescription_params)
        self.assertIn("K", prescription_params)
        self.assertIn("ph", prescription_params)


if __name__ == "__main__":
    unittest.main()
