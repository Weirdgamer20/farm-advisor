"""
Unit tests for src/preprocessing.py
Tests image tensor preprocessing, EXIF/RGB conversions, soil input encoding, and validation bounds.
"""

import io
import unittest
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, SOIL_CROPS, SOIL_FEATURES
from src.preprocessing import (
    prepare_soil_inputs,
    preprocess_image,
    validate_soil_readings,
)


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image_pil_rgb(self):
        """Verify preprocessing of standard PIL RGB image."""
        img = Image.new("RGB", (300, 400), color=(100, 150, 200))
        tensor = preprocess_image(img, target_size=(224, 224))

        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_image_rgba_to_rgb(self):
        """Verify RGBA 4-channel image is converted to 3-channel RGB tensor."""
        img = Image.new("RGBA", (150, 150), color=(255, 0, 0, 128))
        tensor = preprocess_image(img, target_size=(224, 224))

        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_preprocess_image_from_bytes(self):
        """Verify preprocessing from raw image bytes."""
        img = Image.new("RGB", (100, 100), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()

        tensor = preprocess_image(raw_bytes, target_size=(224, 224))
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_prepare_soil_inputs_valid(self):
        """Verify shape, dtypes, and values of prepared soil neural tensors."""
        values = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9,
        }
        inputs = prepare_soil_inputs(values, "tomato")

        self.assertIn("numeric", inputs)
        self.assertIn("crop_index", inputs)

        self.assertEqual(inputs["numeric"].shape, (1, len(SOIL_FEATURES)))
        self.assertEqual(inputs["numeric"].dtype, np.float32)

        self.assertEqual(inputs["crop_index"].shape, (1, 1))
        self.assertEqual(inputs["crop_index"].dtype, np.int32)
        self.assertEqual(inputs["crop_index"][0, 0], SOIL_CROPS.index("tomato"))

    def test_prepare_soil_inputs_invalid_crop(self):
        """Verify ValueError is raised when crop is not in SOIL_CROPS."""
        values = {f: 50.0 for f in SOIL_FEATURES}
        with self.assertRaises(ValueError):
            prepare_soil_inputs(values, "unsupported_crop_xyz")

    def test_validate_soil_readings(self):
        """Verify soil validation bounds and warning detections."""
        valid_readings = {
            "N": 60.0,
            "P": 50.0,
            "K": 60.0,
            "temperature": 25.0,
            "humidity": 70.0,
            "ph": 6.5,
            "rainfall": 100.0,
        }
        is_valid, warnings = validate_soil_readings(valid_readings)
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)

        # Test out of bounds pH and temperature
        invalid_readings = valid_readings.copy()
        invalid_readings["ph"] = 16.0
        invalid_readings["temperature"] = -40.0
        is_valid, warnings = validate_soil_readings(invalid_readings)
        self.assertFalse(is_valid)
        self.assertEqual(len(warnings), 2)

        # Test missing parameter
        missing_readings = valid_readings.copy()
        del missing_readings["N"]
        is_valid, warnings = validate_soil_readings(missing_readings)
        self.assertFalse(is_valid)
        self.assertTrue(any("Missing parameter" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
