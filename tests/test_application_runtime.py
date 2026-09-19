"""
Unit tests for src/application_runtime.py and hardware inspection diagnostics.
"""

import unittest

from src.application_runtime import ApplicationRuntime
from src.utils import inspect_hardware_and_runtime, setup_device


class TestApplicationRuntime(unittest.TestCase):
    def test_singleton_instance(self):
        """Verify ApplicationRuntime enforces process-level singleton."""
        rt1 = ApplicationRuntime.get_instance()
        rt2 = ApplicationRuntime.get_instance()
        self.assertIs(rt1, rt2)

    def test_runtime_snapshot_structure(self):
        """Verify get_snapshot returns all required diagnostic telemetry keys."""
        runtime = ApplicationRuntime.get_instance()
        snapshot = runtime.get_snapshot()

        self.assertIn("status", snapshot)
        self.assertIn("progress_message", snapshot)
        self.assertIn("stages", snapshot)
        self.assertIn("hardware", snapshot)
        self.assertIn("elapsed", snapshot)

        stages = snapshot["stages"]
        self.assertIn("tensorflow", stages)
        self.assertIn("disease_model", stages)
        self.assertIn("soil_model", stages)
        self.assertIn("profiles", stages)
        self.assertIn("warmup", stages)

    def test_inspect_hardware_and_runtime_truthfulness(self):
        """Verify inspect_hardware_and_runtime strictly decouples physical GPU from TF CUDA."""
        hw = inspect_hardware_and_runtime()

        self.assertIsInstance(hw, dict)
        self.assertIn("physical_gpu", hw)
        self.assertIn("hardware_label", hw)
        self.assertIn("tf_cuda_active", hw)
        self.assertIn("tf_device_summary", hw)
        self.assertIn("cuda_status", hw)

        # Integrity invariant: If TF CUDA is False, cuda_status must not claim Active acceleration
        if not hw["tf_cuda_active"]:
            self.assertNotEqual(hw["cuda_status"], "Active")
            self.assertIn("CPU", hw["tf_device_summary"])

    def test_setup_device_truthfulness(self):
        """Verify setup_device does not falsely claim CUDA acceleration when TF has no GPUs."""
        is_cuda, msg = setup_device()
        self.assertIsInstance(is_cuda, bool)
        self.assertIsInstance(msg, str)
        if not is_cuda:
            self.assertTrue("CPU" in msg)


if __name__ == "__main__":
    unittest.main()
