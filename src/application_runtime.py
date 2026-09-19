"""
Application runtime manager for Farmer Crop Advisory System.
Manages asynchronous background initialization of TensorFlow, neural networks,
empirical distribution profiles, and inference engine warm-up.
Ensures the Streamlit UI renders immediately without blocking on startup.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from config import SOIL_FEATURES
from src.data_loader import load_models, load_soil_profiles
from src.utils import inspect_hardware_and_runtime

logger = logging.getLogger(__name__)


class ApplicationRuntime:
    _instance: Optional[ApplicationRuntime] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.status: str = "NOT_STARTED"  # NOT_STARTED, INITIALIZING, READY, FAILED
        self.progress_message: str = "Awaiting startup trigger..."
        self.stages: Dict[str, bool] = {
            "tensorflow": False,
            "disease_model": False,
            "soil_model": False,
            "profiles": False,
            "warmup": False,
        }
        self.image_model: Any = None
        self.image_classes: List[str] = []
        self.soil_model: Any = None
        self.soil_profiles: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.hardware: Dict[str, Any] = inspect_hardware_and_runtime()
        self.start_time: float = 0.0
        self.ready_time: float = 0.0

        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> ApplicationRuntime:
        """Process-level singleton retrieval."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_initialization(self, warmup: bool = True) -> None:
        """Starts asynchronous model initialization in a background daemon thread."""
        with self._lock:
            if self.status != "NOT_STARTED":
                return
            self.status = "INITIALIZING"
            self.progress_message = "Initializing background runtime..."
            self.start_time = time.time()
            self._thread = threading.Thread(
                target=self._run_initialization,
                args=(warmup,),
                name="AI-Runtime-Worker",
                daemon=True,
            )
            self._thread.start()

    def _run_initialization(self, warmup: bool) -> None:
        try:
            logger.info("Background runtime initialization started.")

            # Stage 1: TensorFlow initialization
            self.progress_message = "Initializing TensorFlow runtime..."
            import tensorflow as tf
            # Refresh hardware status once TF is active
            self.hardware = inspect_hardware_and_runtime()
            self.stages["tensorflow"] = True

            # Stage 2: Load neural models and taxonomy
            self.progress_message = "Loading plant pathology and soil suitability models..."
            img_m, classes, soil_m = load_models(warmup=False)
            self.image_model = img_m
            self.image_classes = classes
            self.soil_model = soil_m
            self.stages["disease_model"] = True
            self.stages["soil_model"] = True

            # Stage 3: Load empirical quantile profiles
            self.progress_message = "Loading empirical quantile profiles..."
            self.soil_profiles = load_soil_profiles()
            self.stages["profiles"] = True

            # Stage 4: Inference warm-up
            if warmup:
                self.progress_message = "Warming inference execution graphs..."
                try:
                    import numpy as np
                    _ = self.image_model(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
                    _ = self.soil_model(
                        {
                            "numeric": np.zeros((1, len(SOIL_FEATURES)), dtype=np.float32),
                            "crop_index": np.zeros((1, 1), dtype=np.int32),
                        },
                        training=False,
                    )
                except Exception as warm_exc:
                    logger.warning(f"Inference warm-up encountered non-fatal notice: {warm_exc}")
            self.stages["warmup"] = True

            self.ready_time = time.time()
            self.status = "READY"
            self.progress_message = f"All neural models and profiles ready ({self.ready_time - self.start_time:.1f}s)"
            logger.info(f"AI Runtime ready in {self.ready_time - self.start_time:.2f} seconds.")

        except Exception as exc:
            self.status = "FAILED"
            self.error = str(exc)
            self.progress_message = f"Initialization failed: {exc}"
            logger.error(f"Application runtime initialization failed: {exc}", exc_info=True)
        finally:
            self._ready_event.set()

    def is_ready(self) -> bool:
        """Non-blocking check if runtime is ready for inference."""
        return self.status == "READY"

    def wait_until_ready(self, timeout: Optional[float] = 30.0) -> bool:
        """Blocks waiting for runtime readiness with timeout."""
        self._ready_event.wait(timeout=timeout)
        return self.is_ready()

    def get_snapshot(self) -> Dict[str, Any]:
        """Thread-safe status snapshot for UI rendering."""
        return {
            "status": self.status,
            "progress_message": self.progress_message,
            "stages": dict(self.stages),
            "error": self.error,
            "hardware": dict(self.hardware),
            "elapsed": time.time() - self.start_time if self.start_time > 0 else 0.0,
        }
