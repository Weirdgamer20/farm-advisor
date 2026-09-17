"""
Farmer Crop Advisory System — Compatibility Shim
Redirects execution to modular main.py entry point.
"""

from __future__ import annotations

import runpy
import pathlib

if __name__ == "__main__" or "streamlit" in __name__:
    main_path = pathlib.Path(__file__).resolve().parent / "main.py"
    runpy.run_path(str(main_path), run_name="__main__")
