"""
Farmer Crop Advisory System — Core Package.
Provides modular components:
- data_loader: cached dataset and model loading
- preprocessing: image and numeric tensor transformations
- analysis: inference, empirical quantile diagnostics, and advisory generation
- visualization: UI components, charts, and status indicators
- utils: crop mappings, treatments, and hardware configuration
"""

from src.analysis import (
    classify_leaf,
    diagnose_soil,
    generate_advisory_summary,
    predict_soil,
    recommend_crops,
)
from src.data_loader import (
    load_crop_data,
    load_models,
    load_soil_profiles,
)
from src.preprocessing import (
    prepare_soil_inputs,
    preprocess_image,
    validate_soil_readings,
)
from src.utils import (
    CROP_ALIASES,
    PRETTY_CROP_NAMES,
    canonical_crop,
    get_disease_treatment,
    pretty_crop,
    setup_device,
)
from src.visualization import (
    format_status_badge,
    plot_crop_recommendations,
    plot_soil_parameters_bar,
    render_advisory,
    render_device_status,
    render_disclaimer,
    render_disease_detection,
    render_header,
    render_system_status_sidebar,
    render_workflow_diagram,
)

__all__ = [
    # data_loader
    "load_crop_data",
    "load_models",
    "load_soil_profiles",
    # preprocessing
    "prepare_soil_inputs",
    "preprocess_image",
    "validate_soil_readings",
    # analysis
    "classify_leaf",
    "diagnose_soil",
    "generate_advisory_summary",
    "predict_soil",
    "recommend_crops",
    # visualization
    "format_status_badge",
    "plot_crop_recommendations",
    "plot_soil_parameters_bar",
    "render_advisory",
    "render_device_status",
    "render_disease_detection",
    "render_header",
    # utils
    "CROP_ALIASES",
    "PRETTY_CROP_NAMES",
    "canonical_crop",
    "get_disease_treatment",
    "pretty_crop",
    "setup_device",
]
