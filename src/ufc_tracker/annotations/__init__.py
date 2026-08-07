"""Temporal annotation contracts and window generation for strike detection."""

from ufc_tracker.annotations.contracts import (
    ALLOWED_LABELS,
    SCHEMA_VERSION,
    AnnotationConfig,
    load_annotation_config,
    validate_annotation_rows,
)
from ufc_tracker.annotations.windows import generate_annotation_windows

__all__ = [
    "ALLOWED_LABELS",
    "SCHEMA_VERSION",
    "AnnotationConfig",
    "generate_annotation_windows",
    "load_annotation_config",
    "validate_annotation_rows",
]

