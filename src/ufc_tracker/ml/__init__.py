"""MLflow tracking and model registry helpers."""

from ufc_tracker.ml.registry import (
    DEFAULT_ALIAS,
    PERSON_DETECTOR_DESCRIPTION,
    PERSON_DETECTOR_NAME,
    configure_mlflow,
    register_person_detector,
    resolve_person_detector_weight,
)

__all__ = [
    "DEFAULT_ALIAS",
    "PERSON_DETECTOR_DESCRIPTION",
    "PERSON_DETECTOR_NAME",
    "configure_mlflow",
    "register_person_detector",
    "resolve_person_detector_weight",
]
