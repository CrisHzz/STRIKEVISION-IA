"""MLflow tracking and model registry helpers."""

from ufc_tracker.ml.registry import (
    DEFAULT_ALIAS,
    DEFAULT_POSE_WEIGHT_FILENAME,
    PERSON_DETECTOR_DESCRIPTION,
    PERSON_DETECTOR_NAME,
    POSE_ESTIMATOR_DESCRIPTION,
    POSE_ESTIMATOR_MERGE_DESCRIPTION,
    POSE_ESTIMATOR_MERGE_NAME,
    POSE_ESTIMATOR_NAME,
    POSE_EXPERIMENT_NAME,
    configure_mlflow,
    register_person_detector,
    register_pose_estimator,
    register_pose_estimator_merge,
    resolve_person_detector_weight,
    resolve_pose_estimator_weight,
)

__all__ = [
    "DEFAULT_ALIAS",
    "DEFAULT_POSE_WEIGHT_FILENAME",
    "PERSON_DETECTOR_DESCRIPTION",
    "PERSON_DETECTOR_NAME",
    "POSE_ESTIMATOR_DESCRIPTION",
    "POSE_ESTIMATOR_MERGE_DESCRIPTION",
    "POSE_ESTIMATOR_MERGE_NAME",
    "POSE_ESTIMATOR_NAME",
    "POSE_EXPERIMENT_NAME",
    "configure_mlflow",
    "register_person_detector",
    "register_pose_estimator",
    "register_pose_estimator_merge",
    "resolve_person_detector_weight",
    "resolve_pose_estimator_weight",
]
