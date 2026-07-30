"""Local MLflow Model Registry for StrikeVision detectors."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from typing import Any

from ufc_tracker.detection.weights import project_root, resolve_pretrained_weight, weights_dir

PERSON_DETECTOR_NAME = "PersonDetector"
DEFAULT_WEIGHT_FILENAME = "yolo11n-seg.pt"
DEFAULT_ALIAS = "production"
EXPERIMENT_NAME = "person-detection"

PERSON_DETECTOR_DESCRIPTION = textwrap.dedent(
    """
    StrikeVision PersonDetector — MMA/UFC fighter person detection and tracking.

    Context
    -------
    Production component of the StrikeVision pipeline that finds the two fighters
    in a fight-round video. It uses YOLO11 instance segmentation (COCO person
    class), ByteTrack IDs, and torso skin filters to drop referees, staff, and
    crowd. Annotated silhouettes are drawn blue/red (left-to-right).

    Implementation: ufc_tracker.detection.personDetection
    Entry point:    send_prediction(path, frames)

    Inputs
    ------
    video_path (string)
        Path to the fight video to analyze. Absolute path, or relative to the
        StrikeVision project root (for example data/splits/.../round.mp4).

    frames (integer)
        Number of frames to analyze from the start of the video. Must be > 0.

    Output
    ------
    output_video_path (string)
        Path to the annotated MP4 written under outputs/predictions/, named
        {video_stem}__first_{frames}_tracked.mp4.
    """
).strip()

PERSON_DETECTOR_TAGS = {
    "component": "detection",
    "model_name": PERSON_DETECTOR_NAME,
    "project": "StrikeVision",
    "task": "person-detection-tracking",
    "domain": "mma-ufc",
    "framework": "ultralytics-yolo",
    "entry_point": "ufc_tracker.detection.personDetection.send_prediction",
    "input_video_path": "Path to the fight video to analyze",
    "input_frames": "Number of frames to analyze from the start of the video",
    "output_video_path": "Annotated MP4 under outputs/predictions/",
}

# Defaults mirrored from personDetection.py (documented at registration time).
PERSON_DETECTOR_DEFAULT_PARAMS = {
    "weight_filename": DEFAULT_WEIGHT_FILENAME,
    "detector": "yolo11n-seg",
    "class_id": 0,
    "pipeline": "personDetection.send_prediction",
    "tracker": "bytetrack.yaml",
    "keep_top_n": 2,
    "min_area_frac": 0.002,
    "select_shirtless": True,
    "skin_frac_min": 0.35,
    "fighter_skin_min": 0.50,
    "fighter_min_persistence": 0.05,
    "fighter_min_area": 0.01,
    "input_fields": "video_path,frames",
    "output_fields": "output_video_path",
}

# Keep stage names for CLI/UX; they map to lowercase aliases in MLflow 2.9+.
_STAGE_TO_ALIAS = {
    "None": "none",
    "Staging": "staging",
    "Production": "production",
    "Archived": "archived",
}


def configure_mlflow(root: Path | None = None) -> Path:
    """Point MLflow at local ``mlruns`` / ``mlartifacts`` under the project root."""
    import mlflow

    base = (root or project_root()).resolve()
    tracking_dir = base / "mlruns"
    artifact_dir = base / "mlartifacts"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_dir


def _person_detector_signature():
    """MLflow signature: video_path + frames -> output_video_path."""
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, Schema

    return ModelSignature(
        inputs=Schema(
            [
                ColSpec("string", "video_path"),
                ColSpec("long", "frames"),
            ]
        ),
        outputs=Schema(
            [
                ColSpec("string", "output_video_path"),
            ]
        ),
    )


def _person_detector_input_example():
    """Example batch matching send_prediction(path, frames)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "video_path": (
                    "data/splits/aggressive_men/"
                    "adesanya_pereira_1__israel_adesanya_vs_alex_pereira_1__"
                    "aggressive_men_round1.mp4"
                ),
                "frames": 1000,
            }
        ]
    )


def _as_input_dataframe(model_input: Any):
    import pandas as pd

    if isinstance(model_input, pd.DataFrame):
        df = model_input.copy()
    else:
        df = pd.DataFrame(model_input)

    missing = {"video_path", "frames"} - set(df.columns)
    if missing:
        raise ValueError(
            "PersonDetector expects columns video_path and frames "
            f"(path to the video and number of frames to analyze). Missing: {sorted(missing)}"
        )
    return df


def register_person_detector(
    weight_path: Path | str | None = None,
    *,
    stage: str = "Production",
    params: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Any:
    """Log YOLO-seg weights and register/promote them in the Model Registry.

    Stores an English model description, input/output signature
    (``video_path``, ``frames`` → ``output_video_path``), tags, and params.

    ``stage`` is mapped to an MLflow alias (``production``, ``staging``, ...).

    Returns:
        The registered ``ModelVersion``.
    """
    import mlflow
    from mlflow.pyfunc import PythonModel
    from mlflow.tracking import MlflowClient

    class PersonDetectorModel(PythonModel):
        """Registered StrikeVision person detector (weights + I/O contract)."""

        def load_context(self, context):
            self.weights_path = context.artifacts["weights"]

        def predict(self, context, model_input):
            """Validate inputs and return the planned annotated-video path.

            Production inference should call
            ``ufc_tracker.detection.personDetection.send_prediction(path, frames)``.
            This predict documents the registry contract without re-running the
            full video pipeline during MLflow logging/validation.
            """
            import pandas as pd

            df = _as_input_dataframe(model_input)
            outputs: list[str] = []
            for _, row in df.iterrows():
                video_path = Path(str(row["video_path"]))
                frames = int(row["frames"])
                if frames <= 0:
                    raise ValueError(f"frames must be greater than 0, received: {frames}")
                outputs.append(
                    f"outputs/predictions/{video_path.stem}__first_{frames}_tracked.mp4"
                )
            return pd.DataFrame({"output_video_path": outputs})

    base = (root or project_root()).resolve()
    configure_mlflow(base)

    source = Path(weight_path) if weight_path else resolve_pretrained_weight(
        DEFAULT_WEIGHT_FILENAME, root=base
    )
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Weight file not found: {source}")

    alias = _STAGE_TO_ALIAS.get(stage, stage.lower())
    run_params = {
        **PERSON_DETECTOR_DEFAULT_PARAMS,
        "weight_filename": source.name,
        "alias": alias,
        **(params or {}),
    }
    signature = _person_detector_signature()
    input_example = _person_detector_input_example()

    with mlflow.start_run(run_name=f"register-{source.stem}"):
        mlflow.log_params({k: str(v) for k, v in run_params.items()})
        mlflow.set_tags(PERSON_DETECTOR_TAGS)
        mlflow.set_tag("mlflow.note.content", PERSON_DETECTOR_DESCRIPTION)

        card_path = base / "mlartifacts" / "_tmp_person_detector_card.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(
            "# PersonDetector\n\n" + PERSON_DETECTOR_DESCRIPTION + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(card_path), artifact_path="docs")
        card_path.unlink(missing_ok=True)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PersonDetectorModel(),
            artifacts={"weights": str(source)},
            signature=signature,
            input_example=input_example,
            registered_model_name=PERSON_DETECTOR_NAME,
            metadata={
                "description": PERSON_DETECTOR_DESCRIPTION,
                "inputs": {
                    "video_path": "Path to the fight video to analyze",
                    "frames": "Number of frames to analyze from the start of the video",
                },
                "outputs": {
                    "output_video_path": (
                        "Annotated video path under outputs/predictions/"
                    ),
                },
                "entry_point": "send_prediction(path, frames)",
            },
        )

    client = MlflowClient()
    try:
        client.create_registered_model(
            PERSON_DETECTOR_NAME,
            tags=PERSON_DETECTOR_TAGS,
            description=PERSON_DETECTOR_DESCRIPTION,
        )
    except Exception:
        client.update_registered_model(
            PERSON_DETECTOR_NAME,
            description=PERSON_DETECTOR_DESCRIPTION,
        )
        for key, value in PERSON_DETECTOR_TAGS.items():
            try:
                client.set_registered_model_tag(PERSON_DETECTOR_NAME, key, value)
            except Exception:
                pass

    versions = client.search_model_versions(f"name='{PERSON_DETECTOR_NAME}'")
    model_version = max(versions, key=lambda v: int(v.version))

    version_description = textwrap.dedent(
        f"""
        {PERSON_DETECTOR_DESCRIPTION}

        Weight file: {source.name}
        Alias: {alias}
        """
    ).strip()
    client.update_model_version(
        name=PERSON_DETECTOR_NAME,
        version=model_version.version,
        description=version_description,
    )
    for key, value in PERSON_DETECTOR_TAGS.items():
        try:
            client.set_model_version_tag(
                PERSON_DETECTOR_NAME,
                model_version.version,
                key,
                value,
            )
        except Exception:
            pass

    client.set_registered_model_alias(
        PERSON_DETECTOR_NAME,
        alias,
        model_version.version,
    )

    # Best-effort legacy stage label for the MLflow UI (deprecated API).
    if stage in _STAGE_TO_ALIAS and stage != "None":
        try:
            client.transition_model_version_stage(
                name=PERSON_DETECTOR_NAME,
                version=model_version.version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
        except Exception:
            pass

    return client.get_model_version(PERSON_DETECTOR_NAME, model_version.version)


def _find_weight_file(directory: Path, preferred_name: str) -> Path | None:
    preferred = directory / preferred_name
    if preferred.is_file():
        return preferred
    matches = sorted(directory.rglob("*.pt"))
    return matches[0] if matches else None


def resolve_person_detector_weight(
    *,
    model_name: str = PERSON_DETECTOR_NAME,
    stage: str = "Production",
    fallback_filename: str = DEFAULT_WEIGHT_FILENAME,
    root: Path | None = None,
) -> Path:
    """Resolve Production weights from MLflow; fallback to local ``models/weights``.

    Downloads the registered ``.pt`` into ``models/weights/`` so YOLO can load it.
    """
    base = (root or project_root()).resolve()
    cache = weights_dir(base)
    alias = _STAGE_TO_ALIAS.get(stage, stage.lower())

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        configure_mlflow(base)
        client = MlflowClient()

        try:
            client.get_model_version_by_alias(model_name, alias)
            model_uri = f"models:/{model_name}@{alias}"
        except Exception:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise FileNotFoundError(f"No versions registered for {model_name}")
            version = max(versions, key=lambda v: int(v.version))
            model_uri = f"models:/{model_name}/{version.version}"

        download_root = Path(mlflow.artifacts.download_artifacts(model_uri))
        weight_file = _find_weight_file(download_root, fallback_filename)
        if weight_file is None:
            raise FileNotFoundError(
                f"Registered model {model_name} has no .pt artifact ({model_uri})"
            )

        cached = cache / weight_file.name
        if not cached.exists() or cached.stat().st_size != weight_file.stat().st_size:
            shutil.copy2(weight_file, cached)
        return cached
    except Exception:
        return resolve_pretrained_weight(fallback_filename, root=base)
