"""Build a provenance-preserving corpus from completed human action annotations.

This module deliberately does not infer labels from unlabeled time.  Specific
actions supervise only the technique(s) explicitly observed for their attacker.
Confirmed no-strike windows supervise all techniques as absent.  Ambiguous
records are carried into the audit but never become a predicted class or a
negative example.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ufc_tracker.action_recognition.contracts import (
    ACTION_CLASSES,
    ACTION_SCHEMA_VERSION,
    AMBIGUOUS_LABEL,
    FIGHTER_IDS,
    PRIMARY_ACTION_CLASSES,
    intervals_overlap,
    multi_hot,
)
from ufc_tracker.action_recognition.features import AuditedPose, PoseFeatureSpec, has_active_attacker
from ufc_tracker.annotations.actions import strike_intervals


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_fights(root: Path, manifest_path: Path) -> dict[str, dict[str, str]]:
    with (root / manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {Path(row["relative_path"]).stem: row for row in rows}


def _validated_action_payload(
    root: Path,
    source_path: Path,
    action_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_bytes = source_path.read_bytes()
    source_rows = _read_jsonl(source_path)
    payload = json.loads(action_path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "strike_actions_v1":
        raise ValueError(f"Unsupported action annotation schema: {action_path.name}")
    if payload.get("source_sha256") != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError(f"Action annotation source changed: {action_path.name}")
    if not source_rows:
        raise ValueError(f"Empty binary annotation source: {source_path.name}")
    first = source_rows[0]
    for key in ("video_id", "video_path", "fps"):
        if payload.get(key) != first.get(key):
            raise ValueError(f"Action/source contract mismatch for {key}: {action_path.name}")
    events = {event["event_id"]: event for event in strike_intervals(source_rows)}
    completed = set(payload.get("completed", []))
    if completed != set(events):
        missing = sorted(set(events) - completed)
        extra = sorted(completed - set(events))
        raise ValueError(f"Incomplete or stale action round {action_path.name}: missing={missing[:3]} extra={extra[:3]}")
    return payload, source_rows, list(events.values())


def _action_record(
    action: dict[str, Any], *, round_data: dict[str, Any], fight: dict[str, str], source_path: Path,
) -> dict[str, Any]:
    label = action.get("label")
    attacker = str(action.get("attacker", "unknown"))
    start, end = int(action["start_frame"]), int(action["end_frame"])
    if end < start:
        raise ValueError(f"Invalid action interval: {action.get('action_id')}")
    return {
        "record_type": "action",
        "action_id": action["action_id"],
        "event_id": action["event_id"],
        "round_id": round_data["video_id"],
        "fight_id": fight["fight_id"],
        "category": fight["category"],
        "video_path": round_data["video_path"],
        "fps": float(round_data["fps"]),
        "start_frame": start,
        "end_frame": end,
        "start_seconds": start / float(round_data["fps"]),
        "end_seconds": end / float(round_data["fps"]),
        "attacker": attacker,
        "label": label,
        "source_window_ids": list(action.get("source_window_ids", [])),
        "annotated_at": action.get("annotated_at"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _direct_round(
    root: Path, source: dict[str, Any], spec: PoseFeatureSpec,
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    supervised_classes = source.get("supervised_classes", list(ACTION_CLASSES))
    if (not isinstance(supervised_classes, list) or not supervised_classes
            or any(label not in ACTION_CLASSES for label in supervised_classes)):
        raise ValueError("Direct supervised_classes must be a nonempty list of action classes")
    annotation = root / source["annotation"]
    review_manifest = root / source["review_manifest"]
    refinement = root / source["refinement"]
    payload = _read_json(annotation)
    review = _read_json(review_manifest)
    inventory = _read_json(refinement / "overlay_inventory.json")
    geometry = _read_json(refinement / "geometry_report.json")
    validation = _read_json(refinement / "v4_validation.json")
    mask_path = refinement / "geometry_masks.npz"
    annotation_hash, review_hash, mask_hash = map(sha256, (annotation, review_manifest, mask_path))
    identity_path = refinement / "identity_decisions.json"
    identity_hash = sha256(identity_path)
    exclusions = refinement / "joint_exclusions.json"
    video = (root / payload["video"]).resolve()
    if (
        payload["schema_version"] != "direct_strike_actions_v1"
        or inventory["annotations_sha256"] != annotation_hash
        or not payload["source_sha256"] == review_hash == inventory["manifest_sha256"]
        or not payload["revision"] == inventory["source_revision"] == geometry["source_revision"] == validation["source_revision"]
        or not payload["frame_count"] == review["decoded_frames"] == inventory["frame_count"]
        or not payload["fps"] == review["fps"] == inventory["fps"]
        or (root / review["video"]).resolve() != video
        or sha256(video) != inventory["video_sha256"]
        or not inventory["visual_annotation_complete"]
        or not geometry["identity_audit_complete"]
        or not identity_hash == geometry["identity_decisions_sha256"] == validation["identity_decisions_sha256"]
        or (sha256(exclusions) if exclusions.is_file() else None) != geometry["joint_exclusions_sha256"]
        or mask_hash != validation["masks_sha256"]
        or geometry["pose_sha256"] != inventory["pose_sha256"]
    ):
        raise ValueError(f"Incomplete or stale direct annotation refinement: {annotation.name}")
    cursor = 0
    for first, last in sorted((row["start_frame"], row["end_frame"]) for row in payload["reviewed_ranges"]
                              if row["status"] == "reviewed"):
        if not 0 <= first <= cursor <= payload["frame_count"] or not first <= last < payload["frame_count"]:
            raise ValueError(f"Invalid or incomplete direct review coverage: {annotation.name}")
        cursor = max(cursor, last + 1)
    if cursor != payload["frame_count"]:
        raise ValueError(f"Incomplete direct review coverage: {annotation.name}")

    fighters = dict(zip(source["fighters"], FIGHTER_IDS, strict=True))
    if set(fighters) != set(inventory["fighter_counts"]):
        raise ValueError(f"Direct fighter identities do not match refinement: {annotation.name}")
    windows = {row["action_id"]: row for row in inventory["actions"]}
    actions = {row["action_id"]: row for row in payload["actions"]}
    if len(actions) != len(payload["actions"]) or set(actions) != set(windows) or video.stem != annotation.stem:
        raise ValueError(f"Direct action IDs do not match refinement: {annotation.name}")
    provenance = {
        "action_annotation": annotation.relative_to(root).as_posix(), "action_sha256": annotation_hash,
        "review_manifest": review_manifest.relative_to(root).as_posix(), "review_manifest_sha256": review_hash,
        "pose": geometry["pose_path"], "pose_sha256": geometry["pose_sha256"],
        "pose_masks_path": mask_path.relative_to(root).as_posix(), "pose_masks_sha256": mask_hash,
        "identity_decisions_sha256": identity_hash,
    }
    pose_fields = {key: provenance[key] for key in ("pose_sha256", "pose_masks_path", "pose_masks_sha256")}
    pose_fields["pose_path"] = provenance["pose"]
    audited = AuditedPose(root, pose_fields)
    if set(audited.by_action) != set(actions):
        raise ValueError(f"Direct action IDs do not match pose masks: {annotation.name}")
    round_data = {"video_id": video.stem, "video_path": video.relative_to(root.resolve()).as_posix(), "fps": payload["fps"]}
    records = []
    for action_id, action in actions.items():
        window = windows[action_id]
        if any(action[key] != window[key] for key in ("label", "fighter", "start_frame", "end_frame")):
            raise ValueError(f"Direct action interval or identity mismatch: {action_id}")
        record = _action_record(
            {**action, "attacker": fighters[action["fighter"]], "event_id": action_id},
            round_data=round_data, fight=source, source_path=annotation,
        )
        record.update(pose_fields, pose_action_id=action_id, pose_review_window=window["review_window"],
                      fighter=action["fighter"], pose_usable=True)
        if record["label"] == AMBIGUOUS_LABEL:
            record["supervision"] = "excluded_ambiguous"
        elif record["label"] in ACTION_CLASSES:
            sample = {**record, "sample_id": f"action:{action_id}"}
            record["supervision"] = ("positive" if has_active_attacker(sample, audited.sequence(sample, spec))
                                     else "excluded_no_usable_attacker_pose")
            if record["supervision"] == "positive" and record["label"] not in supervised_classes:
                record["supervision"] = "excluded_by_class_scope"
        else:
            raise ValueError(f"Unknown direct action label: {record['label']}")
        records.append(record)
    return video.stem, records, provenance


def _choose_test_fights(
    fight_counts: dict[str, Counter[str]],
) -> tuple[str, str]:
    """Choose a fixed pair using labels only, before any model is evaluated."""
    fights = sorted(fight_counts)
    total = Counter()
    for counts in fight_counts.values():
        total.update(counts)
    candidates: list[tuple[tuple[float, float, tuple[str, str]], tuple[str, str]]] = []
    for pair in itertools.combinations(fights, 2):
        test = fight_counts[pair[0]] + fight_counts[pair[1]]
        train = total - test
        if any(train[label] == 0 for label in PRIMARY_ACTION_CLASSES):
            continue
        test_coverage = sum(test[label] > 0 for label in PRIMARY_ACTION_CLASSES)
        balance_error = sum(abs(test[label] / total[label] - 0.25) for label in PRIMARY_ACTION_CLASSES)
        candidates.append(((-float(test_coverage), balance_error, pair), pair))
    if not candidates:
        raise ValueError("Cannot reserve a two-fight test set with primary-class coverage in training")
    return min(candidates, key=lambda item: item[0])[1]


def _balanced_development_folds(
    fight_counts: dict[str, Counter[str]], test_fights: tuple[str, str],
) -> dict[str, int]:
    """Pair six development fights into three deterministic, label-balanced folds."""
    fights = tuple(fight for fight in sorted(fight_counts) if fight not in test_fights)
    if len(fights) != 6:
        raise ValueError("Expected six development fights after reserving two test fights")
    total = Counter()
    for fight in fights:
        total.update(fight_counts[fight])
    pairings: list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]] = []
    for first in itertools.combinations(fights, 2):
        rest = tuple(fight for fight in fights if fight not in first)
        for second in itertools.combinations(rest, 2):
            third = tuple(fight for fight in rest if fight not in second)
            pairs = tuple(sorted((tuple(sorted(first)), tuple(sorted(second)), tuple(sorted(third)))))
            if pairs not in pairings:
                pairings.append(pairs)
    def score(pairs: tuple[tuple[str, str], ...]) -> tuple[float, tuple[tuple[str, str], ...]]:
        error = 0.0
        for pair in pairs:
            counts = fight_counts[pair[0]] + fight_counts[pair[1]]
            for label in PRIMARY_ACTION_CLASSES:
                error += abs(counts[label] / total[label] - 1 / 3)
        return error, pairs
    selected = min(pairings, key=score)
    return {fight: fold for fold, pair in enumerate(selected) for fight in pair}


def build_action_corpus(root: Path, config_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read completed action annotations and return samples, audit records and report.

    The first list contains supervised examples. The second contains every human
    record, including ambiguous and excluded records, for auditability.
    """
    root, config_path = Path(root), Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("Unsupported action-recognition config schema")
    if tuple(config.get("classes", ())) != ACTION_CLASSES:
        raise ValueError("Action classes do not match the stable model contract")
    positive_interval_policy = config.get("positive_interval_policy", "positive_labels_only")
    if positive_interval_policy not in {
        "positive_labels_only",
        "exhaustive_unless_uncertain_overlap",
    }:
        raise ValueError(f"Unsupported positive interval policy: {positive_interval_policy}")
    fights = _load_fights(root, Path(config["splits_manifest"]))
    configured_action_dirs = config.get("action_annotations_dirs")
    if configured_action_dirs is None:
        configured_action_dirs = [config["action_annotations_dir"]]
    if not configured_action_dirs:
        raise ValueError("At least one action annotations directory is required")
    action_dirs = [root / Path(path) for path in configured_action_dirs]
    source_dir = root / config["binary_annotations_dir"]
    action_paths = sorted(path for directory in action_dirs for path in directory.glob("*.json"))
    direct_rounds = config.get("direct_rounds", [])
    if not action_paths and not direct_rounds:
        raise ValueError("No action annotations found")
    duplicate_stems = [
        stem for stem, count in Counter(path.stem for path in action_paths).items() if count > 1
    ]
    if duplicate_stems:
        raise ValueError(f"Duplicate action annotation rounds across directories: {duplicate_stems}")

    samples: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    per_round_actions: dict[str, list[dict[str, Any]]] = {}
    per_round_source: dict[str, list[dict[str, Any]]] = {}
    per_round_fights: dict[str, str] = {}
    per_round_templates: dict[str, dict[str, Any]] = {}
    source_provenance: dict[str, dict[str, str]] = {}
    corrections = []
    for action_path in action_paths:
        source_path = source_dir / f"{action_path.stem}.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing binary source for {action_path.name}")
        payload, source_rows, _ = _validated_action_payload(root, source_path, action_path)
        video_id = payload["video_id"]
        if video_id not in fights:
            raise ValueError(f"Action video missing from split manifest: {video_id}")
        if not (root / payload["video_path"]).is_file():
            raise FileNotFoundError(f"Missing video for {video_id}")
        pose_path = Path(source_rows[0]["pose_path"])
        if not (root / pose_path).is_file():
            raise FileNotFoundError(f"Missing pose file for {video_id}")
        pose_usable = video_id not in config.get("untrusted_pose_videos", {})
        round_data = {**payload, "pose_path": pose_path.as_posix(), "pose_usable": pose_usable}
        records = []
        for action in payload.get("actions", []):
            record = _action_record(action, round_data=round_data, fight=fights[video_id], source_path=source_path)
            record["pose_path"] = pose_path.as_posix()
            record["pose_usable"] = pose_usable
            if record["label"] not in ACTION_CLASSES + (AMBIGUOUS_LABEL,):
                raise ValueError(f"Unknown action label in {action_path.name}: {record['label']}")
            if record["attacker"] not in FIGHTER_IDS + ("unknown",):
                raise ValueError(f"Unknown attacker in {action_path.name}: {record['attacker']}")
            if record["label"] == AMBIGUOUS_LABEL:
                record["supervision"] = "excluded_ambiguous"
            elif record["attacker"] == "unknown":
                record["supervision"] = "excluded_unknown_attacker"
            else:
                record["supervision"] = "positive"
            audit.append(record)
            records.append(record)
        per_round_actions[video_id] = records
        per_round_source[video_id] = source_rows
        per_round_fights[video_id] = fights[video_id]["fight_id"]
        per_round_templates[video_id] = {
            "fight_id": fights[video_id]["fight_id"], "category": fights[video_id]["category"],
            "video_path": payload["video_path"], "pose_path": pose_path.as_posix(),
            "pose_usable": pose_usable, "fps": float(payload["fps"]),
        }
        for correction in payload.get("source_corrections", []):
            if correction.get("label") != "no_strike":
                raise ValueError(f"Unsupported source correction in {action_path.name}")
            corrections.append({
                "record_type": "source_correction", "round_id": video_id, "fight_id": fights[video_id]["fight_id"],
                "start_frame": int(correction["start_frame"]), "end_frame": int(correction["end_frame"]),
                "correction_id": correction["correction_id"], "reason": correction.get("reason"),
            })
        source_provenance[video_id] = {
            "action_annotation": action_path.relative_to(root).as_posix(),
            "action_sha256": sha256(action_path),
            "binary_annotation": source_path.relative_to(root).as_posix(),
            "binary_sha256": sha256(source_path),
            "pose": pose_path.as_posix(), "pose_sha256": sha256(root / pose_path),
        }

    direct_spec = PoseFeatureSpec(**config.get("direct_pose_features", {}))
    for source in direct_rounds:
        video_id, records, provenance = _direct_round(root, source, direct_spec)
        if video_id in per_round_actions:
            raise ValueError(f"Duplicate action annotation round: {video_id}")
        per_round_actions[video_id] = records
        # The action masks only cover action contexts, not no-strike intervals.
        per_round_source[video_id] = []
        per_round_fights[video_id] = source["fight_id"]
        source_provenance[video_id] = provenance
        audit.extend(records)

    for video_id, records in per_round_actions.items():
        known = [record for record in records if record["supervision"] == "positive"]
        for record in known:
            positives = {
                other["label"] for other in known
                if other["attacker"] == record["attacker"] and intervals_overlap(
                    record["start_frame"], record["end_frame"], other["start_frame"], other["end_frame"],
                )
            }
            uncertain_overlap = any(
                other["supervision"] in {"excluded_ambiguous", "excluded_unknown_attacker"}
                and other["attacker"] in {record["attacker"], "unknown"}
                and intervals_overlap(
                    record["start_frame"], record["end_frame"],
                    other["start_frame"], other["end_frame"],
                )
                for other in records
            )
            positive_target = multi_hot(positives)
            fully_supervised = (
                positive_interval_policy == "exhaustive_unless_uncertain_overlap"
                and not uncertain_overlap
                and "pose_masks_path" not in record
            )
            audited_fields = (
                "pose_masks_path", "pose_masks_sha256", "pose_sha256", "pose_action_id",
                "pose_review_window", "fighter",
            )
            samples.append({
                "sample_id": f"action:{record['action_id']}", "sample_type": "positive_action",
                **{key: record[key] for key in (
                    "round_id", "fight_id", "category", "video_path", "pose_path", "pose_usable", "fps",
                    "start_frame", "end_frame", "start_seconds", "end_seconds", "attacker",
                )},
                **{key: record[key] for key in audited_fields if key in record},
                "target_labels": [label for label in ACTION_CLASSES if label in positives],
                "target": positive_target,
                "target_mask": [1] * len(ACTION_CLASSES) if fully_supervised else positive_target,
                "action_supervision": "complete" if fully_supervised else "positive_only",
                "uncertain_overlap": uncertain_overlap,
                "source_action_ids": sorted(
                    other["action_id"] for other in known if other["label"] in positives and other["attacker"] == record["attacker"]
                    and intervals_overlap(record["start_frame"], record["end_frame"], other["start_frame"], other["end_frame"])
                ),
            })

        corrections_for_round = [row for row in corrections if row["round_id"] == video_id]
        negative_intervals = [
            (int(row["start_frame"]), int(row["end_frame"]), "binary", f"binary:{row['window_id']}")
            for row in per_round_source[video_id] if row.get("label") == "no_strike"
        ]
        negative_intervals.extend(
            (row["start_frame"], row["end_frame"], "correction", f"correction:{row['correction_id']}")
            for row in corrections_for_round
        )
        for start, end, negative_kind, origin in negative_intervals:
            if negative_kind == "binary" and any(
                intervals_overlap(start, end, action["start_frame"], action["end_frame"])
                for action in records if action["label"] in ACTION_CLASSES
            ):
                continue
            template = per_round_templates[video_id]
            for attacker in FIGHTER_IDS:
                samples.append({
                    "sample_id": f"no_strike:{video_id}:{start}-{end}:{attacker}:{origin}",
                    "sample_type": "confirmed_no_strike", "round_id": video_id, "fight_id": template["fight_id"],
                    "category": template["category"], "video_path": template["video_path"],
                    "pose_path": template["pose_path"], "pose_usable": template["pose_usable"], "fps": template["fps"],
                    "start_frame": start, "end_frame": end, "start_seconds": start / template["fps"],
                    "end_seconds": end / template["fps"], "attacker": attacker,
                    "target_labels": [], "target": [0] * len(ACTION_CLASSES),
                    "target_mask": [1] * len(ACTION_CLASSES), "source_action_ids": [],
                })

    fight_counts: dict[str, Counter[str]] = {
        fight_id: Counter() for fight_id in per_round_fights.values()
    }
    for sample in samples:
        if sample["sample_type"] == "positive_action":
            for label in sample["target_labels"]:
                fight_counts[sample["fight_id"]][label] += 1
    split_config = config.get("fixed_splits")
    train_only_fights: set[str] = set()
    if split_config is None:
        test_fights = _choose_test_fights(fight_counts)
        folds = _balanced_development_folds(fight_counts, test_fights)
    else:
        test_fights = tuple(split_config.get("test_fights", ()))
        folds = {str(fight): int(fold) for fight, fold in split_config.get("development_folds", {}).items()}
        train_only_fights = set(split_config.get("train_only_fights", ()))
        assigned = set(test_fights) | set(folds) | train_only_fights
        available = set(fight_counts)
        if assigned != available:
            raise ValueError(
                "Fixed split fights do not match annotated fights: "
                f"missing={sorted(available - assigned)}, unknown={sorted(assigned - available)}"
            )
        if set(test_fights) & (set(folds) | train_only_fights) or set(folds) & train_only_fights:
            raise ValueError("Fixed test, development and train-only fights must be disjoint")
        if not test_fights or not folds:
            raise ValueError("Fixed splits require test fights and development folds")
    for sample in samples:
        sample["split"] = "test" if sample["fight_id"] in test_fights else "development"
        sample["development_fold"] = (
            None if sample["split"] == "test" or sample["fight_id"] in train_only_fights
            else folds[sample["fight_id"]]
        )

    report = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": list(ACTION_CLASSES), "primary_classes": list(PRIMARY_ACTION_CLASSES),
        "rounds": len(per_round_actions), "fights": len(fight_counts),
        "audit_action_counts": dict(Counter(record["label"] for record in audit)),
        "supervised_samples": len(samples),
        "positive_samples": sum(row["sample_type"] == "positive_action" for row in samples),
        "confirmed_no_strike_samples": sum(row["sample_type"] == "confirmed_no_strike" for row in samples),
        "excluded_ambiguous": sum(row["supervision"] == "excluded_ambiguous" for row in audit),
        "excluded_unknown_attacker": sum(row["supervision"] == "excluded_unknown_attacker" for row in audit),
        "excluded_no_usable_attacker_pose": sum(
            row["supervision"] == "excluded_no_usable_attacker_pose" for row in audit
        ),
        "excluded_by_class_scope": sum(row["supervision"] == "excluded_by_class_scope" for row in audit),
        "source_corrections": len(corrections), "test_fights": list(test_fights),
        "development_folds": folds, "train_only_fights": sorted(train_only_fights),
        "sources": source_provenance,
        "label_policy": "ambiguous_is_excluded_not_a_class_or_negative",
        "positive_interval_policy": positive_interval_policy,
        "fully_supervised_action_samples": sum(
            row.get("action_supervision") == "complete" for row in samples
        ),
        "positive_only_action_samples": sum(
            row.get("action_supervision") == "positive_only" for row in samples
        ),
        "negative_policy": "binary_no_strike_without_specific_action_overlap_or_explicit_source_correction_with_priority",
    }
    return samples, audit + corrections, report


def write_action_corpus(root: Path, config_path: Path, destination: Path) -> dict[str, Any]:
    """Write generated artifacts; source annotations are read-only inputs."""
    samples, audit, report = build_action_corpus(root, config_path)
    destination = Path(destination)
    _atomic_jsonl(destination / "samples.jsonl", samples)
    _atomic_jsonl(destination / "audit_records.jsonl", audit)
    report = {
        **report,
        "samples_sha256": sha256(destination / "samples.jsonl"),
        "audit_records_sha256": sha256(destination / "audit_records.jsonl"),
    }
    _atomic_json(destination / "report.json", report)
    return report
