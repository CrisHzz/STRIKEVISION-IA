"""Prepare the audited local dataset; no training, cloud access or relabeling."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ufc_tracker.annotations.dataset import inspect_dataset, sha256
from ufc_tracker.annotations.repair import atomic_jsonl, refresh_quality_file

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-quality", action="store_true")
    args = parser.parse_args()
    policy = ROOT / "configs/data/strike_dataset_v1.yaml"
    print("Validating included rounds and recomputing pose quality...", flush=True)
    manifest, refreshed, provenance = inspect_dataset(ROOT, policy)
    changed = {}
    if args.refresh_quality:
        for path, rows in refreshed.items():
            count = refresh_quality_file(
                path, rows,
                expected_sha256=provenance["sources"][rows[0]["video_id"]][
                    "annotation_sha256_before"
                ],
                backup_dir=ROOT / "outputs/dataset-repair-20260903/annotations_before",
            )
            if count:
                changed[path.name] = count
    for path, rows in refreshed.items():
        provenance["sources"][rows[0]["video_id"]]["annotation_sha256"] = sha256(path)
    binary = [row for row in manifest if row["binary_eligible"]]
    review = [row for row in manifest if not row["binary_eligible"]]
    destination = ROOT / "data/processed/datasets/strike_dataset_v1"
    atomic_jsonl(destination / "binary_windows.jsonl", binary)
    atomic_jsonl(destination / "review_windows.jsonl", review)
    report = {
        "schema_version": "strike_dataset_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rounds": len(refreshed), "fights": len({r["fight_id"] for r in manifest}),
        "labels": dict(Counter(row["label"] for row in manifest)),
        "binary_windows": len(binary), "review_windows": len(review),
        "binary_low_pose_quality": sum(r["quality"]["status"] == "auto_unknown" for r in binary),
        "binary_untrusted_pose": sum(not r["pose_usable"] for r in binary),
        "untrusted_pose_videos": provenance["policy"].get("untrusted_pose_videos", {}),
        "quality_fields_updated": changed,
        "policy_sha256": sha256(policy),
        "binary_manifest_sha256": sha256(destination / "binary_windows.jsonl"),
        "review_manifest_sha256": sha256(destination / "review_windows.jsonl"),
        "excluded_files": provenance["policy"]["excluded_files"],
        "temporal_policy": "original_intervals_source_fps_no_extra_context",
        "sources": provenance["sources"],
    }
    (destination / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
