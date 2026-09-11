"""Rebuild old Adesanya/Pereira poses in isolation; never install them automatically."""
from __future__ import annotations

import hashlib
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from ufc_tracker.annotations.windows import read_jsonl
from ufc_tracker.pose.pipeline import run_pose_pipeline

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs" / "dataset-repair-20260903"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(folder: Path) -> dict:
    meta = json.loads((folder / "pose_metrics.json").read_text(encoding="utf-8"))
    visible: Counter = Counter()
    valid: Counter = Counter()
    for row in read_jsonl(folder / "pose.jsonl"):
        visible[row["frame_index"]] += bool(row["visible"])
        valid[row["frame_index"]] += bool(row["visible"] and row["pose_valid"])
    return {
        "frames": meta["frame_count"], "fps": meta["fps"],
        "two_visible": sum(n == 2 for n in visible.values()),
        "two_valid_pose": sum(n == 2 for n in valid.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()
    annotation_dir = ROOT / "data/annotations/strike_annotations_v1"
    backup_dir = WORK / "annotations_before"
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Immutable snapshot: reruns must not overwrite the original backup.
    for source in annotation_dir.glob("*.jsonl"):
        backup = backup_dir / source.name
        if not backup.exists():
            shutil.copy2(source, backup)
    report_path = WORK / "staged_comparison.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    sources = sorted(annotation_dir.glob("adesanya_pereira_1__*round*.jsonl"))
    for index, annotation in enumerate(sources, 1):
        if report.get(annotation.stem, {}).get("status", "").startswith("rejected"):
            print(f"SKIP {annotation.stem}: rejected by visual review", flush=True)
            continue
        rows = read_jsonl(annotation)
        old = ROOT / Path(rows[0]["pose_path"]).parent
        metadata = json.loads((old / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["tracking"].get("gap_recovery"):
            print(f"SKIP {annotation.stem}: recovery already present", flush=True)
            continue
        staged = WORK / "staged" / annotation.stem
        before_hash = digest(annotation)
        print(f"ROUND {index}/{len(sources)} {annotation.stem}", flush=True)
        if not (staged / "run_metadata.json").is_file():
            def progress(stage: str, current: int, total: int) -> None:
                if current == 1 or current == total or current % 600 == 0:
                    print(f"[{index}/5] {stage} {current}/{total}", flush=True)
            run_pose_pipeline(
                ROOT / rows[0]["video_path"], staged, merge_track_fragments=True,
                tracking_confidence=metadata["tracking"]["confidence"],
                progress_callback=progress,
                generate_preview=not args.skip_preview,
            )
        before, after = coverage(old), coverage(staged)
        if before_hash != digest(annotation):
            raise RuntimeError("Annotations changed concurrently; refusing to continue")
        if before["frames"] != after["frames"] or abs(before["fps"]-after["fps"]) > 1e-6:
            raise RuntimeError("Rebuild changed video frame/time contract")
        report[annotation.stem] = {
            "before": before, "after": after, "annotation_sha256": before_hash,
            "coverage_improved": after["two_visible"] >= before["two_visible"]
            and after["two_valid_pose"] >= before["two_valid_pose"],
            "status": "staged_awaiting_visual_review",
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("COMPARISON " + json.dumps(report[annotation.stem]), flush=True)
    print("STAGING FINISHED; original artifacts and labels were not replaced", flush=True)


if __name__ == "__main__":
    main()
