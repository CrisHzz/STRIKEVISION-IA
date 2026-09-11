"""Install one quantitatively and visually reviewed staged round, retaining rollback data."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.annotations.windows import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "outputs/dataset-repair-20260903"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, type=int, choices=range(1, 6))
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--visual-approved", action="store_true")
    decision.add_argument("--reject", help="Reason for rejecting the staged poses without installation")
    args = parser.parse_args()
    report_path = WORK / "staged_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = [name for name in report if name.endswith(f"round{args.round}")]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one reviewed round")
    name = matches[0]
    entry = report[name]
    if args.reject:
        if entry["status"].startswith("installed"):
            raise RuntimeError("Cannot mark installed artifacts as uninstalled")
        entry["status"] = "rejected_visual_identity_errors"
        entry["rejection_reason"] = args.reject
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(entry, indent=2))
        return
    if not entry["coverage_improved"] or entry["status"] != "staged_awaiting_visual_review":
        raise RuntimeError("Staged round is not eligible for installation")
    if not (WORK / f"qa_round{args.round}.jpg").is_file():
        raise RuntimeError("Visual comparison has not been generated")
    target = (ROOT / "data/processed/poses/aggressive_men" / name).resolve()
    staged = (WORK / "staged" / name).resolve()
    backup = (WORK / "poses_before" / name).resolve()
    # Validate exact parents before moving entire directories, never overwrite backups.
    for path, parent in (
        (target, ROOT / "data/processed/poses/aggressive_men"),
        (staged, WORK / "staged"), (backup, WORK / "poses_before"),
    ):
        if path.parent != parent.resolve() or not path.is_relative_to(ROOT.resolve()):
            raise RuntimeError(f"Unsafe artifact target: {path}")
    if backup.exists() or not staged.is_dir() or not target.is_dir():
        raise RuntimeError("Missing artifacts or existing backup; refusing destructive overwrite")
    annotation = ROOT / "data/annotations/strike_annotations_v1" / f"{name}.jsonl"
    if sha256(annotation) != entry["annotation_sha256"]:
        raise RuntimeError("Annotation file changed since staging; inspect before installing")
    before = read_jsonl(annotation)
    backup.parent.mkdir(parents=True, exist_ok=True)
    target.rename(backup)
    try:
        staged.rename(target)
    except BaseException:
        backup.rename(target)
        raise
    if read_jsonl(annotation) != before:
        raise RuntimeError("Concurrent annotation change during installation")
    entry["status"] = "installed_after_visual_sample_review"
    entry["installed_at"] = datetime.now(timezone.utc).isoformat()
    entry["backup"] = backup.relative_to(ROOT).as_posix()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
