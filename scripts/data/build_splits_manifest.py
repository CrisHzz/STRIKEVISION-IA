from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

CATEGORY_METADATA: dict[str, dict[str, str]] = {
    "aggressive_men": {
        "gender": "men",
        "activity_level": "high",
        "label": "aggressive_striking",
        "primary_use": "positive striking examples, high-volume exchanges",
    },
    "aggressive_women": {
        "gender": "women",
        "activity_level": "high",
        "label": "aggressive_striking",
        "primary_use": "female positive striking examples, high-volume exchanges",
    },
    "low_kicks_men": {
        "gender": "men",
        "activity_level": "medium_high",
        "label": "low_kick_heavy",
        "primary_use": "low kick, leg kick and kick-heavy striking examples",
    },
    "low_kicks_women": {
        "gender": "women",
        "activity_level": "medium_high",
        "label": "low_kick_heavy",
        "primary_use": "female low kick, leg kick and kick-heavy striking examples",
    },
    "normal_men": {
        "gender": "men",
        "activity_level": "medium",
        "label": "normal_technical_striking",
        "primary_use": "balanced technical striking and regular fight rhythm",
    },
    "passive_men": {
        "gender": "men",
        "activity_level": "low",
        "label": "passive_feints_defensive",
        "primary_use": "no_strike, feints, distance management and false positives",
    },
    "passive_women": {
        "gender": "women",
        "activity_level": "low_medium",
        "label": "technical_defensive_striking",
        "primary_use": "female no_strike, feints, distance management and counters",
    },
    "striking_light_grappling_men": {
        "gender": "men",
        "activity_level": "medium_high",
        "label": "striking_with_light_grappling",
        "primary_use": "hard negatives for clinch, scrambling and out-of-scope transitions",
    },
}


def run_ffprobe(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration,bit_rate:stream=width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return round(float(Fraction(value)), 3)
    except (ValueError, ZeroDivisionError):
        return None


def parse_name(path: Path) -> dict[str, Any]:
    parts = path.stem.split("__")
    fight_id = parts[0] if parts else path.stem
    fight_name = parts[1] if len(parts) > 1 else fight_id
    round_number = None
    if "_round" in path.stem:
        suffix = path.stem.rsplit("_round", 1)[-1]
        if suffix.isdigit():
            round_number = int(suffix)
    return {
        "fight_id": fight_id,
        "fight_name": fight_name,
        "round_number": round_number,
    }


def duration_quality_flag(duration_sec: float | None) -> str:
    if duration_sec is None:
        return "unknown_duration"
    if duration_sec < 240:
        return "short_round_review"
    if duration_sec > 360:
        return "long_round_review"
    return "ok"


def build_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    videos = sorted(SPLITS_DIR.glob("*/*.mp4"))

    for video_path in videos:
        category = video_path.parent.name
        category_info = CATEGORY_METADATA.get(category, {})
        name_info = parse_name(video_path)
        ffprobe_data = run_ffprobe(video_path)
        stream = (ffprobe_data.get("streams") or [{}])[0]
        fmt = ffprobe_data.get("format") or {}
        duration = stream.get("duration") or fmt.get("duration")
        duration_sec = round(float(duration), 3) if duration else None
        fps = parse_fps(stream.get("avg_frame_rate")) or parse_fps(stream.get("r_frame_rate"))
        file_size_bytes = video_path.stat().st_size
        relative_path = video_path.relative_to(PROJECT_ROOT).as_posix()

        rows.append(
            {
                "category": category,
                "label": category_info.get("label", ""),
                "gender": category_info.get("gender", ""),
                "activity_level": category_info.get("activity_level", ""),
                "primary_use": category_info.get("primary_use", ""),
                "fight_id": name_info["fight_id"],
                "fight_name": name_info["fight_name"],
                "round_number": name_info["round_number"],
                "file_name": video_path.name,
                "relative_path": relative_path,
                "dvc_tracked_path": video_path.parent.relative_to(PROJECT_ROOT).as_posix(),
                "file_size_bytes": file_size_bytes,
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
                "duration_sec": duration_sec,
                "duration_min": round(duration_sec / 60, 3) if duration_sec else None,
                "duration_quality_flag": duration_quality_flag(duration_sec),
                "fps": fps,
                "width": stream.get("width"),
                "height": stream.get("height"),
                "codec_name": stream.get("codec_name"),
                "pix_fmt": stream.get("pix_fmt"),
                "bit_rate": fmt.get("bit_rate") or stream.get("bit_rate"),
                "nb_frames": stream.get("nb_frames"),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    summary: list[dict[str, Any]] = []
    for category, items in sorted(groups.items()):
        total_duration = sum(float(item["duration_sec"] or 0) for item in items)
        total_size = sum(int(item["file_size_bytes"]) for item in items)
        fights = sorted({item["fight_id"] for item in items})
        labels = sorted({item["label"] for item in items if item["label"]})
        summary.append(
            {
                "category": category,
                "label": ", ".join(labels),
                "round_count": len(items),
                "fight_count": len(fights),
                "fights": ", ".join(fights),
                "total_duration_sec": round(total_duration, 3),
                "total_duration_min": round(total_duration / 60, 3),
                "total_size_mb": round(total_size / (1024 * 1024), 3),
                "avg_duration_sec": round(total_duration / len(items), 3),
            }
        )
    return summary


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    summary = build_summary(rows)
    generated_at = datetime.now(timezone.utc).isoformat()

    write_csv(METADATA_DIR / "splits_manifest.csv", rows)
    write_csv(METADATA_DIR / "splits_summary_by_category.csv", summary)

    payload = {
        "generated_at": generated_at,
        "dataset_root": SPLITS_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "video_count": len(rows),
        "category_count": len({row["category"] for row in rows}),
        "manifest": rows,
        "summary_by_category": summary,
    }
    (METADATA_DIR / "splits_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Generated {len(rows)} video rows")
    print(METADATA_DIR / "splits_manifest.csv")
    print(METADATA_DIR / "splits_summary_by_category.csv")
    print(METADATA_DIR / "splits_manifest.json")


if __name__ == "__main__":
    main()
