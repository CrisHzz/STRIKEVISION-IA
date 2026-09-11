"""Create fixed-model diagnostics from validation predictions only."""
import argparse
import json
from pathlib import Path

import yaml

from ufc_tracker.strike.diagnostics import analyze_validation

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="head-v1")
    args = parser.parse_args()
    run = ROOT / "outputs/strike-detection" / args.run
    source = run / "validation_predictions.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    config = yaml.safe_load(
        (ROOT / "configs/models/strike_x3d_s.yaml").read_text(encoding="utf-8")
    )
    report = analyze_validation(rows, set(config["split"]["validation"]))
    output = run / "validation_diagnostics.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output.relative_to(ROOT)}")
