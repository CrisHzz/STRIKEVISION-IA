from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_tournament_summary_orders_development_score(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/ml/summarize_action_tournament.py"
    spec = importlib.util.spec_from_file_location("action_tournament", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    first = tmp_path / "first.json"; second = tmp_path / "second.json"
    first.write_text(json.dumps({"family": "one", "validation_fold": 0, "metrics": {"primary_macro_f1": .5, "all_class_macro_f1": .4, "primary_macro_average_precision": .5, "classes": {}}}), encoding="utf-8")
    second.write_text(json.dumps({"family": "two", "validation_fold": 0, "metrics": {"primary_macro_f1": .8, "all_class_macro_f1": .7, "primary_macro_average_precision": .8, "classes": {}}}), encoding="utf-8")
    assert module._summary(second)["primary_macro_f1"] > module._summary(first)["primary_macro_f1"]
