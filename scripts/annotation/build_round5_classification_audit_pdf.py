from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).with_name("build_round4_classification_audit_pdf.py")
spec = importlib.util.spec_from_file_location("classification_audit_pdf", BASE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

name = "zhang_joanna_1__zhang_weili_vs_joanna_jedrzejczyk_1__aggressive_women_round5"
module.ANNOTATION = ROOT / "data/annotations/strike_actions_zhang_joanna_trial" / f"{name}.json"
module.OUTPUT = ROOT / "output/pdf/zhang_vs_joanna_round5_auditoria_clasificacion.pdf"
module.ROUND_NUMBER = 5
module.EXPECTED_FRAGMENTS = 71
module.REVIEW_DIR = ROOT / "tmp_round5_audit"


if __name__ == "__main__":
    module.main()
