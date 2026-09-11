"""Clean-process verification of a validation-selected head (not final test evaluation)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ufc_tracker.annotations.dataset import sha256
from ufc_tracker.strike.cache import atomic_json, read_json
from ufc_tracker.strike.training import probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    ready = read_json(directory / "READY.json")
    if sha256(directory / "candidate.pt") != ready["candidate_sha256"]:
        raise ValueError("Candidate changed since validation")
    if sha256(directory / "reload_reference.npz") != ready["reload_reference_sha256"]:
        raise ValueError("Reference features changed since validation")
    if sha256(directory / "validation.json") != ready["validation_sha256"]:
        raise ValueError("Validation report changed")
    artifact = torch.load(directory / "candidate.pt", map_location="cpu", weights_only=True)
    if artifact["schema"] != "strike_head_candidate_v1" or artifact["class_map"] != {"no_strike": 0, "strike": 1}:
        raise ValueError("Unsupported candidate")
    head = torch.nn.Linear(2048, 2)
    head.load_state_dict(artifact["head"], strict=True)
    head.eval()
    with np.load(directory / "reload_reference.npz", allow_pickle=False) as reference:
        actual = probabilities(head, torch.from_numpy(reference["embeddings"]))
        expected = reference["probabilities"]
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise ValueError("Reloaded head does not reproduce reference probabilities")
    report = {"head_reload_ok": True, "max_abs_difference": float(np.max(np.abs(actual - expected))),
              "candidate_sha256": ready["candidate_sha256"],
              "scope": "Fresh-process head on reference embeddings; RGB export and test evaluation remain later stages"}
    atomic_json(directory / "reload_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
