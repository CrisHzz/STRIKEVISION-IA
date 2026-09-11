from pathlib import Path

import pytest
import torch
import yaml

from ufc_tracker.strike.model import build_model, build_projection_tune_model

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model():
    config = yaml.safe_load((ROOT / "configs/models/strike_x3d_s.yaml").read_text())
    weights = ROOT / config["extractor"]["weights_path"]
    if not weights.exists():
        pytest.skip("Download the pinned official weights before this local integration test")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_model(weights, config["extractor"]["weights_sha256"])


def test_head_backward_keeps_backbone_frozen(model):
    model.train()
    before = {k: v.clone() for k, v in model.backbone.named_buffers()}
    x = torch.rand(2, 3, 13, 182, 182)
    embedding = model.embeddings(x)
    assert embedding.shape == (2, 2048)  # not 400 Kinetics scores
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert not model.backbone.training
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.backbone.named_buffers())
    assert torch.equal(embedding, model.embeddings(x))


def test_local_state_roundtrip_and_input_contract(model, tmp_path):
    model.eval()
    x = torch.rand(1, 3, 13, 182, 182)
    with torch.no_grad():
        expected = model(x)
    file = tmp_path / "state.pt"
    torch.save(model.state_dict(), file)
    with torch.no_grad():
        model.head.weight.add_(1)
    model.load_state_dict(torch.load(file, weights_only=True))
    with torch.no_grad():
        assert torch.equal(model(x), expected)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(1, 3, 24, 182, 182))


def test_weights_hash_rejected(tmp_path):
    file = tmp_path / "bad.pyth"
    file.write_bytes(b"bad weights")
    with pytest.raises(ValueError, match="hash"):
        build_model(file, "wrong-hash")


def test_projection_tune_scope(model):
    config = yaml.safe_load((ROOT / "configs/models/strike_x3d_s.yaml").read_text())
    tuned = build_projection_tune_model(
        ROOT / config["extractor"]["weights_path"],
        config["extractor"]["weights_sha256"],
        model.head.state_dict(),
    )
    trainable = [name for name, value in tuned.backbone.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("blocks.5.") for name in trainable)
    assert sum(value.numel() for value in tuned.parameters() if value.requires_grad) == 972642
    tuned.train()
    assert not tuned.backbone.training
