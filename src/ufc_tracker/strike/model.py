"""Pinned X3D-S backbone and two raw logits; no online code downloads."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from pytorchvideo.models.hub import x3d_s

from ufc_tracker.annotations.dataset import sha256

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth"


class StrikeX3DS(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = nn.Linear(feature_dim, 2)
        self.feature_dim = feature_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def embeddings(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 13, 182, 182):
            raise ValueError("Expected [batch, 3, 13, 182, 182]")
        with torch.no_grad():
            result = self.backbone(clips)
        if result.shape != (clips.shape[0], self.feature_dim) or not torch.isfinite(result).all():
            raise ValueError("Invalid pre-classifier embeddings")
        return result

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.head(self.embeddings(clips))


class StrikeX3DSProjectionTune(nn.Module):
    """X3D-S with only its final projection block and binary head trainable."""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.backbone.blocks[-1].requires_grad_(True)
        self.head = nn.Linear(feature_dim, 2)
        self.feature_dim = feature_dim

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep every BatchNorm running statistic fixed for the small MMA dataset.
        self.backbone.eval()
        self.head.train(mode)
        return self

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, 13, 182, 182):
            raise ValueError("Expected [batch, 3, 13, 182, 182]")
        features = self.backbone(clips)
        if features.shape != (clips.shape[0], self.feature_dim):
            raise ValueError("Invalid pre-classifier embeddings")
        return self.head(features)


def build_model(weights: Path, expected_sha256: str) -> StrikeX3DS:
    if sha256(weights) != expected_sha256:
        raise ValueError("Pretrained weights hash mismatch")
    backbone = x3d_s(pretrained=False)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["model_state"], strict=True)
    head = backbone.blocks[-1]
    if not isinstance(head.proj, nn.Linear) or head.proj.out_features != 400:
        raise ValueError("Unexpected Kinetics head; inspect implementation before adapting")
    feature_dim = head.proj.in_features
    head.proj = nn.Identity()
    head.activation = None
    head.dropout = None
    return StrikeX3DS(backbone, feature_dim)


def build_projection_tune_model(
    weights: Path, expected_sha256: str, initial_head: dict[str, torch.Tensor] | None = None,
) -> StrikeX3DSProjectionTune:
    frozen = build_model(weights, expected_sha256)
    model = StrikeX3DSProjectionTune(frozen.backbone, frozen.feature_dim)
    if initial_head is not None:
        model.head.load_state_dict(initial_head, strict=True)
    return model
