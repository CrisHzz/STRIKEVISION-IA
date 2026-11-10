"""Compact CPU-friendly pose models for the first action-recognition tournament."""

from __future__ import annotations

import torch
from torch import nn

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.pose.estimator import KEYPOINT_NAMES


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Binary loss over known labels only; unknown labels contribute no gradient."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    denominator = mask.sum().clamp_min(1).to(loss.dtype)
    return (loss * mask.to(loss.dtype)).sum() / denominator


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class PoseTCN(nn.Module):
    """Temporal CNN over attacker-relative pose, rival context and validity masks."""
    architecture = "pose_tcn_v1"

    def __init__(self, feature_dim: int, class_count: int, channels: int = 64, blocks: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.input = nn.Conv1d(feature_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(TemporalBlock(channels, dropout) for _ in range(blocks)))
        self.output = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, class_count))

    def pooled_features(self, features: torch.Tensor, attacker_frame_mask: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.input(features.transpose(1, 2))).transpose(1, 2)
        mask = attacker_frame_mask.to(values.dtype).unsqueeze(-1)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(self, features: torch.Tensor, attacker_frame_mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.pooled_features(features, attacker_frame_mask))


class AttentivePoseTCN(nn.Module):
    """TCN that preserves brief motion peaks while learning where to attend."""

    architecture = "pose_tcn_attention_v2"

    def __init__(self, feature_dim: int, class_count: int, channels: int = 64, blocks: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.input = nn.Conv1d(feature_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(*(TemporalBlock(channels, dropout) for _ in range(blocks)))
        self.attention = nn.Linear(channels, 1)
        self.output = nn.Sequential(nn.LayerNorm(channels * 2), nn.Linear(channels * 2, class_count))

    def forward(self, features: torch.Tensor, attacker_frame_mask: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.input(features.transpose(1, 2))).transpose(1, 2)
        valid = attacker_frame_mask.bool()
        scores = self.attention(values).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=1) * valid.to(values.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        attended = (values * weights.unsqueeze(-1)).sum(dim=1)
        masked_values = values.masked_fill(~valid.unsqueeze(-1), -1e4)
        peak = masked_values.amax(dim=1)
        peak = torch.where(valid.any(dim=1, keepdim=True), peak, torch.zeros_like(peak))
        return self.output(torch.cat((attended, peak), dim=1))


class GraphBlock(nn.Module):
    def __init__(self, channels: int, adjacency: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("bctv,vw->bctw", values, self.adjacency)
        return values + self.dropout(torch.nn.functional.gelu(self.norm(self.conv(spatial))))


def _skeleton_adjacency(joints: int) -> torch.Tensor:
    edges = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
    adjacency = torch.eye(joints)
    for first, second in edges:
        adjacency[first, second] = adjacency[second, first] = 1
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1)


class CompactSTGCN(nn.Module):
    """A local graph baseline; no external ST-GCN package or checkpoint required."""
    architecture = "compact_stgcn_v1"

    def __init__(self, feature_dim: int, class_count: int, channels: int = 48, blocks: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        joints = len(KEYPOINT_NAMES)
        self.joints, self.position_dim, self.confidence_offset = joints, joints * 4, joints * 8
        self.mask_offset = self.confidence_offset + joints * 2
        self.input = nn.Conv2d(6, channels, kernel_size=1)
        adjacency = _skeleton_adjacency(joints)
        self.blocks = nn.Sequential(*(GraphBlock(channels, adjacency, dropout) for _ in range(blocks)))
        self.output = nn.Linear(channels, class_count)

    def forward(self, features: torch.Tensor, attacker_frame_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, dimension = features.shape
        if dimension < self.mask_offset + self.joints * 2:
            raise ValueError("Feature dimensionality is incompatible with CompactSTGCN")
        positions = features[:, :, :self.position_dim].reshape(batch, steps, 2, self.joints, 2)
        masks = features[:, :, self.mask_offset:self.mask_offset + self.joints * 2].reshape(batch, steps, 2, self.joints, 1)
        values = torch.cat((positions, masks), dim=-1).permute(0, 2, 4, 1, 3).reshape(batch, 6, steps, self.joints)
        values = self.blocks(self.input(values)).mean(dim=-1).transpose(1, 2)
        mask = attacker_frame_mask.to(values.dtype).unsqueeze(-1)
        pooled = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.output(pooled)


class PunchHeadHybridTCN(nn.Module):
    """Keep the reference logits except for punch_head, which uses a second TCN."""

    def __init__(self, feature_dim: int, class_count: int, *, reference_parameters: dict, donor_parameters: dict) -> None:
        super().__init__()
        self.reference = PoseTCN(feature_dim, class_count, **reference_parameters)
        self.donor = PoseTCN(feature_dim, class_count, **donor_parameters)

    def forward(self, features: torch.Tensor, attacker_frame_mask: torch.Tensor) -> torch.Tensor:
        reference = self.reference(features, attacker_frame_mask)
        donor = self.donor(features, attacker_frame_mask)
        head = ACTION_CLASSES.index("punch_head")
        return torch.cat((reference[:, :head], donor[:, head:head + 1], reference[:, head + 1:]), dim=1)


def build_torch_model(name: str, feature_dim: int, class_count: int, **parameters: object) -> nn.Module:
    if name == "tcn":
        return PoseTCN(feature_dim, class_count, **parameters)
    if name == "tcn_punch_head_hybrid":
        return PunchHeadHybridTCN(feature_dim, class_count, **parameters)
    if name == "hierarchical_tcn":
        return AttentivePoseTCN(feature_dim, class_count, **parameters)
    if name == "stgcn_compact":
        return CompactSTGCN(feature_dim, class_count, **parameters)
    raise ValueError(f"Unsupported local pose model: {name}")
