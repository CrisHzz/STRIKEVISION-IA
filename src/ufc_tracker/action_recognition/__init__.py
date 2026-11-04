"""Datasets and evaluation contracts for per-fighter MMA action recognition."""

from ufc_tracker.action_recognition.contracts import ACTION_CLASSES
from ufc_tracker.action_recognition.dataset import build_action_corpus, write_action_corpus

__all__ = ["ACTION_CLASSES", "build_action_corpus", "write_action_corpus"]
