from __future__ import annotations

from pathlib import Path

from ufc_tracker.ui.split_annotation_app import SplitAnnotationApp, discover_split_categories


def _touch_mp4(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_split_categories_lists_subfolders_with_videos(tmp_path: Path) -> None:
    splits = tmp_path / "data" / "splits"
    _touch_mp4(splits / "aggressive_men" / "fight_a_round1.mp4")
    _touch_mp4(splits / "normal_men" / "fight_b_round1.mp4")
    _touch_mp4(splits / "normal_men" / "fight_b_round2.mp4")
    (splits / "empty_category").mkdir(parents=True)

    categories = discover_split_categories(splits)

    assert list(categories) == ["aggressive_men", "normal_men"]
    assert categories["aggressive_men"] == (splits / "aggressive_men").resolve()


def test_discover_split_categories_accepts_a_single_category_folder(tmp_path: Path) -> None:
    category = tmp_path / "data" / "splits" / "aggressive_men"
    _touch_mp4(category / "fight_a_round1.mp4")

    categories = discover_split_categories(category)

    assert list(categories) == ["aggressive_men"]
    assert categories["aggressive_men"] == category.resolve()


def test_split_app_lists_videos_per_folder_and_keeps_pose_paths(tmp_path: Path) -> None:
    splits = tmp_path / "data" / "splits"
    _touch_mp4(splits / "aggressive_men" / "fight_a_round1.mp4")
    _touch_mp4(splits / "normal_men" / "fight_b_round2.mp4")
    controller = SplitAnnotationApp(
        root=tmp_path,
        split_dir=splits,
        config=None,  # type: ignore[arg-type]
        pose_config={},
        pose_root=tmp_path / "data" / "processed" / "poses",
        annotation_root=tmp_path / "data" / "annotations",
        media=None,  # type: ignore[arg-type]
    )

    assert controller.category_names == ["aggressive_men", "normal_men"]
    assert controller.videos_in("normal_men") == ["fight_b_round2.mp4"]
    video = controller._video_path("aggressive_men", "fight_a_round1.mp4")
    assert controller._pose_dir(video) == (
        tmp_path / "data" / "processed" / "poses" / "aggressive_men" / "fight_a_round1"
    ).resolve()
    assert controller._annotation_path(video).name == "fight_a_round1.jsonl"
