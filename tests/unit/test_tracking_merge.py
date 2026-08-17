from __future__ import annotations

import numpy as np

from ufc_tracker.tracking.merge import (
    FighterSlot,
    TrackFragment,
    merge_fragments,
    recover_missing_fighter_records,
    resolve_fighter_slots,
)
from ufc_tracker.tracking.contracts import TrackObservation, TrackingRecord


def _fragment(
    track_id: int,
    first: int,
    last: int,
    center_x: float,
    *,
    diagonal: float = 400.0,
) -> TrackFragment:
    return TrackFragment(
        track_id=track_id,
        frames=frozenset(range(first, last + 1)),
        first_frame=first,
        last_frame=last,
        first_centroid=(center_x, 300.0),
        last_centroid=(center_x, 300.0),
        diagonal=diagonal,
        mean_cx=center_x,
        skin=0.8,
        area=0.1,
    )


def test_resolve_slots_recovers_fighter_fragments_split_by_camera_cuts() -> None:
    fragments = {
        fragment.track_id: fragment
        for fragment in (
            _fragment(1, 0, 170, 400.0),
            _fragment(5, 1, 28, 600.0),
            _fragment(89, 50, 871, 700.0),
            _fragment(233, 172, 221, 410.0),
            _fragment(364, 234, 426, 420.0),
            # Large camera-cut jump: the greedy distance merge creates a new slot.
            _fragment(750, 428, 859, 900.0),
            _fragment(1195, 873, 977, 850.0),
            # Reappears opposite track 1195 and must join track 89's identity.
            _fragment(1254, 909, 981, 300.0),
        )
    }

    initial_slots = merge_fragments(fragments)
    fighters, unassigned = resolve_fighter_slots(initial_slots)

    identities = [set(slot.track_ids) for slot in fighters]
    assert {1, 233, 364, 750, 1195} in identities
    assert {5, 89, 1254} in identities
    assert unassigned == []
    assert sum(len(slot.track_ids) for slot in fighters) == len(fragments)


def test_resolve_slots_never_places_overlapping_fragments_in_one_identity() -> None:
    slots = [
        FighterSlot.from_fragment(_fragment(1, 0, 100, 300.0)),
        FighterSlot.from_fragment(_fragment(2, 0, 100, 700.0)),
        FighterSlot.from_fragment(_fragment(3, 20, 80, 500.0)),
    ]

    fighters, unassigned = resolve_fighter_slots(slots)

    assert len(fighters) == 2
    assert [slot.track_ids for slot in unassigned] == [[3]]
    for fighter in fighters:
        assert len(fighter.track_ids) == 1


def test_absorb_slot_supports_a_fragment_chain_before_the_anchor() -> None:
    later = FighterSlot.from_fragment(_fragment(20, 100, 150, 800.0))
    earlier = FighterSlot.from_fragment(_fragment(10, 0, 90, 300.0))

    later.absorb_slot(earlier)

    assert later.track_ids == [10, 20]
    assert later.first_frame == 0
    assert later.last_frame == 150
    assert len(later.frames) == 142


def test_resolve_slots_keeps_disconnected_early_camera_shot() -> None:
    slots = [
        FighterSlot.from_fragment(_fragment(1, 0, 100, 300.0)),
        FighterSlot.from_fragment(_fragment(2, 0, 100, 900.0)),
        # The longer late shot becomes the anchor pair.
        FighterSlot.from_fragment(_fragment(10, 500, 900, 320.0)),
        FighterSlot.from_fragment(_fragment(20, 500, 900, 880.0)),
        # Simultaneous third person must never enter either fighter identity.
        FighterSlot.from_fragment(_fragment(99, 20, 80, 600.0)),
    ]

    fighters, unassigned = resolve_fighter_slots(slots)

    identities = [set(slot.track_ids) for slot in fighters]
    assert {1, 10} in identities
    assert {2, 20} in identities
    assert [slot.track_ids for slot in unassigned] == [[99]]
    assert all(fighter.first_frame == 0 for fighter in fighters)


def test_gap_recovery_prefers_large_dimly_lit_fighter_over_spectator() -> None:
    records = [
        TrackingRecord(0, 0.0, "1", 10, (0.0, 0.0, 100.0, 200.0), 0.9, True),
        TrackingRecord(0, 0.0, "2", -1, None, None, False),
    ]
    per_frame = [
        {
            10: TrackObservation(np.array([0, 0, 100, 200]), None, 0.9),
            20: TrackObservation(np.array([200, 0, 340, 240]), None, 0.9),
            30: TrackObservation(np.array([400, 0, 440, 80]), None, 0.9),
        }
    ]
    stats = {
        10: {"n": 30, "skin": 18.0, "area": 3.0, "cx": 0.0},
        20: {"n": 30, "skin": 10.2, "area": 2.4, "cx": 0.0},
        30: {"n": 30, "skin": 24.0, "area": 0.6, "cx": 0.0},
    }

    recovered, count = recover_missing_fighter_records(records, per_frame, stats)

    assert count == 1
    assert recovered[1].visible is True
    assert recovered[1].track_id == 20
