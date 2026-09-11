import pytest

from wodvision.tracking.athlete_selector import select_athlete_track


def test_select_by_click_matches_overlapping_track() -> None:
    tracks = {
        1: [(0, (0.0, 0.0, 10.0, 10.0)), (1, (1.0, 1.0, 11.0, 11.0))],
        2: [(0, (100.0, 100.0, 120.0, 130.0)), (1, (101.0, 101.0, 121.0, 131.0))],
    }
    chosen = select_athlete_track(tracks, click_bbox=(2.0, 2.0, 12.0, 12.0))
    assert chosen == 1


def test_select_by_click_raises_when_no_overlap() -> None:
    tracks = {1: [(0, (0.0, 0.0, 10.0, 10.0))]}
    with pytest.raises(ValueError):
        select_athlete_track(tracks, click_bbox=(1000.0, 1000.0, 1010.0, 1010.0))


def test_select_by_bar_proximity_prefers_closest_track() -> None:
    tracks = {
        1: [(0, (0.0, 0.0, 10.0, 10.0))],  # centre (5, 5)
        2: [(0, (90.0, 90.0, 110.0, 110.0))],  # centre (100, 100)
    }
    bar_center_by_frame = {0: (95.0, 95.0)}
    chosen = select_athlete_track(tracks, bar_center_by_frame=bar_center_by_frame)
    assert chosen == 2


def test_select_by_largest_average_bbox() -> None:
    tracks = {
        1: [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0))],  # aire 100
        2: [(0, (0.0, 0.0, 50.0, 50.0)), (1, (0.0, 0.0, 50.0, 50.0))],  # aire 2500
    }
    chosen = select_athlete_track(tracks)
    assert chosen == 2


def test_click_takes_priority_over_bar_proximity() -> None:
    tracks = {
        1: [(0, (0.0, 0.0, 10.0, 10.0))],
        2: [(0, (90.0, 90.0, 110.0, 110.0))],
    }
    # La barre est plus proche de la piste 2, mais le clic vise la piste 1.
    chosen = select_athlete_track(
        tracks, click_bbox=(0.0, 0.0, 10.0, 10.0), bar_center_by_frame={0: (95.0, 95.0)}
    )
    assert chosen == 1


def test_select_raises_on_empty_tracks() -> None:
    with pytest.raises(ValueError):
        select_athlete_track({})
