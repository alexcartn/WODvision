from wodvision.tracking.sticky_target import StickyTarget
from wodvision.tracking.tracker import TrackedBox


def _box(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackedBox:
    return TrackedBox(track_id=track_id, bbox_xyxy=(x1, y1, x2, y2), confidence=0.9)


def test_sticky_target_follows_same_id_when_present() -> None:
    target = StickyTarget(last_bbox=(0.0, 0.0, 10.0, 10.0), current_track_id=1)
    result = target.update([_box(1, 1.0, 1.0, 11.0, 11.0), _box(2, 500.0, 500.0, 520.0, 520.0)])
    assert result == (1.0, 1.0, 11.0, 11.0)
    assert target.current_track_id == 1


def test_sticky_target_reacquires_by_iou_when_id_switches() -> None:
    # La cible avait la piste 1, qui disparait (occlusion) ; une nouvelle
    # piste 7 apparait au meme endroit -> on doit s'y raccrocher.
    target = StickyTarget(last_bbox=(0.0, 0.0, 10.0, 10.0), current_track_id=1)
    result = target.update([_box(7, 0.5, 0.5, 10.5, 10.5), _box(9, 500.0, 500.0, 520.0, 520.0)])
    assert result == (0.5, 0.5, 10.5, 10.5)
    assert target.current_track_id == 7


def test_sticky_target_falls_back_to_nearest_center_within_range() -> None:
    # Pas de chevauchement IoU (mouvement rapide), mais le centre le plus
    # proche reste dans la tolerance -> on s'y raccroche quand meme.
    target = StickyTarget(
        last_bbox=(0.0, 0.0, 10.0, 10.0), current_track_id=1, max_center_jump_px=50.0
    )
    result = target.update([_box(3, 20.0, 0.0, 30.0, 10.0), _box(4, 500.0, 500.0, 520.0, 520.0)])
    assert result == (20.0, 0.0, 30.0, 10.0)
    assert target.current_track_id == 3


def test_sticky_target_gives_up_when_nothing_close_enough() -> None:
    target = StickyTarget(
        last_bbox=(0.0, 0.0, 10.0, 10.0), current_track_id=1, max_center_jump_px=5.0
    )
    result = target.update([_box(3, 500.0, 500.0, 520.0, 520.0)])
    assert result is None


def test_sticky_target_returns_none_when_no_detections() -> None:
    target = StickyTarget(last_bbox=(0.0, 0.0, 10.0, 10.0), current_track_id=1)
    assert target.update([]) is None


def test_sticky_target_rejects_reacquisition_with_mismatched_size() -> None:
    # Meme piste 1 disparue ; une piste 8 chevauche legerement au meme
    # endroit mais est beaucoup plus petite (personne d'arriere-plan) -> ne
    # doit pas s'y raccrocher malgre l'IoU > 0.
    target = StickyTarget(last_bbox=(0.0, 0.0, 100.0, 100.0), current_track_id=1)
    result = target.update([_box(8, 90.0, 90.0, 100.0, 100.0)])
    assert result is None


def test_sticky_target_accepts_reacquisition_with_similar_size() -> None:
    target = StickyTarget(last_bbox=(0.0, 0.0, 100.0, 100.0), current_track_id=1)
    result = target.update([_box(8, 5.0, 5.0, 105.0, 105.0)])
    assert result == (5.0, 5.0, 105.0, 105.0)
    assert target.current_track_id == 8
