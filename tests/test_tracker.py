from wodvision.detection.person import PersonDetection
from wodvision.tracking.tracker import Tracker

Bbox = tuple[float, float, float, float]


def _shift(bbox: Bbox, dx: float, dy: float) -> Bbox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def test_tracker_keeps_stable_ids_for_non_overlapping_moving_boxes() -> None:
    tracker = Tracker(frame_rate=30.0, lost_track_buffer=10)

    box_a = (0.0, 0.0, 50.0, 100.0)
    box_b = (500.0, 0.0, 550.0, 100.0)

    track_id_a: int | None = None
    track_id_b: int | None = None

    for frame_idx in range(10):
        detections = [
            PersonDetection(bbox_xyxy=_shift(box_a, frame_idx * 2.0, 0.0), confidence=0.9),
            PersonDetection(bbox_xyxy=_shift(box_b, -frame_idx * 2.0, 0.0), confidence=0.9),
        ]
        tracked = tracker.update(detections)
        assert len(tracked) == 2

        by_x = sorted(tracked, key=lambda t: t.bbox_xyxy[0])
        if track_id_a is None:
            track_id_a, track_id_b = by_x[0].track_id, by_x[1].track_id
        else:
            assert by_x[0].track_id == track_id_a
            assert by_x[1].track_id == track_id_b


def test_tracker_handles_empty_detections() -> None:
    tracker = Tracker()
    assert tracker.update([]) == []
