"""Suivi multi-objets (ByteTrack).

`supervision.ByteTrack` est deprecie depuis supervision 0.28.0 au profit du
paquet separe `trackers` (classe `ByteTrackTracker`, methode `update()` au
lieu de `update_with_detections()`), et sera retire en 0.31.0. La version
installee (0.30.x) le fournit encore : on l'utilise pour l'instant, isole
dans ce wrapper pour ne toucher qu'un seul endroit le jour de la migration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from wodvision.detection.person import PersonDetection


@dataclass(slots=True)
class TrackedBox:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class Tracker:
    """Associe des detections a des identifiants de piste stables dans le temps."""

    def __init__(self, frame_rate: float = 30.0, lost_track_buffer: int = 30) -> None:
        self._tracker = sv.ByteTrack(frame_rate=frame_rate, lost_track_buffer=lost_track_buffer)

    def update(self, detections: list[PersonDetection]) -> list[TrackedBox]:
        if detections:
            xyxy = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            confidence = np.zeros((0,), dtype=np.float32)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence)
        tracked = self._tracker.update_with_detections(sv_detections)

        return [
            TrackedBox(
                track_id=int(track_id),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
            )
            for (x1, y1, x2, y2), track_id, conf in zip(
                tracked.xyxy, tracked.tracker_id, tracked.confidence, strict=True
            )
        ]

    def reset(self) -> None:
        self._tracker.reset()
