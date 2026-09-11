"""Selection de l'athlete a suivre parmi les pistes produites par le Tracker.

Ordre de priorite (section 7.3 de CLAUDE.md) :

1. clic utilisateur sur l'athlete dans la premiere frame (`--athlete-click`) ;
2. piste dont les poignets sont les plus proches de la barre detectee ;
3. piste ayant la plus grande bounding box moyenne sur toute la video.

Le detecteur de barre (section 7.5) n'est pas encore implemente, et les
poignets exacts necessiteraient la pose (qui n'est calculee qu'apres la
selection de l'athlete dans le pipeline). En attendant, la priorite 2 utilise
le centre de la bbox comme proxy des poignets — a affiner une fois la barre
detectable.

Une "piste" est representee par une liste de `(frame_index, bbox_xyxy)`
triee par frame, ce qui permet a la priorite 3 de faire une vraie moyenne
sur toute la duree de la piste plutot qu'un choix frame par frame instable.
"""

from __future__ import annotations

import math

Bbox = tuple[float, float, float, float]
Track = list[tuple[int, Bbox]]


def bbox_area(bbox: Bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _select_by_click(tracks: dict[int, Track], click_bbox: Bbox) -> int:
    best_track_id, best_iou = None, 0.0
    for track_id, frames in tracks.items():
        _, first_bbox = min(frames, key=lambda item: item[0])
        pair_iou = iou(first_bbox, click_bbox)
        if pair_iou > best_iou:
            best_track_id, best_iou = track_id, pair_iou
    if best_track_id is None:
        raise ValueError("aucune piste ne chevauche le clic sur la premiere frame")
    return best_track_id


def _select_by_bar_proximity(
    tracks: dict[int, Track], bar_center_by_frame: dict[int, tuple[float, float]]
) -> int:
    best_track_id, best_mean_dist = None, math.inf
    for track_id, frames in tracks.items():
        distances = []
        for frame_idx, bbox in frames:
            if frame_idx not in bar_center_by_frame:
                continue
            cx, cy = bbox_center(bbox)
            bx, by = bar_center_by_frame[frame_idx]
            distances.append(math.hypot(cx - bx, cy - by))
        if not distances:
            continue
        mean_dist = sum(distances) / len(distances)
        if mean_dist < best_mean_dist:
            best_track_id, best_mean_dist = track_id, mean_dist
    if best_track_id is None:
        raise ValueError("aucune piste ne chevauche temporellement les positions de barre")
    return best_track_id


def _select_by_largest_average_bbox(tracks: dict[int, Track]) -> int:
    best_track_id, best_mean_area = None, -1.0
    for track_id, frames in tracks.items():
        mean_area = sum(bbox_area(bbox) for _, bbox in frames) / len(frames)
        if mean_area > best_mean_area:
            best_track_id, best_mean_area = track_id, mean_area
    if best_track_id is None:
        raise ValueError("aucune piste disponible pour la selection")
    return best_track_id


def select_athlete_track(
    tracks: dict[int, Track],
    click_bbox: Bbox | None = None,
    bar_center_by_frame: dict[int, tuple[float, float]] | None = None,
) -> int:
    """Choisit le track_id de l'athlete selon l'ordre de priorite documente."""
    if not tracks:
        raise ValueError("aucune piste a selectionner")
    if click_bbox is not None:
        return _select_by_click(tracks, click_bbox)
    if bar_center_by_frame is not None:
        return _select_by_bar_proximity(tracks, bar_center_by_frame)
    return _select_by_largest_average_bbox(tracks)
