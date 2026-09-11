"""Suivi "collant" d'un athlete unique a travers les changements d'identifiant.

ByteTrack peut perdre puis recreer un identifiant de piste lors d'occlusions
prolongees (pas de re-identification visuelle). Pour une video ou l'on suit
UN athlete precis pendant plusieurs minutes, cela suffit a faire deriver le
compteur d'IDs (observe : 19 IDs pour 9 personnes en 6 secondes sur cette
video). `StickyTarget` maintient un identifiant logique unique en se
re-accrochant, a chaque frame, a la piste dont la bbox est la plus proche
(IoU, a defaut distance de centre) de la derniere position connue.

Approche pragmatique, pas de la reidentification visuelle : elle suppose que
la cible ne "saute" pas trop d'une frame a l'autre, ce qui est vrai tant que
le detecteur tourne a la cadence video normale.

Deux garde-fous avant d'accepter une re-acquisition :
- ratio d'aire (`max_area_ratio`) : rejette les candidats dont la taille de
  bbox est trop differente de la derniere connue, puisqu'une personne plus
  loin/proche de la camera a une bbox nettement plus petite/grande ;
- seuil d'IoU minimal (`min_iou_accept`) : un simple chevauchement non nul
  (observe jusqu'a IoU=0.02-0.25 sur cette video) ne suffit pas — un
  spectateur d'arriere-plan qui frole la derniere position connue de la
  cible passait sinon la garde. En dessous de ce seuil, on retombe sur la
  distance centre-a-centre plutot que d'accepter un chevauchement marginal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wodvision.tracking.athlete_selector import Bbox, bbox_area, bbox_center, iou
from wodvision.tracking.tracker import TrackedBox


@dataclass(slots=True)
class StickyTarget:
    last_bbox: Bbox
    current_track_id: int | None = None
    max_center_jump_px: float = 60.0
    max_area_ratio: float = 2.2
    min_iou_accept: float = 0.5

    def update(self, tracked_boxes: list[TrackedBox]) -> Bbox | None:
        """Renvoie la bbox retenue pour cette frame, ou None si la cible est perdue."""
        if not tracked_boxes:
            return None

        if self.current_track_id is not None:
            match = next(
                (tb for tb in tracked_boxes if tb.track_id == self.current_track_id), None
            )
            if match is not None:
                self.last_bbox = match.bbox_xyxy
                return self.last_bbox

        candidates = [tb for tb in tracked_boxes if self._area_compatible(tb.bbox_xyxy)]
        if not candidates:
            return None

        best_by_iou = max(candidates, key=lambda tb: iou(tb.bbox_xyxy, self.last_bbox))
        if iou(best_by_iou.bbox_xyxy, self.last_bbox) >= self.min_iou_accept:
            self.current_track_id = best_by_iou.track_id
            self.last_bbox = best_by_iou.bbox_xyxy
            return self.last_bbox

        cx, cy = bbox_center(self.last_bbox)

        def dist_to_last(tb: TrackedBox) -> float:
            tx, ty = bbox_center(tb.bbox_xyxy)
            return math.hypot(tx - cx, ty - cy)

        nearest = min(candidates, key=dist_to_last)
        if dist_to_last(nearest) > self.max_center_jump_px:
            return None

        self.current_track_id = nearest.track_id
        self.last_bbox = nearest.bbox_xyxy
        return self.last_bbox

    def _area_compatible(self, bbox: Bbox) -> bool:
        area_last = bbox_area(self.last_bbox)
        area_candidate = bbox_area(bbox)
        if area_last <= 0 or area_candidate <= 0:
            return False
        ratio = max(area_last, area_candidate) / min(area_last, area_candidate)
        return ratio <= self.max_area_ratio
