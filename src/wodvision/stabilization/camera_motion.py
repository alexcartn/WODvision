"""Estimation du mouvement de camera pour garder des mesures coherentes.

Les smartphones filment parfois a la main (ou la camera derive / tourne). On
suit des points du fond (hors athlete) par flot optique de Lucas-Kanade, puis
on estime une transformation rigide (translation + rotation) robuste par RANSAC
entre frames consecutives. Les transformations sont cumulees pour ramener
chaque frame dans le repere de la premiere frame : les keypoints "stabilises"
sont alors comparables dans le temps.

Limites assumees : modele rigide (rotation + translation, pas de zoom ni de
perspective : l'echelle estimee derivait sur des fonds peu textures), derive lente
possible sur de longues videos, et un fond sans texture ou entierement masque
par l'athlete donne une estimation peu fiable (signalee par `confidence`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _to_rigid(m: np.ndarray, p_cur: np.ndarray, p_prev: np.ndarray) -> np.ndarray:
    """Retire l'echelle d'une similitude (le zoom cumule derive) : rotation + translation.

    La translation est recalculee sur les points apparies (mediane) pour rester
    coherente une fois l'echelle fixee a 1.
    """
    angle = np.arctan2(m[1, 0], m[0, 0])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    t = np.median(p_prev - p_cur @ rot.T, axis=0)
    out = np.eye(3)
    out[:2, :2], out[:2, 2] = rot, t
    return out


@dataclass(slots=True)
class MotionStep:
    to_ref: np.ndarray  # (3, 3) : coordonnees frame courante -> frame de reference
    n_inliers: int
    reliable: bool


class CameraMotionEstimator:
    """Estime, frame apres frame, la transformation vers la frame de reference."""

    def __init__(
        self,
        analysis_width_px: int = 640,
        max_features: int = 300,
        min_inliers: int = 12,
        bbox_margin_fraction: float = 0.15,
    ) -> None:
        self._width = analysis_width_px
        self._max_features = max_features
        self._min_inliers = min_inliers
        self._margin = bbox_margin_fraction
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._scale = 1.0
        self._to_ref = np.eye(3)
        self._max_step_deg = 3.0  # rotation max entre deux frames
        self._max_step_frac = 0.1  # translation max entre deux frames, fraction de la largeur

    def _prepare(self, image_bgr: np.ndarray, bbox: tuple[float, float, float, float] | None):
        h, w = image_bgr.shape[:2]
        self._scale = min(1.0, self._width / w)
        small = image_bgr
        if self._scale < 1:
            small = cv2.resize(image_bgr, None, fx=self._scale, fy=self._scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            mx, my = (x2 - x1) * self._margin, (y2 - y1) * self._margin
            p1 = (int(max(0, (x1 - mx) * self._scale)), int(max(0, (y1 - my) * self._scale)))
            p2 = (int((x2 + mx) * self._scale), int((y2 + my) * self._scale))
            cv2.rectangle(mask, p1, p2, 0, thickness=-1)
        return gray, mask

    def _plausible(self, step: np.ndarray, width: int) -> bool:
        """Rejette les sauts irrealistes d'une frame a l'autre (mauvais appariement)."""
        angle = np.degrees(np.arctan2(step[1, 0], step[0, 0]))
        shift = float(np.hypot(*step[:2, 2]))
        return abs(angle) <= self._max_step_deg and shift <= self._max_step_frac * width

    def update(
        self, image_bgr: np.ndarray, athlete_bbox: tuple[float, float, float, float] | None
    ) -> MotionStep:
        gray, mask = self._prepare(image_bgr, athlete_bbox)
        if self._prev_gray is None:
            self._prev_gray, self._prev_mask = gray, mask
            return MotionStep(self._to_ref.copy(), 0, True)

        pts_prev = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self._max_features, qualityLevel=0.01,
            minDistance=8, mask=self._prev_mask,
        )
        step = np.eye(3)
        n_inliers = 0
        if pts_prev is not None and len(pts_prev) >= self._min_inliers:
            pts_cur, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts_prev, None)
            ok = status.ravel() == 1
            p0, p1 = pts_prev[ok].reshape(-1, 2), pts_cur[ok].reshape(-1, 2)
            # Ne garder que les points qui n'atterrissent pas sur l'athlete.
            inside = mask[
                np.clip(p1[:, 1].astype(int), 0, mask.shape[0] - 1),
                np.clip(p1[:, 0].astype(int), 0, mask.shape[1] - 1),
            ] > 0
            p0, p1 = p0[inside], p1[inside]
            if len(p0) >= self._min_inliers:
                m, inl = cv2.estimateAffinePartial2D(
                    p1, p0, method=cv2.RANSAC, ransacReprojThreshold=2.0
                )
                if m is not None:
                    candidate = _to_rigid(m, p1[inl.ravel() == 1], p0[inl.ravel() == 1])
                    if self._plausible(candidate, gray.shape[1]):
                        n_inliers = int(inl.sum())
                        step = candidate
        reliable = n_inliers >= self._min_inliers

        # step : coords (reduites) frame courante -> frame precedente.
        s = np.diag([self._scale, self._scale, 1.0])
        step_full = np.linalg.inv(s) @ step @ s
        self._to_ref = self._to_ref @ step_full
        self._prev_gray, self._prev_mask = gray, mask
        return MotionStep(self._to_ref.copy(), n_inliers, reliable)


def apply_transform(to_ref: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Applique une matrice 3x3 a des points (..., 2)."""
    xy = np.asarray(xy, dtype=float)
    return xy @ to_ref[:2, :2].T + to_ref[:2, 2]


def motion_amplitude_px(transforms: np.ndarray, width: int, height: int) -> float:
    """Amplitude maximale du deplacement du centre de l'image (px) : 0 = camera fixe."""
    center = np.array([width / 2, height / 2])
    moved = np.array([apply_transform(t, center) for t in transforms])
    return float(np.max(np.linalg.norm(moved - center, axis=1)))
