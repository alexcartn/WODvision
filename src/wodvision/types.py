"""Dataclasses partagees entre les modules du pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DepthStatus = Literal["ok", "no", "uncertain"]


@dataclass(slots=True)
class Frame:
    """Une frame video avec son timestamp reel (PTS)."""

    index: int
    pts_s: float
    image_bgr: np.ndarray


@dataclass(slots=True)
class Keypoints:
    """Pose 2D au format COCO-17."""

    xy: np.ndarray  # shape (17, 2), pixels
    score: np.ndarray  # shape (17,)


@dataclass(slots=True)
class BarObservation:
    """Observation de la barre sur une frame."""

    center_px: tuple[float, float]
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float
    confidence: float


@dataclass(slots=True)
class RepMetrics:
    """Metriques calculees pour une repetition."""

    rep_index: int
    t_start: float
    t_bottom: float
    t_end: float
    depth_status: DepthStatus
    knee_angle_min: float
    hip_angle_min: float
    mean_concentric_velocity: float
    peak_velocity: float
    vertical_displacement: float
    bar_horizontal_deviation_max: float
    eccentric_duration: float
    concentric_duration: float
    quality_score: float


@dataclass(slots=True)
class SessionSummary:
    """Resume d'une seance analysee."""

    video_id: str
    n_reps: int
    n_invalid_reps: int
    velocity_loss_pct: float
    reps: list[RepMetrics] = field(default_factory=list)
