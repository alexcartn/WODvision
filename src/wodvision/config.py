"""Configuration du pipeline, chargee depuis un unique fichier YAML.

Aucun seuil ne doit etre code en dur ailleurs dans le projet : tout parametre
utilise par plusieurs modules ou sujet a reglage doit vivre ici.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class CalibrationConfig(BaseModel):
    plate_diameter_mm: float = 450.0
    min_ellipse_ratio: float = 0.9  # minor/major sous ce seuil -> avertissement


class SignalConfig(BaseModel):
    resample_fps: float = 60.0
    max_gap_frames: int = 5  # comblement des trous courts uniquement
    butterworth_order: int = 4
    butterworth_cutoff_hz: float = 6.0
    min_keypoint_confidence: float = 0.3


class RepSegmentationConfig(BaseModel):
    peak_prominence_fraction: float = 0.5  # fraction du diametre du disque
    min_rep_distance_s: float = 0.5


class SquatConfig(BaseModel):
    depth_margin_px: float = 5.0
    lockout_angle_deg: float = 165.0


class PoseConfig(BaseModel):
    backend: str = "rtmlib"
    mode: str = "balanced"  # lightweight | balanced | performance


class TrackingConfig(BaseModel):
    max_occlusion_frames: int = 15


class Config(BaseModel):
    calibration: CalibrationConfig = CalibrationConfig()
    signal: SignalConfig = SignalConfig()
    rep_segmentation: RepSegmentationConfig = RepSegmentationConfig()
    squat: SquatConfig = SquatConfig()
    pose: PoseConfig = PoseConfig()
    tracking: TrackingConfig = TrackingConfig()


def load_config(path: str | Path | None = None) -> Config:
    """Charge la config depuis un fichier YAML, ou renvoie les valeurs par defaut."""
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
