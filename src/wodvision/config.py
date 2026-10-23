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


class CameraMotionConfig(BaseModel):
    enabled: bool = True
    analysis_width_px: int = 640  # largeur de travail pour l'estimation de mouvement
    max_features: int = 300
    min_inliers: int = 12  # en dessous, la frame est marquee peu fiable
    bbox_margin_fraction: float = 0.15  # marge autour de l'athlete exclue du fond


class RowingConfig(BaseModel):
    """Seuils de l'analyse aviron. Ce sont des hypotheses de depart a calibrer
    sur des videos annotees par un coach (voir docs/rowing_concept.md)."""

    analysis_fps: float = 30.0  # cadence max d'inference (frames sautees au-dela)
    detect_every_n_frames: int = 8  # detecteur de personnes ; entre deux, bbox deduite de la pose
    bbox_from_pose_margin: float = 0.25  # marge autour des keypoints pour la bbox deduite
    cutoff_hz: float = 6.0
    stroke_prominence_fraction: float = 0.35  # fraction de l'amplitude du signal poignee
    min_stroke_period_s: float = 1.2
    min_stroke_rate_spm: float = 10.0
    max_stroke_rate_spm: float = 50.0
    phase_onset_fraction: float = 0.1  # debut d'un segment = 10 % de son excursion
    min_leg_range_deg: float = 20.0
    min_trunk_range_deg: float = 10.0
    min_arm_range_deg: float = 30.0
    flagged_stroke_fraction: float = 0.3  # part de coups fautifs pour lever un constat
    # Seuils de regles
    min_leg_fraction_at_trunk_onset: float = 0.4
    min_leg_fraction_at_arm_onset: float = 0.75
    max_shin_beyond_vertical_deg: float = 5.0
    max_layback_deg: float = 35.0
    min_layback_deg: float = 5.0
    min_handle_height_at_catch: float = -0.25  # en longueurs de tronc, vs genou
    max_handle_height_at_finish: float = 0.75  # 0 = hanche, 1 = epaule
    max_chain_angle_deg: float = 8.0
    max_knee_break_fraction: float = 0.25
    min_recovery_drive_ratio: float = 1.3
    max_trunk_drift_deg: float = 6.0  # derive de l'angle du tronc au catch (fatigue)
    max_view_shoulder_ratio: float = 0.5  # largeur d'epaules / tronc : au-dela, vue non profil
    view_frontal_ratio: float = 1.1  # au-dela : face ou dos, analyse minimale
    max_travel_loss_pct: float = 15.0  # raccourcissement du drive avec la fatigue
    max_rate_cv_pct: float = 12.0  # irregularite du rythme (CV de la periode des coups)


class Pm5Config(BaseModel):
    """Couplage PM5 : appariement video/PM5 et regles sur la courbe de force."""

    record_phase: str = "auto"  # auto | finish | catch : moment du coup ou le PM5 notifie
    max_lag: int = 12  # decalage max (en coups) cherche entre les deux sequences
    ambiguity_confidence: float = 0.5  # en dessous, l'appariement est declare ambigu
    min_period_std_s: float = 0.05  # variation de periode minimale pour apparier par sequence
    phase_tolerance_s: float = 0.3  # tolerance pour deduire la phase depuis les etats 100 ms
    peak_early_pos: float = 0.2  # pic de force avant 20 % de la course = catch explosif
    multipeak_prominence: float = 0.15  # fraction du pic pour compter un pic secondaire
    min_force_points: int = 8
    force_curve_points: int = 100  # reechantillonnage de la forme


class Config(BaseModel):
    calibration: CalibrationConfig = CalibrationConfig()
    signal: SignalConfig = SignalConfig()
    rep_segmentation: RepSegmentationConfig = RepSegmentationConfig()
    squat: SquatConfig = SquatConfig()
    pose: PoseConfig = PoseConfig()
    tracking: TrackingConfig = TrackingConfig()
    camera_motion: CameraMotionConfig = CameraMotionConfig()
    rowing: RowingConfig = RowingConfig()
    pm5: Pm5Config = Pm5Config()


def load_config(path: str | Path | None = None) -> Config:
    """Charge la config depuis un fichier YAML, ou renvoie les valeurs par defaut."""
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
