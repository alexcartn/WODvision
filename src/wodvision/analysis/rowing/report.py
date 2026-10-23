"""Analyse complete a partir du cache d'inference : keypoints -> coups -> constats."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wodvision.analysis.angles import pick_visible_side
from wodvision.analysis.rowing.frame import MachineFrame, estimate_rail_axis
from wodvision.analysis.rowing.metrics import StrokeMetrics, compute_stroke_metrics
from wodvision.analysis.rowing.rules import (
    TIMING_ONLY_CODES,
    Finding,
    evaluate_rules,
    trunk_drift,
)
from wodvision.analysis.rowing.series import RowSeries, prepare_series
from wodvision.analysis.rowing.strokes import Stroke, detect_strokes
from wodvision.config import Config


@dataclass(slots=True)
class RowingReport:
    side: str
    n_strokes: int
    stroke_rate_spm: float  # mediane
    drive_time_s: float  # mediane
    recovery_drive_ratio: float  # mediane
    trunk_drift_deg: float
    camera_motion_px: float  # amplitude du mouvement de camera estime (0 = fixe)
    n_masked_findings: int = 0  # constats angulaires masques (vue non profil)
    view_shoulder_ratio: float = float("nan")  # ~0.1-0.3 en profil, ~1 de face ou de dos
    warnings: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    strokes: list[StrokeMetrics] = field(default_factory=list)
    series: RowSeries | None = None
    frame: MachineFrame | None = None
    stroke_events: list[Stroke] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "side": self.side, "n_strokes": self.n_strokes,
            "stroke_rate_spm": self.stroke_rate_spm, "drive_time_s": self.drive_time_s,
            "recovery_drive_ratio": self.recovery_drive_ratio,
            "trunk_drift_deg": self.trunk_drift_deg,
            "camera_motion_px": self.camera_motion_px,
            "view_shoulder_ratio": self.view_shoulder_ratio,
            "n_masked_findings": self.n_masked_findings, "warnings": list(self.warnings),
        }


def shoulder_width_ratio(xy: np.ndarray, score: np.ndarray, min_conf: float) -> float:
    """Mediane de |epaule G - epaule D| / longueur du tronc : indicateur de vue de profil."""
    ok = np.all(score[:, [5, 6, 11, 12]] > min_conf, axis=1)
    if ok.sum() < 10:
        return float("nan")
    width = np.linalg.norm(xy[ok, 5] - xy[ok, 6], axis=1)
    torso = np.linalg.norm((xy[ok, 5] + xy[ok, 6]) / 2 - (xy[ok, 11] + xy[ok, 12]) / 2, axis=1)
    return float(np.median(width / np.maximum(torso, 1e-6)))


def analyze_keypoints(
    t_obs_s: np.ndarray,
    xy_stab: np.ndarray,
    score: np.ndarray,
    cfg: Config,
    camera_motion_px: float = 0.0,
    side: str | None = None,
) -> RowingReport:
    """Chaine d'analyse pure (aucun modele) : utilisable depuis le cache.

    Args:
        t_obs_s: PTS des frames analysees (n,).
        xy_stab: keypoints COCO-17 stabilises (n, 17, 2).
        score: confiances (n, 17).
        camera_motion_px: amplitude du mouvement de camera (information).
        side: 'left' | 'right', ou None pour choisir le meilleur score moyen.
    """
    side = side or pick_visible_side(score)
    warnings: list[str] = []
    series = prepare_series(t_obs_s, xy_stab, score, side, cfg)

    frame = MachineFrame.from_axis(estimate_rail_axis(series.pts["hip"]))
    handle_s = frame.to_machine(series.pts["wrist"])[:, 0]
    r = cfg.rowing
    strokes, sign = detect_strokes(
        series.t_s, handle_s, r.stroke_prominence_fraction, r.min_stroke_period_s,
        r.min_stroke_rate_spm, r.max_stroke_rate_spm,
    )
    if sign == -1:
        frame = frame.flipped()
    if len(strokes) < 3:
        warnings.append(
            f"Seulement {len(strokes)} coup(s) complet(s) detecte(s) : analyse peu fiable. "
            "Verifier la vue de profil, la visibilite du rameur et la duree de la video."
        )

    ratio = shoulder_width_ratio(xy_stab, score, cfg.signal.min_keypoint_confidence)
    not_profile = ratio > r.max_view_shoulder_ratio

    metrics = compute_stroke_metrics(series, frame, strokes, r) if strokes else []
    findings = evaluate_rules(metrics, r)
    drift, drift_finding = trunk_drift(metrics, r)
    if drift_finding:
        findings.append(drift_finding)
    n_masked = 0
    if not_profile:
        kept = [f for f in findings if f.code in TIMING_ONLY_CODES]
        n_masked = len(findings) - len(kept)
        findings = kept
        warnings.append(
            f"La vue n'est pas un profil (largeur d'epaules / tronc = {ratio:.2f}, seuil "
            f"{r.max_view_shoulder_ratio:g}) : les mesures angulaires sont biaisees, "
            f"{n_masked} constat(s) de technique masque(s). Cadence et ratio drive/recuperation "
            "restent valables. Filmer perpendiculairement au rail, a hauteur de hanche."
        )
    if series.missing.mean() > 0.2:
        warnings.append("Plus de 20 % du signal est interpole (rameur peu visible).")

    def med(attr: str) -> float:
        vals = [getattr(m, attr) for m in metrics]
        return float(np.median(vals)) if vals else float("nan")

    return RowingReport(
        side=side, n_strokes=len(metrics),
        stroke_rate_spm=med("stroke_rate_spm"), drive_time_s=med("drive_time_s"),
        recovery_drive_ratio=med("recovery_drive_ratio"), trunk_drift_deg=drift,
        camera_motion_px=camera_motion_px, view_shoulder_ratio=ratio, n_masked_findings=n_masked,
        warnings=warnings, findings=findings,
        strokes=metrics, series=series, frame=frame, stroke_events=strokes,
    )
