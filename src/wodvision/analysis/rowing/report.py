"""Analyse complete a partir du cache d'inference : keypoints -> coups -> constats."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wodvision.analysis.angles import pick_visible_side
from wodvision.analysis.rowing.frame import MachineFrame, estimate_rail_axis
from wodvision.analysis.rowing.metrics import (
    StrokeMetrics,
    compute_stroke_metrics,
    knee_angle_series,
)
from wodvision.analysis.rowing.rules import (
    INDICATIVE_CODES,
    TIMING_CODES,
    Finding,
    evaluate_rules,
    rhythm_cv,
    travel_drift,
    trunk_drift,
)
from wodvision.analysis.rowing.series import RowSeries, prepare_series
from wodvision.analysis.rowing.strokes import Stroke, detect_strokes_multi
from wodvision.config import Config

if TYPE_CHECKING:
    from wodvision.analysis.rowing.pm5_sync import Pm5Analysis

MODE_LABELS = {
    "full": "Analyse complete (vue de profil)",
    "degraded": "Analyse degradee (vue oblique)",
    "minimal": "Analyse minimale (vue de face ou de dos)",
}


@dataclass(slots=True)
class RowingReport:
    side: str
    n_strokes: int
    stroke_rate_spm: float  # mediane
    drive_time_s: float  # mediane
    recovery_drive_ratio: float  # mediane
    trunk_drift_deg: float
    camera_motion_px: float  # amplitude du mouvement de camera estime (0 = fixe)
    mode: str = "full"  # full | degraded | minimal, selon la vue
    signal_used: str = "wrist"  # signal de detection des coups
    rate_cv_pct: float = float("nan")
    travel_change_pct: float = float("nan")
    n_masked_findings: int = 0  # constats angulaires masques (vue non profil)
    view_shoulder_ratio: float = float("nan")  # ~0.1-0.3 en profil, ~1 de face ou de dos
    warnings: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    strokes: list[StrokeMetrics] = field(default_factory=list)
    series: RowSeries | None = None
    frame: MachineFrame | None = None
    stroke_events: list[Stroke] = field(default_factory=list)
    pm5: Pm5Analysis | None = None  # couplage PM5 (optionnel)
    stabilized: bool = True  # False si l'analyse a ete faite sur les keypoints bruts (repli)

    @property
    def mode_label(self) -> str:
        return MODE_LABELS[self.mode]

    def summary_dict(self) -> dict:
        return {
            "mode": self.mode, "mode_label": self.mode_label, "side": self.side,
            "signal_used": self.signal_used, "n_strokes": self.n_strokes,
            "stroke_rate_spm": self.stroke_rate_spm, "drive_time_s": self.drive_time_s,
            "recovery_drive_ratio": self.recovery_drive_ratio,
            "rate_cv_pct": self.rate_cv_pct, "travel_change_pct": self.travel_change_pct,
            "trunk_drift_deg": self.trunk_drift_deg,
            "camera_motion_px": self.camera_motion_px,
            "view_shoulder_ratio": self.view_shoulder_ratio,
            "n_masked_findings": self.n_masked_findings, "warnings": list(self.warnings),
            **({"pm5": self.pm5.summary_dict()} if self.pm5 else {}),
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
    r = cfg.rowing
    series = prepare_series(t_obs_s, xy_stab, score, side, cfg)

    ratio = shoulder_width_ratio(xy_stab, score, cfg.signal.min_keypoint_confidence)
    mode = "full"
    if ratio > r.view_frontal_ratio:
        mode = "minimal"
    elif ratio > r.max_view_shoulder_ratio:
        mode = "degraded"

    up_hint = np.median(series.pts["shoulder"] - series.pts["hip"], axis=0)
    frame = MachineFrame.from_axis(estimate_rail_axis(series.pts["hip"]), up_hint)
    signals = {
        name: frame.to_machine(series.pts[name])[:, 0] for name in ("wrist", "hip", "shoulder")
    }
    signal_used, strokes, sign = detect_strokes_multi(
        series.t_s, signals, r.stroke_prominence_fraction, r.min_stroke_period_s,
        r.min_stroke_rate_spm, r.max_stroke_rate_spm,
        knee_angle=knee_angle_series(series), min_knee_diff_deg=r.min_knee_orientation_diff_deg,
    )
    if sign == -1:
        frame = frame.flipped()
    if len(strokes) < 3:
        warnings.append(
            f"Seulement {len(strokes)} coup(s) complet(s) detecte(s) : analyse peu fiable. "
            "Verifier la visibilite du rameur, la vue (profil) et la duree de la video."
        )

    metrics = compute_stroke_metrics(series, frame, strokes, r) if strokes else []
    findings = evaluate_rules(metrics, r)
    drift, drift_finding = trunk_drift(metrics, r)
    if drift_finding:
        findings.append(drift_finding)
    travel_change, travel_finding = travel_drift(metrics, r)
    if travel_finding:
        findings.append(travel_finding)
    cv, cv_finding = rhythm_cv(metrics, r)
    if cv_finding:
        findings.append(cv_finding)

    n_masked = 0
    if mode != "full":
        allowed = TIMING_CODES if mode == "minimal" else TIMING_CODES | INDICATIVE_CODES
        kept = [f for f in findings if f.code in allowed]
        for f in kept:
            if f.code in INDICATIVE_CODES:
                f.reliability = "indicative"
        n_masked = len(findings) - len(kept)
        findings = kept
        if mode == "degraded":
            warnings.append(
                f"{MODE_LABELS[mode]} : la vue n'est pas un profil (largeur d'epaules / tronc = "
                f"{ratio:.2f}, seuil {r.max_view_shoulder_ratio:g}). Les angles et hauteurs "
                f"absolus sont deformes : {n_masked} constat(s) masque(s). Restent fiables : "
                "cadence, ratio drive/recuperation, regularite. Restent indicatifs : sequence "
                "jambes/dos/bras et evolution de l'amplitude. Pour une analyse complete : filmer "
                "perpendiculairement au rail, camera a hauteur de hanche, rameur entier visible."
            )
        else:
            warnings.append(
                f"{MODE_LABELS[mode]} (largeur d'epaules / tronc = {ratio:.2f}, seuil "
                f"{r.view_frontal_ratio:g}) : seuls le comptage des coups, la cadence et le "
                f"rythme sont exploitables ; {n_masked} constat(s) de technique masque(s). "
                "Filmer de profil pour une analyse complete."
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
        camera_motion_px=camera_motion_px, mode=mode, signal_used=signal_used or "wrist",
        rate_cv_pct=cv, travel_change_pct=travel_change, view_shoulder_ratio=ratio,
        n_masked_findings=n_masked, warnings=warnings, findings=findings,
        strokes=metrics, series=series, frame=frame, stroke_events=strokes,
    )
