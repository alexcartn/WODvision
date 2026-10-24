"""Metriques par coup, calculees dans le repere machine (voir `frame.py`).

Les angles articulaires sont invariants au repere ; les angles "par rapport a
la verticale" utilisent la perpendiculaire du rail, donc independants du
roulis de la camera. Les hauteurs sont normalisees par la longueur du tronc
(echelle pixel -> unite corporelle) : aucune calibration metrique requise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from wodvision.analysis.rowing.frame import MachineFrame, signed_lean_deg
from wodvision.analysis.rowing.series import RowSeries
from wodvision.analysis.rowing.strokes import Stroke
from wodvision.config import RowingConfig

NAN = float("nan")


@dataclass(slots=True)
class StrokeMetrics:
    index: int
    t_catch: float
    t_finish: float
    t_next_catch: float
    stroke_rate_spm: float
    drive_time_s: float
    recovery_time_s: float
    recovery_drive_ratio: float
    target_recovery_ratio: float = NAN  # repere de ratio pour la cadence de ce coup
    recovery_ratio_vs_target: float = NAN  # ratio mesure / repere (1 = conforme)
    knee_angle_catch: float = NAN
    knee_angle_finish: float = NAN
    hip_angle_catch: float = NAN
    hip_angle_finish: float = NAN
    elbow_angle_catch: float = NAN
    elbow_angle_finish: float = NAN
    trunk_lean_catch: float = NAN  # deg, + = vers l'arriere (layback)
    trunk_lean_finish: float = NAN
    shin_angle_catch: float = NAN  # deg, - = tibia au-dela de la verticale (vers le volant)
    leg_fraction_at_trunk_onset: float = NAN  # 0..1, extension des jambes quand le dos s'ouvre
    leg_fraction_at_arm_onset: float = NAN
    handle_height_at_catch: float = NAN  # longueurs de tronc, 0 = genou
    handle_height_at_finish: float = NAN  # 0 = hanche, 1 = epaule
    chain_angle_deg: float = NAN  # pente de la poignee vs rail pendant le drive, + = monte
    knee_break_fraction: float = NAN  # flexion deja faite quand les mains passent les genoux
    handle_travel: float = NAN  # amplitude de la poignee le long du rail, en longueurs de tronc
    low_quality: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle au sommet b (degres), vectorise sur (m, 2)."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cos = np.einsum("...i,...i->...", ba, bc) / np.where(denom < 1e-9, np.nan, denom)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def target_recovery_ratio(rate_spm: float, table: list[list[float]]) -> float:
    """Repere recuperation / drive pour une cadence, par interpolation (borne aux extremites)."""
    rates, ratios = zip(*sorted((float(r), float(v)) for r, v in table), strict=True)
    return float(np.interp(rate_spm, rates, ratios))


def knee_angle_series(series: RowSeries) -> np.ndarray:
    """Angle du genou (hanche-genou-cheville, degres) sur la grille des series filtrees."""
    p = series.pts
    return _angle(p["hip"], p["knee"], p["ankle"])


def _fraction(x: np.ndarray, x0: float, x1: float, min_range: float) -> np.ndarray | None:
    """Progression normalisee (0 -> 1) de x entre x0 et x1, None si l'excursion est trop faible."""
    if not np.isfinite(x0) or not np.isfinite(x1) or abs(x1 - x0) < min_range:
        return None
    return (x - x0) / (x1 - x0)


def _first_crossing(f: np.ndarray, threshold: float) -> int | None:
    idx = np.flatnonzero(f >= threshold)
    return int(idx[0]) if len(idx) else None


def compute_stroke_metrics(
    series: RowSeries, frame: MachineFrame, strokes: list[Stroke], cfg: RowingConfig
) -> list[StrokeMetrics]:
    """Calcule les metriques de chaque coup detecte."""
    t = series.t_s
    m = {name: frame.to_machine(p) for name, p in series.pts.items()}
    torso = float(np.nanmedian(np.linalg.norm(m["shoulder"] - m["hip"], axis=1)))

    knee_angle = _angle(m["hip"], m["knee"], m["ankle"])
    hip_angle = _angle(m["shoulder"], m["hip"], m["knee"])
    elbow_angle = _angle(m["shoulder"], m["elbow"], m["wrist"])
    trunk_lean = signed_lean_deg(m["hip"], m["shoulder"])
    shin_angle = signed_lean_deg(m["ankle"], m["knee"])

    out: list[StrokeMetrics] = []
    for stroke in strokes:
        ic = int(np.argmin(np.abs(t - stroke.t_catch)))
        i_f = int(np.argmin(np.abs(t - stroke.t_finish)))
        i_n = int(np.argmin(np.abs(t - stroke.t_next_catch)))
        drive = slice(ic, i_f + 1)
        recovery = slice(i_f, i_n + 1)

        sm = StrokeMetrics(
            index=stroke.index,
            t_catch=stroke.t_catch,
            t_finish=stroke.t_finish,
            t_next_catch=stroke.t_next_catch,
            stroke_rate_spm=stroke.rate_spm,
            drive_time_s=stroke.drive_time,
            recovery_time_s=stroke.recovery_time,
            recovery_drive_ratio=stroke.recovery_time / stroke.drive_time,
            target_recovery_ratio=target_recovery_ratio(stroke.rate_spm, cfg.ratio_targets),
            recovery_ratio_vs_target=(stroke.recovery_time / stroke.drive_time)
            / target_recovery_ratio(stroke.rate_spm, cfg.ratio_targets),
            knee_angle_catch=float(knee_angle[ic]),
            knee_angle_finish=float(knee_angle[i_f]),
            hip_angle_catch=float(hip_angle[ic]),
            hip_angle_finish=float(hip_angle[i_f]),
            elbow_angle_catch=float(elbow_angle[ic]),
            elbow_angle_finish=float(elbow_angle[i_f]),
            trunk_lean_catch=float(trunk_lean[ic]),
            trunk_lean_finish=float(trunk_lean[i_f]),
            shin_angle_catch=float(shin_angle[ic]),
            low_quality=bool(series.missing[ic : i_n + 1].mean() > 0.2),
        )

        # Sequence du drive : progression de chaque segment entre catch et finish.
        f_leg = _fraction(knee_angle[drive], sm.knee_angle_catch, sm.knee_angle_finish,
                          cfg.min_leg_range_deg)
        f_trunk = _fraction(trunk_lean[drive], sm.trunk_lean_catch, sm.trunk_lean_finish,
                            cfg.min_trunk_range_deg)
        f_arm = _fraction(elbow_angle[drive], sm.elbow_angle_catch, sm.elbow_angle_finish,
                          cfg.min_arm_range_deg)
        if f_leg is not None:
            onset = cfg.phase_onset_fraction
            k = _first_crossing(f_trunk, onset) if f_trunk is not None else None
            if k is not None:
                sm.leg_fraction_at_trunk_onset = float(np.clip(f_leg[k], 0.0, 1.0))
            k = _first_crossing(f_arm, onset) if f_arm is not None else None
            if k is not None:
                sm.leg_fraction_at_arm_onset = float(np.clip(f_leg[k], 0.0, 1.0))

        # Hauteur de poignee (poignet = proxy), en longueurs de tronc.
        if torso > 0:
            sm.handle_travel = float(abs(m["wrist"][i_f, 0] - m["wrist"][ic, 0]) / torso)
            sm.handle_height_at_catch = float((m["wrist"][ic, 1] - m["knee"][ic, 1]) / torso)
        span = m["shoulder"][i_f, 1] - m["hip"][i_f, 1]
        if abs(span) > 1e-6:
            sm.handle_height_at_finish = float((m["wrist"][i_f, 1] - m["hip"][i_f, 1]) / span)

        # Alignement chaine/rail : pente de la trajectoire du poignet pendant le drive.
        s_w, h_w = m["wrist"][drive, 0], m["wrist"][drive, 1]
        if np.ptp(s_w) > 0.3 * torso:
            slope = np.polyfit(s_w, h_w, 1)[0]
            sm.chain_angle_deg = float(np.degrees(np.arctan(slope)))

        # Genoux qui plient avant que les mains passent les genoux (recuperation).
        gone = m["wrist"][recovery, 0] <= m["knee"][recovery, 0]
        f_knee_rec = _fraction(knee_angle[recovery], sm.knee_angle_finish,
                               float(knee_angle[i_n]), cfg.min_leg_range_deg)
        if f_knee_rec is not None and gone.any():
            sm.knee_break_fraction = float(np.clip(f_knee_rec[int(np.argmax(gone))], 0.0, 1.0))

        out.append(sm)
    return out
