"""Mode fantome : posture de reference superposee a la posture reelle.

Le modele est cinematique et plan, dans le repere machine (s le long du rail, h vers le
haut). Il est mis a l'echelle des segments du rameur (cuisse, tibia, tronc, bras), ancre
sur sa cheville (le cale-pieds ne bouge pas) et a sa hauteur d'assise, puis synchronise
sur sa phase de coup : progression du drive (catch -> finish) ou de la recuperation.

Sequence de reference (consignes Concept2) : au drive jambes, puis dos, puis bras ;
a la recuperation bras, buste, puis jambes. Tibias verticaux au catch, jambes presque
tendues et leger recul au finish, poignee a hauteur de genoux avec une chaine horizontale.
Ce n'est pas un modele valide sur des rameurs d'elite : c'est un guide visuel dont les
parametres (`GhostConfig`) sont des hypotheses a calibrer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wodvision.analysis.rowing.frame import MachineFrame
from wodvision.analysis.rowing.series import RowSeries
from wodvision.config import GhostConfig

GHOST_EDGES = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
               ("shoulder", "elbow"), ("elbow", "wrist")]


@dataclass(slots=True)
class Anthropometry:
    thigh: float
    shin: float
    torso: float
    upper_arm: float
    forearm: float
    ankle: np.ndarray  # (s, h) absolu, repere machine
    seat_h: float  # hauteur de la hanche au-dessus de la cheville


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 3 * x**2 - 2 * x**3


def _circle_point(c0: np.ndarray, r0: float, c1: np.ndarray, r1: float, upper: bool) -> np.ndarray:
    """Intersection de deux cercles (genou ou coude), cote haut ou bas."""
    d = c1 - c0
    dist = float(np.linalg.norm(d))
    dist = min(max(dist, abs(r0 - r1) + 1e-6), r0 + r1 - 1e-6)
    a = (r0**2 - r1**2 + dist**2) / (2 * dist)
    h = np.sqrt(max(r0**2 - a**2, 0.0))
    base = c0 + a * d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    cands = [base + h * perp, base - h * perp]
    return max(cands, key=lambda p: p[1]) if upper else min(cands, key=lambda p: p[1])


def generic_anthropometry() -> Anthropometry:
    """Proportions moyennes (m) pour dessiner un schema quand aucun rameur n'est mesure."""
    return Anthropometry(thigh=0.45, shin=0.45, torso=0.50, upper_arm=0.30, forearm=0.30,
                         ankle=np.array([0.0, 0.15]), seat_h=0.12)


def estimate_anthropometry(series: RowSeries, frame: MachineFrame) -> Anthropometry | None:
    """Longueurs de segments (mediane) et ancrage de la cheville depuis les series filtrees."""
    p = series.pts

    def seg(a: str, b: str) -> float:
        return float(np.nanmedian(np.linalg.norm(p[a] - p[b], axis=1)))

    m = {k: frame.to_machine(v) for k, v in p.items()}
    ankle = np.nanmedian(m["ankle"], axis=0)
    seat_h = float(np.nanmedian(m["hip"][:, 1]) - ankle[1])
    anthro = Anthropometry(
        thigh=seg("hip", "knee"), shin=seg("knee", "ankle"), torso=seg("shoulder", "hip"),
        upper_arm=seg("shoulder", "elbow"), forearm=seg("elbow", "wrist"),
        ankle=ankle, seat_h=seat_h,
    )
    lengths = [anthro.thigh, anthro.shin, anthro.torso, anthro.upper_arm, anthro.forearm]
    if not all(np.isfinite(lengths)) or min(lengths) <= 0:
        return None
    if not 0.05 * anthro.torso < seat_h < 0.99 * (anthro.thigh + anthro.shin):
        return None
    return anthro


class GhostModel:
    """Posture de reference en fonction de la phase du coup."""

    def __init__(self, anthro: Anthropometry, cfg: GhostConfig) -> None:
        self.a, self.cfg = anthro, cfg
        a = anthro
        shin_c = np.radians(cfg.shin_catch_deg)
        self._knee_catch = a.shin * np.array([np.sin(shin_c), np.cos(shin_c)])
        dh = self._knee_catch[1] - a.seat_h
        self._hip_catch_s = self._knee_catch[0] + np.sqrt(max(a.thigh**2 - dh**2, 0.0))
        straight = cfg.leg_straight_fraction * (a.thigh + a.shin)
        self._hip_finish_s = max(np.sqrt(max(straight**2 - a.seat_h**2, 0.0)), self._hip_catch_s)
        # Chaine horizontale : la poignee ne depasse ni la hauteur des genoux au catch ni, au
        # finish, la hauteur "sous les cotes" (handle_finish_ratio) : la reference respecte
        # ainsi les regles appliquees au rameur (handle_drop, handle_high_finish).
        lean_f = np.radians(cfg.lean_finish_deg)
        shoulder_h_finish = a.seat_h + a.torso * np.cos(lean_f)
        rib_h = a.seat_h + cfg.handle_finish_ratio * (shoulder_h_finish - a.seat_h)
        self._handle_h = float(min(self._knee_catch[1], rib_h))

    def _progress(self, kind: str, x: float) -> tuple[float, float, float]:
        """(jambes, buste, bras) : 0 = catch, 1 = finish."""
        c = self.cfg
        if kind == "drive":
            return (_smooth(x / c.leg_end),
                    _smooth((x - c.trunk_onset) / (c.trunk_end - c.trunk_onset)),
                    _smooth((x - c.arm_onset) / (1.0 - c.arm_onset)))
        return (1 - _smooth((x - c.knee_break) / (1.0 - c.knee_break)),
                1 - _smooth((x - 0.2) / 0.4),
                1 - _smooth(x / 0.3))

    def pose(self, kind: str, x: float) -> dict[str, np.ndarray]:
        """Articulations (s, h) absolues ; `kind` 'drive' ou 'recovery', x dans [0, 1]."""
        a, c = self.a, self.cfg
        leg, trunk, arm = self._progress(kind, x)
        hip = np.array([self._hip_catch_s + (self._hip_finish_s - self._hip_catch_s) * leg,
                        a.seat_h])
        knee = _circle_point(np.zeros(2), a.shin, hip, a.thigh, upper=True)
        lean = np.radians(c.lean_catch_deg + (c.lean_finish_deg - c.lean_catch_deg) * trunk)
        shoulder = hip + a.torso * np.array([np.sin(lean), np.cos(lean)])
        reach = (a.upper_arm + a.forearm) * (0.99 - 0.5 * arm)
        dh = min(abs(shoulder[1] - self._handle_h), 0.999 * reach)
        wrist = np.array([shoulder[0] - np.sqrt(reach**2 - dh**2), self._handle_h])
        elbow = _circle_point(shoulder, a.upper_arm, wrist, a.forearm, upper=False)
        pts = {"ankle": np.zeros(2), "knee": knee, "hip": hip, "shoulder": shoulder,
               "elbow": elbow, "wrist": wrist}
        return {k: v + a.ankle for k, v in pts.items()}

    @staticmethod
    def trunk_lean_deg(pose: dict[str, np.ndarray]) -> float:
        d = pose["shoulder"] - pose["hip"]
        return float(np.degrees(np.arctan2(d[0], d[1])))


def phase_of(stroke_events, t: float) -> tuple[str, float] | None:
    """('drive', progression) ou ('recovery', progression) a l'instant t, None hors coups."""
    for s in stroke_events:
        if s.t_catch <= t < s.t_finish:
            return "drive", (t - s.t_catch) / (s.t_finish - s.t_catch)
        if s.t_finish <= t < s.t_next_catch:
            return "recovery", (t - s.t_finish) / (s.t_next_catch - s.t_finish)
    return None


def build_ghost(report, cfg: GhostConfig) -> GhostModel | None:
    """Modele fantome du rameur analyse, ou None (desactive, vue non profil, donnees faibles).

    En vue non profil les angles sont deformes : une posture de reference y serait trompeuse.
    """
    if not cfg.enabled or report.mode != "full" or report.series is None or report.frame is None:
        return None
    anthro = estimate_anthropometry(report.series, report.frame)
    return GhostModel(anthro, cfg) if anthro is not None else None


DEVIATION_JOINTS = ("knee", "hip", "shoulder", "wrist")


def joint_gaps(report, ghost: GhostModel, t: float) -> dict[str, float] | None:
    """Ecart (en longueurs de tronc) de chaque articulation au fantome a l'instant t."""
    ph = phase_of(report.stroke_events, t)
    if ph is None:
        return None
    series, frame = report.series, report.frame
    i = int(np.argmin(np.abs(series.t_s - t)))
    pose = ghost.pose(*ph)
    out = {}
    for name in ("knee", "hip", "shoulder", "elbow", "wrist"):
        actual = frame.to_machine(series.pts[name][i])
        out[name] = float(np.linalg.norm(actual - pose[name]) / ghost.a.torso)
    return out


def stroke_deviations(report, ghost: GhostModel) -> list[float]:
    """Ecart moyen au fantome de chaque coup (en longueurs de tronc) : sert a choisir le
    meilleur coup et le coup a ameliorer."""
    series = report.series
    out = []
    for s in report.stroke_events:
        sel = (series.t_s >= s.t_catch) & (series.t_s < s.t_next_catch)
        gaps = [joint_gaps(report, ghost, float(t)) for t in series.t_s[sel]]
        vals = [np.mean([g[j] for j in DEVIATION_JOINTS]) for g in gaps if g]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out
