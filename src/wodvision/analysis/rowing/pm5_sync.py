"""Couplage video / PM5 : appariement des coups, decalage temporel, validation, force.

Principes :
- Appariement par sequence. Les coups video et PM5 sont mis en correspondance par un
  decalage entier `lag` (en coups) qui minimise l'ecart des periodes et des durees de drive
  sur les coups apparies. Si le rythme est trop regulier, plusieurs decalages se valent :
  l'appariement est declare ambigu (avertissement) et le decalage de plus petit |offset|
  est retenu. `lag` peut etre impose.
- Decalage temporel. Le PM5 notifie un coup a un instant fixe de son cycle (`phase`) :
  fin du drive ("finish") ou debut du suivant ("catch"). La phase est deduite des
  echantillons d'etat a 100 ms si l'enregistrement les contient, sinon "finish" par defaut
  (le PM5 ne connait peak force et drive length qu'a la fin du drive). Le decalage
  (pm = video + offset) est la mediane des ecarts ; son ecart-type mesure la gigue.
- Le drive du PM5 est defini par l'acceleration du volant : il commence un peu apres le catch
  video et finit un peu avant le finish. Un biais systematique sur le temps de drive est
  normal ; c'est sa dispersion qui valide la detection video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wodvision.analysis.rowing.force_curve import (
    ForceFeatures,
    curve_features,
    evaluate_force_rules,
)
from wodvision.analysis.rowing.rules import Finding
from wodvision.config import Config
from wodvision.pm5 import spec
from wodvision.pm5.session import Pm5Session


@dataclass(slots=True)
class Pm5Analysis:
    lag: int  # indice PM5 = indice video + lag
    ambiguous: bool
    confidence: float
    phase: str  # finish | catch
    phase_source: str  # status | default | forced
    offset_s: float  # temps PM5 = temps video + offset
    offset_jitter_s: float
    n_matched: int
    validation: dict[str, float] = field(default_factory=dict)
    rows: dict[int, dict] = field(default_factory=dict)  # indice video -> mesures PM5 + force
    features: dict[int, ForceFeatures] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    session: Pm5Session | None = None

    def summary_dict(self) -> dict:
        return {
            "lag": self.lag, "ambiguous": self.ambiguous, "confidence": self.confidence,
            "phase": self.phase, "phase_source": self.phase_source, "offset_s": self.offset_s,
            "offset_jitter_s": self.offset_jitter_s, "n_matched": self.n_matched,
            "n_pm5_strokes": len(self.session.strokes) if self.session else 0,
            "drag_factor": self.session.drag_factor if self.session else float("nan"),
            **{f"validation_{k}": v for k, v in self.validation.items()},
        }


def infer_record_phase(session: Pm5Session, tolerance_s: float) -> str | None:
    """Phase de notification des coups, deduite des transitions d'etat (si enregistrees)."""
    if len(session.status) < 10:
        return None
    t = np.array([s.t_pm_s for s in session.status])
    state = np.array([s.stroke_state for s in session.status])
    driving = state == spec.StrokeState.DRIVING
    catches = t[1:][driving[1:] & ~driving[:-1]]
    finishes = t[1:][~driving[1:] & driving[:-1]]
    if len(catches) < 3 or len(finishes) < 3:
        return None
    tau = np.array([s.t_pm_s for s in session.strokes])

    def dist(events: np.ndarray) -> float:
        return float(np.median([np.min(np.abs(events - x)) for x in tau]))

    d_finish, d_catch = dist(finishes), dist(catches)
    if min(d_finish, d_catch) > tolerance_s:
        return None
    return "finish" if d_finish <= d_catch else "catch"


def _video_arrays(report) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ev = report.stroke_events
    period = np.array([s.t_next_catch - s.t_catch for s in ev])
    drive = np.array([s.drive_time for s in ev])
    event_t = np.array([s.t_next_catch for s in ev])  # utilise si phase == catch
    finish_t = np.array([s.t_finish for s in ev])
    return period, drive, finish_t, event_t


def _lag_score(v_per, v_drv, p_per, p_drv, lag: int) -> tuple[float, int]:
    """Ecart moyen des periodes et des durees de drive (biais retire) pour un decalage."""
    i0, i1 = max(0, -lag), min(len(v_per), len(p_per) - lag)
    if i1 - i0 < 4:
        return np.inf, 0
    sl_v, sl_p = slice(i0, i1), slice(i0 + lag, i1 + lag)
    d_per = np.abs(v_per[sl_v] - p_per[sl_p]).mean()
    diff = v_drv[sl_v] - p_drv[sl_p]
    d_drv = np.abs(diff - np.median(diff)).mean()
    return float(d_per + d_drv), i1 - i0


def couple_pm5(
    report, session: Pm5Session, cfg: Config, lag: int | None = None, phase: str | None = None
) -> Pm5Analysis:
    """Apparie les coups video et PM5, estime le decalage, valide et analyse la force."""
    c = cfg.pm5
    warnings: list[str] = []
    strokes = [s for s in session.strokes if np.isfinite(s.period_s)]
    if len(report.stroke_events) < 4 or len(strokes) < 4:
        return Pm5Analysis(0, True, 0.0, "finish", "default", float("nan"), float("nan"), 0,
                           warnings=["Trop peu de coups (video ou PM5) pour apparier."],
                           session=session)

    v_per, v_drv, finish_t, catch_next_t = _video_arrays(report)
    p_per = np.array([s.period_s for s in strokes])
    p_drv = np.array([s.drive_time_s for s in strokes])
    tau = np.array([s.t_pm_s for s in strokes])

    # Phase de notification.
    phase_source = "forced"
    chosen = phase or (c.record_phase if c.record_phase != "auto" else None)
    if chosen is None:
        inferred = infer_record_phase(session, c.phase_tolerance_s)
        chosen, phase_source = (inferred, "status") if inferred else ("finish", "default")
        if phase_source == "default":
            warnings.append(
                "Phase de notification du PM5 non verifiee (pas d'etats a 100 ms dans "
                "l'enregistrement) : 'fin du drive' suppose. Cela decale l'horloge d'au plus "
                "un drive, sans effet sur l'appariement des coups."
            )
    video_event = finish_t if chosen == "finish" else catch_next_t

    def offset_for(k: int) -> tuple[float, float, int]:
        i0, i1 = max(0, -k), min(len(v_per), len(tau) - k)
        diffs = tau[i0 + k : i1 + k] - video_event[i0:i1]
        return float(np.median(diffs)), float(np.std(diffs)), i1 - i0

    scores = {k: _lag_score(v_per, v_drv, p_per, p_drv, k)[0]
              for k in range(-c.max_lag, c.max_lag + 1)}
    ranked = sorted(scores, key=scores.get)
    finite = [v for v in scores.values() if np.isfinite(v)]
    ambiguous, confidence = False, 1.0
    if lag is None:
        best = ranked[0]
        # Confiance : le meilleur decalage se detache-t-il du niveau "typique" des autres ?
        # Un rythme trop regulier donne des scores tous voisins (bruit de mesure seulement).
        typical = float(np.median(finite))
        confidence = float(1.0 - scores[best] / typical) if typical > 0 else 1.0
        # Rythme trop uniforme (variation inferieure au bruit de mesure) : non informatif.
        if np.std(v_per) < c.min_period_std_s:
            confidence = min(confidence, 0.0)
        if confidence < c.ambiguity_confidence:
            ambiguous = True
            limit = scores[best] + 0.5 * (typical - scores[best])
            close = [k for k in ranked if scores[k] <= limit]
            lag = min(close, key=lambda k: abs(offset_for(k)[0]))
            warnings.append(
                f"Appariement ambigu (confiance {confidence:.2f}) : le rythme est trop regulier "
                f"pour determiner le decalage de coups ; {lag:+d} retenu (plus petit decalage "
                "d'horloge). Imposer --pm5-lag si les coups de depart sont connus."
            )
        else:
            lag = best
    offset, jitter, n_pairs = offset_for(lag)
    if n_pairs < 4:
        warnings.append("Moins de 4 coups apparies : validation peu fiable.")

    # Coups apparies et mesures.
    rows: dict[int, dict] = {}
    features: dict[int, ForceFeatures] = {}
    for i in range(len(report.strokes)):
        j = i + lag
        if not (0 <= j < len(strokes)):
            continue
        s = strokes[j]
        row = {
            "pm5_stroke_count": s.stroke_count, "pm5_t_s": s.t_pm_s,
            "pm5_drive_time_s": s.drive_time_s, "pm5_recovery_time_s": s.recovery_time_s,
            "pm5_drive_length_m": s.drive_length_m, "pm5_peak_force_lbf": s.peak_force_lbf,
            "pm5_avg_force_lbf": s.avg_force_lbf, "pm5_work_j": s.work_j,
            "pm5_power_w": s.power_w, "force_curve": s.force_curve,
        }
        f = curve_features(s.force_curve, c)
        if f is not None:
            features[i] = f
            row.update(force_peak_pos=f.peak_pos, force_peak_to_mean=f.peak_to_mean,
                       force_n_peaks=f.n_peaks, force_front_load=f.front_load)
        rows[i] = row

    validation = _validation(report, rows)
    findings = evaluate_force_rules(list(features.items()), c, cfg.rowing)
    if session.strokes and not session.has_force_curves:
        warnings.append("Pas de courbe de force dans l'enregistrement (PM5v1, ou notification "
                        "0x003D non souscrite) : regles de force non evaluees.")
    return Pm5Analysis(
        lag=lag, ambiguous=ambiguous, confidence=confidence, phase=chosen,
        phase_source=phase_source, offset_s=offset, offset_jitter_s=jitter,
        n_matched=len(rows), validation=validation, rows=rows, features=features,
        findings=findings, warnings=warnings, session=session,
    )


def _validation(report, rows: dict[int, dict]) -> dict[str, float]:
    if len(rows) < 4:
        return {}
    idx = sorted(rows)
    v_drive = np.array([report.strokes[i].drive_time_s for i in idx])
    p_drive = np.array([rows[i]["pm5_drive_time_s"] for i in idx])
    v_period = np.array([report.stroke_events[i].t_next_catch - report.stroke_events[i].t_catch
                         for i in idx])
    p_period = np.array([rows[i]["pm5_drive_time_s"] + rows[i]["pm5_recovery_time_s"]
                         for i in idx])
    out = {
        "period_mae_ms": float(np.mean(np.abs(v_period - p_period)) * 1000),
        "drive_bias_ms": float(np.mean(v_drive - p_drive) * 1000),
        "drive_std_ms": float(np.std(v_drive - p_drive) * 1000),
        "drive_corr": float(np.corrcoef(v_drive, p_drive)[0, 1]) if np.std(v_drive) > 0
        and np.std(p_drive) > 0 else float("nan"),
    }
    # Echelle implicite : longueur du tronc (m) = drive length PM5 / amplitude poignee (tronc).
    torso = np.array([rows[i]["pm5_drive_length_m"] / report.strokes[i].handle_travel
                      for i in idx if report.strokes[i].handle_travel > 0.05
                      and rows[i]["pm5_drive_length_m"] > 0])
    if len(torso) >= 4:
        out["implied_torso_m"] = float(np.median(torso))
        out["implied_torso_cv_pct"] = float(np.std(torso) / np.mean(torso) * 100)
    return out
