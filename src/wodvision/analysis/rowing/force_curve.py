"""Forme de la courbe de force du PM5 et regles associees.

L'axe des abscisses est la position de la poignee (echantillons regulierement
espaces) et l'unite de force n'est pas documentee par Concept2 : seules des
grandeurs sans unite, calculees sur la courbe normalisee, sont utilisees.

Repere Concept2 ("Using the Force Curve") : une courbe lisse traduit une force
appliquee progressivement ; un catch explosif donne un pic aigu decale vers le debut ;
plusieurs pics (type "chaine de montagnes") signalent une transition jambes / dos /
bras a travailler. Les seuils numeriques sont des hypotheses a calibrer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from wodvision.analysis.rowing.rules import Finding
from wodvision.config import Pm5Config, RowingConfig


@dataclass(slots=True)
class ForceFeatures:
    peak_pos: float  # 0..1, position du pic le long du drive
    peak_to_mean: float  # pic / moyenne : > 1,6 = courbe pointue, ~1,4 = bien arrondie
    n_peaks: int  # pics significatifs (prominence >= fraction du pic)
    front_load: float  # part de l'impulsion dans le premier tiers du drive


def normalized_curve(curve: np.ndarray, n_points: int) -> np.ndarray | None:
    """Courbe rechantillonnee sur n_points, sans les zeros de remplissage de fin."""
    c = np.asarray(curve, dtype=float)
    nz = np.flatnonzero(c > 0)
    if len(nz) < 3:
        return None
    c = c[nz[0] : nz[-1] + 1]
    x_old = np.linspace(0.0, 1.0, len(c))
    return np.interp(np.linspace(0.0, 1.0, n_points), x_old, c)


def curve_features(curve: np.ndarray | None, cfg: Pm5Config) -> ForceFeatures | None:
    if curve is None or len(curve) < cfg.min_force_points:
        return None
    c = normalized_curve(curve, cfg.force_curve_points)
    if c is None or c.max() <= 0:
        return None
    smooth = np.convolve(c, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, prominence=cfg.multipeak_prominence * smooth.max())
    n = len(c)
    return ForceFeatures(
        peak_pos=float(np.argmax(smooth) / (n - 1)),
        peak_to_mean=float(smooth.max() / smooth.mean()),
        n_peaks=int(max(1, len(peaks))),
        front_load=float(c[: n // 3].sum() / c.sum()),
    )


def evaluate_force_rules(
    features: list[tuple[int, ForceFeatures]], pm5: Pm5Config, rowing: RowingConfig
) -> list[Finding]:
    """Constats sur la courbe de force ; `features` = (indice de coup video, formes)."""
    if not features:
        return []
    findings: list[Finding] = []
    rules = [
        ("force_peak_early", "Pic de force trop precoce (catch explosif)", "force_peak_pos", "<",
         pm5.peak_early_pos, lambda f: f.peak_pos,
         "Construis la force progressivement des le catch : accroche l'eau sans a-coup, "
         "puis accelere en poussant sur les jambes."),
        ("force_multipeak", "Courbe de force a plusieurs pics (transition saccadee)",
         "force_n_peaks", ">", 1.5, lambda f: float(f.n_peaks),
         "Fais enchainer jambes, dos puis bras sans a-coup : l'objectif est une courbe lisse "
         "et arrondie."),
    ]
    for code, title, metric, cmp, threshold, getter, advice in rules:
        values = [(i, getter(f)) for i, f in features]
        flagged = [i for i, v in values if (v < threshold if cmp == "<" else v > threshold)]
        if len(flagged) / len(values) < rowing.flagged_stroke_fraction:
            continue
        findings.append(Finding(
            code=code, title=title, metric=metric, comparator=cmp, threshold=threshold,
            median_value=float(np.median([v for _, v in values])), n_flagged=len(flagged),
            n_valid=len(values), advice=advice, stroke_indices=flagged,
        ))
    return findings
