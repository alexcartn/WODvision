"""Regles explicites de technique : chaque constat cite metrique, valeur et seuil.

Un constat est leve quand une part significative des coups valides
(`flagged_stroke_fraction`) viole la regle, pour ne pas reagir a un coup isole.
Les seuils viennent de `RowingConfig` : ce sont des hypotheses de depart tirees
des consignes Concept2, a calibrer sur des videos annotees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wodvision.analysis.rowing.metrics import StrokeMetrics
from wodvision.config import RowingConfig


@dataclass(slots=True)
class Finding:
    code: str
    title: str
    metric: str
    comparator: str  # '<' : fautif si valeur < seuil ; '>' : fautif si valeur > seuil
    threshold: float
    median_value: float
    n_flagged: int
    n_valid: int
    advice: str
    stroke_indices: list[int] = field(default_factory=list)

    def explain(self) -> str:
        return (
            f"{self.title} : {self.metric} median = {self.median_value:.2f} "
            f"(seuil {self.comparator} {self.threshold:g}), "
            f"{self.n_flagged}/{self.n_valid} coups concernes. {self.advice}"
        )


@dataclass(slots=True)
class _Rule:
    code: str
    title: str
    metric: str
    comparator: str
    threshold_attr: str
    advice: str
    use_abs: bool = False
    threshold_sign: float = 1.0


RULES: list[_Rule] = [
    _Rule("trunk_early", "Dos qui s'ouvre trop tot", "leg_fraction_at_trunk_onset", "<",
          "min_leg_fraction_at_trunk_onset",
          "Pousse d'abord avec les jambes : le dos ne s'ouvre qu'une fois les jambes bien "
          "engagees, sans changer l'angle du buste au debut du drive."),
    _Rule("arms_early", "Bras qui tirent trop tot", "leg_fraction_at_arm_onset", "<",
          "min_leg_fraction_at_arm_onset",
          "Garde les bras tendus jusqu'a ce que jambes et dos aient fait l'essentiel du travail."),
    _Rule("shin_beyond_vertical", "Tibias au-dela de la verticale au catch", "shin_angle_catch",
          "<", "max_shin_beyond_vertical_deg",
          "Au catch, tibias verticaux (ou presque) : ne pas rouler plus loin vers le volant.",
          threshold_sign=-1.0),
    _Rule("layback_excess", "Recul du buste excessif au finish", "trunk_lean_finish", ">",
          "max_layback_deg",
          "Finish avec un leger recul, soutenu par les abdominaux, sans se jeter en arriere."),
    _Rule("layback_missing", "Pas assez de recul du buste au finish", "trunk_lean_finish", "<",
          "min_layback_deg",
          "Termine le drive avec un leger recul du buste pour finir la poussee des jambes."),
    _Rule("handle_drop", "Poignee qui plonge au catch", "handle_height_at_catch", "<",
          "min_handle_height_at_catch",
          "Apres avoir passe les genoux, laisse la poignee remonter legerement vers le catch."),
    _Rule("handle_high_finish", "Poignee trop haute au finish", "handle_height_at_finish", ">",
          "max_handle_height_at_finish",
          "Tire la poignee sous les cotes, pas vers la poitrine ou le menton."),
    _Rule("chain_misaligned", "Chaine non alignee avec le rail (tirage vers le haut)",
          "chain_angle_deg", ">", "max_chain_angle_deg",
          "La poignee doit suivre une ligne droite, parallele au rail, vers le volant : "
          "chaine horizontale, sans tirer vers le haut.", use_abs=True),
    _Rule("knee_break_early", "Genoux qui plient avant le passage des mains",
          "knee_break_fraction", ">", "max_knee_break_fraction",
          "A la recuperation : bras tendus, buste incline, puis seulement flexion des genoux."),
    _Rule("recovery_rushed", "Recuperation trop courte par rapport au drive",
          "recovery_drive_ratio", "<", "min_recovery_drive_ratio",
          "Vise une recuperation environ deux fois plus longue que le drive."),
]


# Constats qui ne dependent que du timing (valables meme si la vue n'est pas un profil).
TIMING_ONLY_CODES = {"recovery_rushed"}


def evaluate_rules(metrics: list[StrokeMetrics], cfg: RowingConfig) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        threshold = rule.threshold_sign * float(getattr(cfg, rule.threshold_attr))
        values = []
        for sm in metrics:
            v = getattr(sm, rule.metric)
            if not sm.low_quality and np.isfinite(v):
                values.append((sm.index, abs(v) if rule.use_abs else v))
        if not values:
            continue
        if rule.comparator == "<":
            flagged = [i for i, v in values if v < threshold]
        else:
            flagged = [i for i, v in values if v > threshold]
        if len(flagged) / len(values) < cfg.flagged_stroke_fraction:
            continue
        findings.append(
            Finding(
                code=rule.code, title=rule.title, metric=rule.metric,
                comparator=rule.comparator, threshold=threshold,
                median_value=float(np.median([v for _, v in values])),
                n_flagged=len(flagged), n_valid=len(values), advice=rule.advice,
                stroke_indices=flagged,
            )
        )
    return findings


def trunk_drift(metrics: list[StrokeMetrics], cfg: RowingConfig) -> tuple[float, Finding | None]:
    """Derive de l'inclinaison du tronc au catch (dernier tiers - premier tiers).

    Indicateur de fatigue (la flexion lombaire au catch augmente avec la fatigue
    d'apres la litterature). Inclinaison vers l'avant = valeur plus negative.
    """
    vals = [sm.trunk_lean_catch for sm in metrics if not sm.low_quality
            and np.isfinite(sm.trunk_lean_catch)]
    if len(vals) < 6:
        return float("nan"), None
    third = len(vals) // 3
    drift = float(np.median(vals[-third:]) - np.median(vals[:third]))
    if -drift <= cfg.max_trunk_drift_deg:
        return drift, None
    return drift, Finding(
        code="trunk_drift", title="Buste de plus en plus penche au catch (fatigue)",
        metric="trunk_lean_catch_drift_deg", comparator="<", threshold=-cfg.max_trunk_drift_deg,
        median_value=drift, n_flagged=len(vals), n_valid=len(vals),
        advice="Perte de position avec la fatigue : garde le buste haut au catch, "
               "reduis la cadence ou l'intensite si besoin.",
    )
