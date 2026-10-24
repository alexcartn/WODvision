"""Synthese pedagogique : criteres (pastilles), points forts, priorites et conseils.

Le contenu (libelles, conseils, exercices) vit dans `configs/conseils.yaml`, un fichier que
l'on peut relire et corriger sans toucher au code. Ce module le charge, le valide et
transforme un `RowingReport` en synthese lisible par un amateur : un statut par critere,
un point fort et au plus deux priorites, chacune avec son conseil.

Statuts : ok (vert), warn (orange), bad (rouge), na (non mesure avec cette vue).
Un constat rouge exige une faute fiable sur au moins 60 % des coups ; un constat indicatif
(vue non profil) ne depasse jamais l'orange.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

BAD_FRACTION = 0.6
STATUS_EMOJI = {"ok": "🟢", "warn": "🟠", "bad": "🔴", "na": "⚪"}


class Criterion(BaseModel):
    label: str
    strength: str
    codes: list[str]
    available_in: list[str]
    needs_pm5: bool = False
    min_duration_s: float = 0.0  # en dessous, le critere n'est pas evaluable (ex. la fatigue)


class Tip(BaseModel):
    title: str
    why: str
    feeling: str
    drill: str
    status: str = "a_valider"


class GlossaryEntry(BaseModel):
    term: str
    text: str


class CoachingContent(BaseModel):
    criteria: dict[str, Criterion]
    tips: dict[str, Tip]
    weights: dict[str, float] = {}
    glossary: list[GlossaryEntry] = []
    filming: list[str] = []


def load_coaching(path: str | Path) -> CoachingContent:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CoachingContent.model_validate(data)


@dataclass(slots=True)
class CriterionStatus:
    key: str
    label: str
    status: str  # ok | warn | bad | na
    note: str
    indicative: bool = False

    @property
    def emoji(self) -> str:
        return STATUS_EMOJI[self.status]


@dataclass(slots=True)
class Priority:
    code: str
    criterion: str
    severity: float  # part des coups concernes (0..1)
    reliability: str
    tip: Tip
    metric_line: str  # mesure brute, pour le mode Coach


@dataclass(slots=True)
class CoachingSummary:
    criteria: list[CriterionStatus]
    strengths: list[str]
    priorities: list[Priority]
    notes: list[str] = field(default_factory=list)


def _criterion_status(key: str, crit: Criterion, report) -> CriterionStatus:
    if report.mode not in crit.available_in:
        why = {"full": "", "degraded": "vue oblique", "minimal": "vue de face ou de dos"}
        return CriterionStatus(key, crit.label, "na",
                               f"Non mesuré avec cette vue ({why.get(report.mode, '')}).")
    duration = float(report.series.t_s[-1] - report.series.t_s[0]) if report.series else 0.0
    if duration < crit.min_duration_s:
        return CriterionStatus(key, crit.label, "na",
                               f"Vidéo trop courte pour en juger (moins de "
                               f"{crit.min_duration_s:.0f} s).")
    if crit.needs_pm5 and not (report.pm5 is not None and report.pm5.n_matched > 0
                               and report.pm5.features):
        return CriterionStatus(key, crit.label, "na", "Pas de données PM5 avec courbe de force.")
    found = [f for f in report.findings if f.code in crit.codes]
    indicative = report.mode == "degraded" and key in {"sequence", "endurance"}
    if not found:
        return CriterionStatus(key, crit.label, "ok", "Rien à signaler.", indicative)
    worst = max(f.n_flagged / max(f.n_valid, 1) for f in found)
    reliable = any(f.reliability == "reliable" for f in found)
    status = "bad" if (reliable and worst >= BAD_FRACTION) else "warn"
    return CriterionStatus(key, crit.label, status, " ; ".join(f.title for f in found),
                           any(f.reliability == "indicative" for f in found) or indicative)


def summarize(report, content: CoachingContent, max_priorities: int = 2) -> CoachingSummary:
    """Synthese pedagogique d'un rapport."""
    criteria = [_criterion_status(k, c, report) for k, c in content.criteria.items()]
    code_to_criterion = {code: k for k, c in content.criteria.items() for code in c.codes}
    not_evaluable = {c.key for c in criteria if c.status == "na"}

    ranked = []
    for f in report.findings:
        if f.code not in content.tips or code_to_criterion.get(f.code) in not_evaluable:
            continue
        severity = f.n_flagged / max(f.n_valid, 1)
        weight = content.weights.get(f.code, 1.0)
        reliable = f.reliability == "reliable"
        ranked.append((reliable, severity * weight, f))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)

    priorities, seen = [], set()
    for _reliable, _score, f in ranked:  # une priorite par critere d'abord, pour varier
        crit = code_to_criterion.get(f.code, "")
        if crit in seen:
            continue
        seen.add(crit)
        priorities.append(Priority(
            code=f.code, criterion=crit, severity=f.n_flagged / max(f.n_valid, 1),
            reliability=f.reliability, tip=content.tips[f.code],
            metric_line=f"{f.metric} médian = {f.median_value:.2f} (seuil {f.comparator} "
                        f"{f.threshold:g}), {f.n_flagged}/{f.n_valid} coups concernés",
        ))
        if len(priorities) == max_priorities:
            break

    strengths = [content.criteria[c.key].strength for c in criteria
                 if c.status == "ok" and not c.indicative]
    notes = []
    if report.mode != "full":
        notes.append("Vue non profil : certaines mesures ne sont pas disponibles. Filme de profil "
                     "pour une analyse complète.")
    return CoachingSummary(criteria, strengths, priorities, notes)
