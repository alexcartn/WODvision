"""Session PM5 : coups fusionnes (stroke data + puissance + courbe de force) et leur I/O.

Deux entrees :
- journal d'evenements bruts `.jsonl` ecrit par l'enregistreur (`pm5-record`) : une
  ligne d'en-tete, puis {"t": heure hote (s), "c": id caracteristique, "d": octets en hex} ;
- tableau de coups `.csv` (saisie manuelle ou export) avec les colonnes de `CSV_COLUMNS`.

Le journal brut est la source la plus fidele : le decodage se rejoue hors ligne.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wodvision.pm5 import spec
from wodvision.pm5.parsers import (
    ForceCurveAssembler,
    parse_additional_status_1,
    parse_additional_stroke_data,
    parse_general_status,
    parse_stroke_data,
    split_multiplexed,
)

EVENT_LOG_FORMAT = "wodvision-pm5-events"
CSV_COLUMNS = [
    "stroke_count", "elapsed_s", "drive_time_s", "recovery_time_s", "drive_length_m",
    "peak_force_lbf", "avg_force_lbf", "work_j", "power_w", "force_curve",
]
NAN = float("nan")


@dataclass(slots=True)
class Pm5Stroke:
    index: int
    stroke_count: int
    t_pm_s: float  # temps ecoule PM5 de la notification de coup
    drive_time_s: float
    recovery_time_s: float
    drive_length_m: float
    peak_force_lbf: float
    avg_force_lbf: float
    work_j: float
    power_w: float = NAN
    force_curve: np.ndarray | None = None
    t_host_s: float | None = None

    @property
    def period_s(self) -> float:
        return self.drive_time_s + self.recovery_time_s


@dataclass(slots=True)
class StatusSample:
    t_pm_s: float
    stroke_state: int
    drag_factor: int


@dataclass(slots=True)
class Pm5Session:
    strokes: list[Pm5Stroke] = field(default_factory=list)
    status: list[StatusSample] = field(default_factory=list)
    drag_factor: float = NAN
    rate_samples_spm: list[float] = field(default_factory=list)
    source: str = ""

    @property
    def has_force_curves(self) -> bool:
        return any(s.force_curve is not None for s in self.strokes)


# --- Journal d'evenements bruts --------------------------------------------------------------


def write_event_header(fh, device: str = "") -> None:
    fh.write(json.dumps({"format": EVENT_LOG_FORMAT, "version": 1, "device": device}) + "\n")


def write_event(fh, t_host_s: float, char_id: int, data: bytes) -> None:
    fh.write(json.dumps({"t": round(t_host_s, 4), "c": char_id, "d": bytes(data).hex()}) + "\n")


def read_events(path: str | Path) -> list[tuple[float, int, bytes]]:
    events = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") == EVENT_LOG_FORMAT:
            continue
        events.append((float(row["t"]), int(row["c"]), bytes.fromhex(row["d"])))
    return events


def session_from_events(events: list[tuple[float, int, bytes]], source: str = "") -> Pm5Session:
    """Decode un journal d'evenements bruts (ordre d'arrivee) en session de coups."""
    session = Pm5Session(source=source)
    assembler = ForceCurveAssembler()
    power_by_count: dict[int, int] = {}
    pending_curves: list[tuple[float, np.ndarray]] = []
    drag: list[int] = []

    for t_host, char_id, data in events:
        if char_id == spec.MULTIPLEXED:
            try:
                char_id, data = split_multiplexed(data)
            except ValueError:
                continue
        try:
            if char_id == spec.STROKE_DATA:
                rec = parse_stroke_data(data)
                session.strokes.append(Pm5Stroke(
                    index=len(session.strokes), stroke_count=rec.stroke_count,
                    t_pm_s=rec.elapsed_s, drive_time_s=rec.drive_time_s,
                    recovery_time_s=rec.recovery_time_s, drive_length_m=rec.drive_length_m,
                    peak_force_lbf=rec.peak_force_lbf, avg_force_lbf=rec.avg_force_lbf,
                    work_j=rec.work_j, t_host_s=t_host,
                ))
            elif char_id == spec.ADDITIONAL_STROKE_DATA:
                add = parse_additional_stroke_data(data)
                power_by_count[add.stroke_count] = add.power_w
            elif char_id == spec.FORCE_CURVE:
                curve = assembler.feed(data)
                if curve is not None:
                    pending_curves.append((t_host, curve))
            elif char_id == spec.GENERAL_STATUS:
                gs = parse_general_status(data)
                session.status.append(StatusSample(gs.elapsed_s, gs.stroke_state, gs.drag_factor))
                if gs.drag_factor:
                    drag.append(gs.drag_factor)
            elif char_id == spec.ADDITIONAL_STATUS_1:
                st = parse_additional_status_1(data)
                if st.stroke_rate_spm:
                    session.rate_samples_spm.append(float(st.stroke_rate_spm))
        except ValueError:
            continue  # notification tronquee : ignoree

    for stroke in session.strokes:
        if stroke.stroke_count in power_by_count:
            stroke.power_w = float(power_by_count[stroke.stroke_count])
    _attach_curves(session, pending_curves)
    session.drag_factor = float(np.median(drag)) if drag else NAN
    return session


def _attach_curves(session: Pm5Session, curves: list[tuple[float, np.ndarray]]) -> None:
    """Associe chaque courbe au coup dont la notification la precede ou l'accompagne.

    La courbe arrive apres (ou avec) la notification de coup : on prend le dernier
    coup recu a une heure hote <= celle de la courbe, a 1,5 s pres.
    """
    for t_curve, curve in curves:
        candidates = [s for s in session.strokes
                      if s.t_host_s is not None and 0.0 <= t_curve - s.t_host_s <= 1.5
                      and s.force_curve is None]
        if candidates:
            candidates[-1].force_curve = curve


# --- Tableau de coups (CSV) ------------------------------------------------------------------


def session_from_csv(path: str | Path) -> Pm5Session:
    import pandas as pd

    df = pd.read_csv(path)
    missing = {"elapsed_s", "drive_time_s", "recovery_time_s"} - set(df.columns)
    if missing:
        raise ValueError(f"colonnes manquantes dans {path}: {sorted(missing)}")
    session = Pm5Session(source=str(path))
    for i, row in df.reset_index(drop=True).iterrows():
        def get(name: str, default: float = NAN, row=row) -> float:
            return float(row[name]) if name in row and not pd.isna(row[name]) else default

        curve = None
        if "force_curve" in row and isinstance(row["force_curve"], str) and row["force_curve"]:
            curve = np.array([float(v) for v in row["force_curve"].split(";")])
        session.strokes.append(Pm5Stroke(
            index=int(i), stroke_count=int(get("stroke_count", i + 1)),
            t_pm_s=get("elapsed_s"), drive_time_s=get("drive_time_s"),
            recovery_time_s=get("recovery_time_s"), drive_length_m=get("drive_length_m"),
            peak_force_lbf=get("peak_force_lbf"), avg_force_lbf=get("avg_force_lbf"),
            work_j=get("work_j"), power_w=get("power_w"), force_curve=curve,
        ))
    return session


def session_to_csv(session: Pm5Session, path: str | Path) -> Path:
    import pandas as pd

    rows = [{
        "stroke_count": s.stroke_count, "elapsed_s": s.t_pm_s, "drive_time_s": s.drive_time_s,
        "recovery_time_s": s.recovery_time_s, "drive_length_m": s.drive_length_m,
        "peak_force_lbf": s.peak_force_lbf, "avg_force_lbf": s.avg_force_lbf, "work_j": s.work_j,
        "power_w": s.power_w,
        "force_curve": (
            ";".join(f"{v:g}" for v in s.force_curve) if s.force_curve is not None else ""
        ),
    } for s in session.strokes]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(out, index=False)
    return out


def load_pm5(path: str | Path) -> Pm5Session:
    """Charge un fichier PM5 : journal brut .jsonl ou tableau de coups .csv."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        session = session_from_csv(path)
    elif suffix in {".jsonl", ".json", ".log"}:
        session = session_from_events(read_events(path), source=str(path))
    else:
        raise ValueError(f"format PM5 non reconnu : {path.suffix} (attendu .jsonl ou .csv)")
    if not session.strokes:
        raise ValueError(f"aucun coup PM5 dans {path}")
    return session
