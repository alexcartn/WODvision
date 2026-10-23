"""Decodage des notifications BLE du PM5 (octets little-endian, format de la spec)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _u(data: bytes, start: int, size: int) -> int:
    return int.from_bytes(data[start : start + size], "little")


def _check(data: bytes, size: int, name: str) -> None:
    if len(data) != size:
        raise ValueError(f"{name}: {size} octets attendus, {len(data)} recus")


@dataclass(slots=True)
class GeneralStatus:
    elapsed_s: float
    distance_m: float
    workout_state: int
    rowing_state: int
    stroke_state: int
    drag_factor: int


@dataclass(slots=True)
class AdditionalStatus1:
    elapsed_s: float
    speed_mps: float
    stroke_rate_spm: int
    heart_rate_bpm: int | None
    pace_s_per_500m: float


@dataclass(slots=True)
class StrokeRecord:
    elapsed_s: float
    distance_m: float
    drive_length_m: float
    drive_time_s: float
    recovery_time_s: float
    stroke_distance_m: float
    peak_force_lbf: float
    avg_force_lbf: float
    work_j: float
    stroke_count: int


@dataclass(slots=True)
class AdditionalStrokeRecord:
    elapsed_s: float
    power_w: int
    calories_per_hr: int
    stroke_count: int


def parse_general_status(data: bytes) -> GeneralStatus:
    _check(data, 19, "general status (0x0031)")
    return GeneralStatus(
        elapsed_s=_u(data, 0, 3) * 0.01, distance_m=_u(data, 3, 3) * 0.1,
        workout_state=data[8], rowing_state=data[9], stroke_state=data[10],
        drag_factor=data[18],
    )


def parse_additional_status_1(data: bytes) -> AdditionalStatus1:
    _check(data, 17, "additional status 1 (0x0032)")
    hr = data[6]
    return AdditionalStatus1(
        elapsed_s=_u(data, 0, 3) * 0.01, speed_mps=_u(data, 3, 2) * 0.001,
        stroke_rate_spm=data[5], heart_rate_bpm=None if hr == 255 else hr,
        pace_s_per_500m=_u(data, 7, 2) * 0.01,
    )


def parse_stroke_data(data: bytes) -> StrokeRecord:
    _check(data, 20, "stroke data (0x0035)")
    return StrokeRecord(
        elapsed_s=_u(data, 0, 3) * 0.01, distance_m=_u(data, 3, 3) * 0.1,
        drive_length_m=data[6] * 0.01, drive_time_s=data[7] * 0.01,
        recovery_time_s=_u(data, 8, 2) * 0.01, stroke_distance_m=_u(data, 10, 2) * 0.01,
        peak_force_lbf=_u(data, 12, 2) * 0.1, avg_force_lbf=_u(data, 14, 2) * 0.1,
        work_j=_u(data, 16, 2) * 0.1, stroke_count=_u(data, 18, 2),
    )


def parse_additional_stroke_data(data: bytes) -> AdditionalStrokeRecord:
    _check(data, 18, "additional stroke data (0x0036)")
    return AdditionalStrokeRecord(
        elapsed_s=_u(data, 0, 3) * 0.01, power_w=_u(data, 3, 2),
        calories_per_hr=_u(data, 5, 2), stroke_count=_u(data, 7, 2),
    )


class ForceCurveAssembler:
    """Reassemble une courbe de force a partir des notifications 0x003D.

    Chaque notification : octet 0 = (nb total de notifications << 4) | nb de mots 16 bits ;
    octet 1 = numero de sequence ; puis les mots en little-endian. Une courbe est
    complete quand ses `total` notifications sont recues. Hypothese (a verifier sur
    materiel) : le numero de sequence repart de 0 a chaque courbe ; en cas de perte
    ou de desordre, la courbe en cours est abandonnee plutot que reconstruite de travers.
    L'axe est la position de la poignee (echantillons regulierement espaces, ~3 cm
    d'apres le forum Concept2) et l'unite de force n'est pas documentee : seule la
    forme est exploitee en aval.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._total = 0

    def feed(self, data: bytes) -> np.ndarray | None:
        if len(data) < 2:
            return None
        total, n_words = data[0] >> 4, data[0] & 0x0F
        seq = data[1]
        if len(data) < 2 + 2 * n_words or total == 0:
            return None
        if seq == 0 or total != self._total or seq != len(self._chunks):
            self._chunks, self._total = [], total
            if seq != 0:
                return None  # debut de courbe manque
        words = np.frombuffer(data[2 : 2 + 2 * n_words], dtype="<u2").astype(float)
        self._chunks.append(words)
        if len(self._chunks) == self._total:
            curve = np.concatenate(self._chunks)
            self._chunks, self._total = [], 0
            return curve
        return None


def split_multiplexed(data: bytes) -> tuple[int, bytes]:
    """0x0080 : (identifiant de la caracteristique d'origine, donnees)."""
    if not data:
        raise ValueError("notification multiplexee vide")
    return data[0], data[1:]  # l'identifiant est l'octet de poids faible (0x31 -> 0x0031)
