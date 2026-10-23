"""Detection des coups d'aviron (catch, finish) sur le signal de la poignee.

Signal : projection du poignet sur l'axe du rail (s). Valleys = catch (poignee
au plus pres du volant), pics = finish. Le sens de l'axe est lui-meme
deduit des donnees : le drive (catch -> finish) est plus court que la
recuperation (finish -> catch suivant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(slots=True)
class Stroke:
    index: int
    t_catch: float
    t_finish: float
    t_next_catch: float

    @property
    def drive_time(self) -> float:
        return self.t_finish - self.t_catch

    @property
    def recovery_time(self) -> float:
        return self.t_next_catch - self.t_finish

    @property
    def rate_spm(self) -> float:
        return 60.0 / (self.t_next_catch - self.t_catch)


def _alternating_extrema(
    t_s: np.ndarray, y: np.ndarray, prominence: float, distance: int
) -> list[tuple[float, str, float]]:
    """Valleys ('catch') et pics ('finish') tries par temps, en alternance stricte."""
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    valleys, _ = find_peaks(-y, prominence=prominence, distance=distance)
    events = [(float(t_s[i]), "finish", float(y[i])) for i in peaks]
    events += [(float(t_s[i]), "catch", float(y[i])) for i in valleys]
    events.sort(key=lambda e: e[0])

    merged: list[tuple[float, str, float]] = []
    for event in events:
        if merged and merged[-1][1] == event[1]:
            # Deux extrema du meme type d'affilee : garder le plus marque.
            prev = merged[-1][2]
            keep_new = event[2] > prev if event[1] == "finish" else event[2] < prev
            if keep_new:
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def _strokes_from_events(events: list[tuple[float, str, float]]) -> list[Stroke]:
    strokes: list[Stroke] = []
    for i in range(len(events) - 2):
        a, b, c = events[i], events[i + 1], events[i + 2]
        if a[1] == "catch" and b[1] == "finish" and c[1] == "catch":
            strokes.append(Stroke(len(strokes), a[0], b[0], c[0]))
    return strokes


def detect_strokes(
    t_s: np.ndarray,
    handle_s: np.ndarray,
    prominence_fraction: float = 0.35,
    min_period_s: float = 1.2,
    min_rate_spm: float = 10.0,
    max_rate_spm: float = 50.0,
) -> tuple[list[Stroke], int]:
    """Detecte les coups complets (catch -> finish -> catch suivant).

    Args:
        t_s: grille temporelle uniforme (s).
        handle_s: position de la poignee le long du rail (filtree).
        prominence_fraction: prominence minimale, fraction de l'amplitude.
        min_period_s: distance minimale entre deux extrema de meme type.
        min_rate_spm, max_rate_spm: coups hors de cette plage rejetes.

    Returns:
        (coups, sign) ou sign = +1 si l'axe fourni est dans le bon sens
        (catch = minimum), -1 s'il a fallu l'inverser. Les temps sont ceux
        du signal oriente.
    """
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(handle_s, dtype=float)
    amplitude = float(np.max(y) - np.min(y))
    if amplitude <= 0:
        return [], 1

    fps = 1.0 / float(np.median(np.diff(t_s)))
    distance = max(1, int(round(min_period_s * fps)))
    prominence = prominence_fraction * amplitude

    best: tuple[list[Stroke], int] | None = None
    best_score = np.inf
    for sign in (1, -1):
        events = _alternating_extrema(t_s, sign * y, prominence, distance)
        strokes = _strokes_from_events(events)
        if not strokes:
            continue
        drive = np.median([s.drive_time for s in strokes])
        recovery = np.median([s.recovery_time for s in strokes])
        score = drive / recovery  # le bon sens a un drive plus court que la recuperation
        if score < best_score:
            best, best_score = (strokes, sign), score
    if best is None:
        return [], 1

    strokes, sign = best
    kept = [s for s in strokes if min_rate_spm <= s.rate_spm <= max_rate_spm]
    for i, s in enumerate(kept):
        s.index = i
    return kept, sign


def _period_cv(strokes: list[Stroke]) -> float:
    periods = np.array([s.t_next_catch - s.t_catch for s in strokes])
    return float(np.std(periods) / np.mean(periods))


def detect_strokes_multi(
    t_s: np.ndarray,
    signals: dict[str, np.ndarray],
    prominence_fraction: float = 0.35,
    min_period_s: float = 1.2,
    min_rate_spm: float = 10.0,
    max_rate_spm: float = 50.0,
    max_period_cv: float = 0.35,
) -> tuple[str | None, list[Stroke], int]:
    """Essaie plusieurs signaux (poignet, hanche, epaule) et garde le plus regulier.

    Utile quand le poignet est mal detecte (occultation, vue oblique) : la hanche
    et l'epaule suivent aussi le cycle du siege. A nombre de coups comparable
    (>= 80 % du maximum), le signal dont la periode est la plus reguliere gagne ;
    a egalite, l'ordre du dictionnaire (poignet d'abord) departage.

    Returns:
        (nom du signal retenu, coups, sens) ; (None, [], 1) si aucun ne convient.
    """
    results = []
    for name, y in signals.items():
        strokes, sign = detect_strokes(
            t_s, y, prominence_fraction, min_period_s, min_rate_spm, max_rate_spm
        )
        if len(strokes) >= 3 and _period_cv(strokes) <= max_period_cv:
            results.append((name, strokes, sign, _period_cv(strokes)))
    if not results:
        # Aucun signal regulier : renvoyer le meilleur nombre de coups, pour le diagnostic.
        best = max(
            ((n, *detect_strokes(t_s, y, prominence_fraction, min_period_s,
                                 min_rate_spm, max_rate_spm)) for n, y in signals.items()),
            key=lambda r: len(r[1]),
        )
        return best[0], best[1], best[2]
    most = max(len(r[1]) for r in results)
    eligible = [r for r in results if len(r[1]) >= 0.8 * most]
    order = list(signals)
    name, strokes, sign, _ = min(eligible, key=lambda r: (round(r[3], 2), order.index(r[0])))
    return name, strokes, sign
