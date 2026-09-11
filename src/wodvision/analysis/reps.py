"""Segmentation des repetitions a partir de la position verticale de la barre.

Machine a etats conceptuelle : STANDING -> ECCENTRIC -> BOTTOM -> CONCENTRIC ->
LOCKOUT. En pratique, une repetition est entierement decrite par trois
instants : `t_start` (fin de STANDING / debut ECCENTRIC), `t_bottom` (BOTTOM),
`t_end` (fin de CONCENTRIC / LOCKOUT) — ce sont les trois champs que
`RepMetrics` porte deja.

Les points bas sont detectes par prominence (fraction du diametre du disque)
sur le signal filtre ; les points hauts (debut/fin) sont le maximum local
entre deux points bas consecutifs, ce qui reste robuste meme quand la phase
STANDING est un plateau plat touchant le bord du signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(slots=True)
class RepWindow:
    rep_index: int
    t_start: float
    t_bottom: float
    t_end: float
    i_start: int
    i_bottom: int
    i_end: int


def segment_reps(
    t_s: np.ndarray,
    y: np.ndarray,
    plate_diameter_m: float,
    peak_prominence_fraction: float,
    min_rep_distance_s: float,
) -> list[RepWindow]:
    """Segmente les repetitions a partir de la trajectoire verticale filtree.

    Args:
        t_s: grille temporelle uniforme (secondes).
        y: position verticale filtree (m, axe vers le haut), meme longueur que t_s.
        plate_diameter_m: diametre du disque (m), utilise pour convertir la
            prominence relative en unites physiques.
        peak_prominence_fraction: prominence minimale des points bas, exprimee
            en fraction du diametre du disque.
        min_rep_distance_s: distance temporelle minimale entre deux points bas.

    Returns:
        Liste de `RepWindow`, cycles incomplets en debut/fin exclus.
    """
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)
    fps = 1.0 / float(np.median(np.diff(t_s)))

    prominence = peak_prominence_fraction * plate_diameter_m
    distance_samples = max(1, int(round(min_rep_distance_s * fps)))

    bottom_idx, _ = find_peaks(-y, prominence=prominence, distance=distance_samples)
    if len(bottom_idx) == 0:
        return []

    # Hauteur de reference du lockout (STANDING). Un depart/arrivee de rep qui
    # n'atteint pas cette hauteur a une tolerance pres signale un cycle
    # tronque en debut/fin de segment (video qui commence/finit en mouvement).
    global_top = float(np.max(y))
    lockout_tolerance = 0.5 * prominence

    windows: list[RepWindow] = []
    n = len(bottom_idx)
    for rep_index, i_bottom in enumerate(bottom_idx):
        left_bound = bottom_idx[rep_index - 1] if rep_index > 0 else 0
        right_bound = bottom_idx[rep_index + 1] if rep_index < n - 1 else len(y) - 1

        if left_bound == i_bottom or right_bound == i_bottom:
            continue

        i_start = left_bound + int(np.argmax(y[left_bound : i_bottom + 1]))
        i_end = i_bottom + int(np.argmax(y[i_bottom : right_bound + 1]))

        if i_start == i_bottom or i_end == i_bottom:
            continue  # pas de remontee/descente distincte : mouvement partiel

        if global_top - y[i_start] > lockout_tolerance:
            continue  # pas de lockout complet visible avant : cycle tronque
        if global_top - y[i_end] > lockout_tolerance:
            continue  # pas de lockout complet visible apres : cycle tronque

        windows.append(
            RepWindow(
                rep_index=len(windows),
                t_start=float(t_s[i_start]),
                t_bottom=float(t_s[i_bottom]),
                t_end=float(t_s[i_end]),
                i_start=i_start,
                i_bottom=i_bottom,
                i_end=i_end,
            )
        )

    return windows
