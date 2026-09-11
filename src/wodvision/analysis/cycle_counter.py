"""Comptage generique de cycles (repetitions) sur un signal periodique filtre.

Contrairement a `analysis.reps.segment_reps` (specifique au squat : prominence
en fraction du diametre du disque, machine a etats STANDING/.../LOCKOUT), ce
compteur s'applique a n'importe quel signal cyclique (angle articulaire,
position d'un keypoint) sans calibration physique externe : la prominence
est exprimee en fraction de l'amplitude observee du signal lui-meme.

Usage prevu : compteurs de mouvements ad hoc (thruster, row) qui n'ont pas
encore de module d'analyse dedie (section 9, Phase 3).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def count_cycles(
    t_s: np.ndarray,
    y: np.ndarray,
    prominence_fraction: float = 0.3,
    min_distance_s: float = 0.5,
) -> list[float]:
    """Compte les cycles d'un signal periodique par detection de pics.

    Args:
        t_s: grille temporelle uniforme (secondes).
        y: signal filtre (angle en degres, position en pixels, etc.).
        prominence_fraction: prominence minimale des pics, en fraction de
            l'amplitude observee (max - min) du signal.
        min_distance_s: distance temporelle minimale entre deux cycles.

    Returns:
        Instants (s) des pics detectes, un par cycle.
    """
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)

    amplitude = float(np.max(y) - np.min(y))
    if amplitude <= 0:
        return []

    fps = 1.0 / float(np.median(np.diff(t_s)))
    prominence = prominence_fraction * amplitude
    distance_samples = max(1, int(round(min_distance_s * fps)))

    peaks, _ = find_peaks(y, prominence=prominence, distance=distance_samples)
    return t_s[peaks].tolist()


def find_bottoms_between(t_s: np.ndarray, y: np.ndarray, top_times: list[float]) -> list[float]:
    """Trouve le minimum du signal entre chaque paire de sommets consecutifs.

    Sert a decomposer un cycle deja compte par `count_cycles` (qui ne renvoie
    que les sommets) en phases montante/descendante : phase descendante =
    top[i] -> bottom[i], phase montante = bottom[i] -> top[i+1]. Un seul
    minimum entre chaque paire de sommets, donc `len(top_times) - 1` valeurs.

    Args:
        t_s: grille temporelle uniforme (secondes).
        y: signal filtre, meme longueur que t_s.
        top_times: instants des sommets (sortie de `count_cycles`), tries.

    Returns:
        Instants (s) du minimum entre chaque paire de sommets consecutifs.
    """
    if len(top_times) < 2:
        return []
    t_s = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)
    top_indices = [int(np.argmin(np.abs(t_s - t))) for t in top_times]

    bottoms = []
    for i_left, i_right in zip(top_indices[:-1], top_indices[1:], strict=True):
        i_bottom = i_left + int(np.argmin(y[i_left : i_right + 1]))
        bottoms.append(float(t_s[i_bottom]))
    return bottoms
