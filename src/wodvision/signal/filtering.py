"""Filtrage passe-bas a phase nulle, applique apres reechantillonnage uniforme.

Le filtrage suppose une grille temporelle uniforme : toujours appeler
`resample_uniform` avant `butterworth_lowpass`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_lowpass(
    y: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Filtre passe-bas Butterworth a phase nulle (filtfilt).

    Args:
        y: signal echantillonne uniformement (voir `resample_uniform`).
        fps: frequence d'echantillonnage de `y`.
        cutoff_hz: frequence de coupure.
        order: ordre du filtre (avant application filtfilt aller-retour).
    """
    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} doit etre < frequence de Nyquist ({nyquist})"
        )
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(y) <= padlen:
        raise ValueError(
            f"signal trop court ({len(y)} echantillons) pour filtfilt (padlen={padlen})"
        )
    return filtfilt(b, a, y)
