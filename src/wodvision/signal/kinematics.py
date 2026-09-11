"""Derivees cinematiques (vitesse, acceleration) sur un signal filtre."""

from __future__ import annotations

import numpy as np


def velocity(y: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Vitesse par derivation numerique (np.gradient) sur une grille uniforme."""
    return np.gradient(y, t_s)


def acceleration(v: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Acceleration par derivation numerique de la vitesse."""
    return np.gradient(v, t_s)
