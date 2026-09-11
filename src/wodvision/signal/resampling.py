"""Reechantillonnage sur une grille temporelle uniforme a partir de PTS reels.

Les smartphones (Android surtout) enregistrent souvent en VFR : le temps ne
doit jamais etre deduit de `index / fps`, toujours des PTS observes. Les
points a faible confiance sont exclus avant interpolation ; les trous plus
longs que `max_gap_frames` (a la cadence nominale d'origine) restent marques
`missing`, meme si une valeur interpolee est renvoyee pour permettre un
filtrage continu en aval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ResampleResult:
    t_s: np.ndarray
    y: np.ndarray
    missing: np.ndarray  # True = trou trop long, valeur interpolee peu fiable


def resample_uniform(
    t_obs_s: np.ndarray,
    y_obs: np.ndarray,
    valid_obs: np.ndarray,
    target_fps: float,
    max_gap_frames: int,
) -> ResampleResult:
    """Reechantillonne un signal irregulier (VFR, trous) sur une grille uniforme.

    Args:
        t_obs_s: timestamps observes (secondes, PTS reels), pas necessairement
            uniformes.
        y_obs: valeurs observees, meme longueur que t_obs_s.
        valid_obs: masque bool, False pour les frames a exclure (faible
            confiance, detection manquante).
        target_fps: frequence de la grille de sortie.
        max_gap_frames: au-dela de ce nombre de frames manquantes consecutives
            (a la cadence nominale d'origine), les points de la grille dans ce
            trou sont marques `missing`.
    """
    t_obs_s = np.asarray(t_obs_s, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    valid_obs = np.asarray(valid_obs, dtype=bool)

    if valid_obs.sum() < 2:
        raise ValueError("resample_uniform: moins de 2 observations valides")

    t_valid = t_obs_s[valid_obs]
    y_valid = y_obs[valid_obs]

    n_samples = int(round((t_obs_s[-1] - t_obs_s[0]) * target_fps)) + 1
    t_grid = t_obs_s[0] + np.arange(n_samples) / target_fps

    y_grid = np.interp(t_grid, t_valid, y_valid)

    dt_nominal = float(np.median(np.diff(t_obs_s)))
    max_gap_s = max_gap_frames * dt_nominal

    # Bracket [idx_le, idx_ge] : derniere observation valide <= t_grid et
    # premiere >= t_grid. Si t_grid coincide exactement avec une observation
    # valide, les deux indices coincident et l'ecart est nul (pas un trou).
    idx_le = np.clip(np.searchsorted(t_valid, t_grid, side="right") - 1, 0, len(t_valid) - 1)
    idx_ge = np.clip(np.searchsorted(t_valid, t_grid, side="left"), 0, len(t_valid) - 1)
    gap_s = t_valid[idx_ge] - t_valid[idx_le]
    missing = gap_s > max_gap_s

    return ResampleResult(t_s=t_grid, y=y_grid, missing=missing)
