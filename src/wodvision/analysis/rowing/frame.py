"""Repere "machine" de l'ergometre, deduit de la video elle-meme.

La camera peut etre inclinee (roulis) ou bouger : l'horizontale de l'image
n'est donc pas une reference fiable. Le rail de l'ergometre, lui, est
horizontal dans le monde. Le siege coulisse sur le rail : la direction
principale du deplacement de la hanche (ACP) donne l'axe du rail dans
l'image, quel que soit le roulis de la camera (apres stabilisation du
mouvement de camera, voir `stabilization/camera_motion.py`).

Convention des coordonnees machine (s, h) :
- s : le long du rail, oriente du volant vers le siege (sens du drive pour
  le siege). Le volant est donc du cote des s negatifs.
- h : perpendiculaire au rail, vers le haut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MachineFrame:
    u: np.ndarray  # axe du rail, vecteur unitaire (image, pixels)
    v: np.ndarray  # perpendiculaire, vers le haut

    @classmethod
    def from_axis(cls, u: np.ndarray) -> MachineFrame:
        u = np.asarray(u, dtype=float)
        u = u / np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        if v[1] > 0:  # y image vers le bas : on veut v vers le haut
            v = -v
        return cls(u=u, v=v)

    def flipped(self) -> MachineFrame:
        return MachineFrame.from_axis(-self.u)

    def to_machine(self, xy: np.ndarray) -> np.ndarray:
        """Convertit des points image (..., 2) en coordonnees machine (s, h)."""
        xy = np.asarray(xy, dtype=float)
        return np.stack([xy @ self.u, xy @ self.v], axis=-1)


def estimate_rail_axis(hip_xy: np.ndarray) -> np.ndarray:
    """Axe principal (ACP) du deplacement de la hanche, vecteur unitaire non oriente.

    Args:
        hip_xy: shape (n, 2), position de la hanche cote visible, pixels.
    """
    pts = np.asarray(hip_xy, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 3:
        raise ValueError("estimate_rail_axis: trop peu de points de hanche valides")
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def signed_lean_deg(frame_sh_from: np.ndarray, frame_sh_to: np.ndarray) -> np.ndarray:
    """Inclinaison signee du segment from->to par rapport a la verticale machine.

    Positif = le point `to` est decale vers +s (cote siege, "en arriere" par
    rapport au volant). Accepte des tableaux (..., 2) en coordonnees (s, h).
    """
    d = np.asarray(frame_sh_to, dtype=float) - np.asarray(frame_sh_from, dtype=float)
    return np.degrees(np.arctan2(d[..., 0], d[..., 1]))
