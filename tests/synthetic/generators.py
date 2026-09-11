"""Generateurs de trajectoires synthetiques pour tester signal/analysis sans video.

Modele : position verticale de la barre au cours d'un squat = un cosinus
sureleve (demi-periode par phase excentrique/concentrique), enchaine sur
N repetitions avec un plateau (STANDING) entre chaque rep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SyntheticTrajectory:
    """Trajectoire de reference et son signal degrade."""

    t_true_s: np.ndarray  # grille reguliere "verite terrain", haute resolution
    y_true_m: np.ndarray  # position verticale vraie (m), axe vers le haut
    t_obs_s: np.ndarray  # timestamps observes (VFR eventuel), sous-echantillon
    y_obs_m: np.ndarray  # position observee (bruitee, avec trous eventuels)
    valid_obs: np.ndarray  # bool, False = frame manquante (a exclure)
    bottom_times_s: list[float]  # verite terrain : instant du point bas par rep
    n_reps: int


def _squat_cycle(t: np.ndarray, t0: float, duration: float, depth_m: float) -> np.ndarray:
    """Un cycle descente-remontee en cosinus sureleve, nul hors de [t0, t0+duration]."""
    y = np.zeros_like(t)
    mask = (t >= t0) & (t <= t0 + duration)
    phase = (t[mask] - t0) / duration  # 0 -> 1
    y[mask] = -depth_m * (1 - np.cos(2 * np.pi * phase)) / 2
    return y


def generate_squat_trajectory(
    n_reps: int = 5,
    fps: float = 60.0,
    duration_per_rep_s: float = 2.0,
    rest_between_reps_s: float = 1.0,
    lead_in_s: float = 1.0,
    depth_m: float = 0.5,
    noise_std_m: float = 0.0,
    dropout_fraction: float = 0.0,
    vfr_jitter_s: float = 0.0,
    seed: int = 0,
) -> SyntheticTrajectory:
    """Genere une trajectoire verticale de barre synthetique pour n_reps repetitions.

    Args:
        n_reps: nombre de repetitions.
        fps: frequence d'echantillonnage nominale des observations.
        duration_per_rep_s: duree d'un cycle descente+remontee.
        rest_between_reps_s: plateau STANDING entre deux reps.
        lead_in_s: plateau STANDING avant la premiere rep.
        depth_m: profondeur du squat (amplitude verticale).
        noise_std_m: ecart-type du bruit gaussien ajoute a l'observation.
        dropout_fraction: fraction de frames observees marquees manquantes.
        vfr_jitter_s: bruit uniforme ajoute aux timestamps observes (simulate VFR).
        seed: graine aleatoire pour la reproductibilite.
    """
    rng = np.random.default_rng(seed)

    n_rest = n_reps - 1
    total_duration = (
        lead_in_s + n_reps * duration_per_rep_s + n_rest * rest_between_reps_s + 1.0
    )

    t_true = np.linspace(0.0, total_duration, int(total_duration * 1000))
    y_true = np.zeros_like(t_true)
    bottom_times: list[float] = []
    t_cursor = lead_in_s
    for _ in range(n_reps):
        y_true += _squat_cycle(t_true, t_cursor, duration_per_rep_s, depth_m)
        bottom_times.append(t_cursor + duration_per_rep_s / 2)
        t_cursor += duration_per_rep_s + rest_between_reps_s

    n_obs = int(total_duration * fps)
    t_obs = np.arange(n_obs) / fps
    if vfr_jitter_s > 0:
        t_obs = t_obs + rng.uniform(-vfr_jitter_s, vfr_jitter_s, size=n_obs)
        t_obs = np.sort(t_obs)
        t_obs = np.clip(t_obs, 0.0, total_duration)

    y_obs = np.interp(t_obs, t_true, y_true)
    if noise_std_m > 0:
        y_obs = y_obs + rng.normal(0.0, noise_std_m, size=n_obs)

    valid_obs = np.ones(n_obs, dtype=bool)
    if dropout_fraction > 0:
        n_drop = int(n_obs * dropout_fraction)
        drop_idx = rng.choice(n_obs, size=n_drop, replace=False)
        valid_obs[drop_idx] = False

    return SyntheticTrajectory(
        t_true_s=t_true,
        y_true_m=y_true,
        t_obs_s=t_obs,
        y_obs_m=y_obs,
        valid_obs=valid_obs,
        bottom_times_s=bottom_times,
        n_reps=n_reps,
    )
