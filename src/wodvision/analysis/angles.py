"""Angles articulaires 2D a partir de keypoints COCO-17.

Reutilisable par le squat (genou, hanche, section 7.9) comme par les
compteurs de mouvements ad hoc (thruster, row).
"""

from __future__ import annotations

import math

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = range(5)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

LEFT_LIMB = (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE)
RIGHT_LIMB = (R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE)


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle au sommet `b`, forme par les segments b->a et b->c (degres)."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def knee_angle_deg(keypoints_xy: np.ndarray, side: str = "left") -> float:
    hip, knee, ankle = (L_HIP, L_KNEE, L_ANKLE) if side == "left" else (R_HIP, R_KNEE, R_ANKLE)
    return joint_angle_deg(keypoints_xy[hip], keypoints_xy[knee], keypoints_xy[ankle])


def hip_angle_deg(keypoints_xy: np.ndarray, side: str = "left") -> float:
    shoulder, hip, knee = (
        (L_SHOULDER, L_HIP, L_KNEE) if side == "left" else (R_SHOULDER, R_HIP, R_KNEE)
    )
    return joint_angle_deg(keypoints_xy[shoulder], keypoints_xy[hip], keypoints_xy[knee])


def elbow_angle_deg(keypoints_xy: np.ndarray, side: str = "left") -> float:
    shoulder, elbow, wrist = (
        (L_SHOULDER, L_ELBOW, L_WRIST) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST)
    )
    return joint_angle_deg(keypoints_xy[shoulder], keypoints_xy[elbow], keypoints_xy[wrist])


def angle_from_vertical_deg(p_near: np.ndarray, p_far: np.ndarray) -> float:
    """Angle du segment p_near->p_far par rapport a la verticale (degres).

    0 = segment vertical (aligne avec l'axe image haut-bas), 90 = horizontal.
    Non signe : ne distingue pas une inclinaison avant d'une inclinaison
    arriere, seulement son amplitude — a interpreter avec le contexte du
    mouvement (ex. sens de la barre/percentage de la rep).
    """
    dx, dy = np.asarray(p_far, dtype=float) - np.asarray(p_near, dtype=float)
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return float("nan")
    # Axe vertical image = (0, -1) ou (0, 1) selon le sens ; seule l'amplitude
    # compte ici, donc on prend l'angle par rapport a l'axe des y.
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def trunk_lean_deg(keypoints_xy: np.ndarray, side: str = "left") -> float:
    """Inclinaison du tronc (epaule->hanche) par rapport a la verticale."""
    shoulder, hip = (L_SHOULDER, L_HIP) if side == "left" else (R_SHOULDER, R_HIP)
    return angle_from_vertical_deg(keypoints_xy[shoulder], keypoints_xy[hip])


def shin_angle_deg(keypoints_xy: np.ndarray, side: str = "left") -> float:
    """Inclinaison du tibia (genou->cheville) par rapport a la verticale."""
    knee, ankle = (L_KNEE, L_ANKLE) if side == "left" else (R_KNEE, R_ANKLE)
    return angle_from_vertical_deg(keypoints_xy[knee], keypoints_xy[ankle])


def pick_visible_side(scores_over_time: np.ndarray) -> str:
    """Choisit 'left' ou 'right' selon le meilleur score moyen sur toute la sequence.

    Args:
        scores_over_time: shape (n_frames, 17), scores de confiance COCO-17.
    """
    left_mean = float(np.mean(scores_over_time[:, LEFT_LIMB]))
    right_mean = float(np.mean(scores_over_time[:, RIGHT_LIMB]))
    return "left" if left_mean >= right_mean else "right"
