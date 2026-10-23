"""Rameur synthetique : keypoints COCO-17 d'un cote, dans le repere machine puis image.

Modele cinematique plan (metres) : cheville fixe sur le cale-pieds, jambes /
buste / bras pilotes par des progressions 0..1 qui commencent a des instants
parametrables du drive. Permet de fabriquer un geste "propre" et des fautes
connues (dos precoce, bras precoces, chaine qui monte, ...) sans video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Indices COCO-17 cote gauche (le droit reste a score 0).
L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE = 5, 7, 9, 11, 13, 15

THIGH, SHIN, TORSO, ARM = 0.45, 0.45, 0.50, 0.30


@dataclass(slots=True)
class RowerParams:
    rate_spm: float = 24.0
    drive_fraction: float = 1.0 / 3.0
    duration_s: float = 40.0
    fps: float = 30.0
    trunk_onset: float = 0.5  # debut du dos, en fraction du drive
    arm_onset: float = 0.8
    knee_break: float = 0.6  # fraction de la recuperation avant flexion des genoux
    shin_catch_deg: float = 0.0  # < 0 : tibia au-dela de la verticale
    chain_deg: float = 0.0  # pente de la poignee pendant le drive
    handle_h0: float = 0.62
    lean_catch_deg: float = -30.0
    lean_finish_deg: float = 25.0
    fatigue_drift_deg: float = 0.0  # inclinaison supplementaire du buste au catch en fin de serie
    noise_px: float = 1.5
    roll_deg: float = 0.0  # roulis de camera (rotation de l'image)
    px_per_m: float = 300.0
    origin_px: tuple[float, float] = (400.0, 600.0)  # position de la cheville dans l'image
    seed: int = 0


def _smooth(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3


SEAT_H = 0.35  # le siege coulisse a hauteur constante sur le rail


def _knee_from_hip(ankle: np.ndarray, hip: np.ndarray) -> np.ndarray:
    """Genou par intersection de cercles (cuisse/tibia), du cote haut."""
    d = hip - ankle
    dist = min(np.linalg.norm(d), (THIGH + SHIN) * 0.999)
    a = (SHIN**2 - THIGH**2 + dist**2) / (2 * dist)
    hh = np.sqrt(max(SHIN**2 - a**2, 0.0))
    base = ankle + a * d / np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    cands = [base + hh * perp, base - hh * perp]
    return max(cands, key=lambda q: q[1])


def _arm(shoulder: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Coude par IK a 2 segments egaux, du cote bas."""
    mid = (shoulder + wrist) / 2
    d = wrist - shoulder
    r = np.linalg.norm(d)
    off = np.sqrt(max(ARM**2 - (r / 2) ** 2, 0.0))
    perp = np.array([-d[1], d[0]]) / max(r, 1e-9)
    cands = [mid + off * perp, mid - off * perp]
    return min(cands, key=lambda p: p[1])


def generate_rower(p: RowerParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Renvoie (t_s, xy (n,17,2) en pixels image, score (n,17), verite terrain)."""
    rng = np.random.default_rng(p.seed)
    n = int(p.duration_s * p.fps)
    t = np.arange(n) / p.fps
    period = 60.0 / p.rate_spm
    d = p.drive_fraction

    ankle = np.array([0.0, 0.25])
    xy_m = np.zeros((n, 17, 2))
    wrist_s_catch = None

    for i, ti in enumerate(t):
        phase = (ti % period) / period
        if phase < d:
            u = phase / d
            leg = _smooth(u / 0.7)
            trunk = _smooth((u - p.trunk_onset) / (0.9 - p.trunk_onset))
            arm = _smooth((u - p.arm_onset) / (1.0 - p.arm_onset))
        else:
            q = (phase - d) / (1 - d)
            arm = 1 - _smooth(q / 0.3)
            trunk = 1 - _smooth((q - 0.2) / 0.4)
            leg = 1 - _smooth((q - p.knee_break) / (1 - p.knee_break))

        shin_c = np.radians(p.shin_catch_deg)
        knee_c = ankle + SHIN * np.array([np.sin(shin_c), np.cos(shin_c)])
        s_hip_catch = knee_c[0] + np.sqrt(THIGH**2 - (knee_c[1] - SEAT_H) ** 2)
        hip = np.array([s_hip_catch + (0.88 - s_hip_catch) * leg, SEAT_H])
        knee = _knee_from_hip(ankle, hip)
        drift = p.fatigue_drift_deg * (ti / p.duration_s)
        lean_c = p.lean_catch_deg - drift
        lean = np.radians(lean_c + (p.lean_finish_deg - lean_c) * trunk)
        shoulder = hip + TORSO * np.array([np.sin(lean), np.cos(lean)])

        reach = 0.595 - 0.295 * arm
        # Hauteur de poignee pilotee par la pente de chaine, s deduit de la portee.
        if wrist_s_catch is None:
            wrist_s_catch = shoulder[0] - 0.55
        h_try = p.handle_h0
        for _ in range(3):  # point fixe : la hauteur depend de s, s de la hauteur
            dh = min(abs(shoulder[1] - h_try), reach * 0.999)
            ws = shoulder[0] - np.sqrt(reach**2 - dh**2)
            h_try = p.handle_h0 + np.tan(np.radians(p.chain_deg)) * (ws - wrist_s_catch)
        wrist = np.array([ws, h_try])
        elbow = _arm(shoulder, wrist)

        for idx, pt in ((L_SHOULDER, shoulder), (L_ELBOW, elbow), (L_WRIST, wrist),
                        (L_HIP, hip), (L_KNEE, knee), (L_ANKLE, ankle)):
            xy_m[i, idx] = pt

    # Metres (s, h) -> image (x, y vers le bas), roulis de camera, bruit.
    img = np.stack([xy_m[..., 0], -xy_m[..., 1]], axis=-1) * p.px_per_m
    a = np.radians(p.roll_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    img = img @ rot.T + np.array(p.origin_px)
    img = img + rng.normal(0, p.noise_px, img.shape)

    score = np.zeros((n, 17))
    for idx in (L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE):
        score[:, idx] = 0.9
    truth = {"period_s": period, "n_strokes_full": int(p.duration_s // period)}
    return t, img, score, truth
