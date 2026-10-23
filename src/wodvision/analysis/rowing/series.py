"""Preparation des series de keypoints : reechantillonnage uniforme puis filtrage.

Entree : keypoints stabilises (cache d'inference). Sortie : series filtrees
du cote visible sur une grille uniforme, plus un masque des instants dont le
signal est interpole sur un trou trop long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wodvision.analysis.angles import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from wodvision.config import Config
from wodvision.signal.filtering import butterworth_lowpass
from wodvision.signal.resampling import resample_uniform

SIDE_INDICES = {
    "left": {
        "shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST,
        "hip": L_HIP, "knee": L_KNEE, "ankle": L_ANKLE,
    },
    "right": {
        "shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST,
        "hip": R_HIP, "knee": R_KNEE, "ankle": R_ANKLE,
    },
}


@dataclass(slots=True)
class RowSeries:
    t_s: np.ndarray  # (m,)
    pts: dict[str, np.ndarray]  # nom -> (m, 2), pixels (repere stabilise), filtre
    missing: np.ndarray  # (m,) True si un des keypoints est interpole sur un long trou
    fps: float


def prepare_series(
    t_obs_s: np.ndarray,
    xy: np.ndarray,
    score: np.ndarray,
    side: str,
    cfg: Config,
) -> RowSeries:
    """Reechantillonne puis filtre (Butterworth phase nulle) les 6 keypoints du cote retenu.

    Args:
        t_obs_s: PTS reels des frames analysees, shape (n,).
        xy: keypoints COCO-17, shape (n, 17, 2), repere stabilise.
        score: confiances, shape (n, 17).
        side: 'left' | 'right'.
    """
    fps = cfg.signal.resample_fps
    min_conf = cfg.signal.min_keypoint_confidence
    pts: dict[str, np.ndarray] = {}
    missing = None
    t_grid = None
    for name, idx in SIDE_INDICES[side].items():
        valid = (score[:, idx] > min_conf) & np.all(np.isfinite(xy[:, idx]), axis=1)
        cols = []
        for axis in (0, 1):
            res = resample_uniform(
                t_obs_s, xy[:, idx, axis], valid, fps, cfg.signal.max_gap_frames
            )
            cols.append(
                butterworth_lowpass(
                    res.y,
                    fps=fps,
                    cutoff_hz=cfg.rowing.cutoff_hz,
                    order=cfg.signal.butterworth_order,
                )
            )
            missing = res.missing if missing is None else (missing | res.missing)
            t_grid = res.t_s
        pts[name] = np.stack(cols, axis=1)
    assert t_grid is not None and missing is not None
    return RowSeries(t_s=t_grid, pts=pts, missing=missing, fps=fps)
