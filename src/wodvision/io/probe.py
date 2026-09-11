"""Inspection d'un fichier video : resolution, fps, VFR, rotation, duree, codec.

Les timestamps reels (PTS) sont utilises pour detecter le VFR, plutot que de
faire confiance au fps nominal declare par le conteneur — frequent sur
smartphone Android.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

logger = logging.getLogger(__name__)

# Coefficient de variation des intervalles PTS au-dela duquel la video est
# consideree a frequence d'images variable (VFR).
_VFR_CV_THRESHOLD = 0.05


@dataclass(slots=True)
class VideoProbe:
    path: str
    width: int
    height: int
    fps_nominal: float
    is_vfr: bool
    pts_interval_cv: float
    duration_s: float
    rotation_deg: int
    codec: str
    n_frames_sampled: int


def _rotation_from_stream(stream: av.video.stream.VideoStream) -> int:
    """Recupere la rotation (degres) depuis les side data DISPLAYMATRIX si presentes."""
    for side_data in getattr(stream.codec_context, "coded_side_data", None) or []:
        rotation = getattr(side_data, "rotation", None)
        if rotation is not None:
            return int(round(-rotation)) % 360
    # Repli : certains conteneurs exposent une metadonnee "rotate" classique.
    rotate_tag = stream.metadata.get("rotate")
    if rotate_tag is not None:
        try:
            return int(rotate_tag) % 360
        except ValueError:
            pass
    return 0


def probe(path: str | Path, max_frames_for_vfr: int = 2000) -> VideoProbe:
    """Analyse un fichier video et renvoie ses caracteristiques.

    Args:
        path: chemin du fichier video.
        max_frames_for_vfr: nombre max de frames decodees pour estimer le VFR
            (evite de decoder integralement une video tres longue).
    """
    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        codec = stream.codec_context.name
        fps_nominal = float(stream.average_rate) if stream.average_rate else 0.0
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
            if container.duration is not None
            else 0.0
        )
        rotation_deg = _rotation_from_stream(stream)

        pts_list: list[float] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts_list.append(float(frame.pts * stream.time_base))
            if len(pts_list) >= max_frames_for_vfr:
                break

    if len(pts_list) >= 2:
        intervals = np.diff(np.array(pts_list))
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
        pts_interval_cv = std_interval / mean_interval if mean_interval > 0 else 0.0
    else:
        pts_interval_cv = 0.0

    is_vfr = pts_interval_cv > _VFR_CV_THRESHOLD

    return VideoProbe(
        path=str(path),
        width=width,
        height=height,
        fps_nominal=fps_nominal,
        is_vfr=is_vfr,
        pts_interval_cv=pts_interval_cv,
        duration_s=duration_s,
        rotation_deg=rotation_deg,
        codec=codec,
        n_frames_sampled=len(pts_list),
    )
