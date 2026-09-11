"""Extraction de frames a partir des PTS reels (instant unique ou fenetre)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np


def iter_frames(
    path: str | Path, t_start_s: float, t_end_s: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode les frames dont le PTS reel tombe dans [t_start_s, t_end_s].

    Utilise `seek` vers le keyframe precedent puis filtre par PTS reel,
    correct meme en VFR.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target_pts = int(t_start_s / stream.time_base)
        container.seek(target_pts, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t_s = float(frame.pts * stream.time_base)
            if t_s < t_start_s:
                continue
            if t_s > t_end_s:
                break
            yield t_s, frame.to_ndarray(format="bgr24")


def grab_frame(path: str | Path, t_s: float) -> np.ndarray:
    """Renvoie l'image BGR de la premiere frame dont le PTS est >= t_s.

    Utilise `seek` (vers le keyframe precedent) puis decode jusqu'a l'instant
    cible, pour rester correct sur les fichiers a frequence d'images variable.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target_pts = int(t_s / stream.time_base)
        container.seek(target_pts, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts * stream.time_base) >= t_s:
                return frame.to_ndarray(format="bgr24")
    raise ValueError(f"aucune frame trouvee a t={t_s}s dans {path}")
