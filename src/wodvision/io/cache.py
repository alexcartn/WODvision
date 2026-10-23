"""Cache disque de l'inference (npz) : l'analyse se rejoue sans relancer les modeles.

Cle = hash(fichier video + parametres d'inference). Le hash du fichier porte
sur la taille et des blocs de debut / milieu / fin (rapide sur de grosses
videos, suffisant pour distinguer deux fichiers differents).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE_VERSION = 1


@dataclass(slots=True)
class RowingTrace:
    """Sortie de la partie "inference" : tout ce qu'il faut pour analyser, sans modele."""

    t_s: np.ndarray  # (n,) PTS reels
    xy_raw: np.ndarray  # (n, 17, 2) pixels image (apres rotation metadonnee)
    xy_stab: np.ndarray  # (n, 17, 2) repere de la premiere frame (mouvement de camera retire)
    score: np.ndarray  # (n, 17)
    bbox: np.ndarray  # (n, 4) NaN si athlete perdu
    to_ref: np.ndarray  # (n, 3, 3)
    cam_reliable: np.ndarray  # (n,) bool
    meta: dict = field(default_factory=dict)


def file_fingerprint(path: str | Path) -> str:
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        for offset in (0, max(0, size // 2 - 32768), max(0, size - 65536)):
            f.seek(offset)
            h.update(f.read(65536))
    return h.hexdigest()


def cache_key(video: str | Path, params: dict) -> str:
    payload = json.dumps({"v": CACHE_VERSION, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256((file_fingerprint(video) + payload).encode()).hexdigest()[:20]


def save_trace(trace: RowingTrace, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, t_s=trace.t_s, xy_raw=trace.xy_raw, xy_stab=trace.xy_stab, score=trace.score,
        bbox=trace.bbox, to_ref=trace.to_ref, cam_reliable=trace.cam_reliable,
        meta=json.dumps(trace.meta, default=str),
    )
    return path


def load_trace(path: str | Path) -> RowingTrace:
    with np.load(path, allow_pickle=False) as z:
        return RowingTrace(
            t_s=z["t_s"], xy_raw=z["xy_raw"], xy_stab=z["xy_stab"], score=z["score"],
            bbox=z["bbox"], to_ref=z["to_ref"], cam_reliable=z["cam_reliable"],
            meta=json.loads(str(z["meta"])),
        )
