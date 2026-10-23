"""Images clés et ralentis : le meilleur coup et le coup à améliorer, avec le fantôme.

Pour un amateur, une image vaut mieux qu'un tableau : au catch et au finish du coup le plus
proche de la posture de référence et du coup qui s'en écarte le plus, on voit la posture réelle
(vert), la référence (jaune) et une flèche vers l'articulation qui s'écarte le plus. Chaque coup
est aussi rendu en ralenti (x4). Disponible en analyse complète (le fantome en est la base).
Les textes incrustés sont en ASCII (limite de cv2.putText) ; les légendes accentuées sont
ajoutées par l'application.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from wodvision.analysis.rowing.ghost import GhostModel, joint_gaps, phase_of, stroke_deviations
from wodvision.analysis.rowing.report import RowingReport
from wodvision.analysis.rowing.series import SIDE_INDICES
from wodvision.io.cache import RowingTrace
from wodvision.io.frames import grab_frame, iter_frames
from wodvision.io.video_writer import H264Writer
from wodvision.rowing_pipeline import apply_rotation
from wodvision.viz.rowing_dashboard import (
    DRIVE_COLOR,
    GHOST_COLOR,
    KP_THR,
    _draw_athlete,
    _draw_ghost,
    ghost_points_image,
)

JOINT_FR = {"knee": "genou", "hip": "hanche", "shoulder": "épaule", "elbow": "coude",
            "wrist": "poignet"}
SLOWMO = 4
OUT_WIDTH = 640
MIN_STROKES = 3
# Ecart (longueurs de tronc) entre meilleur et pire coup sous lequel les coups sont "similaires".
SIMILAR_GAP = 0.03


@dataclass(slots=True)
class Highlights:
    images: dict[str, bytes] = field(default_factory=dict)  # best_catch, best_finish, ...
    clips: dict[str, bytes] = field(default_factory=dict)  # best, worst (mp4, ralenti)
    meta: dict[str, dict] = field(default_factory=dict)  # coup, ecart, articulation par cle


def _crop_box(trace: RowingTrace, i0: int, i1: int, shape: tuple[int, int]) -> tuple[int, ...]:
    """Cadre stable autour du rameur sur les echantillons [i0, i1]."""
    xy = trace.xy_raw[i0 : i1 + 1].reshape(-1, 2)  # (n * 17, 2)
    ok = (trace.score[i0 : i1 + 1] > KP_THR).reshape(-1)
    pts = xy[ok & np.all(np.isfinite(xy), axis=1)]
    h, w = shape
    if len(pts) == 0:
        return 0, 0, w, h
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    mx, my = max((x2 - x1) * 0.5, 60), max((y2 - y1) * 0.35, 60)
    box = np.array([x1 - mx, y1 - my, x2 + mx, y2 + my])
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
    return tuple(int(v) for v in box)


def _crop_resize(image: np.ndarray, box: tuple[int, ...]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    scale = OUT_WIDTH / max(crop.shape[1], 1)
    return cv2.resize(crop, (OUT_WIDTH, max(2, round(crop.shape[0] * scale))))


def _legend(image: np.ndarray, title: str) -> None:
    cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.circle(image, (18, 52), 6, DRIVE_COLOR[::-1], -1, cv2.LINE_AA)
    cv2.putText(image, "Toi", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.circle(image, (88, 52), 6, GHOST_COLOR[::-1], -1, cv2.LINE_AA)
    cv2.putText(image, "Reference", (100, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                1, cv2.LINE_AA)


def _annotate(image, kp, sc, pose, report, to_ref, joint: str | None) -> None:
    _draw_ghost(image, pose, report, to_ref)
    if joint is not None:  # fleche de l'articulation reelle vers sa position de reference
        idx = SIDE_INDICES[report.side][joint]
        if sc[idx] > KP_THR and np.all(np.isfinite(kp[idx])):
            target = ghost_points_image(pose, report, to_ref)[joint]
            cv2.arrowedLine(image, tuple(int(v) for v in kp[idx]), target, (255, 255, 255),
                            3, cv2.LINE_AA, tipLength=0.25)
    _draw_athlete(image, kp, sc, [], DRIVE_COLOR)


def _nearest(trace: RowingTrace, t: float) -> int:
    return int(np.clip(np.argmin(np.abs(trace.t_s - t)), 0, len(trace.t_s) - 1))


def _still(video, trace, report, ghost, t: float, x: float, title: str):
    """Image du catch (x=0) ou du finish (x=1) au temps t, avec fantome et fleche."""
    i = _nearest(trace, t)
    image = apply_rotation(grab_frame(video, t), int(trace.meta.get("rotation_deg", 0)))
    pose = ghost.pose("drive", x)
    gaps = joint_gaps(report, ghost, t + (0.02 if x == 0 else -0.02))
    joint = max(gaps, key=gaps.get) if gaps else None
    box = _crop_box(trace, max(0, i - 8), min(len(trace.t_s) - 1, i + 8), image.shape[:2])
    _annotate(image, trace.xy_raw[i], trace.score[i], pose, report, trace.to_ref[i], joint)
    out = _crop_resize(image, box)
    _legend(out, title)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), joint


def _clip(video, trace, report, ghost, stroke, title: str) -> bytes:
    """Ralenti (x4) d'un coup, rameur recadre, fantome et squelette lisses dans le temps."""
    info_fps = float(trace.meta.get("fps", 30.0))
    t0 = max(float(trace.t_s[0]), stroke.t_catch - 0.3)
    t1 = min(float(trace.t_s[-1]), stroke.t_next_catch + 0.3)
    i0, i1 = _nearest(trace, t0), _nearest(trace, t1)
    h, w = int(trace.meta["height"]), int(trace.meta["width"])
    box = _crop_box(trace, i0, i1, (h, w))
    rotation = int(trace.meta.get("rotation_deg", 0))
    xy = trace.xy_raw
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        writer = None
        for t, image in iter_frames(video, t0, t1):
            image = apply_rotation(image, rotation)
            i = _nearest(trace, t)
            kp = np.stack([np.interp(t, trace.t_s, xy[:, k, a]) for k in range(17)
                           for a in (0, 1)]).reshape(17, 2)
            ph = phase_of(report.stroke_events, t)
            pose = ghost.pose(*ph) if ph else None
            if pose is not None:
                _draw_ghost(image, pose, report, trace.to_ref[i])
            _draw_athlete(image, kp, trace.score[i], [], DRIVE_COLOR)
            frame = _crop_resize(image, box)
            _legend(frame, title)
            cv2.putText(frame, f"Ralenti x{SLOWMO}", (frame.shape[1] - 170, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = H264Writer(path, frame.shape[1], frame.shape[0],
                                    max(6.0, info_fps / SLOWMO), crf=26)
            writer.write(frame)
        if writer is None:
            raise ValueError("aucune frame dans le coup")
        writer.close()
        return path.read_bytes()


def render_highlights(
    video: str | Path, trace: RowingTrace, report: RowingReport, ghost: GhostModel | None
) -> Highlights | None:
    """Images clés et ralentis du meilleur coup et du coup à améliorer, ou None.

    None si le fantôme est indisponible (vue non profil) ou s'il y a trop peu de coups.
    """
    if ghost is None or len(report.stroke_events) < MIN_STROKES:
        return None
    dev = stroke_deviations(report, ghost)
    valid = [i for i, d in enumerate(dev) if np.isfinite(d)]
    if len(valid) < MIN_STROKES:
        return None
    best = min(valid, key=lambda i: dev[i])
    worst = max(valid, key=lambda i: dev[i])
    out = Highlights()
    similar = dev[worst] - dev[best] < SIMILAR_GAP
    for key, idx, title in (("best", best, "Ton meilleur coup"),
                            ("worst", worst, "Coup a comparer" if similar else "Coup a ameliorer")):
        s = report.stroke_events[idx]
        catch, joint_c = _still(video, trace, report, ghost, s.t_catch + 0.02, 0.0, title)
        finish, joint_f = _still(video, trace, report, ghost, s.t_finish - 0.02, 1.0, title)
        out.images[f"{key}_catch"], out.images[f"{key}_finish"] = catch, finish
        out.clips[key] = _clip(video, trace, report, ghost, s, title)
        out.meta[key] = {"stroke": idx + 1, "deviation": dev[idx], "similar": bool(similar),
                         "catch_joint": JOINT_FR.get(joint_c, ""),
                         "finish_joint": JOINT_FR.get(joint_f, "")}
    return out


def save_highlights(hl: Highlights, out_dir: str | Path) -> Path:
    """Ecrit images (.jpg), ralentis (.mp4) et meta.json dans `out_dir`."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in hl.images.items():
        (out / f"{name}.jpg").write_bytes(data)
    for name, data in hl.clips.items():
        (out / f"{name}.mp4").write_bytes(data)
    (out / "meta.json").write_text(json.dumps(hl.meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return out


def load_highlights(in_dir: str | Path) -> Highlights:
    import json

    d = Path(in_dir)
    hl = Highlights(meta=json.loads((d / "meta.json").read_text(encoding="utf-8")))
    for key in ("best", "worst"):
        for step in ("catch", "finish"):
            hl.images[f"{key}_{step}"] = (d / f"{key}_{step}.jpg").read_bytes()
        hl.clips[key] = (d / f"{key}.mp4").read_bytes()
    return hl
