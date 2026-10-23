"""Pipeline aviron : video -> trace d'inference (cache) -> analyse -> sorties.

Separation inference / analyse (CLAUDE.md section 4) : `run_inference` est la
partie lourde (detection, pose, mouvement de camera) ; `analyze_trace` ne
depend d'aucun modele et rejoue depuis le cache.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from wodvision.analysis.rowing.pm5_sync import couple_pm5
from wodvision.analysis.rowing.report import RowingReport, analyze_keypoints
from wodvision.config import Config
from wodvision.io.cache import RowingTrace, cache_key, load_trace, save_trace
from wodvision.io.frames import iter_frames
from wodvision.io.probe import probe
from wodvision.pm5.session import Pm5Session
from wodvision.stabilization.camera_motion import (
    CameraMotionEstimator,
    apply_transform,
    motion_amplitude_px,
)
from wodvision.tracking.athlete_selector import bbox_area, bbox_center
from wodvision.tracking.sticky_target import StickyTarget
from wodvision.tracking.tracker import TrackedBox, Tracker

logger = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


class Detector(Protocol):
    def detect(self, image: np.ndarray): ...


class Pose(Protocol):
    def infer(self, image: np.ndarray, bbox: tuple[float, float, float, float]): ...


def apply_rotation(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Applique la rotation de metadonnee (degres horaires, voir io/probe.py)."""
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(image, code[rotation_deg]) if rotation_deg in code else image


def _first_bbox(tracked: list[TrackedBox], click_xy: tuple[float, float] | None):
    if click_xy is not None:
        inside = [tb for tb in tracked if tb.bbox_xyxy[0] <= click_xy[0] <= tb.bbox_xyxy[2]
                  and tb.bbox_xyxy[1] <= click_xy[1] <= tb.bbox_xyxy[3]]
        pool = inside or tracked
        return min(pool, key=lambda tb: math.dist(bbox_center(tb.bbox_xyxy), click_xy)).bbox_xyxy
    return max(tracked, key=lambda tb: bbox_area(tb.bbox_xyxy)).bbox_xyxy


def bbox_from_keypoints(
    xy: np.ndarray, score: np.ndarray, margin: float, shape: tuple[int, int], min_conf: float,
    min_points: int = 6,
) -> tuple[float, float, float, float] | None:
    """Bbox serree autour des keypoints fiables (+ marge), None si trop peu de points."""
    ok = (score > min_conf) & np.all(np.isfinite(xy), axis=1)
    if ok.sum() < min_points:
        return None
    x1, y1 = xy[ok].min(axis=0)
    x2, y2 = xy[ok].max(axis=0)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = shape
    return (float(max(0, x1 - mx)), float(max(0, y1 - my)),
            float(min(w - 1, x2 + mx)), float(min(h - 1, y2 + my)))


def build_default_models(mode: str = "balanced"):
    """Detecteur + pose rtmlib (imports lourds differes)."""
    from wodvision.detection.person import PersonDetector
    from wodvision.pose.rtmlib_backend import RTMLibPoseEstimator, select_device

    device = select_device()
    return PersonDetector(mode=mode, device=device), RTMLibPoseEstimator(mode=mode, device=device)


def run_inference(
    video: str | Path,
    cfg: Config,
    detector: Detector,
    pose: Pose,
    click_xy: tuple[float, float] | None = None,
    max_seconds: float | None = None,
    progress: Progress | None = None,
    start_s: float = 0.0,
) -> RowingTrace:
    """Partie lourde : athlete -> pose par frame, avec stabilisation du mouvement de camera."""
    info = probe(video)
    fps = info.fps_nominal or 30.0
    stride = max(1, round(fps / cfg.rowing.analysis_fps))
    end = start_s + max_seconds if max_seconds else float("inf")
    duration = min(info.duration_s, end) - start_s
    detect_every = max(1, cfg.rowing.detect_every_n_frames)
    tracker = Tracker(frame_rate=fps / stride / detect_every)
    cam_cfg = cfg.camera_motion
    estimator = CameraMotionEstimator(
        cam_cfg.analysis_width_px, cam_cfg.max_features, cam_cfg.min_inliers,
        cam_cfg.bbox_margin_fraction,
    )
    sticky: StickyTarget | None = None

    t_l, xy_l, sc_l, bb_l, tr_l, rel_l = [], [], [], [], [], []
    width = height = 0
    prev_xy, prev_score = np.full((17, 2), np.nan), np.zeros(17)
    min_conf = cfg.signal.min_keypoint_confidence
    n_proc = 0
    for i, (t_s, image) in enumerate(iter_frames(video, start_s, end)):
        if i % stride:
            continue
        image = apply_rotation(image, info.rotation_deg)
        height, width = image.shape[:2]
        bbox = None
        if sticky is not None and n_proc % detect_every:
            bbox = bbox_from_keypoints(
                prev_xy, prev_score, cfg.rowing.bbox_from_pose_margin, (height, width), min_conf
            )
            if bbox is not None:
                sticky.last_bbox = bbox
        if bbox is None:
            tracked = tracker.update(detector.detect(image))
            if sticky is None and tracked:
                sticky = StickyTarget(last_bbox=_first_bbox(tracked, click_xy))
            if sticky is not None:
                bbox = sticky.update(tracked)
        n_proc += 1

        if bbox is not None:
            kp = pose.infer(image, bbox)
            xy, score = np.asarray(kp.xy, float), np.asarray(kp.score, float)
        else:
            xy, score = np.full((17, 2), np.nan), np.zeros(17)
        prev_xy, prev_score = xy, score

        if cam_cfg.enabled:
            step = estimator.update(image, bbox if bbox is not None else
                                    (sticky.last_bbox if sticky else None))
            to_ref, reliable = step.to_ref, step.reliable
        else:
            to_ref, reliable = np.eye(3), True

        t_l.append(t_s)
        xy_l.append(xy)
        sc_l.append(score)
        bb_l.append(bbox if bbox is not None else (np.nan,) * 4)
        tr_l.append(to_ref)
        rel_l.append(reliable)
        if progress and duration > 0 and len(t_l) % 10 == 0:
            progress(min(0.99, (t_s - start_s) / duration), f"Analyse de la pose : {t_s:.1f} s")

    if len(t_l) < 10 or not np.any(np.stack(sc_l) > cfg.signal.min_keypoint_confidence):
        raise ValueError("Athlete non detecte : verifier la video (rameur visible, vue de profil).")

    xy_raw = np.stack(xy_l)
    to_ref = np.stack(tr_l)
    xy_stab = np.stack([apply_transform(m, p) for m, p in zip(to_ref, xy_raw, strict=True)])
    if progress:
        progress(1.0, "Pose terminee")
    return RowingTrace(
        t_s=np.array(t_l), xy_raw=xy_raw, xy_stab=xy_stab, score=np.stack(sc_l),
        bbox=np.array(bb_l, dtype=float), to_ref=to_ref, cam_reliable=np.array(rel_l),
        meta={"video": str(video), "width": width, "height": height, "fps": fps,
              "stride": stride, "rotation_deg": info.rotation_deg, "vfr": info.is_vfr,
              "camera_motion_px": motion_amplitude_px(to_ref, width, height)},
    )


def analyze_trace(
    trace: RowingTrace, cfg: Config, side: str | None = None, pm5: Pm5Session | None = None,
    pm5_lag: int | None = None, pm5_phase: str | None = None,
) -> RowingReport:
    """Analyse depuis la trace. Repli sur les keypoints bruts si la stabilisation derive.

    Si `pm5` est fourni, les coups video et PM5 sont apparies (voir `pm5_sync`) : validation
    de la detection video, courbe de force et constats associes.
    """
    motion = float(trace.meta.get("camera_motion_px", 0.0))
    limit = float(max(trace.meta.get("width", 0), trace.meta.get("height", 0)))
    unreliable = limit > 0 and motion > limit
    xy = trace.xy_raw if unreliable else trace.xy_stab
    report = analyze_keypoints(trace.t_s, xy, trace.score, cfg, camera_motion_px=motion, side=side)
    if unreliable:
        report.warnings.append(
            f"Mouvement de camera estime invraisemblable ({motion:.0f} px) : stabilisation "
            "ignoree, analyse sur les keypoints bruts (valable si la camera est fixe)."
        )
    if pm5 is not None:
        coupling = couple_pm5(report, pm5, cfg, lag=pm5_lag, phase=pm5_phase)
        report.pm5 = coupling
        report.findings.extend(coupling.findings)
        report.warnings.extend(coupling.warnings)
    return report


def get_trace(
    video: str | Path, cfg: Config, mode: str = "balanced", click_xy=None,
    max_seconds: float | None = None, cache_dir: str | Path = "data/cache",
    detector: Detector | None = None, pose: Pose | None = None, progress: Progress | None = None,
    start_s: float = 0.0,
) -> RowingTrace:
    """Charge la trace depuis le cache, ou lance l'inference puis la met en cache."""
    params = {"mode": mode, "click": click_xy, "max_seconds": max_seconds,
              "start_s": start_s, "analysis_fps": cfg.rowing.analysis_fps,
              "detect_every": cfg.rowing.detect_every_n_frames,
              "camera": cfg.camera_motion.model_dump(),
              "backend": cfg.pose.backend}
    cache_path = Path(cache_dir) / f"rowing_{cache_key(video, params)}.npz"
    if cache_path.exists():
        logger.info("trace chargee depuis le cache : %s", cache_path)
        return load_trace(cache_path)
    if detector is None or pose is None:
        detector, pose = build_default_models(mode)
    trace = run_inference(video, cfg, detector, pose, click_xy, max_seconds, progress, start_s)
    save_trace(trace, cache_path)
    return trace


def export_run(report: RowingReport, out_dir: str | Path, cfg: Config) -> Path:
    """Ecrit strokes.csv, summary.json et la config utilisee dans `out_dir`."""
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in report.strokes:
        row = s.to_dict()
        if report.pm5 is not None and s.index in report.pm5.rows:
            extra = dict(report.pm5.rows[s.index])
            curve = extra.pop("force_curve", None)
            row.update(extra)
            row["force_curve"] = ";".join(f"{v:g}" for v in curve) if curve is not None else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "strokes.csv", index=False)
    summary = report.summary_dict()
    summary["findings"] = [
        {**{k: getattr(f, k) for k in ("code", "title", "metric", "comparator", "threshold",
                                        "median_value", "n_flagged", "n_valid", "advice",
                                        "reliability")},
         "explanation": f.explain()}
        for f in report.findings
    ]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                      encoding="utf-8")
    (out / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return out


def new_run_dir(video: str | Path, root: str | Path = "data/runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / Path(video).stem / stamp
