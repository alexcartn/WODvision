"""Exporte les clips annotes + donnees de signal pour le dashboard de performance.

Ad hoc (hors perimetre Phase 1) : suit UN athlete deja identifie par clic sur
deux segments d'une meme video (row, thruster), dessine uniquement son
squelette (aucune boite pour les autres personnes detectees : avec 9
athletes dans le champ, les boites supplementaires rendaient la video
illisible), et exporte :

- un clip mp4 annote par segment (squelette seul, pas de boite),
- un signal brut/filtre + les cycles detectes par segment (pour les
  graphiques de performance),
- les keypoints bruts par frame (pour un panneau "squelette seul" anime,
  independant de la video, recentre/mis a l'echelle a chaque frame),
- un JSON combinant tout, avec des timestamps relatifs a une timeline unique
  (segment 1 puis segment 2 concatenes) pour tout piloter en synchro avec la
  video combinee.

Usage : python scripts/export_performance.py
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rtmlib import draw_skeleton

from wodvision.analysis.angles import (
    L_WRIST,
    R_WRIST,
    elbow_angle_deg,
    knee_angle_deg,
    pick_visible_side,
    shin_angle_deg,
    trunk_lean_deg,
)
from wodvision.analysis.cycle_counter import count_cycles, find_bottoms_between
from wodvision.config import load_config
from wodvision.detection.person import PersonDetector
from wodvision.io.frames import iter_frames
from wodvision.io.probe import probe as probe_video
from wodvision.pose.rtmlib_backend import RTMLibPoseEstimator, select_device
from wodvision.signal.filtering import butterworth_lowpass
from wodvision.signal.kinematics import velocity
from wodvision.signal.resampling import resample_uniform
from wodvision.tracking.athlete_selector import iou
from wodvision.tracking.sticky_target import StickyTarget
from wodvision.tracking.tracker import Tracker

VIDEO_PATH = "data/raw/videoplayback.mp4"
OUT_DIR = Path("data/runs/performance")
FFMPEG = (
    "C:/Users/alexandre_carton/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
)


@dataclass(slots=True)
class SegmentSpec:
    name: str
    movement: str  # "row" | "thruster"
    start_s: float
    duration_s: float
    click_xy: tuple[float, float]
    click_size: float = 140.0


SEGMENTS = [
    SegmentSpec(
        name="row", movement="row", start_s=65.0, duration_s=20.0, click_xy=(315.0, 200.0)
    ),
    SegmentSpec(
        name="thruster",
        movement="thruster",
        start_s=225.0,
        duration_s=20.0,
        click_xy=(355.0, 160.0),
        click_size=180.0,
    ),
]


def click_bbox_of(spec: SegmentSpec) -> tuple[float, float, float, float]:
    cx, cy = spec.click_xy
    half = spec.click_size / 2
    return (cx - half, cy - half, cx + half, cy + half)


def compute_rep_details(
    movement: str,
    t_obs: np.ndarray,
    xy_obs: np.ndarray,
    side: str,
    tops: list[float],
    bottoms: list[float],
) -> list[dict]:
    """Decompose chaque cycle en phases montante/descendante + angles cles.

    Unites en pixels/degres (proxy non calibre : pas de detection barre/poignee
    ni de calibration metrique dans ce script ad hoc) :
    - phase_down_s / phase_up_s : duree top->bottom et bottom->top suivant ;
    - peak_speed_up_px_s : vitesse 2D max du poignet pendant la phase montante
      (proxy de vitesse de barre/poignee, en px/s, PAS m/s) ;
    - path_deviation_px : amplitude du poignet sur l'axe "hors trajectoire"
      pendant la phase montante — proxy de wobble poignee (row, axe vertical
      pendant une traction censee etre horizontale) ou d'efficacite de
      trajectoire de barre (thruster, axe horizontal pendant une montee
      censee etre verticale). Reutilise le poignet comme proxy de la
      barre/poignee, comme peak_speed_up_px_s ;
    - lockout_wobble_px (thruster) : ecart-type de la position du poignet
      dans une fenetre de 0.3s autour du lockout, proxy de stabilite en haut ;
    - angles articulaires au bas et au haut de chaque cycle.
    """
    wrist = L_WRIST if side == "left" else R_WRIST
    hip_idx, knee_idx = (11, 13) if side == "left" else (12, 14)

    wrist_x = xy_obs[:, wrist, 0]
    wrist_y = xy_obs[:, wrist, 1]
    vx = velocity(wrist_x, t_obs)
    vy = velocity(wrist_y, t_obs)
    speed = np.hypot(vx, vy)

    def nearest_idx(t: float) -> int:
        return int(np.argmin(np.abs(t_obs - t)))

    reps = []
    for i, t_bottom in enumerate(bottoms):
        t_top_before, t_top_after = tops[i], tops[i + 1]
        idx_bottom = nearest_idx(t_bottom)
        idx_top_after = nearest_idx(t_top_after)
        xy_bottom, xy_top_after = xy_obs[idx_bottom], xy_obs[idx_top_after]

        window_up = (t_obs >= t_bottom) & (t_obs <= t_top_after)
        peak_speed_up = float(np.max(speed[window_up])) if window_up.any() else float("nan")

        off_axis = wrist_y[window_up] if movement == "row" else wrist_x[window_up]
        if window_up.any():
            path_deviation = float(np.max(off_axis) - np.min(off_axis))
        else:
            path_deviation = float("nan")

        rep = {
            "t_top_before": round(t_top_before, 3),
            "t_bottom": round(t_bottom, 3),
            "t_top_after": round(t_top_after, 3),
            "phase_down_s": round(t_bottom - t_top_before, 3),
            "phase_up_s": round(t_top_after - t_bottom, 3),
            "peak_speed_up_px_s": round(peak_speed_up, 1),
            "path_deviation_px": round(path_deviation, 1),
            "trunk_lean_bottom_deg": round(trunk_lean_deg(xy_bottom, side), 1),
        }
        if movement == "row":
            rep["knee_angle_catch_deg"] = round(knee_angle_deg(xy_bottom, side), 1)
            rep["knee_angle_finish_deg"] = round(knee_angle_deg(xy_top_after, side), 1)
            rep["trunk_lean_finish_deg"] = round(trunk_lean_deg(xy_top_after, side), 1)
            rep["shin_angle_catch_deg"] = round(shin_angle_deg(xy_bottom, side), 1)
        else:
            hip_knee_depth = float(xy_bottom[hip_idx, 1] - xy_bottom[knee_idx, 1])
            rep["hip_knee_depth_px"] = round(hip_knee_depth, 1)
            rep["elbow_angle_bottom_deg"] = round(elbow_angle_deg(xy_bottom, side), 1)
            rep["elbow_angle_top_deg"] = round(elbow_angle_deg(xy_top_after, side), 1)

            window_lockout = (t_obs >= t_top_after - 0.15) & (t_obs <= t_top_after + 0.15)
            if window_lockout.sum() >= 2:
                wx, wy = wrist_x[window_lockout], wrist_y[window_lockout]
                offset = np.hypot(wx - wx.mean(), wy - wy.mean())
                lockout_wobble = float(offset.std())
            else:
                lockout_wobble = float("nan")
            rep["lockout_wobble_px"] = round(lockout_wobble, 2)
        reps.append(rep)
    return reps


def process_segment(
    spec: SegmentSpec,
    detector: PersonDetector,
    pose_estimator: RTMLibPoseEstimator,
    fps: float,
    min_conf: float,
) -> dict:
    tracker = Tracker(frame_rate=fps)
    click_bbox = click_bbox_of(spec)

    raw_path = OUT_DIR / f"{spec.name}_raw.mp4"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    sticky: StickyTarget | None = None
    t_obs, xy_obs, score_obs = [], [], []
    n_frames = 0

    for t_s, image in iter_frames(VIDEO_PATH, spec.start_s, spec.start_s + spec.duration_s):
        tracked = tracker.update(detector.detect(image))
        n_frames += 1

        if sticky is None:
            best = max(tracked, key=lambda tb: iou(tb.bbox_xyxy, click_bbox), default=None)
            if best is not None and iou(best.bbox_xyxy, click_bbox) > 0:
                sticky = StickyTarget(last_bbox=best.bbox_xyxy, current_track_id=best.track_id)

        target_bbox = sticky.update(tracked) if sticky is not None else None

        annotated = image.copy()
        if target_bbox is not None:
            keypoints = pose_estimator.infer(image, target_bbox)
            annotated = draw_skeleton(
                annotated, keypoints.xy[None, :, :], keypoints.score[None, :], kpt_thr=0.3
            )
            t_obs.append(t_s)
            xy_obs.append(keypoints.xy)
            score_obs.append(keypoints.score)

        if writer is None:
            height, width = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        writer.write(annotated)

    if writer is not None:
        writer.release()

    print(f"[{spec.name}] {n_frames} frames, athlete visible sur {len(t_obs)} frames")

    t_obs_arr = np.array(t_obs)
    xy_obs_arr = np.stack(xy_obs)
    score_obs_arr = np.stack(score_obs)
    side = pick_visible_side(score_obs_arr)

    if spec.movement == "thruster":
        wrist = L_WRIST if side == "left" else R_WRIST
        y_raw = -xy_obs_arr[:, wrist, 1]
        valid = score_obs_arr[:, wrist] > min_conf
        y_label = "hauteur du poignet (px)"
    else:
        y_raw = np.array([knee_angle_deg(xy, side=side) for xy in xy_obs_arr])
        hip_idx, knee_idx, ankle_idx = (11, 13, 15) if side == "left" else (12, 14, 16)
        valid = np.min(score_obs_arr[:, [hip_idx, knee_idx, ankle_idx]], axis=1) > min_conf
        y_label = "angle du genou (deg)"

    resampled = resample_uniform(t_obs_arr, y_raw, valid, target_fps=fps, max_gap_frames=5)
    y_filtered = butterworth_lowpass(resampled.y, fps=fps, cutoff_hz=6.0, order=4)
    min_distance_s = 1.0 if spec.movement == "thruster" else 0.6
    cycle_times_abs = count_cycles(
        resampled.t_s, y_filtered, prominence_fraction=0.3, min_distance_s=min_distance_s
    )
    bottom_times_abs = find_bottoms_between(resampled.t_s, y_filtered, cycle_times_abs)
    reps = compute_rep_details(
        spec.movement,
        t_obs_arr - spec.start_s,
        xy_obs_arr,
        side,
        [t - spec.start_s for t in cycle_times_abs],
        [t - spec.start_s for t in bottom_times_abs],
    )

    return {
        "name": spec.name,
        "movement": spec.movement,
        "start_s": spec.start_s,
        "duration_s": spec.duration_s,
        "side": side,
        "y_label": y_label,
        "t_rel": (resampled.t_s - spec.start_s).round(3).tolist(),
        "y_filtered": y_filtered.round(2).tolist(),
        "cycle_t_rel": [round(t - spec.start_s, 3) for t in cycle_times_abs],
        "reps": reps,
        "kp_t_rel": (t_obs_arr - spec.start_s).round(3).tolist(),
        "kp_xy": xy_obs_arr.round(1).tolist(),
        "kp_score": score_obs_arr.round(2).tolist(),
        "raw_video_path": str(raw_path),
    }


def compress_video(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
            "-an", str(dst),
        ],
        check=True,
    )


def concat_videos(parts: list[Path], dst: Path) -> None:
    list_file = OUT_DIR / "concat_list.txt"
    lines = "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts)
    list_file.write_text(lines, encoding="utf-8")
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-an", str(dst),
        ],
        check=True,
    )


def main() -> None:
    cfg = load_config()
    info = probe_video(VIDEO_PATH)
    fps = info.fps_nominal or 30.0
    device = select_device()
    print(f"device={device} fps={fps}")

    detector = PersonDetector(mode="balanced", device=device)
    pose_estimator = RTMLibPoseEstimator(mode="balanced", device=device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    compressed_paths = []
    offset_s = 0.0
    for spec in SEGMENTS:
        min_conf = cfg.signal.min_keypoint_confidence
        result = process_segment(spec, detector, pose_estimator, fps, min_conf)
        result["combined_offset_s"] = offset_s
        offset_s += spec.duration_s
        results.append(result)

        raw_path = Path(result["raw_video_path"])
        compressed_path = OUT_DIR / f"{spec.name}.mp4"
        compress_video(raw_path, compressed_path)
        compressed_paths.append(compressed_path)
        raw_path.unlink()

    combined_path = OUT_DIR / "combined.mp4"
    concat_videos(compressed_paths, combined_path)

    for result in results:
        del result["raw_video_path"]

    (OUT_DIR / "data.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"video combinee : {combined_path} ({combined_path.stat().st_size / 1e6:.1f} MB)")
    print(f"donnees : {OUT_DIR / 'data.json'}")


if __name__ == "__main__":
    main()
