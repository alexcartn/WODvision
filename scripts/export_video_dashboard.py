"""Compose une seule video mp4 (video + squelette seul + graphique + stats en
direct) a partager, sans dependre d'un artifact HTML interactif.

Reutilise data.json (deja calcule par export_performance.py) et les clips
annotes existants (row.mp4, thruster.mp4) : pas de nouvelle inference, juste
un compositing image par image (video a gauche, panneau a droite, tous deux
synchronises sur le meme temps local).

Le panneau reste plus simple que le tableau de bord HTML (un graphique + un
histogramme de phases, pas les 4 mini-graphiques) car un export partageable
recompresse doit rester lisible a la volee.

Usage : python scripts/export_video_dashboard.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wodvision.io.probe import probe as probe_video

OUT_DIR = Path("data/runs/performance")
DATA_PATH = OUT_DIR / "data.json"
SOURCE_VIDEO = "data/raw/videoplayback.mp4"
FFMPEG = (
    "C:/Users/alexandre_carton/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
)

CANVAS_H = 900
VIDEO_H = CANVAS_H
VIDEO_W = round(VIDEO_H * 16 / 9)  # conserve le ratio 16:9 source a cette hauteur
PANEL_W = 480
CANVAS_W = VIDEO_W + PANEL_W
PAD = 24

BG = (18, 21, 27)
BORDER = (42, 47, 61)
TEXT = (232, 229, 222)
MUTED = (141, 144, 160)
COLOR = {"row": (52, 171, 110), "thruster": (229, 80, 63)}
Y_LABEL = {"row": "knee angle (deg)", "thruster": "wrist height (px)"}

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]
SKELETON_SCORE_THR = 0.3

FONT_DIR = Path("C:/Windows/Fonts")
F_LABEL = ImageFont.truetype(str(FONT_DIR / "segoeuib.ttf"), 22)
F_BIG = ImageFont.truetype(str(FONT_DIR / "consolab.ttf"), 60)
F_STAT_LABEL = ImageFont.truetype(str(FONT_DIR / "segoeui.ttf"), 15)
F_STAT_VALUE = ImageFont.truetype(str(FONT_DIR / "consolab.ttf"), 22)
F_SMALL = ImageFont.truetype(str(FONT_DIR / "segoeui.ttf"), 13)


def nearest_idx(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


RGB = tuple[int, int, int]


def mix(color: RGB, bg: RGB, alpha: float) -> RGB:
    return tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(color, bg, strict=True))


def draw_line_chart(draw, x0, y0, w, h, t_rel, y_filtered, local_t, cycle_t_rel, color) -> None:
    t_arr = np.array(t_rel)
    y_arr = np.array(y_filtered)
    y_min, y_max = float(y_arr.min()), float(y_arr.max())
    y_range = (y_max - y_min) or 1.0
    duration = float(t_arr[-1])

    def to_px(t: float, y: float) -> tuple[float, float]:
        px = x0 + (t / duration) * w
        py = y0 + h - ((y - y_min) / y_range) * h
        return px, py

    for f in (0.25, 0.5, 0.75):
        yy = y0 + h * f
        draw.line([(x0, yy), (x0 + w, yy)], fill=BORDER, width=1)

    points = [to_px(t, y) for t, y in zip(t_arr, y_arr, strict=True)]
    draw.line(points, fill=color, width=3, joint="curve")

    for ct in cycle_t_rel:
        idx = nearest_idx(t_arr, ct)
        px, py = to_px(ct, y_arr[idx])
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline=color, width=2, fill=BG)

    if 0 <= local_t <= duration:
        idx = nearest_idx(t_arr, local_t)
        px, py = to_px(local_t, y_arr[idx])
        draw.line([(px, y0), (px, y0 + h)], fill=TEXT, width=2)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=TEXT)


def draw_skeleton_only(draw, x0, y0, w, h, kp_xy, kp_score, color) -> None:
    """Squelette recentre/mis a l'echelle a chaque frame, independant de la
    position de l'athlete dans le champ (meme logique que le panneau
    "skeleton only" du tableau de bord HTML)."""
    valid = [i for i, s in enumerate(kp_score) if s > SKELETON_SCORE_THR]
    if len(valid) < 3:
        draw.text((x0 + w / 2, y0 + h / 2), "no pose", font=F_SMALL, fill=MUTED, anchor="mm")
        return

    xs = [kp_xy[i][0] for i in valid]
    ys = [kp_xy[i][1] for i in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w, bbox_h = (max_x - min_x) or 1.0, (max_y - min_y) or 1.0
    inner_pad = 24
    scale = min((w - 2 * inner_pad) / bbox_w, (h - 2 * inner_pad) / bbox_h)
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    def to_px(x: float, y: float) -> tuple[float, float]:
        return x0 + w / 2 + (x - cx) * scale, y0 + h / 2 + (y - cy) * scale

    for a, b in SKELETON_EDGES:
        if kp_score[a] > SKELETON_SCORE_THR and kp_score[b] > SKELETON_SCORE_THR:
            pa, pb = to_px(*kp_xy[a]), to_px(*kp_xy[b])
            draw.line([pa, pb], fill=color, width=5)
    for i in valid:
        px, py = to_px(*kp_xy[i])
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)


def draw_phase_bars(draw, x0, y0, w, h, reps_done, color) -> None:
    n = len(reps_done)
    if n == 0:
        draw.text((x0 + w / 2, y0 + h / 2), "no reps yet", font=F_SMALL, fill=MUTED, anchor="mm")
        return
    gap = 4
    bar_w = max(3, (w - gap * (n - 1)) / n)
    ups = [r["phase_up_s"] for r in reps_done]
    downs = [r["phase_down_s"] for r in reps_done]
    max_v = max(u + d for u, d in zip(ups, downs, strict=True)) or 1.0
    dim = mix(color, BG, 0.42)
    for i, (u, d) in enumerate(zip(ups, downs, strict=True)):
        x = x0 + i * (bar_w + gap)
        h_up = (u / max_v) * h
        h_down = (d / max_v) * h
        draw.rounded_rectangle([x, y0 + h - h_up, x + bar_w, y0 + h], radius=2, fill=color)
        draw.rounded_rectangle(
            [x, y0 + h - h_up - h_down, x + bar_w, y0 + h - h_up], radius=2, fill=dim
        )


def build_panel(
    seg: dict,
    local_t: float,
    reps_done: list[dict],
    movement_label: str,
    kp_xy: np.ndarray,
    kp_score: np.ndarray,
) -> np.ndarray:
    img = Image.new("RGB", (PANEL_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)
    color = COLOR[seg["movement"]]

    draw.ellipse([PAD, PAD + 6, PAD + 12, PAD + 18], fill=color)
    draw.text((PAD + 22, PAD), movement_label.upper(), font=F_LABEL, fill=TEXT)

    n_total = len(seg["reps"])
    word = "strokes" if seg["movement"] == "row" else "reps"
    draw.text((PAD, PAD + 36), f"{len(reps_done)}", font=F_BIG, fill=color)
    draw.text((PAD + 90, PAD + 36 + 32), f"/ {n_total} {word}", font=F_STAT_LABEL, fill=MUTED)

    chart_y = PAD + 122
    chart_h = 170
    draw_line_chart(
        draw, PAD, chart_y, PANEL_W - 2 * PAD, chart_h,
        seg["t_rel"], seg["y_filtered"], local_t, seg["cycle_t_rel"], color,
    )
    draw.text((PAD, chart_y + chart_h + 8), Y_LABEL[seg["movement"]], font=F_SMALL, fill=MUTED)

    skeleton_y = chart_y + chart_h + 32
    skeleton_h = 220
    draw.text((PAD, skeleton_y), "Skeleton only", font=F_STAT_LABEL, fill=MUTED)
    draw_skeleton_only(
        draw, PAD, skeleton_y + 20, PANEL_W - 2 * PAD, skeleton_h - 20, kp_xy, kp_score, color
    )

    if reps_done:
        avg_speed = mean([r["peak_speed_up_px_s"] for r in reps_done])
        angle_key = "knee_angle_catch_deg" if seg["movement"] == "row" else "hip_knee_depth_px"
        avg_angle = mean([r[angle_key] for r in reps_done])
        done_tops = seg["cycle_t_rel"][: len(reps_done) + 1]
        intervals = [b - a for a, b in zip(done_tops[:-1], done_tops[1:], strict=True)]
        cadence = 60 / mean(intervals) if len(intervals) >= 1 else float("nan")
    else:
        avg_speed = avg_angle = cadence = float("nan")

    def stat_row(y: int, label: str, value_str: str) -> None:
        draw.text((PAD, y), label, font=F_STAT_LABEL, fill=MUTED)
        draw.text((PANEL_W - PAD, y - 3), value_str, font=F_STAT_VALUE, fill=TEXT, anchor="ra")

    stats_y = skeleton_y + skeleton_h + 24
    stat_row(stats_y, "Cadence so far", f"{cadence:.0f} /min" if cadence == cadence else "—")
    stat_row(
        stats_y + 32, "Peak speed avg", f"{avg_speed:.0f} px/s" if avg_speed == avg_speed else "—"
    )
    angle_label = "Knee@catch avg" if seg["movement"] == "row" else "Depth avg"
    unit = "°" if seg["movement"] == "row" else "px"
    stat_row(stats_y + 64, angle_label, f"{avg_angle:.1f}{unit}" if avg_angle == avg_angle else "—")

    bars_y = stats_y + 100
    draw.text((PAD, bars_y), "Phase duration per rep", font=F_STAT_LABEL, fill=MUTED)
    draw_phase_bars(draw, PAD, bars_y + 22, PANEL_W - 2 * PAD, 110, reps_done, color)

    prog_y = CANVAS_H - 24
    draw.line([(PAD, prog_y), (PANEL_W - PAD, prog_y)], fill=BORDER, width=4)
    frac = max(0.0, min(1.0, local_t / seg["duration_s"]))
    draw.line([(PAD, prog_y), (PAD + frac * (PANEL_W - 2 * PAD), prog_y)], fill=color, width=4)

    return np.array(img)


def process_segment_video(seg: dict, fps: float) -> Path:
    src_path = OUT_DIR / f"{seg['name']}.mp4"
    cap = cv2.VideoCapture(str(src_path))
    out_path = OUT_DIR / f"{seg['name']}_dashboard_raw.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_W, CANVAS_H)
    )

    movement_label = "Row" if seg["movement"] == "row" else "Thruster"
    kp_t_rel = np.array(seg["kp_t_rel"])
    kp_xy_all = np.array(seg["kp_xy"])
    kp_score_all = np.array(seg["kp_score"])

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        local_t = frame_idx / fps
        reps_done = [r for r in seg["reps"] if r["t_top_after"] <= local_t]
        kp_idx = nearest_idx(kp_t_rel, local_t)

        video_frame = cv2.resize(frame, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        panel_rgb = build_panel(
            seg, local_t, reps_done, movement_label, kp_xy_all[kp_idx], kp_score_all[kp_idx]
        )
        panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)

        canvas = np.hstack([video_frame, panel_bgr])
        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    return out_path


def compress_video(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-an", str(dst),
        ],
        check=True,
    )


def concat_videos(parts: list[Path], dst: Path) -> None:
    list_file = OUT_DIR / "concat_dashboard_list.txt"
    lines = "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts)
    list_file.write_text(lines, encoding="utf-8")
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-an", str(dst),
        ],
        check=True,
    )


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    fps = probe_video(SOURCE_VIDEO).fps_nominal or 24.0

    compressed = []
    for seg in data:
        raw_path = process_segment_video(seg, fps)
        compressed_path = OUT_DIR / f"{seg['name']}_dashboard_c.mp4"
        compress_video(raw_path, compressed_path)
        raw_path.unlink()
        compressed.append(compressed_path)
        print(f"[{seg['name']}] done -> {compressed_path}")

    final_path = OUT_DIR / "wodvision_share.mp4"
    concat_videos(compressed, final_path)
    for p in compressed:
        p.unlink()
    print(f"video finale : {final_path} ({final_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
