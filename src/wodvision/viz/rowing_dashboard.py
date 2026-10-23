"""Video "tableau de bord" : athlete annote a gauche, panneau a droite.

Panneau (synchronise sur la video) : compteur de coups, courbe de la poignee
avec curseur, squelette seul dans le repere de l'ergometre (rail horizontal,
volant a gauche, echelle fixe), indicateurs du dernier coup et alertes de
technique le concernant.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wodvision.analysis.rowing.force_curve import normalized_curve
from wodvision.analysis.rowing.ghost import GHOST_EDGES, GhostModel, phase_of
from wodvision.analysis.rowing.report import RowingReport
from wodvision.io.cache import RowingTrace
from wodvision.io.frames import iter_frames
from wodvision.io.video_writer import H264Writer
from wodvision.rowing_pipeline import apply_rotation

RGB = tuple[int, int, int]
BG: RGB = (18, 21, 27)
BORDER: RGB = (42, 47, 61)
TEXT: RGB = (232, 229, 222)
MUTED: RGB = (141, 144, 160)
DRIVE_COLOR: RGB = (52, 171, 110)
RECOVERY_COLOR: RGB = (74, 144, 226)
ALERT_COLOR: RGB = (229, 120, 63)
OK_COLOR: RGB = (52, 171, 110)

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
KP_THR = 0.3
PANEL_W = 460
CANVAS_H = 720
GHOST_COLOR: RGB = (255, 214, 102)
PM5_CANVAS_H = 900  # panneau plus haut quand un PM5 est couple (lignes + courbe de force)
PAD = 22

_FONT_PATH = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
_FONT_BOLD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_BOLD if bold else _FONT_PATH, size)


def mach_series_shoulder_hip(report: RowingReport) -> np.ndarray:
    """Vecteur hanche -> epaule (s, h) sur la grille des series filtrees."""
    mach = {k: report.frame.to_machine(report.series.pts[k]) for k in ("shoulder", "hip")}
    return mach["shoulder"] - mach["hip"]


def _phase_at(report: RowingReport, t: float) -> str:
    for s in report.stroke_events:
        if s.t_catch <= t < s.t_finish:
            return "drive"
        if s.t_finish <= t < s.t_next_catch:
            return "recovery"
    return "none"


class _Panel:
    def __init__(self, report: RowingReport, trace: RowingTrace, ghost: GhostModel | None = None):
        assert report.series is not None and report.frame is not None
        self.report = report
        series = report.series
        self.t_series = series.t_s
        self.handle_s = report.frame.to_machine(series.pts["wrist"])[:, 0]
        # Squelette seul : coordonnees machine de tous les keypoints, echelle globale fixe.
        mach = report.frame.to_machine(trace.xy_stab)  # (n, 17, 2) -> (s, h)
        self.mach = mach
        valid = trace.score > KP_THR
        pts = mach[valid]
        lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
        self.center = (lo + hi) / 2
        self.extent = np.maximum(hi - lo, 1.0)
        self.trace = trace
        self.ghost = ghost
        # Inclinaison reelle du buste (deg, + = recul) pour l'ecart au fantome en direct.
        sh_hip = mach_series_shoulder_hip(report)
        self.lean_deg = np.degrees(np.arctan2(sh_hip[:, 0], sh_hip[:, 1]))
        pm5 = report.pm5
        self.height = PM5_CANVAS_H if (pm5 is not None and pm5.n_matched) else CANVAS_H
        self.f_label, self.f_big = _font(20, True), _font(56, True)
        self.f_stat, self.f_val, self.f_small = _font(14), _font(20, True), _font(12)

    def _chart(self, draw, x0, y0, w, h, t, color):
        y = self.handle_s
        lo, hi = float(y.min()), float(y.max())
        rng = (hi - lo) or 1.0
        t0, t1 = float(self.t_series[0]), float(self.t_series[-1])

        def pt(tt, yy):
            return x0 + (tt - t0) / (t1 - t0) * w, y0 + h - (yy - lo) / rng * h

        for f in (0.25, 0.5, 0.75):
            draw.line([(x0, y0 + h * f), (x0 + w, y0 + h * f)], fill=BORDER, width=1)
        draw.line([pt(a, b) for a, b in zip(self.t_series, y, strict=True)],
                  fill=MUTED, width=2)
        for s in self.report.stroke_events:
            for tt, col in ((s.t_catch, RECOVERY_COLOR), (s.t_finish, DRIVE_COLOR)):
                px, py = pt(tt, float(np.interp(tt, self.t_series, y)))
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)
        px, py = pt(min(max(t, t0), t1), float(np.interp(t, self.t_series, y)))
        draw.line([(px, y0), (px, y0 + h)], fill=TEXT, width=2)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color)

    def _force_curves(self, draw, x0, y0, w, h, done, color):
        """Courbe de force du dernier coup (trait plein) et des 5 precedents (gris)."""
        pm5 = self.report.pm5
        curves = []
        for s in done[-6:]:
            row = pm5.rows.get(s.index) if pm5 else None
            raw = row.get("force_curve") if row else None
            c = normalized_curve(raw, 60) if raw is not None else None
            if c is not None:
                curves.append(c)
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline=BORDER)
        if not curves:
            draw.text((x0 + w / 2, y0 + h / 2), "pas de courbe de force", font=self.f_small,
                      fill=MUTED, anchor="mm")
            return
        top = max(float(c.max()) for c in curves)
        for k, c in enumerate(curves):
            pts = [(x0 + i / (len(c) - 1) * w, y0 + h - 4 - (v / top) * (h - 10))
                   for i, v in enumerate(c)]
            last = k == len(curves) - 1
            draw.line(pts, fill=color if last else BORDER, width=3 if last else 1)
        draw.text((x0, y0 + h + 2), "courbe de force (position de la poignee)",
                  font=self.f_small, fill=MUTED)

    def _skeleton(self, draw, x0, y0, w, h, idx, color, ghost_pose=None):
        kp, sc = self.mach[idx], self.trace.score[idx]
        scale = min((w - 40) / self.extent[0], (h - 40) / self.extent[1])

        def pt(p):
            return (x0 + w / 2 + (p[0] - self.center[0]) * scale,
                    y0 + h / 2 - (p[1] - self.center[1]) * scale)

        if ghost_pose is not None:  # fantome sous le squelette reel
            for a, b in GHOST_EDGES:
                draw.line([pt(ghost_pose[a]), pt(ghost_pose[b])], fill=GHOST_COLOR, width=7)
            for p in ghost_pose.values():
                px, py = pt(p)
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=GHOST_COLOR)

        for a, b in EDGES:
            if sc[a] > KP_THR and sc[b] > KP_THR:
                draw.line([pt(kp[a]), pt(kp[b])], fill=color, width=5)
        for i in range(17):
            if sc[i] > KP_THR:
                px, py = pt(kp[i])
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
        draw.text((x0 + 4, y0 + h - 16), "volant", font=self.f_small, fill=MUTED)

    def render(self, t: float, idx: int) -> np.ndarray:
        r = self.report
        img = Image.new("RGB", (PANEL_W, self.height), BG)
        draw = ImageDraw.Draw(img)
        phase = _phase_at(r, t)
        color = {"drive": DRIVE_COLOR, "recovery": RECOVERY_COLOR, "none": MUTED}[phase]
        label = {"drive": "DRIVE", "recovery": "RECUPERATION", "none": "-"}[phase]

        done = [s for s in r.strokes if s.t_next_catch <= t]
        last = done[-1] if done else None
        draw.text((PAD, PAD - 4), "AVIRON", font=self.f_label, fill=TEXT)
        if r.mode != "full":
            tag = "MODE DEGRADE" if r.mode == "degraded" else "MODE MINIMAL"
            draw.text((PAD + 100, PAD), tag, font=self.f_small, fill=ALERT_COLOR)
        draw.text((PANEL_W - PAD, PAD - 2), label, font=self.f_label, fill=color, anchor="ra")
        draw.text((PAD, PAD + 26), f"{len(done)}", font=self.f_big, fill=color)
        draw.text((PAD + 96, PAD + 62), f"/ {r.n_strokes} coups", font=self.f_stat, fill=MUTED)

        cy = PAD + 104
        self._chart(draw, PAD, cy, PANEL_W - 2 * PAD, 100, t, color)
        draw.text((PAD, cy + 104), "position poignee (le long du rail)", font=self.f_small,
                  fill=MUTED)

        sy = cy + 126
        phase_info = phase_of(r.stroke_events, t) if self.ghost is not None else None
        ghost_pose = self.ghost.pose(*phase_info) if phase_info else None
        title = "Squelette seul (repere ergometre)"
        draw.text((PAD, sy), title, font=self.f_stat, fill=MUTED)
        if self.ghost is not None:
            draw.text((PANEL_W - PAD, sy + 2), "fantome = reference", font=self.f_small,
                      fill=GHOST_COLOR, anchor="ra")
        self._skeleton(draw, PAD, sy + 20, PANEL_W - 2 * PAD, 170, idx, color, ghost_pose)

        def row(y, name, value):
            draw.text((PAD, y), name, font=self.f_stat, fill=MUTED)
            draw.text((PANEL_W - PAD, y - 2), value, font=self.f_val, fill=TEXT, anchor="ra")

        def fmt(v, unit="", nd=0):
            return f"{v:.{nd}f}{unit}".replace("-0.0", "0.0") if v == v else "-"

        ty = sy + 200
        nan = float("nan")
        row(ty, "Cadence (dernier coup)", fmt(last.stroke_rate_spm if last else nan, " c/min"))
        row(ty + 26, "Recup / drive", fmt(last.recovery_drive_ratio if last else nan, " : 1", 1))
        if r.mode == "full":
            row(ty + 52, "Genou au catch", fmt(last.knee_angle_catch if last else nan, "°"))
            row(ty + 78, "Buste au finish", fmt(last.trunk_lean_finish if last else nan, "°"))
            row(ty + 104, "Angle chaine / rail", fmt(last.chain_angle_deg if last else nan, "°", 1))
        else:
            row(ty + 52, "Regularite (CV periode)", fmt(r.rate_cv_pct, " %", 1))
            travel = last.handle_travel if last else nan
            row(ty + 78, "Amplitude poignee", fmt(travel, " x tronc", 2))
            if r.mode == "degraded":
                row(ty + 104, "~ Jambes quand le dos part",
                    fmt(100 * last.leg_fraction_at_trunk_onset if last else nan, " %"))

        extra = 0
        if self.ghost is not None:
            extra = 26
            gap = nan
            if ghost_pose is not None:
                actual = float(np.interp(t, self.t_series, self.lean_deg))
                gap = actual - GhostModel.trunk_lean_deg(ghost_pose)
            row(ty + 130, "Ecart buste vs fantome", fmt(gap, "°"))
        ay = ty + 142 + extra
        pm_row = r.pm5.rows.get(last.index) if (r.pm5 and last is not None) else None
        if r.pm5 is not None and r.pm5.n_matched:
            y = ty + 130 + extra
            row(y, "Puissance (PM5)", fmt(pm_row["pm5_power_w"] if pm_row else nan, " W"))
            row(y + 26, "Force de pic (PM5)",
                fmt(pm_row["pm5_peak_force_lbf"] if pm_row else nan, " lbf"))
            peak_pos = pm_row.get("force_peak_pos", nan) if pm_row else nan
            row(y + 52, "Pic de force a", fmt(100 * peak_pos, " % du drive"))
            self._force_curves(draw, PAD, y + 84, PANEL_W - 2 * PAD, 80, done, color)
            ay = y + 84 + 80 + 22
        alerts = [f for f in r.findings if last is not None and last.index in f.stroke_indices]
        if last is None:
            draw.text((PAD, ay), "En attente du premier coup complet...", font=self.f_stat,
                      fill=MUTED)
        elif not alerts:
            draw.text((PAD, ay), "Dernier coup : pas d'alerte", font=self.f_stat, fill=OK_COLOR)
        else:
            draw.text((PAD, ay), "A corriger sur le dernier coup :", font=self.f_stat,
                      fill=ALERT_COLOR)
            for k, f in enumerate(alerts[:3]):
                prefix = "~ " if f.reliability == "indicative" else "- "
                draw.text((PAD, ay + 22 + 20 * k), f"{prefix}{f.title}", font=self.f_small,
                          fill=TEXT)
        return np.array(img)[:, :, ::-1]  # RGB -> BGR


def _draw_athlete(image, kp, sc, trail, color: RGB):
    bgr = color[::-1]
    for a, b in EDGES:
        if sc[a] > KP_THR and sc[b] > KP_THR:
            cv2.line(image, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), bgr, 3,
                     cv2.LINE_AA)
    for i in range(17):
        if sc[i] > KP_THR:
            cv2.circle(image, tuple(int(v) for v in kp[i]), 4, (255, 255, 255), -1, cv2.LINE_AA)
    for a, b in zip(trail[:-1], trail[1:], strict=False):
        cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b), bgr, 2, cv2.LINE_AA)


def _draw_ghost(image, pose, report: RowingReport, to_ref: np.ndarray) -> None:
    """Superpose le fantome (translucide) a l'image : repere machine -> stabilise -> image."""
    frame = report.frame
    inv = np.linalg.inv(to_ref) if report.stabilized else np.eye(3)
    pts = {}
    for name, (s, h) in pose.items():
        x, y = s * frame.u + h * frame.v
        px = inv @ np.array([x, y, 1.0])
        pts[name] = (int(px[0]), int(px[1]))
    layer = image.copy()
    thick = max(4, round(image.shape[0] / 110))
    bgr = GHOST_COLOR[::-1]
    for a, b in GHOST_EDGES:
        cv2.line(layer, pts[a], pts[b], bgr, thick, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(layer, p, thick, bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.6, image, 0.4, 0, image)


def render_dashboard(
    video: str | Path,
    trace: RowingTrace,
    report: RowingReport,
    out_path: str | Path,
    progress: Callable[[float, str], None] | None = None,
    ghost: GhostModel | None = None,
) -> Path:
    """Ecrit la video composite H.264 (video annotee | panneau).

    `ghost` (voir `analysis.rowing.ghost.build_ghost`) superpose la posture de reference.
    """
    panel = _Panel(report, trace, ghost)
    stride = int(trace.meta.get("stride", 1))
    fps = float(trace.meta.get("fps", 30.0)) / stride
    rotation = int(trace.meta.get("rotation_deg", 0))
    side_idx = [5, 7, 9, 11, 13, 15] if report.side == "left" else [6, 8, 10, 12, 14, 16]
    wrist = side_idx[2]
    t0, t_end = float(trace.t_s[0]), float(trace.t_s[-1])

    writer: H264Writer | None = None
    n_done = 0
    for i, (t, image) in enumerate(iter_frames(video, t0, t_end + 1.0 / fps)):
        if i % stride:
            continue
        image = apply_rotation(image, rotation)
        idx = int(np.clip(np.searchsorted(trace.t_s, t), 0, len(trace.t_s) - 1))
        if idx > 0 and abs(trace.t_s[idx - 1] - t) < abs(trace.t_s[idx] - t):
            idx -= 1
        phase = _phase_at(report, t)
        color = {"drive": DRIVE_COLOR, "recovery": RECOVERY_COLOR, "none": MUTED}[phase]
        recent = np.flatnonzero((trace.t_s <= t) & (trace.t_s > t - 1.0))
        trail = [trace.xy_raw[j, wrist] for j in recent if trace.score[j, wrist] > KP_THR]
        phase_info = phase_of(report.stroke_events, t) if ghost is not None else None
        if phase_info:
            _draw_ghost(image, ghost.pose(*phase_info), report, trace.to_ref[idx])
        if ghost is not None:
            cv2.putText(image, "FANTOME : posture de reference", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, image.shape[0] / 900),
                        GHOST_COLOR[::-1], 2, cv2.LINE_AA)
        _draw_athlete(image, trace.xy_raw[idx], trace.score[idx], trail, color)

        scale = panel.height / image.shape[0]
        video_img = cv2.resize(image, (round(image.shape[1] * scale), panel.height))
        canvas = np.hstack([video_img, panel.render(t, idx)])
        if writer is None:
            writer = H264Writer(out_path, canvas.shape[1], canvas.shape[0], fps)
        writer.write(canvas)
        n_done += 1
        if progress and n_done % 15 == 0:
            progress(min(0.99, (t - t0) / max(t_end - t0, 1e-6)), f"Rendu video : {t:.1f} s")

    if writer is None:
        raise ValueError("aucune frame rendue")
    writer.close()
    if progress:
        progress(1.0, "Video prete")
    return Path(out_path)
