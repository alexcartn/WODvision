"""Courbes superposees coup par coup : moyenne et ecart-type sur le cycle drive / retour.

Chaque coup est reechantillonne sur un axe de phase commun : 0 = catch, 1 = finish, 2 = catch
suivant. Les courbes individuelles sont tracees en fond, la moyenne et la bande +/- 1 ecart-type
par-dessus : on voit d'un regard le geste type et sa regularite.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wodvision.analysis.rowing.frame import signed_lean_deg  # noqa: E402
from wodvision.analysis.rowing.metrics import knee_angle_series  # noqa: E402

N_PER_PHASE = 60
DRIVE_BG, RECOVERY_BG = "#e8f0fb", "#fdf1e3"
LINE, BAND, THIN = "#1f6feb", "#1f6feb", "#9aa3b2"


def _signals(report) -> dict[str, tuple[str, str, np.ndarray]]:
    """nom -> (titre, unite, serie sur la grille de report.series)."""
    series, frame = report.series, report.frame
    m = {k: frame.to_machine(p) for k, p in series.pts.items()}
    torso = float(np.nanmedian(np.linalg.norm(m["shoulder"] - m["hip"], axis=1)))
    catches = [int(np.argmin(np.abs(series.t_s - s.t_catch))) for s in report.stroke_events]
    s0 = float(np.nanmedian(m["wrist"][catches, 0])) if catches else 0.0
    return {
        "handle_s": ("Poignée : avance le long du rail", "longueurs de tronc",
                     (m["wrist"][:, 0] - s0) / torso),
        "handle_h": ("Poignée : hauteur par rapport à la hanche", "longueurs de tronc",
                     (m["wrist"][:, 1] - m["hip"][:, 1]) / torso),
        "knee": ("Angle du genou", "degrés", knee_angle_series(series)),
        "trunk": ("Inclinaison du buste (+ = vers l'arrière)", "degrés",
                  signed_lean_deg(m["hip"], m["shoulder"])),
    }


def cycle_curves(report, values: np.ndarray) -> np.ndarray:
    """(n_coups, 2 * N_PER_PHASE) : la serie de chaque coup sur l'axe de phase commun."""
    t = report.series.t_s
    out = []
    for s in report.stroke_events:
        drive = np.linspace(s.t_catch, s.t_finish, N_PER_PHASE, endpoint=False)
        recovery = np.linspace(s.t_finish, s.t_next_catch, N_PER_PHASE)
        out.append(np.interp(np.concatenate([drive, recovery]), t, values))
    return np.array(out)


def curves_png(report, keys: tuple[str, ...] = ("handle_s", "handle_h", "knee", "trunk"),
               dpi: int = 170) -> bytes | None:
    """Figure PNG des courbes moyennes +/- ecart-type, ou None s'il y a moins de 3 coups."""
    if report.series is None or report.frame is None or len(report.stroke_events) < 3:
        return None
    good = [i for i, sm in enumerate(report.strokes) if not sm.low_quality]
    keep = good if len(good) >= 3 else list(range(len(report.stroke_events)))
    signals = _signals(report)
    x = np.concatenate([np.linspace(0, 1, N_PER_PHASE, endpoint=False),
                        np.linspace(1, 2, N_PER_PHASE)])

    n = len(keys)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.4, 2.15 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys, strict=False):
        title, unit, values = signals[key]
        curves = cycle_curves(report, values)[keep]
        mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        ax.axvspan(0, 1, color=DRIVE_BG, lw=0)
        ax.axvspan(1, 2, color=RECOVERY_BG, lw=0)
        for c in curves:
            ax.plot(x, c, color=THIN, lw=0.6, alpha=0.7)
        ax.fill_between(x, mean - std, mean + std, color=BAND, alpha=0.22, lw=0)
        ax.plot(x, mean, color=LINE, lw=2)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylabel(unit, fontsize=7)
        ax.set_xticks([0, 1, 2], ["catch", "finish", "catch"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlim(0, 2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for xc, label in ((0.5, "poussée"), (1.5, "retour")):
            ax.text(xc, 0.02, label, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=6.5, color="#6b7280")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()
