"""Courbes superposees coup par coup : moyenne et ecart-type sur le cycle drive / retour.

Chaque coup est reechantillonne sur un axe de phase commun : 0 = catch, 1 = finish, 2 = catch
suivant. Les courbes individuelles sont en fond, la moyenne et la bande +/- 1 ecart-type
par-dessus : on voit le geste type et sa regularite. En vue de profil, la posture de reference
(fantome) est tracee en pointille jaune sur les indicateurs qu'elle definit.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wodvision.analysis.rowing.frame import signed_lean_deg  # noqa: E402
from wodvision.analysis.rowing.metrics import _angle, knee_angle_series  # noqa: E402

N_PER_PHASE = 60
DRIVE_BG, RECOVERY_BG = "#e8f0fb", "#fdf1e3"
LINE, THIN, GHOST = "#1f6feb", "#9aa3b2", "#e0a800"
SEG_COLORS = {"Jambes": "#3b78c2", "Dos": "#2f8f5b", "Bras": "#e08a1e"}
PHASE_X = np.concatenate([np.linspace(0, 1, N_PER_PHASE, endpoint=False),
                          np.linspace(1, 2, N_PER_PHASE)])

PANELS = ("handle_s", "handle_h", "handle_v", "knee", "trunk", "elbow")
TITLES = {
    "handle_s": ("Poignée : avance le long du rail", "longueurs de tronc"),
    "handle_h": ("Poignée : hauteur par rapport à la hanche", "longueurs de tronc"),
    "handle_v": ("Poignée : vitesse le long du rail", "longueurs de tronc / s"),
    "knee": ("Angle du genou", "degrés"),
    "trunk": ("Inclinaison du buste (+ = vers l'arrière)", "degrés"),
    "elbow": ("Angle du coude", "degrés"),
}


def _measured(report) -> tuple[dict[str, np.ndarray], float]:
    """Series mesurees sur la grille de report.series, et longueur de tronc (unite)."""
    series = report.series
    m = {k: report.frame.to_machine(p) for k, p in series.pts.items()}
    torso = float(np.nanmedian(np.linalg.norm(m["shoulder"] - m["hip"], axis=1)))
    catches = [int(np.argmin(np.abs(series.t_s - s.t_catch))) for s in report.stroke_events]
    s0 = float(np.nanmedian(m["wrist"][catches, 0]))
    hs = (m["wrist"][:, 0] - s0) / torso
    return {
        "handle_s": hs,
        "handle_h": (m["wrist"][:, 1] - m["hip"][:, 1]) / torso,
        "handle_v": np.gradient(hs, series.t_s),
        "knee": knee_angle_series(series),
        "trunk": signed_lean_deg(m["hip"], m["shoulder"]),
        "elbow": _angle(m["shoulder"], m["elbow"], m["wrist"]),
    }, torso


def _reference(ghost, report) -> dict[str, np.ndarray]:
    """Series du fantome sur l'axe de phase, ramenees dans les memes unites que les mesures."""
    t_drive = float(np.median([s.drive_time for s in report.stroke_events]))
    t_rec = float(np.median([s.recovery_time for s in report.stroke_events]))
    poses = [ghost.pose("drive", x) for x in np.linspace(0, 1, N_PER_PHASE, endpoint=False)]
    poses += [ghost.pose("recovery", x) for x in np.linspace(0, 1, N_PER_PHASE)]
    torso = np.linalg.norm(poses[0]["shoulder"] - poses[0]["hip"])
    w = np.array([p["wrist"] for p in poses])
    hs = (w[:, 0] - w[0, 0]) / torso
    t = np.concatenate([np.linspace(0, t_drive, N_PER_PHASE, endpoint=False),
                        t_drive + np.linspace(0, t_rec, N_PER_PHASE)])
    ang = lambda a, b, c: _angle(np.array([p[a] for p in poses]),  # noqa: E731
                                 np.array([p[b] for p in poses]),
                                 np.array([p[c] for p in poses]))
    return {
        "handle_s": hs,
        "handle_h": np.array([(p["wrist"][1] - p["hip"][1]) / torso for p in poses]),
        "handle_v": np.gradient(hs, t),
        "knee": ang("hip", "knee", "ankle"),
        "trunk": np.array([ghost.trunk_lean_deg(p) for p in poses]),
        "elbow": ang("shoulder", "elbow", "wrist"),
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


def _kept(report) -> list[int]:
    good = [i for i, sm in enumerate(report.strokes) if not sm.low_quality]
    return good if len(good) >= 3 else list(range(len(report.stroke_events)))


def _usable(report) -> bool:
    return (report.series is not None and report.frame is not None
            and len(report.stroke_events) >= 3)


def _png(fig, dpi: int) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def _style(ax) -> None:
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _band(ax, x, curves, color=LINE, thin=True) -> None:
    mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
    if thin:
        for c in curves:
            ax.plot(x, c, color=THIN, lw=0.6, alpha=0.7)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22, lw=0)
    ax.plot(x, mean, color=color, lw=2)


def curves_png(report, ghost=None, dpi: int = 170) -> bytes | None:
    """Six courbes de cycle (poignee, genou, buste, coude), moyenne +/- ecart-type."""
    if not _usable(report):
        return None
    meas, _ = _measured(report)
    ref = _reference(ghost, report) if ghost is not None else {}
    keep = _kept(report)
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 6.3))
    for ax, key in zip(axes.flat, PANELS, strict=True):
        title, unit = TITLES[key]
        ax.axvspan(0, 1, color=DRIVE_BG, lw=0)
        ax.axvspan(1, 2, color=RECOVERY_BG, lw=0)
        _band(ax, PHASE_X, cycle_curves(report, meas[key])[keep])
        if key in ref:
            ax.plot(PHASE_X, ref[key], color=GHOST, lw=1.6, ls="--")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylabel(unit, fontsize=7)
        ax.set_xticks([0, 1, 2], ["catch", "finish", "catch"], fontsize=7)
        ax.set_xlim(0, 2)
        _style(ax)
        for xc, label in ((0.5, "poussée"), (1.5, "retour")):
            ax.text(xc, 0.02, label, transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=6.5, color="#6b7280")
    if ref:
        axes[0, 0].plot([], [], color=GHOST, ls="--", label="référence")
        axes[0, 0].plot([], [], color=LINE, label="toi")
        axes[0, 0].legend(fontsize=6.5, frameon=False, loc="upper left")
    return _png(fig, dpi)


def _progress(curves: np.ndarray, min_range: float) -> np.ndarray:
    """Progression 0 -> 1 de chaque coup pendant le drive ; coups a faible excursion exclus."""
    d = curves[:, :N_PER_PHASE]
    lo, hi = d[:, :1], d[:, -1:]
    ok = np.abs(hi - lo)[:, 0] >= min_range
    return (d[ok] - lo[ok]) / (hi[ok] - lo[ok])


def sequence_png(report, ghost=None, dpi: int = 170) -> bytes | None:
    """Enchainement jambes / dos / bras pendant la poussee, et regularite de chaque indicateur."""
    if not _usable(report):
        return None
    meas, _ = _measured(report)
    ref = _reference(ghost, report) if ghost is not None else {}
    keep = _kept(report)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.linspace(0, 1, N_PER_PHASE, endpoint=False)
    for name, key, rng in (("Jambes", "knee", 20.0), ("Dos", "trunk", 10.0),
                           ("Bras", "elbow", 15.0)):
        prog = _progress(cycle_curves(report, meas[key])[keep], rng)
        if len(prog):
            _band(ax, x, prog, SEG_COLORS[name], thin=False)
            ax.plot([], [], color=SEG_COLORS[name], label=name)
        if key in ref:
            r = ref[key][:N_PER_PHASE]
            if abs(r[-1] - r[0]) > 1e-6:
                ax.plot(x, (r - r[0]) / (r[-1] - r[0]), color=SEG_COLORS[name], lw=1.2, ls=":")
    ax.set_title("Enchaînement pendant la poussée", fontsize=8.5, loc="left")
    ax.set_xlabel("avancement de la poussée (catch → finish)", fontsize=7)
    ax.set_ylabel("segment déplié (0 à 1)", fontsize=7)
    if ref:
        ax.plot([], [], color="#555", lw=1.2, ls=":", label="référence")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    _style(ax)

    labels, values = [], []
    for key, label in (("handle_s", "Avance poignée"), ("handle_h", "Hauteur poignée"),
                       ("knee", "Genou"), ("trunk", "Buste"), ("elbow", "Coude")):
        curves = cycle_curves(report, meas[key])[keep]
        mean = np.nanmean(curves, axis=0)
        span = float(np.ptp(mean))
        if span > 1e-9:
            labels.append(label)
            values.append(100.0 * float(np.nanmean(np.nanstd(curves, axis=0))) / span)
    ax2.barh(labels[::-1], values[::-1], color="#7a8699", height=0.55)
    ax2.set_title("Régularité (écart-type / amplitude)", fontsize=8.5, loc="left")
    ax2.set_xlabel("% : plus c'est bas, plus le geste se répète", fontsize=7)
    for i, v in enumerate(values[::-1]):
        ax2.text(v, i, f" {v:.0f} %", va="center", fontsize=7)
    ax2.set_xlim(0, max(values + [10]) * 1.25)
    _style(ax2)
    return _png(fig, dpi)


def force_png(report, dpi: int = 170) -> bytes | None:
    """Courbe de force PM5 moyenne +/- ecart-type (forme, normalisee), si un PM5 est couple."""
    from wodvision.analysis.rowing.force_curve import normalized_curve

    pm5 = getattr(report, "pm5", None)
    if pm5 is None or not pm5.rows:
        return None
    curves = []
    for row in pm5.rows.values():
        c = normalized_curve(row.get("force_curve"), N_PER_PHASE) if row.get(
            "force_curve") is not None else None
        if c is not None and np.max(c) > 0:
            curves.append(c / np.max(c))
    if len(curves) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    x = np.linspace(0, 1, N_PER_PHASE)
    ax.axvspan(0, 1, color=DRIVE_BG, lw=0)
    _band(ax, x, np.array(curves))
    ax.set_title("Courbe de force du PM5 (forme, normalisée par coup)", fontsize=8.5, loc="left")
    ax.set_xlabel("avancement de la poussée (catch → finish)", fontsize=7)
    ax.set_ylabel("force / pic", fontsize=7)
    ax.set_xlim(0, 1)
    _style(ax)
    return _png(fig, dpi)
