"""Schema pedagogique des phases du coup, dessine avec le modele de posture de reference.

Quatre positions (catch, milieu de la poussee, finish, milieu du retour) sur un ergometre
schematique. Sert de repere visuel aux debutants, dans l'application et dans la fiche PDF.
"""

from __future__ import annotations

import functools
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wodvision.analysis.rowing.ghost import (  # noqa: E402
    GHOST_EDGES,
    GhostModel,
    generic_anthropometry,
)
from wodvision.config import GhostConfig  # noqa: E402

PHASES = [
    ("1. Catch", "drive", 0.0, "Groupé, bras tendus"),
    ("2. Poussée", "drive", 0.45, "Les jambes poussent"),
    ("3. Finish", "drive", 1.0, "Jambes tendues, léger recul"),
    ("4. Retour", "recovery", 0.5, "Bras, buste, puis jambes"),
]
MACHINE = "#8a8d98"
SEGMENT_COLORS = {  # jambes, buste, bras : un code couleur lisible pour un debutant
    ("hip", "knee"): "#3b78c2", ("knee", "ankle"): "#3b78c2",
    ("shoulder", "hip"): "#2f8f5b",
    ("shoulder", "elbow"): "#e08a1e", ("elbow", "wrist"): "#e08a1e",
}


@functools.cache
def _default_diagram() -> bytes:
    return phases_diagram_png(GhostConfig())


def default_phases_diagram() -> bytes:
    """Schema par defaut, calcule une seule fois par processus."""
    return _default_diagram()


def phases_diagram_png(cfg: GhostConfig | None = None, dpi: int = 160) -> bytes:
    """Image PNG des quatre phases du coup (proportions moyennes)."""
    model = GhostModel(generic_anthropometry(), cfg or GhostConfig())
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7), dpi=dpi)
    for ax, (title, kind, x, caption) in zip(axes, PHASES, strict=True):
        pose = model.pose(kind, x)
        hip, shoulder = pose["hip"], pose["shoulder"]
        rail_h = 0.17  # le rail passe sous le siege
        ax.plot([-1.0, 1.3], [-0.06, -0.06], color="#cfd2da", lw=1.5)  # sol
        ax.plot([-0.75, 1.25], [rail_h, rail_h], color=MACHINE, lw=3, solid_capstyle="round")
        ax.plot([-0.75, -0.75], [-0.06, rail_h], color=MACHINE, lw=3)  # pied avant
        ax.plot([1.25, 1.25], [-0.06, rail_h], color=MACHINE, lw=3)  # pied arriere
        ax.add_patch(plt.Circle((-0.8, 0.5), 0.23, color=MACHINE, alpha=0.35))  # volant
        ax.plot([-0.8, -0.8], [rail_h, 0.5 - 0.23], color=MACHINE, lw=3)  # support du volant
        ax.plot([-0.05, 0.12], [0.1, 0.1], color=MACHINE, lw=3)  # cale-pieds
        ax.add_patch(plt.Rectangle((hip[0] - 0.09, hip[1] - 0.07), 0.18, 0.05,
                                   color=MACHINE, alpha=0.6))  # siege
        for a, b in GHOST_EDGES:
            ax.plot([pose[a][0], pose[b][0]], [pose[a][1], pose[b][1]],
                    color=SEGMENT_COLORS[(a, b)], lw=5, solid_capstyle="round", zorder=2)
        up = (shoulder - hip) / ((shoulder - hip) @ (shoulder - hip)) ** 0.5
        ax.add_patch(plt.Circle(shoulder + 0.13 * up, 0.075, color="#2f8f5b", zorder=2))  # tete
        for p in pose.values():
            ax.plot(*p, "o", color="white", mec="#333", ms=4, zorder=3)
        ax.plot([pose["wrist"][0], -0.8], [pose["wrist"][1], 0.5], color="#555", lw=1,
                ls=":")  # chaine
        ax.set_xlim(-1.1, 1.35)
        ax.set_ylim(-0.1, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.12, -0.12, caption, ha="center", va="top", fontsize=8, color="#333")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
