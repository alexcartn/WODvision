"""Genere les exemples de demonstration de l'application (donnees SIMULEES, aucune personne reelle).

Un rameur filaire synthetique est analyse avec le meme pipeline que les vraies videos ;
la trace d'inference, le journal PM5 et la video composite sont ecrits dans app/demo/.
Usage : python scripts/build_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import pm5 as syn_pm5  # noqa: E402
from tests.synthetic.rowing import RowerParams  # noqa: E402
from tests.synthetic.video import make_scene  # noqa: E402

from wodvision.analysis.rowing.ghost import build_ghost  # noqa: E402
from wodvision.config import Config  # noqa: E402
from wodvision.io.cache import save_trace  # noqa: E402
from wodvision.pm5.session import session_from_events, write_event, write_event_header  # noqa: E402
from wodvision.rowing_pipeline import analyze_trace, get_trace  # noqa: E402
from wodvision.viz.rowing_dashboard import render_dashboard  # noqa: E402
from wodvision.viz.rowing_highlights import render_highlights, save_highlights  # noqa: E402

OUT = ROOT / "app" / "demo"
COMMON = {"duration_s": 20.0, "px_per_m": 200.0, "origin_px": (250.0, 420.0), "noise_px": 1.0,
          "period_jitter": 0.06, "rate_spm": 26.0}
DEMOS = {
    # Profil : dos qui s'ouvre tot + chaine qui monte, avec PM5 (catch explosif).
    "profil": {"params": {**COMMON, "trunk_onset": 0.3, "fault_jitter": 0.18, "chain_deg": 12.0},
               "pm5": True},
    # Oblique : technique propre, analyse en mode degrade.
    "oblique": {"params": {**COMMON, "yaw_deg": 50.0}, "pm5": False},
}


def build(name: str, spec: dict) -> None:
    cfg = Config()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / f"{name}.mp4"
        detector, pose, _ = make_scene(video, RowerParams(**spec["params"]))
        trace = get_trace(video, cfg, cache_dir=tmp / "cache", detector=detector, pose=pose)
        trace.meta["video"] = f"demo:{name}"
        report = analyze_trace(trace, cfg)
        session = None
        if spec["pm5"]:
            events = syn_pm5.make_events(report, lag=2, offset_s=14.2,
                                         curve=lambda j: syn_pm5.bell_curve(0.1 if j % 5 else 0.4))
            with (OUT / f"{name}_pm5.jsonl").open("w", encoding="utf-8") as fh:
                write_event_header(fh, "PM5 SIMULE")
                for t, c, d in events:
                    write_event(fh, t, c, d)
            session = session_from_events(events)
            report = analyze_trace(trace, cfg, pm5=session)
        save_trace(trace, OUT / f"{name}_trace.npz")
        ghost = build_ghost(report, cfg.ghost)
        render_dashboard(video, trace, report, OUT / f"{name}.mp4", ghost=ghost)
        hl = render_highlights(video, trace, report, ghost)
        if hl is not None:
            save_highlights(hl, OUT / f"{name}_highlights")
    print(f"{name}: {report.n_strokes} coups, mode {report.mode}, "
          f"{len(report.findings)} constat(s)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for demo_name, demo_spec in DEMOS.items():
        build(demo_name, demo_spec)
