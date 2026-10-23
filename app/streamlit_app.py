"""Application Streamlit : depose une video d'aviron, obtient l'analyse technique.

Lancer : streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from wodvision.config import load_config
from wodvision.io.frames import grab_frame
from wodvision.rowing_pipeline import (
    analyze_trace,
    build_default_models,
    export_run,
    get_trace,
    new_run_dir,
)
from wodvision.viz.rowing_dashboard import render_dashboard

logging.basicConfig(level=logging.INFO)
UPLOAD_DIR = Path("data/raw/uploads")

st.set_page_config(page_title="WODVision Rameur", layout="wide")
st.title("WODVision : technique d'aviron sur Concept2")


@st.cache_resource(show_spinner="Chargement des modeles de pose...")
def models(mode: str):
    return build_default_models(mode)


with st.sidebar:
    st.header("Video")
    upload = st.file_uploader("Depose ta video (MP4, AVI, MOV, MKV)",
                              type=["mp4", "avi", "mov", "mkv"])
    mode = st.selectbox("Precision de la pose", ["lightweight", "balanced", "performance"],
                        index=1, help="Plus precis = plus lent sur CPU.")
    start_s = st.number_input("Debut de l'analyse (s)", 0, 3600, 0,
                              help="Utile si la video contient autre chose que de l'aviron.")
    max_seconds = st.number_input("Duree analysee (s, 0 = jusqu'a la fin)", 0, 3600, 0)
    stabilize = st.checkbox("Compenser le mouvement de camera", value=True,
                            help="A garder active si le telephone n'est pas parfaitement fixe.")
    pick = st.checkbox("Choisir le rameur manuellement (plusieurs personnes dans l'image)")
    click_xy = None
    if pick and upload is not None:
        st.caption("Position du rameur dans la premiere frame, en % de l'image.")
        cx = st.slider("Horizontal (%)", 0, 100, 50)
        cy = st.slider("Vertical (%)", 0, 100, 50)
        click_xy = (cx / 100.0, cy / 100.0)
    go = st.button("Analyser", type="primary", disabled=upload is None)
    st.caption("Vue de profil, rameur entier visible, de preference 30 images/s ou plus.")

if upload is None:
    st.info("Depose une video pour commencer. Les regles de technique sont des hypotheses "
            "de depart (guides Concept2) a calibrer : voir docs/rowing_concept.md.")
    st.stop()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
video_path = UPLOAD_DIR / upload.name
video_path.write_bytes(upload.getbuffer())

if pick:
    frame = grab_frame(video_path, 0.0)
    h, w = frame.shape[:2]
    preview = frame.copy()
    import cv2

    cv2.drawMarker(preview, (int(click_xy[0] * w), int(click_xy[1] * h)), (0, 255, 0),
                   cv2.MARKER_CROSS, 40, 3)
    st.sidebar.image(preview[:, :, ::-1], caption="Repere sur le rameur")
    click_xy = (click_xy[0] * w, click_xy[1] * h)

if go:
    cfg = load_config()
    cfg.camera_motion.enabled = stabilize
    bar = st.progress(0.0, text="Demarrage...")

    def progress(frac: float, msg: str) -> None:
        bar.progress(min(1.0, frac), text=msg)

    try:
        detector, pose = models(mode)
        trace = get_trace(video_path, cfg, mode=mode, click_xy=click_xy,
                          max_seconds=max_seconds or None, detector=detector, pose=pose,
                          progress=progress, start_s=float(start_s))
        report = analyze_trace(trace, cfg)
        run_dir = export_run(report, new_run_dir(video_path), cfg)
        dash = render_dashboard(video_path, trace, report, run_dir / "dashboard.mp4",
                                progress=progress)
    except ValueError as exc:
        bar.empty()
        st.error(str(exc))
        st.stop()
    bar.empty()
    st.session_state["result"] = {"report": report, "dash": str(dash), "run_dir": str(run_dir)}

result = st.session_state.get("result")
if not result:
    st.stop()

report, dash, run_dir = result["report"], Path(result["dash"]), Path(result["run_dir"])

for w in report.warnings:
    st.warning(w)
if report.camera_motion_px > 20:
    st.caption(f"Mouvement de camera detecte (~{report.camera_motion_px:.0f} px) : compense.")

st.video(dash.read_bytes())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Coups", report.n_strokes)
c2.metric("Cadence mediane", f"{report.stroke_rate_spm:.1f} c/min")
c3.metric("Duree du drive", f"{report.drive_time_s:.2f} s")
c4.metric("Recup / drive", f"{report.recovery_drive_ratio:.2f} : 1")

st.subheader("Constats de technique")
if not report.findings:
    st.success("Aucune faute detectee avec les seuils actuels.")
for f in report.findings:
    with st.container(border=True):
        st.markdown(f"{f.title}")
        st.write(f.advice)
        st.caption(f"{f.metric} median = {f.median_value:.2f} (seuil {f.comparator} "
                   f"{f.threshold:g}), {f.n_flagged}/{f.n_valid} coups concernes")

st.subheader("Detail par coup")
df = pd.DataFrame([s.to_dict() for s in report.strokes])
st.dataframe(df.round(2), use_container_width=True)

d1, d2, d3 = st.columns(3)
d1.download_button("Video annotee (mp4)", dash.read_bytes(), "dashboard.mp4", "video/mp4")
d2.download_button("Coups (csv)", (run_dir / "strokes.csv").read_bytes(), "strokes.csv", "text/csv")
d3.download_button("Resume (json)", json.dumps(
    json.loads((run_dir / "summary.json").read_text(encoding="utf-8")), indent=2),
    "summary.json", "application/json")
