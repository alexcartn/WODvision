"""Application Streamlit : depose une video d'aviron, obtient l'analyse technique.

Lancer : streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
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
    pm5_upload = st.file_uploader(
        "Fichier PM5 (optionnel : .jsonl de pm5-record, ou .csv)", type=["jsonl", "csv"],
        help="Couple la video aux donnees du PM5 : validation de la detection, force, puissance.",
    )
    pm5_lag = st.number_input("Decalage PM5 - video (coups, 0 = automatique)", -20, 20, 0,
                              help="A renseigner si l'appariement automatique est ambigu.")
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

    pm5_session = None
    if pm5_upload is not None:
        from wodvision.pm5.session import load_pm5

        pm5_path = UPLOAD_DIR / pm5_upload.name
        pm5_path.write_bytes(pm5_upload.getbuffer())
        try:
            pm5_session = load_pm5(pm5_path)
        except ValueError as exc:
            st.error(f"Fichier PM5 illisible : {exc}")
            st.stop()
    try:
        detector, pose = models(mode)
        trace = get_trace(video_path, cfg, mode=mode, click_xy=click_xy,
                          max_seconds=max_seconds or None, detector=detector, pose=pose,
                          progress=progress, start_s=float(start_s))
        report = analyze_trace(trace, cfg, pm5=pm5_session,
                               pm5_lag=int(pm5_lag) if pm5_lag else None)
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

if report.mode == "full":
    st.success(report.mode_label)
else:
    st.warning(f"Mode : {report.mode_label}")
    with st.expander("Que signifie ce mode ?", expanded=True):
        st.markdown(
            "- Fiable : nombre de coups, cadence, duree du drive, ratio drive/recuperation, "
            "regularite du rythme.\n"
            "- Indicatif (marque ~) : sequence jambes / dos / bras, evolution de l'amplitude de "
            "la poignee. Ces mesures sont relatives, donc moins sensibles a l'angle de vue.\n"
            "- Masque : angles et hauteurs absolus (tibias, recul du buste, hauteur de poignee, "
            "alignement de la chaine), qui sont deformes quand la camera n'est pas "
            "perpendiculaire au rail.\n"
            "- Pour une analyse complete : camera fixe, perpendiculaire au rail, a hauteur de "
            "hanche, rameur entier visible."
        )
for w in report.warnings:
    st.warning(w)
if report.camera_motion_px > 20:
    st.caption(f"Mouvement de camera detecte (~{report.camera_motion_px:.0f} px) : compense.")

st.video(dash.read_bytes())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Coups", report.n_strokes)
c2.metric("Cadence mediane", f"{report.stroke_rate_spm:.1f} c/min")
c3.metric("Duree du drive", f"{report.drive_time_s:.2f} s")
c4.metric("Recup / drive", f"{report.recovery_drive_ratio:.2f} : 1")
c5.metric("Regularite (CV)", f"{report.rate_cv_pct:.1f} %")

st.subheader("Constats de technique")
if not report.findings:
    st.success("Aucune faute detectee avec les seuils actuels.")
for f in report.findings:
    with st.container(border=True):
        st.markdown(f"{f.title}" + ("  (indicatif)" if f.reliability == "indicative" else ""))
        st.write(f.advice)
        st.caption(f"{f.metric} median = {f.median_value:.2f} (seuil {f.comparator} "
                   f"{f.threshold:g}), {f.n_flagged}/{f.n_valid} coups concernes")

if report.pm5 is not None:
    p = report.pm5
    st.subheader("Couplage PM5")
    if p.n_matched == 0:
        st.warning("Aucun coup apparie : verifier le fichier PM5 et la duree de la video.")
    else:
        if p.ambiguous:
            st.warning("Appariement ambigu : renseigne le decalage en coups dans la colonne de "
                       "gauche si tu connais les coups de depart.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Coups apparies", f"{p.n_matched} / {len(p.session.strokes)}")
        c2.metric("Decalage PM5 - video", f"{p.offset_s:+.2f} s",
                  help=f"Gigue {p.offset_jitter_s:.3f} s ; phase {p.phase} ({p.phase_source}).")
        v = p.validation
        if v:
            c3.metric("Ecart de periode", f"{v['period_mae_ms']:.0f} ms",
                      help="Ecart moyen entre la periode des coups video et PM5.")
            c4.metric("Drive : biais / dispersion",
                      f"{v['drive_bias_ms']:+.0f} / {v['drive_std_ms']:.0f} ms",
                      help="Le drive PM5 (volant en acceleration) est plus court que le drive "
                           "video : un biais est normal, la dispersion valide la detection.")
        if v and "implied_torso_m" in v:
            st.caption(f"Longueur de tronc deduite (drive length PM5 / amplitude video) : "
                       f"{v['implied_torso_m']:.2f} m, CV {v['implied_torso_cv_pct']:.0f} %. "
                       "Une valeur proche de 0,45 a 0,60 m confirme l'appariement.")
        curves = [(i, r["force_curve"]) for i, r in p.rows.items()
                  if r.get("force_curve") is not None]
        if curves:
            import matplotlib.pyplot as plt

            from wodvision.analysis.rowing.force_curve import normalized_curve

            fig, ax = plt.subplots(figsize=(7, 3))
            for _, curve in curves:
                c = normalized_curve(curve, 100)
                if c is not None:
                    ax.plot(np.linspace(0, 1, 100), c, color="#4a90e2", alpha=0.25)
            ax.set_xlabel("position de la poignee (fraction du drive)")
            ax.set_ylabel("force (unite PM5)")
            ax.set_title("Courbes de force, tous les coups")
            st.pyplot(fig)
        if p.session.drag_factor == p.session.drag_factor:
            st.caption(f"Drag factor : {p.session.drag_factor:.0f}")

st.subheader("Detail par coup")
df = pd.read_csv(run_dir / "strokes.csv").drop(columns=["force_curve"], errors="ignore")
st.dataframe(df.round(2), use_container_width=True)

d1, d2, d3 = st.columns(3)
d1.download_button("Video annotee (mp4)", dash.read_bytes(), "dashboard.mp4", "video/mp4")
d2.download_button("Coups (csv)", (run_dir / "strokes.csv").read_bytes(), "strokes.csv", "text/csv")
d3.download_button("Resume (json)", json.dumps(
    json.loads((run_dir / "summary.json").read_text(encoding="utf-8")), indent=2),
    "summary.json", "application/json")
