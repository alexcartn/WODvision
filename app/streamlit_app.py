"""Application Streamlit : analyse de la technique d'aviron sur Concept2.

Lancer : streamlit run app/streamlit_app.py

Variable d'environnement WODVISION_HOSTED=1 : version hebergee d'essai (profil
configs/hosted.yaml : modele leger, 30 s max, fichiers supprimes apres analyse).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wodvision.analysis.rowing.report import RowingReport
from wodvision.config import load_config
from wodvision.io.cache import load_trace
from wodvision.rowing_pipeline import (
    analyze_trace,
    build_default_models,
    export_run,
    get_trace,
    new_run_dir,
)
from wodvision.viz.rowing_dashboard import render_dashboard

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent / "demo"
UPLOAD_DIR = Path("data/raw/uploads")
HOSTED = os.environ.get("WODVISION_HOSTED") == "1"
CFG_PATH = ROOT / "configs" / "hosted.yaml" if HOSTED else None

DEMOS = {
    "Profil, avec PM5 : dos precoce et chaine qui monte": "profil",
    "Vue oblique : mode degrade": "oblique",
}

st.set_page_config(page_title="WODVision Rameur", layout="wide")
st.title("WODVision : technique d'aviron sur Concept2")


@st.cache_resource(show_spinner="Chargement des modeles de pose...")
def models(mode: str):
    return build_default_models(mode)


@st.cache_resource(show_spinner="Analyse de l'exemple...")
def demo_result(name: str) -> dict:
    """Analyse instantanee d'un exemple simule (trace et PM5 pre-calcules, video pre-rendue)."""
    from wodvision.pm5.session import load_pm5

    cfg = load_config(CFG_PATH)
    trace = load_trace(DEMO_DIR / f"{name}_trace.npz")
    pm5_path = DEMO_DIR / f"{name}_pm5.jsonl"
    report = analyze_trace(trace, cfg, pm5=load_pm5(pm5_path) if pm5_path.exists() else None)
    return pack_result(report, cfg, (DEMO_DIR / f"{name}.mp4").read_bytes())


def pack_result(report: RowingReport, cfg, dash_bytes: bytes) -> dict:
    """Resultat autonome en memoire (aucun fichier a conserver cote serveur)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = export_run(report, Path(tmp) / "run", cfg)
        csv_bytes = (run_dir / "strokes.csv").read_bytes()
        json_text = (run_dir / "summary.json").read_text(encoding="utf-8")
    return {"report": report, "dash": dash_bytes, "csv": csv_bytes, "json": json_text}


def show_result(res: dict, simulated: bool = False) -> None:
    report: RowingReport = res["report"]
    if simulated:
        st.info("Donnees simulees (rameur filaire synthetique) : aucune personne reelle. "
                "Le pipeline est le meme que pour tes videos.")
    if report.mode == "full":
        st.success(report.mode_label)
    else:
        st.warning(f"Mode : {report.mode_label}")
        with st.expander("Que signifie ce mode ?", expanded=True):
            st.markdown(
                "- Fiable : nombre de coups, cadence, duree du drive, ratio drive/recuperation, "
                "regularite du rythme.\n"
                "- Indicatif (marque ~) : sequence jambes / dos / bras, evolution de l'amplitude "
                "de la poignee. Ces mesures sont relatives, donc moins sensibles a l'angle de "
                "vue.\n"
                "- Masque : angles et hauteurs absolus (tibias, recul du buste, hauteur de "
                "poignee, alignement de la chaine), deformes quand la camera n'est pas "
                "perpendiculaire au rail.\n"
                "- Pour une analyse complete : camera fixe, perpendiculaire au rail, a hauteur "
                "de hanche, rameur entier visible."
            )
    for w in report.warnings:
        st.warning(w)
    if report.camera_motion_px > 20:
        st.caption(f"Mouvement de camera detecte (~{report.camera_motion_px:.0f} px) : compense.")

    st.video(res["dash"])

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
        show_pm5(report)

    st.subheader("Detail par coup")
    df = pd.read_csv(io.BytesIO(res["csv"])).drop(columns=["force_curve"], errors="ignore")
    st.dataframe(df.round(2), width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Video annotee (mp4)", res["dash"], "dashboard.mp4", "video/mp4")
    d2.download_button("Coups (csv)", res["csv"], "strokes.csv", "text/csv")
    d3.download_button("Resume (json)", json.dumps(json.loads(res["json"]), indent=2),
                       "summary.json", "application/json")


def show_pm5(report: RowingReport) -> None:
    p = report.pm5
    st.subheader("Couplage PM5")
    if p.n_matched == 0:
        st.warning("Aucun coup apparie : verifier le fichier PM5 et la duree de la video.")
        return
    if p.ambiguous:
        st.warning("Appariement ambigu : renseigne le decalage en coups si tu connais les coups "
                   "de depart.")
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
    curves = [r["force_curve"] for r in p.rows.values() if r.get("force_curve") is not None]
    if curves:
        import matplotlib.pyplot as plt

        from wodvision.analysis.rowing.force_curve import normalized_curve

        fig, ax = plt.subplots(figsize=(7, 3))
        for curve in curves:
            c = normalized_curve(curve, 100)
            if c is not None:
                ax.plot(np.linspace(0, 1, 100), c, color="#4a90e2", alpha=0.25)
        ax.set_xlabel("position de la poignee (fraction du drive)")
        ax.set_ylabel("force (unite PM5)")
        ax.set_title("Courbes de force, tous les coups")
        st.pyplot(fig)
        plt.close(fig)
    if p.session.drag_factor == p.session.drag_factor:
        st.caption(f"Drag factor : {p.session.drag_factor:.0f}")


def upload_tab() -> None:
    cfg = load_config(CFG_PATH)
    if HOSTED:
        st.warning(
            f"Version d'essai en ligne : {cfg.hosted.max_seconds:.0f} s de video maximum, modele "
            "de pose leger, calcul sur un serveur gratuit partage (compte une a deux minutes). "
            "La video est supprimee du serveur apres l'analyse. Pour des videos longues ou "
            "confidentielles, installe l'outil en local (voir le depot GitHub)."
        )
    upload = st.file_uploader("Depose ta video (MP4, AVI, MOV, MKV)",
                              type=["mp4", "avi", "mov", "mkv"], key="video")
    with st.expander("Reglages et fichier PM5 (optionnel)"):
        pm5_upload = st.file_uploader(
            "Fichier PM5 (.jsonl de pm5-record, ou .csv)", type=["jsonl", "csv"], key="pm5",
            help="Couple la video aux donnees du PM5 : validation, force, puissance.")
        pm5_lag = st.number_input("Decalage PM5 - video (coups, 0 = automatique)", -20, 20, 0)
        if HOSTED:
            mode = cfg.hosted.mode
            st.caption(f"Modele de pose : {mode} (impose par la version en ligne).")
        else:
            mode = st.selectbox("Precision de la pose", ["lightweight", "balanced", "performance"],
                                index=1, help="Plus precis = plus lent sur CPU.")
        start_s = st.number_input("Debut de l'analyse (s)", 0, 3600, 0)
        default_len = int(cfg.hosted.max_seconds) if HOSTED else 0
        max_seconds = st.number_input(
            "Duree analysee (s)" + (" - 30 max en ligne" if HOSTED else ", 0 = jusqu'a la fin"),
            0, 3600, default_len)
        stabilize = st.checkbox("Compenser le mouvement de camera", value=True)
        pick = st.checkbox("Choisir le rameur manuellement (plusieurs personnes dans l'image)")
        click_pct = None
        if pick:
            click_pct = (st.slider("Horizontal (%)", 0, 100, 50),
                         st.slider("Vertical (%)", 0, 100, 50))
    st.caption("Vue de profil, rameur entier visible, de preference 30 images/s ou plus.")
    consent = True
    if HOSTED:
        consent = st.checkbox(
            "Je confirme avoir l'accord des personnes filmees et j'accepte que la video soit "
            "traitee sur un serveur externe (supprimee apres l'analyse).")
    go = st.button("Analyser", type="primary", disabled=upload is None or not consent)

    if go and upload is not None:
        run_analysis(cfg, upload, pm5_upload, int(pm5_lag), mode, float(start_s),
                     float(max_seconds), stabilize, click_pct)
    result = st.session_state.get("result")
    if result:
        show_result(result)


def run_analysis(cfg, upload, pm5_upload, pm5_lag, mode, start_s, max_seconds, stabilize,
                 click_pct) -> None:
    from wodvision.io.frames import grab_frame
    from wodvision.pm5.session import load_pm5

    cfg.camera_motion.enabled = stabilize
    if HOSTED:
        max_seconds = min(max_seconds or cfg.hosted.max_seconds, cfg.hosted.max_seconds)
    work = Path(tempfile.mkdtemp(prefix="wodvision_")) if HOSTED else None
    upload_dir = work / "up" if work else UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / Path(upload.name).name
    video_path.write_bytes(upload.getbuffer())
    bar = st.progress(0.0, text="Demarrage...")

    def progress(frac: float, msg: str) -> None:
        bar.progress(min(1.0, frac), text=msg)

    try:
        pm5_session = None
        if pm5_upload is not None:
            pm5_path = upload_dir / Path(pm5_upload.name).name
            pm5_path.write_bytes(pm5_upload.getbuffer())
            pm5_session = load_pm5(pm5_path)
        click_xy = None
        if click_pct:
            h, w = grab_frame(video_path, start_s).shape[:2]
            click_xy = (click_pct[0] / 100.0 * w, click_pct[1] / 100.0 * h)
        detector, pose = models(mode)
        trace = get_trace(video_path, cfg, mode=mode, click_xy=click_xy,
                          max_seconds=max_seconds or None, detector=detector, pose=pose,
                          progress=progress, start_s=start_s,
                          cache_dir=(work / "cache") if work else "data/cache")
        report = analyze_trace(trace, cfg, pm5=pm5_session, pm5_lag=pm5_lag or None)
        run_dir = new_run_dir(video_path, work / "runs" if work else "data/runs")
        export_run(report, run_dir, cfg)
        dash = render_dashboard(video_path, trace, report, run_dir / "dashboard.mp4",
                                progress=progress)
        st.session_state["result"] = pack_result(report, cfg, dash.read_bytes())
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    finally:
        bar.empty()
        if work:  # version hebergee : rien ne reste sur le serveur
            shutil.rmtree(work, ignore_errors=True)


tab_demo, tab_upload = st.tabs(["Demo instantanee", "Analyser ma video"])
with tab_demo:
    choice = st.radio("Exemple", list(DEMOS), horizontal=True)
    show_result(demo_result(DEMOS[choice]), simulated=True)
with tab_upload:
    upload_tab()

st.caption("Les regles de technique sont des hypotheses de depart (guides Concept2) a calibrer : "
           "voir docs/rowing_concept.md.")
