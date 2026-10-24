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

from wodvision.analysis.rowing.ghost import build_ghost
from wodvision.analysis.rowing.report import MODE_LABELS, RowingReport
from wodvision.coaching import load_coaching, summarize
from wodvision.config import GhostConfig, load_config
from wodvision.io.cache import load_trace
from wodvision.report_pdf import build_fiche_pdf
from wodvision.rowing_pipeline import (
    analyze_trace,
    build_default_models,
    export_run,
    get_trace,
    new_run_dir,
)
from wodvision.viz.rowing_curves import curves_png, force_png, sequence_png
from wodvision.viz.rowing_dashboard import render_dashboard
from wodvision.viz.rowing_diagram import default_phases_diagram
from wodvision.viz.rowing_highlights import load_highlights, render_highlights

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent / "demo"
UPLOAD_DIR = Path("data/raw/uploads")
HOSTED = os.environ.get("WODVISION_HOSTED") == "1"
CFG_PATH = ROOT / "configs" / "hosted.yaml" if HOSTED else None

DEMOS = {
    "Profil, avec PM5 : dos précoce et chaîne qui monte": "profil",
    "Vue oblique : mode dégradé": "oblique",
}

st.set_page_config(page_title="Pimp my row", layout="wide")
st.title("Pimp my row : ta technique d'aviron sur Concept2")


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
    result = pack_result(report, cfg, (DEMO_DIR / f"{name}.mp4").read_bytes())
    result["ghost"] = build_ghost(report, cfg.ghost) is not None
    hl_dir = DEMO_DIR / f"{name}_highlights"
    if hl_dir.exists():
        result["highlights"] = load_highlights(hl_dir)
    attach_pdfs(result, f"demo_{name}")
    return result


def pack_result(report: RowingReport, cfg, dash_bytes: bytes) -> dict:
    """Resultat autonome en memoire (aucun fichier a conserver cote serveur)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = export_run(report, Path(tmp) / "run", cfg)
        csv_bytes = (run_dir / "strokes.csv").read_bytes()
        json_text = (run_dir / "summary.json").read_text(encoding="utf-8")
    return {"report": report, "dash": dash_bytes, "csv": csv_bytes, "json": json_text}


STATUS_WORD = {"ok": "Bien", "warn": "À surveiller", "bad": "À corriger", "na": "Non mesuré"}


@st.cache_resource
def coaching_content():
    return load_coaching(ROOT / "configs" / "conseils.yaml")


def attach_pdfs(result: dict, source_name: str) -> None:
    """Fiche PDF (version coach) construite une fois l'analyse terminée."""
    result["pdf"] = build_fiche_pdf(result["report"], coaching_content(),
                                    result.get("highlights"), "Coach", source_name)


def show_mode_banner(report: RowingReport, level: str) -> None:
    if report.mode == "full":
        st.success(report.mode_label)
        return
    st.warning(f"Mode : {report.mode_label}")
    with st.expander("Que signifie ce mode ?", expanded=(level == "Coach")):
        st.markdown(
            "- Fiable : nombre de coups, cadence, durée du drive, ratio drive/récupération, "
            "régularité du rythme.\n"
            "- Indicatif (marqué ~) : enchaînement jambes / dos / bras, évolution de l'amplitude "
            "de la poignée. Ces mesures sont relatives, donc moins sensibles à l'angle de vue.\n"
            "- Non mesuré : angles et hauteurs absolus (tibias, recul du buste, hauteur de "
            "poignée, alignement de la chaîne), déformés quand la caméra n'est pas "
            "perpendiculaire au rail.\n"
            "- Pour une analyse complète : caméra fixe, perpendiculaire au rail, à hauteur "
            "de hanche, rameur entier visible."
        )


def show_overview(report: RowingReport, summary, level: str) -> None:
    """Pastilles par critère (avec leur légende) et trois chiffres expliqués."""
    st.subheader("En un coup d'œil")
    cols = st.columns(len(summary.criteria))
    for col, c in zip(cols, summary.criteria, strict=True):
        col.markdown(f"{c.emoji} {c.label}")
        col.caption(STATUS_WORD[c.status] + (" (indicatif)" if c.indicative and c.status != "na"
                                             else ""))
        if level == "Coach":
            col.caption(c.note)
    st.caption("🟢 Bien · 🟠 À surveiller · 🔴 À corriger · ⚪ Non mesuré avec cette vidéo")
    target = float(np.nanmedian([m.target_recovery_ratio for m in report.strokes])) \
        if report.strokes else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coups analysés", report.n_strokes)
    c2.metric("Cadence", f"{report.stroke_rate_spm:.0f} coups/min",
              help="Le nombre de coups par minute.")
    c3.metric("Retour / poussée", f"{report.recovery_drive_ratio:.1f} fois",
              help=f"Combien de fois ton retour dure plus longtemps que ta poussée. Repère à ta "
                   f"cadence : environ {target:.1f}.")


def show_priorities(summary, level: str) -> None:
    if summary.strengths:
        st.success("Point fort : " + summary.strengths[0])
    st.subheader("Tes priorités")
    if not summary.priorities:
        st.success("Rien à corriger en priorité avec les seuils actuels.")
    for i, p in enumerate(summary.priorities, 1):
        with st.container(border=True):
            st.markdown(f"##### Priorité {i} : {p.tip.title}"
                        + (" (indicatif)" if p.reliability == "indicative" else ""))
            st.write(p.tip.why)
            st.markdown(f"Ce que tu dois sentir : {p.tip.feeling}")
            st.markdown(f"Exercice : {p.tip.drill}")
            if level == "Coach":
                st.caption(p.metric_line + (" ; conseil à valider par un coach"
                                            if p.tip.status == "a_valider" else ""))
    for note in summary.notes:
        st.caption(note)


def show_plan(summary) -> None:
    """Les exercices des priorités, regroupés en un petit plan pour la prochaine séance."""
    if not summary.priorities:
        return
    st.subheader("Ton plan pour la prochaine séance")
    st.caption("Un échauffement tranquille, puis :")
    for i, p in enumerate(summary.priorities, 1):
        st.markdown(f"{i}. {p.tip.title} : {p.tip.drill}")
    st.caption("Filme-toi de nouveau de profil après ces exercices pour comparer.")


def show_learn(content) -> None:
    """Pour comprendre les termes et bien filmer (replié par défaut)."""
    with st.expander("Mieux comprendre : les phases du coup, le vocabulaire, bien filmer"):
        st.image(default_phases_diagram(), caption="Les quatre phases du coup")
        for entry in content.glossary:
            st.markdown(f"{entry.term} : {entry.text}")
        if content.filming:
            st.markdown("Pour bien filmer ton prochain coup :")
            for tip in content.filming:
                st.markdown(f"- {tip}")


def show_highlights(res: dict) -> None:
    hl = res.get("highlights")
    if hl is None:
        return
    similar = hl.meta["worst"].get("similar", False)
    st.subheader("Un de tes coups comparé à la référence" if similar
                 else "Ton meilleur coup et le coup à améliorer")
    st.caption("Vert : toi. Jaune : posture de référence. La flèche blanche montre l'articulation "
               "qui s'écarte le plus et où elle devrait aller. Les vidéos sont au ralenti."
               + (" Tes coups se ressemblent : un seul est montré." if similar else ""))
    keys = (("worst", "Coup comparé"),) if similar else (("best", "Ton meilleur coup"),
                                                         ("worst", "Coup à améliorer"))
    for col, (key, label) in zip(st.columns(len(keys)), keys, strict=True):
        m = hl.meta[key]
        with col:
            st.markdown(f"{label} (coup n°{m['stroke']})")
            a, b = st.columns(2)
            a.image(hl.images[f"{key}_catch"], caption="Au catch"
                    + (f" : écart surtout au {m['catch_joint']}" if m["catch_joint"] else ""))
            b.image(hl.images[f"{key}_finish"], caption="Au finish"
                    + (f" : écart surtout au {m['finish_joint']}" if m["finish_joint"] else ""))
            st.video(hl.clips[key])


def show_downloads(res: dict, level: str) -> None:
    coach = level == "Coach"
    st.subheader("Garder une trace")
    cols = st.columns(4 if coach else 2)
    pdf = res.get("pdf")
    if pdf:
        cols[0].download_button("Télécharger la fiche PDF", pdf,
                                "fiche_aviron.pdf", "application/pdf",
                                type="primary")
    cols[1].download_button("Vidéo annotée (mp4)", res["dash"], "dashboard.mp4", "video/mp4")
    if coach:
        cols[2].download_button("Coups (csv)", res["csv"], "strokes.csv", "text/csv")
        cols[3].download_button("Résumé (json)", json.dumps(json.loads(res["json"]), indent=2),
                                "summary.json", "application/json")


def show_curves(report) -> None:
    """Enchainement, regularite et courbes superposees de tous les coups."""
    ghost = build_ghost(report, GhostConfig())
    figures = [sequence_png(report, ghost), curves_png(report, ghost), force_png(report)]
    figures = [f for f in figures if f is not None]
    if not figures:
        return
    st.subheader("Ton geste, coup après coup")
    st.caption("Trait gris : un coup. Courbe bleue : la moyenne. Bande : l'écart-type, donc ta "
               "régularité. Fond bleu : poussée, fond orange : retour. Pointillé jaune : "
               "posture de référence.")
    for f in figures:
        st.image(f)


def show_result(res: dict, level: str = "Débutant", simulated: bool = False) -> None:
    report: RowingReport = res["report"]
    coach = level == "Coach"
    content = coaching_content()
    summary = summarize(report, content, max_priorities=3 if coach else 2)
    if simulated:
        st.info("Données simulées (rameur filaire synthétique) : aucune personne réelle. "
                "Le pipeline est le même que pour tes vidéos.")
    show_mode_banner(report, level)
    mode_texts = tuple(MODE_LABELS.values())
    for w in report.warnings:
        if coach or not w.startswith(mode_texts):  # le mode est deja explique plus haut
            st.warning(w)

    st.subheader("Ta vidéo analysée")
    st.video(res["dash"])
    if res.get("ghost"):
        st.caption("Squelette vert : toi. Squelette jaune : la posture de référence (modèle issu "
                   "des consignes Concept2, hypothèses à calibrer), à la même phase du coup.")
    elif res.get("ghost_requested"):
        st.info("Mode fantôme indisponible : il faut une vue de profil (analyse complète).")
    if coach and report.camera_motion_px > 20:
        st.caption(f"Mouvement de caméra détecté (~{report.camera_motion_px:.0f} px) : compensé.")

    show_overview(report, summary, level)
    show_priorities(summary, level)
    if not coach:
        show_plan(summary)
    show_highlights(res)
    show_curves(report)

    if coach:
        st.subheader("Mesures détaillées")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Coups", report.n_strokes)
        c2.metric("Cadence médiane", f"{report.stroke_rate_spm:.1f} c/min")
        c3.metric("Durée du drive", f"{report.drive_time_s:.2f} s")
        c4.metric("Récup / drive", f"{report.recovery_drive_ratio:.2f} : 1")
        c5.metric("Régularité (CV)", f"{report.rate_cv_pct:.1f} %")
        st.subheader("Constats de technique (détail)")
        if not report.findings:
            st.success("Aucune faute détectée avec les seuils actuels.")
        for f in report.findings:
            with st.container(border=True):
                st.markdown(f"{f.title}"
                            + ("  (indicatif)" if f.reliability == "indicative" else ""))
                st.write(f.advice)
                st.caption(f"{f.metric} médian = {f.median_value:.2f} (seuil {f.comparator} "
                           f"{f.threshold:g}), {f.n_flagged}/{f.n_valid} coups concernés")
        if report.pm5 is not None:
            show_pm5(report)
        st.subheader("Détail par coup")
        df = pd.read_csv(io.BytesIO(res["csv"])).drop(columns=["force_curve"], errors="ignore")
        st.dataframe(df.round(2), width="stretch")
    else:
        show_learn(content)

    show_downloads(res, level)


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


def upload_tab(level: str) -> None:
    cfg = load_config(CFG_PATH)
    if HOSTED:
        st.warning(
            f"Version d'essai en ligne : {cfg.hosted.max_seconds:.0f} s de vidéo maximum, modèle "
            "de pose léger, calcul sur un serveur gratuit partagé (compte une à deux minutes). "
            "La vidéo est supprimée du serveur après l'analyse. Pour des vidéos longues ou "
            "confidentielles, installe l'outil en local (voir le dépôt GitHub)."
        )
    upload = st.file_uploader("Dépose ta vidéo (MP4, AVI, MOV, MKV)",
                              type=["mp4", "avi", "mov", "mkv"], key="video")
    with st.expander("Réglages et fichier PM5 (optionnel)"):
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
        ghost_on = st.checkbox(
            "Mode fantome : superposer la posture de reference", value=True,
            help="Squelette jaune = modele de posture issu des consignes Concept2, mis a "
                 "l'echelle du rameur et synchronise sur son coup. Vue de profil seulement.")
        pick = st.checkbox("Choisir le rameur manuellement (plusieurs personnes dans l'image)")
        click_pct = None
        if pick:
            click_pct = (st.slider("Horizontal (%)", 0, 100, 50),
                         st.slider("Vertical (%)", 0, 100, 50))
    st.caption("Vue de profil, rameur entier visible, de préférence 30 images/s ou plus.")
    consent = True
    if HOSTED:
        consent = st.checkbox(
            "Je confirme avoir l'accord des personnes filmées et j'accepte que la vidéo soit "
            "traitée sur un serveur externe (supprimée après l'analyse).")
    go = st.button("Analyser", type="primary", disabled=upload is None or not consent)

    if go and upload is not None:
        cfg.ghost.enabled = ghost_on
        run_analysis(cfg, upload, pm5_upload, int(pm5_lag), mode, float(start_s),
                     float(max_seconds), stabilize, click_pct)
    result = st.session_state.get("result")
    if result:
        show_result(result, level)


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
        ghost_model = build_ghost(report, cfg.ghost)
        dash = render_dashboard(video_path, trace, report, run_dir / "dashboard.mp4",
                                progress=progress, ghost=ghost_model)
        result = pack_result(report, cfg, dash.read_bytes())
        result["highlights"] = None
        if ghost_model is not None:
            progress(0.99, "Images clés et ralentis...")
            try:
                result["highlights"] = render_highlights(video_path, trace, report, ghost_model)
            except Exception:  # les images clés sont un bonus : l'analyse reste affichée
                logging.getLogger(__name__).exception("images clés non générées")
        result["ghost"] = ghost_model is not None
        attach_pdfs(result, Path(upload.name).name)
        result["ghost_requested"] = cfg.ghost.enabled
        st.session_state["result"] = result
    except ValueError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    finally:
        bar.empty()
        if work:  # version hebergee : rien ne reste sur le serveur
            shutil.rmtree(work, ignore_errors=True)


level = st.radio("Affichage", ["Débutant", "Coach"], horizontal=True,
                 help="Débutant : l'essentiel et des conseils simples. Coach : tous les chiffres, "
                      "les seuils et les données détaillées.")
tab_demo, tab_upload = st.tabs(["Démo instantanée", "Analyser ma vidéo"])
with tab_demo:
    choice = st.radio("Exemple", list(DEMOS), horizontal=True)
    show_result(demo_result(DEMOS[choice]), level, simulated=True)
with tab_upload:
    upload_tab(level)

st.caption("Les règles de technique sont des hypothèses de départ (guides Concept2) à calibrer : "
           "voir docs/rowing_concept.md.")
