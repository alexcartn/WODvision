"""Point d'entree CLI de WODVision."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import typer

from wodvision.io.probe import probe as probe_video

app = typer.Typer(add_completion=False, no_args_is_help=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.callback()
def main() -> None:
    """WODVision : analyse video de mouvements CrossFit."""


@app.command()
def probe(path: str) -> None:
    """Inspecte un fichier video : resolution, fps, VFR, rotation, duree, codec."""
    result = probe_video(path)
    typer.echo(f"path            : {result.path}")
    typer.echo(f"resolution      : {result.width}x{result.height}")
    typer.echo(f"fps nominal     : {result.fps_nominal:.3f}")
    typer.echo(f"VFR             : {result.is_vfr} (cv={result.pts_interval_cv:.4f})")
    typer.echo(f"duree           : {result.duration_s:.2f} s")
    typer.echo(f"rotation        : {result.rotation_deg} deg")
    typer.echo(f"codec           : {result.codec}")
    typer.echo(f"frames analysees: {result.n_frames_sampled}")


@app.command("pose-preview")
def pose_preview(
    path: str,
    time_s: float = typer.Option(5.0, "--time", help="Instant (s) de la frame a analyser."),
    mode: str = typer.Option("balanced", help="lightweight | balanced | performance"),
    output: str = typer.Option(
        "data/runs/pose_preview.jpg", "--output", help="Chemin de l'image annotee."
    ),
) -> None:
    """Detecte les personnes et leur pose sur une frame, sauvegarde l'image annotee.

    Utilise `rtmlib.Body` (detection + pose bout en bout) car les modules de
    detection/tracking dedies du pipeline ne sont pas encore branches : sert
    de validation visuelle rapide du backend de pose.
    """
    from rtmlib import Body, draw_skeleton

    from wodvision.io.frames import grab_frame
    from wodvision.pose.rtmlib_backend import select_device

    frame = grab_frame(path, time_s)
    device = select_device()
    typer.echo(f"backend pose : rtmlib mode={mode} device={device}")

    body = Body(mode=mode, to_openpose=False, backend="onnxruntime", device=device)
    keypoints, scores = body(frame)

    annotated = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.3)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    typer.echo(f"t={time_s}s : {len(keypoints)} personne(s) detectee(s)")
    typer.echo(f"image annotee : {out_path}")


@app.command("track-preview")
def track_preview(
    path: str,
    start_s: float = typer.Option(0.0, "--start", help="Debut de la fenetre (s)."),
    duration_s: float = typer.Option(5.0, "--duration", help="Duree de la fenetre (s)."),
    mode: str = typer.Option("balanced", help="lightweight | balanced | performance"),
    output: str = typer.Option(
        "data/runs/track_preview.mp4", "--output", help="Chemin de la video annotee."
    ),
) -> None:
    """Detecte + suit les personnes sur une fenetre, sauvegarde une video annotee.

    L'athlete est selectionne par piste a plus grande bbox moyenne (priorite
    3 : ni clic ni barre detectable pour l'instant) et surligne en vert ; les
    autres pistes restent en orange.
    """
    from wodvision.detection.person import PersonDetector
    from wodvision.io.frames import iter_frames
    from wodvision.pose.rtmlib_backend import select_device
    from wodvision.tracking.athlete_selector import select_athlete_track
    from wodvision.tracking.tracker import Tracker

    info = probe_video(path)
    fps_out = info.fps_nominal or 30.0
    device = select_device()
    typer.echo(f"detecteur : rtmlib YOLOX mode={mode} device={device}")

    detector = PersonDetector(mode=mode, device=device)
    tracker = Tracker(frame_rate=fps_out)

    frames: list = []
    tracks_by_frame: list = []
    tracks: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}

    for frame_idx, (_t_s, image) in enumerate(iter_frames(path, start_s, start_s + duration_s)):
        tracked = tracker.update(detector.detect(image))
        frames.append(image)
        tracks_by_frame.append(tracked)
        for tb in tracked:
            tracks.setdefault(tb.track_id, []).append((frame_idx, tb.bbox_xyxy))

    if not frames:
        typer.echo("aucune frame dans la fenetre demandee")
        raise typer.Exit(1)

    athlete_id = select_athlete_track(tracks)
    typer.echo(
        f"{len(tracks)} piste(s) detectee(s) sur {len(frames)} frames, "
        f"athlete choisi : piste {athlete_id}"
    )

    height, width = frames[0].shape[:2]
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (width, height)
    )

    for image, tracked in zip(frames, tracks_by_frame, strict=True):
        annotated = image.copy()
        for tb in tracked:
            x1, y1, x2, y2 = (int(v) for v in tb.bbox_xyxy)
            color = (0, 200, 0) if tb.track_id == athlete_id else (0, 140, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_pos = (x1, max(0, y1 - 8))
            cv2.putText(
                annotated, f"#{tb.track_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        writer.write(annotated)

    writer.release()
    typer.echo(f"video annotee : {out_path}")


@app.command("movement-count")
def movement_count(
    path: str,
    movement: str = typer.Argument(..., help="thruster | row"),
    start_s: float = typer.Option(..., "--start", help="Debut de la fenetre (s)."),
    duration_s: float = typer.Option(..., "--duration", help="Duree de la fenetre (s)."),
    click_x: float = typer.Option(..., "--click-x", help="X approx. de l'athlete, 1ere frame."),
    click_y: float = typer.Option(..., "--click-y", help="Y approx. de l'athlete, 1ere frame."),
    click_size: float = typer.Option(80.0, "--click-size", help="Cote (px) de la bbox de clic."),
    prominence_fraction: float = typer.Option(
        0.3, help="Prominence min., fraction de l'amplitude."
    ),
    min_distance_s: float = typer.Option(1.0, help="Distance temporelle min. entre deux cycles."),
    mode: str = typer.Option("balanced", help="lightweight | balanced | performance"),
    output: str = typer.Option(
        "data/runs/movement_count.png", "--output", help="Chemin du graphique de sortie."
    ),
) -> None:
    """Compte les cycles (reps thruster / coups d'aviron) d'un athlete choisi par clic.

    Version allegee, ad hoc (hors perimetre Phase 1) : pas de module
    d'analyse de mouvement dedie encore, juste angle/position de keypoint ->
    signal.filtering -> analysis.cycle_counter. L'athlete est verrouille par
    IoU avec la bbox de clic sur la premiere frame de la fenetre, puis suivi
    par ByteTrack sur le reste de la fenetre.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from wodvision.analysis.angles import L_WRIST, R_WRIST, knee_angle_deg, pick_visible_side
    from wodvision.analysis.cycle_counter import count_cycles
    from wodvision.config import load_config
    from wodvision.detection.person import PersonDetector
    from wodvision.io.frames import iter_frames
    from wodvision.pose.rtmlib_backend import RTMLibPoseEstimator, select_device
    from wodvision.signal.filtering import butterworth_lowpass
    from wodvision.signal.resampling import resample_uniform
    from wodvision.tracking.athlete_selector import select_athlete_track
    from wodvision.tracking.tracker import Tracker

    if movement not in {"thruster", "row"}:
        typer.echo(f"movement doit etre 'thruster' ou 'row', recu {movement!r}")
        raise typer.Exit(1)

    cfg = load_config()
    info = probe_video(path)
    fps = info.fps_nominal or 30.0
    device = select_device()
    typer.echo(f"backend : rtmlib mode={mode} device={device}")

    detector = PersonDetector(mode=mode, device=device)
    pose_estimator = RTMLibPoseEstimator(mode=mode, device=device)
    tracker = Tracker(frame_rate=fps)

    click_bbox = (
        click_x - click_size / 2,
        click_y - click_size / 2,
        click_x + click_size / 2,
        click_y + click_size / 2,
    )

    frames: list = []
    tracks_by_frame: list = []
    tracks: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}

    for frame_idx, (t_s, image) in enumerate(iter_frames(path, start_s, start_s + duration_s)):
        tracked = tracker.update(detector.detect(image))
        frames.append((t_s, image))
        tracks_by_frame.append(tracked)
        for tb in tracked:
            tracks.setdefault(tb.track_id, []).append((frame_idx, tb.bbox_xyxy))

    if not frames:
        typer.echo("aucune frame dans la fenetre demandee")
        raise typer.Exit(1)

    athlete_id = select_athlete_track(tracks, click_bbox=click_bbox)
    typer.echo(f"athlete verrouille sur la piste {athlete_id}")

    t_obs, xy_obs, score_obs = [], [], []
    for (t_s, image), tracked in zip(frames, tracks_by_frame, strict=True):
        bbox = next((tb.bbox_xyxy for tb in tracked if tb.track_id == athlete_id), None)
        if bbox is None:
            continue
        keypoints = pose_estimator.infer(image, bbox)
        t_obs.append(t_s)
        xy_obs.append(keypoints.xy)
        score_obs.append(keypoints.score)

    if len(t_obs) < 2:
        typer.echo("athlete pas assez souvent visible dans la fenetre pour analyser")
        raise typer.Exit(1)

    t_obs_arr = np.array(t_obs)
    xy_obs_arr = np.stack(xy_obs)
    score_obs_arr = np.stack(score_obs)
    side = pick_visible_side(score_obs_arr)
    typer.echo(f"cote visible retenu : {side}")

    min_conf = cfg.signal.min_keypoint_confidence
    if movement == "thruster":
        wrist = L_WRIST if side == "left" else R_WRIST
        y_raw = -xy_obs_arr[:, wrist, 1]  # pixel y vers le bas -> on remonte vers le haut
        valid = score_obs_arr[:, wrist] > min_conf
    else:
        y_raw = np.array([knee_angle_deg(xy, side=side) for xy in xy_obs_arr])
        hip_idx = 11 if side == "left" else 12
        knee_idx = 13 if side == "left" else 14
        ankle_idx = 15 if side == "left" else 16
        valid = np.min(score_obs_arr[:, [hip_idx, knee_idx, ankle_idx]], axis=1) > min_conf

    if valid.sum() < 2:
        typer.echo("keypoints pertinents trop peu confiants sur cette fenetre")
        raise typer.Exit(1)

    resampled = resample_uniform(
        t_obs_arr, y_raw, valid, target_fps=fps, max_gap_frames=cfg.signal.max_gap_frames
    )
    y_filtered = butterworth_lowpass(
        resampled.y, fps=fps, cutoff_hz=cfg.signal.butterworth_cutoff_hz, order=4
    )
    cycle_times = count_cycles(
        resampled.t_s,
        y_filtered,
        prominence_fraction=prominence_fraction,
        min_distance_s=min_distance_s,
    )

    typer.echo(f"{len(cycle_times)} cycle(s) detecte(s) : {[round(x, 2) for x in cycle_times]}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resampled.t_s, y_filtered, label="signal filtre")
    cycle_y = np.interp(cycle_times, resampled.t_s, y_filtered)
    ax.scatter(cycle_times, cycle_y, color="red", zorder=5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("pixels (thruster)" if movement == "thruster" else "angle genou (deg)")
    ax.set_title(f"{movement} : {len(cycle_times)} cycle(s), piste {athlete_id}, cote {side}")
    ax.legend()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    typer.echo(f"graphique : {out_path}")


def _echo_coaching(report) -> None:
    """Synthese lisible par un amateur (criteres, point fort, priorites avec conseil)."""
    from pathlib import Path

    from wodvision.coaching import load_coaching, summarize

    path = Path(__file__).resolve().parents[2] / "configs" / "conseils.yaml"
    if not path.exists():
        return
    s = summarize(report, load_coaching(path))
    typer.echo("--- ce qu'il faut retenir ---")
    typer.echo("  ".join(f"{c.emoji} {c.label}" for c in s.criteria))
    if s.strengths:
        typer.echo(f"point fort : {s.strengths[0]}")
    for i, p in enumerate(s.priorities, 1):
        tag = " (indicatif : vue non profil)" if p.reliability == "indicative" else ""
        typer.echo(f"priorite {i} : {p.tip.title}{tag}")
        typer.echo(f"  pourquoi : {p.tip.why}")
        typer.echo(f"  exercice : {p.tip.drill}")


def _write_fiches(report, highlights, out, source_name: str, athlete: str = "") -> None:
    """Fiche PDF recapitulative (version coach) : fiche.pdf."""
    from wodvision.coaching import load_coaching
    from wodvision.report_pdf import build_fiche_pdf

    path = Path(__file__).resolve().parents[2] / "configs" / "conseils.yaml"
    if not path.exists():
        return
    content = load_coaching(path)
    (out / "fiche.pdf").write_bytes(
        build_fiche_pdf(report, content, highlights, "Coach", source_name, athlete))
    typer.echo(f"fiche PDF : {out / 'fiche.pdf'}")


def _echo_pm5(p) -> None:
    typer.echo("--- couplage PM5 ---")
    typer.echo(f"coups PM5      : {len(p.session.strokes)} (apparies : {p.n_matched})")
    typer.echo(f"decalage coups : {p.lag:+d}  (confiance {p.confidence:.2f}"
               f"{', AMBIGU' if p.ambiguous else ''})")
    typer.echo(f"phase PM5      : {p.phase} ({p.phase_source})")
    typer.echo(f"decalage horloge: PM5 = video {p.offset_s:+.2f} s "
               f"(gigue {p.offset_jitter_s:.3f} s)")
    for key, value in p.validation.items():
        typer.echo(f"  validation {key}: {value:.2f}")


@app.command("pm5-record")
def pm5_record(
    output: str = typer.Option("data/pm5/session.jsonl", "--output", help="Journal .jsonl."),
    duration_s: float = typer.Option(None, "--duration", help="Duree (s) ; Ctrl-C pour arreter."),
    address: str = typer.Option(None, "--address", help="Adresse BLE du PM5 (sinon scan)."),
) -> None:
    """Enregistre les notifications Bluetooth du PM5 (extra `pm5`, sans-fil du PM5 actif).

    Lancer l'enregistrement, puis demarrer l'effort et la video. Les coups sont apparies
    a la video apres coup (`row-analyze --pm5`), le decalage d'horloge est estime.
    """
    import asyncio

    from wodvision.pm5.recorder import record

    try:
        n = asyncio.run(record(output, duration_s, address))
    except KeyboardInterrupt:
        typer.echo("arrete par l'utilisateur")
        return
    except RuntimeError as exc:
        typer.echo(f"erreur : {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"{n} notifications enregistrees : {output}")


@app.command("pm5-info")
def pm5_info(path: str) -> None:
    """Decode un fichier PM5 et resume ses coups (verification avant couplage)."""
    import numpy as np

    from wodvision.pm5.session import load_pm5

    s = load_pm5(path)
    st = s.strokes
    typer.echo(f"coups          : {len(st)}")
    typer.echo(f"periode mediane: {np.median([x.period_s for x in st]):.2f} s")
    typer.echo(f"drive median   : {np.median([x.drive_time_s for x in st]):.2f} s")
    typer.echo(f"drag factor    : {s.drag_factor:.0f}")
    typer.echo(f"courbes force  : {sum(x.force_curve is not None for x in st)}/{len(st)}")
    typer.echo(f"etats 100 ms   : {len(s.status)} echantillons")


@app.command("row-analyze")
def row_analyze(
    path: str,
    mode: str = typer.Option("balanced", help="lightweight | balanced | performance"),
    athlete: str = typer.Option("", "--athlete", help="Prenom affiche sur la fiche PDF."),
    click_x: float = typer.Option(None, "--click-x", help="X approx. du rameur (px, 1ere frame)."),
    click_y: float = typer.Option(None, "--click-y", help="Y approx. du rameur (px, 1ere frame)."),
    start_s: float = typer.Option(0.0, "--start", help="Debut de la fenetre analysee (s)."),
    max_seconds: float = typer.Option(None, "--max-seconds", help="Limiter la duree analysee."),
    config: str = typer.Option(None, "--config", help="YAML de config (defaut : interne)."),
    output_dir: str = typer.Option(None, "--output-dir", help="Defaut : data/runs/<video>/<date>."),
    no_video: bool = typer.Option(False, "--no-video", help="Ne pas rendre la video annotee."),
    from_cache: str = typer.Option(None, "--from-cache", help="Trace .npz deja calculee."),
    ghost: bool = typer.Option(True, "--ghost/--no-ghost",
                               help="Superpose la posture de reference (profil seulement)."),
    pm5: str = typer.Option(None, "--pm5", help="Fichier PM5 (.jsonl de pm5-record, ou .csv)."),
    pm5_lag: int = typer.Option(None, "--pm5-lag", help="Decalage PM5 - video, en coups."),
    pm5_phase: str = typer.Option(None, "--pm5-phase", help="finish | catch (defaut : auto)."),
) -> None:
    """Analyse technique d'un rameur sur ergometre (vue de profil, camera fixe ou mobile).

    Ecrit strokes.csv, summary.json, config.json et dashboard.mp4 (video annotee
    + squelette seul + indicateurs en direct) dans le dossier de run.
    """
    from wodvision.config import load_config
    from wodvision.io.cache import load_trace
    from wodvision.rowing_pipeline import analyze_trace, export_run, get_trace, new_run_dir
    from wodvision.viz.rowing_dashboard import render_dashboard

    cfg = load_config(config)
    click = (click_x, click_y) if click_x is not None and click_y is not None else None

    def progress(frac: float, msg: str) -> None:
        typer.echo(f"[{frac * 100:3.0f} %] {msg}")

    if from_cache:
        trace = load_trace(from_cache)
        path = trace.meta["video"]
    else:
        trace = get_trace(path, cfg, mode=mode, click_xy=click, max_seconds=max_seconds,
                          progress=progress, start_s=start_s)
    session = None
    if pm5:
        from wodvision.pm5.session import load_pm5

        session = load_pm5(pm5)
    report = analyze_trace(trace, cfg, pm5=session, pm5_lag=pm5_lag, pm5_phase=pm5_phase)
    out = export_run(report, Path(output_dir) if output_dir else new_run_dir(path), cfg)

    typer.echo(f"mode         : {report.mode_label}")
    typer.echo(f"cote visible : {report.side} (signal de coups : {report.signal_used})")
    typer.echo(f"coups        : {report.n_strokes}")
    typer.echo(f"cadence      : {report.stroke_rate_spm:.1f} coups/min (mediane)")
    typer.echo(f"recup/drive  : {report.recovery_drive_ratio:.2f}")
    typer.echo(f"mouvement cam: {report.camera_motion_px:.0f} px")
    typer.echo(f"vue (epaules/tronc) : {report.view_shoulder_ratio:.2f}")
    for w in report.warnings:
        typer.echo(f"ATTENTION : {w}")
    for f in report.findings:
        tag = " [indicatif]" if f.reliability == "indicative" else ""
        typer.echo(f"- {f.explain()}{tag}")
    _echo_coaching(report)
    if report.pm5 is not None:
        _echo_pm5(report.pm5)
    hl = None
    if not no_video:
        from wodvision.analysis.rowing.ghost import build_ghost

        cfg.ghost.enabled = cfg.ghost.enabled and ghost
        ghost_model = build_ghost(report, cfg.ghost)
        if ghost and ghost_model is None:
            typer.echo("fantome indisponible (vue non profil ou donnees insuffisantes)")
        render_dashboard(path, trace, report, out / "dashboard.mp4", progress=progress,
                         ghost=ghost_model)
        if ghost_model is not None:
            from wodvision.viz.rowing_highlights import render_highlights, save_highlights

            try:
                hl = render_highlights(path, trace, report, ghost_model)
                if hl is not None:
                    save_highlights(hl, out / "highlights")
                    typer.echo(f"images cles et ralentis : {out / 'highlights'}")
            except Exception as exc:  # bonus : ne bloque pas l'analyse
                typer.echo(f"images cles non generees ({exc})")
    _write_fiches(report, hl, out, Path(path).name, athlete)
    typer.echo(f"sorties : {out}")


if __name__ == "__main__":
    app()
