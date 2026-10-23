import numpy as np
import pytest

from tests.synthetic.rowing import RowerParams
from tests.synthetic.video import make_scene
from wodvision.config import Config
from wodvision.io.cache import load_trace, save_trace
from wodvision.io.probe import probe
from wodvision.rowing_pipeline import analyze_trace, export_run, get_trace
from wodvision.viz.rowing_dashboard import render_dashboard

PARAMS = RowerParams(duration_s=20.0, px_per_m=200.0, origin_px=(250.0, 420.0), noise_px=0.0)


def run(tmp_path, name, **cam):
    video = tmp_path / f"{name}.mp4"
    detector, pose, _ = make_scene(video, PARAMS, **cam)
    cfg = Config()
    trace = get_trace(video, cfg, cache_dir=tmp_path / "cache", detector=detector, pose=pose)
    return video, cfg, trace, analyze_trace(trace, cfg)


def test_static_pipeline_counts_strokes(tmp_path):
    _, _, trace, report = run(tmp_path, "static")
    assert report.n_strokes >= 6
    assert report.stroke_rate_spm == pytest.approx(24.0, abs=1.0)
    assert trace.meta["camera_motion_px"] < 3.0
    assert not report.findings


def test_moving_camera_gives_same_result_as_static(tmp_path):
    _, _, _, static = run(tmp_path, "static")
    _, _, trace, moving = run(tmp_path, "moving", pan_px_s=8.0, roll_deg_s=0.8)
    assert trace.meta["camera_motion_px"] > 40  # le mouvement a bien ete estime
    assert moving.n_strokes == static.n_strokes
    assert moving.stroke_rate_spm == pytest.approx(static.stroke_rate_spm, abs=0.5)
    ref = np.median([s.trunk_lean_finish for s in static.strokes])
    got = np.median([s.trunk_lean_finish for s in moving.strokes])
    assert got == pytest.approx(ref, abs=4.0)
    assert not moving.findings, [f.explain() for f in moving.findings]


def test_cache_roundtrip_and_hit(tmp_path):
    video, cfg, trace, _ = run(tmp_path, "static")
    path = save_trace(trace, tmp_path / "t.npz")
    again = load_trace(path)
    assert np.allclose(again.xy_stab, trace.xy_stab, equal_nan=True)
    # 2e appel : cache, pas de modeles requis (detector/pose absents, deleguerait sinon a rtmlib)
    cached = get_trace(video, cfg, cache_dir=tmp_path / "cache", detector=object(), pose=object())
    assert np.allclose(cached.t_s, trace.t_s)


def test_dashboard_video_and_exports(tmp_path):
    video, cfg, trace, report = run(tmp_path, "static")
    out = render_dashboard(video, trace, report, tmp_path / "dash.mp4")
    info = probe(out)
    assert info.width > 640 and info.height == 480 + 240  # 720 de haut
    assert info.duration_s > 15
    run_dir = export_run(report, tmp_path / "run", cfg)
    assert (run_dir / "strokes.csv").exists() and (run_dir / "summary.json").exists()


def test_detector_runs_only_every_n_frames(tmp_path):
    video = tmp_path / "v.mp4"
    detector, pose, _ = make_scene(video, PARAMS)
    cfg = Config()
    trace = get_trace(video, cfg, cache_dir=tmp_path / "cache", detector=detector, pose=pose)
    n = len(trace.t_s)
    assert detector.calls <= n // cfg.rowing.detect_every_n_frames + 2
    assert analyze_trace(trace, cfg).n_strokes >= 6


def test_pm5_coupling_in_pipeline_and_dashboard(tmp_path):
    from tests.synthetic import pm5 as syn
    from wodvision.pm5.session import session_from_events

    video = tmp_path / "v.mp4"
    params = RowerParams(duration_s=30.0, px_per_m=200.0, origin_px=(250.0, 420.0),
                         noise_px=0.0, period_jitter=0.08)
    detector, pose, _ = make_scene(video, params)
    cfg = Config()
    trace = get_trace(video, cfg, cache_dir=tmp_path / "cache", detector=detector, pose=pose)
    base = analyze_trace(trace, cfg)
    session = session_from_events(syn.make_events(base, lag=2, curve=syn.bell_curve(0.08)))
    report = analyze_trace(trace, cfg, pm5=session)
    assert report.pm5 is not None and report.pm5.lag == 2
    assert "force_peak_early" in {f.code for f in report.findings}
    out = render_dashboard(video, trace, report, tmp_path / "dash_pm5.mp4")
    assert probe(out).height == 900
    run_dir = export_run(report, tmp_path / "run", cfg)
    import pandas as pd

    df = pd.read_csv(run_dir / "strokes.csv")
    assert {"pm5_power_w", "pm5_peak_force_lbf", "force_peak_pos"} <= set(df.columns)
    assert "pm5" in (run_dir / "summary.json").read_text()


def test_downscaled_inference_matches_full_resolution(tmp_path):
    video = tmp_path / "v.mp4"
    detector, pose, _ = make_scene(video, PARAMS)
    full = Config()
    full.rowing.inference_max_width = 0
    t_full = get_trace(video, full, cache_dir=tmp_path / "c_full", detector=detector, pose=pose)

    detector2, pose2, _ = make_scene(video, PARAMS)  # reinitialise les compteurs des faux
    small = Config()
    small.rowing.inference_max_width = 320  # image reduite de moitie
    t_small = get_trace(video, small, cache_dir=tmp_path / "c_small",
                        detector=detector2, pose=pose2)
    assert t_small.meta["width"] == 640  # la trace reste dans le repere d'origine
    both = ~np.isnan(t_full.xy_raw[:, 5, 0]) & ~np.isnan(t_small.xy_raw[:, 5, 0])
    err = np.abs(t_full.xy_raw[both] - t_small.xy_raw[both])
    assert np.nanmax(err) < 2.0
    a, b = analyze_trace(t_full, full), analyze_trace(t_small, small)
    assert a.n_strokes == b.n_strokes


def test_hosted_profile_loads():
    from wodvision.config import load_config

    cfg = load_config("configs/hosted.yaml")
    assert cfg.rowing.analysis_fps == 15.0 and cfg.hosted.max_seconds == 30.0
    assert cfg.rowing.cutoff_hz == 6.0  # defaut conserve


def test_ghost_is_drawn_on_the_video_and_in_the_panel(tmp_path):
    import cv2

    from wodvision.analysis.rowing.ghost import build_ghost

    video = tmp_path / "v.mp4"
    detector, pose, _ = make_scene(video, PARAMS)
    cfg = Config()
    trace = get_trace(video, cfg, cache_dir=tmp_path / "cache", detector=detector, pose=pose)
    report = analyze_trace(trace, cfg)
    ghost = build_ghost(report, cfg.ghost)
    assert ghost is not None
    plain = render_dashboard(video, trace, report, tmp_path / "plain.mp4")
    with_ghost = render_dashboard(video, trace, report, tmp_path / "ghost.mp4", ghost=ghost)

    def frame(path, k):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, img = cap.read()
        assert ok
        return img.astype(int)

    a, b = frame(plain, 150), frame(with_ghost, 150)
    assert a.shape == b.shape
    video_part = np.abs(a - b)[:, : a.shape[1] - 460].mean()
    panel_part = np.abs(a - b)[:, a.shape[1] - 460 :].mean()
    assert video_part > 0.05 and panel_part > 0.05  # le fantome modifie la video et le panneau
