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
