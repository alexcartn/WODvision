import numpy as np
import pytest

from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.analysis.rowing.ghost import (
    GhostModel,
    build_ghost,
    estimate_anthropometry,
    phase_of,
)
from wodvision.analysis.rowing.report import analyze_keypoints
from wodvision.config import Config, GhostConfig

# Parametres du rameur synthetique "propre" (tests/synthetic/rowing.py) pour comparer le modele.
MATCHING = GhostConfig(lean_catch_deg=-30.0, lean_finish_deg=25.0, trunk_onset=0.5, trunk_end=0.9,
                       arm_onset=0.8, knee_break=0.6, leg_end=0.7, shin_catch_deg=0.0)


def analyze(**kw):
    t, xy, score, _ = generate_rower(RowerParams(noise_px=0.0, **kw))
    return analyze_keypoints(t, xy, score, Config())


def body_error(report, ghost: GhostModel, joints=("knee", "hip", "shoulder")) -> float:
    """Distance moyenne (en longueurs de tronc) entre posture reelle et fantome, sur les coups."""
    series, frame = report.series, report.frame
    mach = {k: frame.to_machine(v) for k, v in series.pts.items()}
    torso = ghost.a.torso
    errs = []
    for i, t in enumerate(series.t_s):
        ph = phase_of(report.stroke_events, t)
        if ph is None:
            continue
        pose = ghost.pose(*ph)
        errs.append(np.mean([np.linalg.norm(mach[j][i] - pose[j]) for j in joints]) / torso)
    return float(np.mean(errs))


def test_anthropometry_is_estimated_from_the_athlete():
    report = analyze()
    a = estimate_anthropometry(report.series, report.frame)
    # Segments du rameur synthetique : cuisse 0.45, tibia 0.45, tronc 0.5 (m) a 300 px/m.
    assert a.thigh == pytest.approx(0.45 * 300, rel=0.03)
    assert a.torso == pytest.approx(0.50 * 300, rel=0.03)
    assert a.seat_h == pytest.approx((0.35 - 0.25) * 300, rel=0.1)  # hanche 10 cm au-dessus


def test_ghost_matches_a_rower_that_follows_the_same_model():
    report = analyze()
    ghost = GhostModel(estimate_anthropometry(report.series, report.frame), MATCHING)
    assert body_error(report, ghost) < 0.08  # moins de 8 % de longueur de tronc en moyenne


def test_ghost_deviates_when_the_back_opens_early():
    clean = analyze()
    early = analyze(trunk_onset=0.05)
    g_clean = GhostModel(estimate_anthropometry(clean.series, clean.frame), MATCHING)
    g_early = GhostModel(estimate_anthropometry(early.series, early.frame), MATCHING)
    err_early = body_error(early, g_early, ("shoulder",))
    assert err_early > 2 * body_error(clean, g_clean, ("shoulder",))


def test_ghost_pose_keeps_the_anchor_and_segment_lengths():
    report = analyze()
    anthro = estimate_anthropometry(report.series, report.frame)
    ghost = GhostModel(anthro, GhostConfig())
    for kind in ("drive", "recovery"):
        for x in np.linspace(0, 1, 11):
            p = ghost.pose(kind, x)
            assert np.allclose(p["ankle"], anthro.ankle)
            torso = np.linalg.norm(p["shoulder"] - p["hip"])
            assert torso == pytest.approx(anthro.torso, rel=1e-6)
            assert np.linalg.norm(p["hip"] - p["knee"]) == pytest.approx(anthro.thigh, rel=1e-3)
            assert np.linalg.norm(p["knee"] - p["ankle"]) == pytest.approx(anthro.shin, rel=1e-3)
            forearm = np.linalg.norm(p["elbow"] - p["wrist"])
            assert forearm == pytest.approx(anthro.forearm, rel=1e-3)


def test_ghost_follows_the_reference_sequence():
    anthro = estimate_anthropometry(*[getattr(analyze(), k) for k in ("series", "frame")])
    g = GhostModel(anthro, GhostConfig())
    mid = g.pose("drive", 0.5)
    catch, finish = g.pose("drive", 0.0), g.pose("drive", 1.0)
    # Mi-drive : jambes largement etendues, dos a peine ouvert, bras encore tendus.
    leg_done = (mid["hip"][0] - catch["hip"][0]) / (finish["hip"][0] - catch["hip"][0])
    lean = lambda p: GhostModel.trunk_lean_deg(p)  # noqa: E731
    trunk_done = (lean(mid) - lean(catch)) / (lean(finish) - lean(catch))
    assert leg_done > 0.6 and trunk_done < 0.2
    assert lean(finish) == pytest.approx(25.0, abs=0.5)
    assert lean(catch) == pytest.approx(-35.0, abs=0.5)


def test_phase_of():
    report = analyze()
    s = report.stroke_events[1]
    assert phase_of(report.stroke_events, s.t_catch + 0.01)[0] == "drive"
    assert phase_of(report.stroke_events, s.t_finish + 0.01)[0] == "recovery"
    assert phase_of(report.stroke_events, report.stroke_events[0].t_catch - 1.0) is None


def test_no_ghost_outside_profile_view_or_when_disabled():
    p = RowerParams(yaw_deg=50.0)
    t, xy, score, _ = generate_rower(p)
    oblique = analyze_keypoints(t, xy, score, Config())
    assert build_ghost(oblique, GhostConfig()) is None
    assert build_ghost(analyze(), GhostConfig(enabled=False)) is None
    assert build_ghost(analyze(), GhostConfig()) is not None


@pytest.mark.parametrize("roll", [15.0, 90.0])
def test_ghost_is_independent_of_camera_roll(roll):
    base, rolled = analyze(), analyze(roll_deg=roll)
    gb = GhostModel(estimate_anthropometry(base.series, base.frame), MATCHING)
    gr = GhostModel(estimate_anthropometry(rolled.series, rolled.frame), MATCHING)
    assert body_error(rolled, gr) == pytest.approx(body_error(base, gb), abs=0.03)
