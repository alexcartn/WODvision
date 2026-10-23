import numpy as np
import pytest

from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.analysis.rowing.report import analyze_keypoints
from wodvision.config import Config


def run(**kw):
    p = RowerParams(**kw)
    t, xy, score, truth = generate_rower(p)
    return analyze_keypoints(t, xy, score, Config()), truth, p


def codes(report):
    return {f.code for f in report.findings}


def test_clean_stroke_counts_and_rate():
    report, truth, p = run()
    assert report.side == "left"
    assert abs(report.n_strokes - (truth["n_strokes_full"] - 1)) <= 1
    assert report.stroke_rate_spm == pytest.approx(24.0, abs=0.7)
    assert report.recovery_drive_ratio == pytest.approx(2.0, abs=0.25)


def test_clean_stroke_has_no_findings():
    report, _, _ = run()
    assert codes(report) == set(), [f.explain() for f in report.findings]


@pytest.mark.parametrize("roll", [0.0, 12.0, -20.0, 90.0])
def test_camera_roll_does_not_change_result(roll):
    base, _, _ = run()
    rolled, _, _ = run(roll_deg=roll)
    assert rolled.n_strokes == base.n_strokes
    assert rolled.stroke_rate_spm == pytest.approx(base.stroke_rate_spm, abs=0.2)
    assert codes(rolled) == set()
    a = np.median([s.trunk_lean_finish for s in base.strokes])
    b = np.median([s.trunk_lean_finish for s in rolled.strokes])
    assert a == pytest.approx(b, abs=1.5)


def test_layback_matches_ground_truth():
    report, _, _ = run()
    assert np.median([s.trunk_lean_finish for s in report.strokes]) == pytest.approx(25.0, abs=3.0)
    assert np.median([s.trunk_lean_catch for s in report.strokes]) == pytest.approx(-30.0, abs=3.0)


def test_detects_early_trunk():
    report, _, _ = run(trunk_onset=0.05)
    assert "trunk_early" in codes(report)


def test_detects_early_arms():
    report, _, _ = run(arm_onset=0.1)
    assert "arms_early" in codes(report)


def test_detects_chain_pulling_up():
    report, _, _ = run(chain_deg=15.0)
    assert "chain_misaligned" in codes(report)


def test_detects_shin_beyond_vertical():
    report, _, _ = run(shin_catch_deg=-20.0)
    assert "shin_beyond_vertical" in codes(report)


def test_detects_early_knee_break():
    report, _, _ = run(knee_break=0.0)
    assert "knee_break_early" in codes(report)


def test_detects_layback_excess():
    report, _, _ = run(lean_finish_deg=50.0)
    assert "layback_excess" in codes(report)


def test_detects_fatigue_drift():
    report, _, _ = run(fatigue_drift_deg=15.0, duration_s=60.0)
    assert "trunk_drift" in codes(report)


def test_too_short_video_warns():
    report, _, _ = run(duration_s=3.0)
    assert report.warnings


def test_non_profile_view_warns():
    p = RowerParams()
    t, xy, score, _ = generate_rower(p)
    # Epaule et hanche droites decalees : largeur d'epaules ~0.8 x tronc (vue oblique).
    offset = np.array([0.0, 0.8 * 0.5 * p.px_per_m])
    xy[:, 6], xy[:, 12] = xy[:, 5] + offset, xy[:, 11] + offset
    score[:, [6, 12]] = 0.9
    report = analyze_keypoints(t, xy, score, Config(), side="left")
    assert any("profil" in w for w in report.warnings)
    assert report.view_shoulder_ratio > 0.5
    # Faute angulaire injectee : masquee puisque la vue n'est pas un profil.
    t, xy, score, _ = generate_rower(RowerParams(chain_deg=15.0))
    xy[:, 6], xy[:, 12] = xy[:, 5] + offset, xy[:, 11] + offset
    score[:, [6, 12]] = 0.9
    masked = analyze_keypoints(t, xy, score, Config(), side="left")
    assert "chain_misaligned" not in {f.code for f in masked.findings}
    assert masked.n_masked_findings >= 1


def test_profile_view_does_not_warn():
    report, _, _ = run()
    assert not any("profil" in w for w in report.warnings)
