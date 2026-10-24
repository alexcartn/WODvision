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
    assert np.median([s.trunk_lean_finish for s in report.strokes]) == pytest.approx(45.0, abs=3.0)
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
    report, _, _ = run(lean_finish_deg=62.0)
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


# --- Vues non profil : mode degrade ---------------------------------------------------------


def run_view(yaw=50.0, **kw):
    p = RowerParams(yaw_deg=yaw, **kw)
    t, xy, score, truth = generate_rower(p)
    return analyze_keypoints(t, xy, score, Config()), p


def test_oblique_view_is_degraded_but_counts_strokes():
    base, _, _ = run()
    report, _ = run_view()
    assert report.mode == "degraded"
    assert any("dégradée" in w for w in report.warnings)
    assert report.n_strokes == base.n_strokes
    assert report.stroke_rate_spm == pytest.approx(24.0, abs=0.7)
    assert report.recovery_drive_ratio == pytest.approx(base.recovery_drive_ratio, abs=0.3)


def test_oblique_clean_stroke_has_no_findings():
    report, _ = run_view()
    assert codes(report) == set(), [f.explain() for f in report.findings]


def test_oblique_masks_absolute_angle_findings():
    report, _ = run_view(chain_deg=15.0, lean_finish_deg=62.0)
    assert not codes(report) & {"chain_misaligned", "layback_excess", "shin_beyond_vertical"}
    assert report.n_masked_findings >= 1


def test_oblique_keeps_relative_findings_as_indicative():
    report, _ = run_view(trunk_onset=0.05)
    early = [f for f in report.findings if f.code == "trunk_early"]
    assert early and early[0].reliability == "indicative"


def test_oblique_keeps_timing_findings_as_reliable():
    report, _ = run_view(rate_spm=24.0, drive_fraction=0.6)
    rushed = [f for f in report.findings if f.code == "recovery_rushed"]
    assert rushed and rushed[0].reliability == "reliable"


def test_frontal_view_is_minimal():
    p = RowerParams(yaw_deg=50.0)
    t, xy, score, _ = generate_rower(p)
    xy[:, [6, 8, 10, 12, 14, 16], 0] += 0.8 * 0.5 * p.px_per_m  # epaules tres ecartees
    report = analyze_keypoints(t, xy, score, Config())
    assert report.mode == "minimal"
    assert report.n_strokes >= 5
    assert not codes(report) & {"trunk_early", "arms_early", "knee_break_early"}


def test_falls_back_to_hip_when_wrist_is_unusable():
    p = RowerParams()
    t, xy, score, _ = generate_rower(p)
    rng = np.random.default_rng(0)
    xy[:, 9] += rng.normal(0, 400.0, (len(t), 2))  # poignet detruit : bien au-dela du filtrage
    report = analyze_keypoints(t, xy, score, Config())
    assert report.signal_used in ("hip", "shoulder")
    assert report.n_strokes >= 5
    assert report.stroke_rate_spm == pytest.approx(24.0, abs=1.0)


def test_detects_travel_loss_with_fatigue():
    p = RowerParams(duration_s=60.0)
    t, xy, score, _ = generate_rower(p)
    # Raccourcit progressivement l'amplitude de la poignee (derive de fatigue).
    shrink = 1.0 - 0.4 * (t / t[-1])
    centre = xy[:, [9]].mean(axis=0, keepdims=True)
    xy[:, 9, 0] = centre[0, 0, 0] + (xy[:, 9, 0] - centre[0, 0, 0]) * shrink
    report = analyze_keypoints(t, xy, score, Config())
    assert "travel_shortening" in codes(report)


def test_rhythm_cv_flags_irregular_periods():
    from wodvision.analysis.rowing.metrics import StrokeMetrics
    from wodvision.analysis.rowing.rules import rhythm_cv
    from wodvision.config import RowingConfig

    def make(periods):
        t, out = 0.0, []
        for i, per in enumerate(periods):
            out.append(
                StrokeMetrics(i, t, t + per / 3, t + per, 60 / per, per / 3, per * 2 / 3, 2.0)
            )
            t += per
        return out

    cv, finding = rhythm_cv(make([2.5, 2.5, 2.5, 2.5, 2.5, 2.5]), RowingConfig())
    assert finding is None and cv < 1
    cv, finding = rhythm_cv(make([1.5, 3.0, 1.8, 3.2, 1.6, 3.0]), RowingConfig())
    assert finding is not None and cv > 12


@pytest.mark.parametrize("drive_fraction", [0.45, 0.5, 0.55, 0.6])
def test_catch_and_finish_are_not_swapped_when_drive_and_recovery_last_the_same(drive_fraction):
    """Rythme rapide : drive et recuperation de duree voisine. Le sens est donne par le genou
    (fléchi au catch, tendu au finish), pas par la comparaison des durees."""
    report, _, _ = run(drive_fraction=drive_fraction, rate_spm=32.0, duration_s=40.0)
    assert report.n_strokes >= 8
    for s in report.strokes:
        assert s.knee_angle_catch < s.knee_angle_finish - 60.0
        assert s.hip_angle_catch < s.hip_angle_finish
    assert report.recovery_drive_ratio == pytest.approx((1 - drive_fraction) / drive_fraction,
                                                        abs=0.25)


def test_recovery_threshold_follows_the_stroke_rate():
    from wodvision.analysis.rowing.metrics import target_recovery_ratio

    table = Config().rowing.ratio_targets
    assert target_recovery_ratio(20.7, table) == pytest.approx(1.40)  # Kleshnev 2005
    assert target_recovery_ratio(32.1, table) == pytest.approx(0.93)
    assert 0.93 < target_recovery_ratio(26.0, table) < 1.40
    assert target_recovery_ratio(10, table) == pytest.approx(1.40)  # borne basse
    assert target_recovery_ratio(50, table) == pytest.approx(0.93)  # borne haute


def test_same_ratio_is_a_fault_when_slow_but_fine_when_fast():
    # Drive = recuperation (ratio 1,0) : trop court a 20 coups/min (repere 1,4), normal a 36.
    slow, _, _ = run(rate_spm=20.0, drive_fraction=0.5, duration_s=60.0)
    fast, _, _ = run(rate_spm=36.0, drive_fraction=0.5, duration_s=60.0)
    assert "recovery_rushed" in codes(slow)
    assert "recovery_rushed" not in codes(fast)
    # Ce qui compte c'est le ratio rapporte au repere de la cadence.
    assert slow.strokes[2].recovery_ratio_vs_target < 0.75
    assert fast.strokes[2].recovery_ratio_vs_target >= 0.75
