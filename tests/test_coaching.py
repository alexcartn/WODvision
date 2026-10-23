from pathlib import Path

import pytest

from tests.synthetic.pm5 import bell_curve, make_events
from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.analysis.rowing.pm5_sync import couple_pm5
from wodvision.analysis.rowing.report import analyze_keypoints
from wodvision.analysis.rowing.rules import RULES
from wodvision.coaching import load_coaching, summarize
from wodvision.config import Config
from wodvision.pm5.session import session_from_events

CONTENT = load_coaching(Path(__file__).resolve().parents[1] / "configs" / "conseils.yaml")


def report_for(**kw):
    t, xy, score, _ = generate_rower(RowerParams(noise_px=0.0, duration_s=40.0, **kw))
    return analyze_keypoints(t, xy, score, Config())


def status(summary, key):
    return next(c.status for c in summary.criteria if c.key == key)


def test_every_rule_has_a_tip_and_a_criterion():
    codes = {r.code for r in RULES} | {"trunk_drift", "travel_shortening", "rhythm_irregular",
                                       "force_peak_early", "force_multipeak"}
    in_criteria = {c for crit in CONTENT.criteria.values() for c in crit.codes}
    assert codes <= set(CONTENT.tips), codes - set(CONTENT.tips)
    assert codes <= in_criteria, codes - in_criteria


def test_tips_are_complete_and_flagged_for_validation():
    for code, tip in CONTENT.tips.items():
        assert tip.title and tip.why and tip.feeling and tip.drill, code
        assert tip.status in {"a_valider", "valide"}, code


def test_clean_stroke_is_green_with_strengths():
    s = summarize(report_for(), CONTENT)
    assert [c.status for c in s.criteria if c.key != "force"] == ["ok"] * 4
    assert status(s, "force") == "na" and not s.priorities
    assert len(s.strengths) == 4


def test_faults_become_priorities_with_tips_and_red_status():
    s = summarize(report_for(trunk_onset=0.05, arm_onset=0.1), CONTENT)
    assert status(s, "sequence") == "bad"
    codes = [p.code for p in s.priorities]
    assert codes and codes[0] in {"trunk_early", "arms_early"}
    assert s.priorities[0].tip.drill and s.priorities[0].reliability == "reliable"
    assert len(s.priorities) <= 2


def test_priorities_are_limited_and_diverse():
    s = summarize(report_for(trunk_onset=0.05, arm_onset=0.1, chain_deg=15.0,
                             shin_catch_deg=-20.0), CONTENT, max_priorities=2)
    assert len({p.criterion for p in s.priorities}) == len(s.priorities) == 2


def test_oblique_view_marks_positions_not_measured_and_findings_indicative():
    t, xy, score, _ = generate_rower(RowerParams(yaw_deg=50.0, trunk_onset=0.05, duration_s=40.0))
    s = summarize(analyze_keypoints(t, xy, score, Config()), CONTENT)
    assert status(s, "positions") == "na"
    seq = next(c for c in s.criteria if c.key == "sequence")
    assert seq.status in {"warn", "ok"} and seq.indicative  # jamais rouge en vue oblique
    assert any("profil" in n for n in s.notes)


def test_force_criterion_needs_pm5_and_uses_force_findings():
    report = report_for(period_jitter=0.08)
    events = make_events(report, lag=2, curve=bell_curve(0.06))
    report.pm5 = couple_pm5(report, session_from_events(events), Config())
    report.findings.extend(report.pm5.findings)
    s = summarize(report, CONTENT)
    assert status(s, "force") in {"warn", "bad"}
    assert any(p.code == "force_peak_early" for p in s.priorities) or len(s.priorities) == 2


def test_invalid_content_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("criteria: {}\ntips: {x: {title: t}}\n")
    with pytest.raises(ValueError):
        load_coaching(bad)
