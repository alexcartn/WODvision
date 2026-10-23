import asyncio

import numpy as np
import pytest

from tests.synthetic import pm5 as syn
from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.analysis.rowing.force_curve import curve_features
from wodvision.analysis.rowing.pm5_sync import couple_pm5, infer_record_phase
from wodvision.analysis.rowing.report import analyze_keypoints
from wodvision.config import Config, Pm5Config
from wodvision.pm5 import parsers, spec
from wodvision.pm5.recorder import record_events
from wodvision.pm5.session import (
    load_pm5,
    read_events,
    session_from_events,
    session_to_csv,
    write_event,
    write_event_header,
)

# --- Parseurs : aller-retour avec l'encodeur de test (format de la spec) -------------------


def test_stroke_data_roundtrip():
    raw = syn.enc_stroke_data(123.45, 410.0, 1.32, 0.82, 1.63, 9.6, 412.3, 230.5, 288.0, 77)
    rec = parsers.parse_stroke_data(raw)
    assert len(raw) == 20
    assert rec.elapsed_s == pytest.approx(123.45) and rec.stroke_count == 77
    assert rec.drive_length_m == pytest.approx(1.32) and rec.drive_time_s == pytest.approx(0.82)
    assert rec.recovery_time_s == pytest.approx(1.63)
    assert rec.peak_force_lbf == pytest.approx(412.3)
    assert rec.work_j == pytest.approx(288.0)


def test_additional_and_status_roundtrip():
    add = parsers.parse_additional_stroke_data(syn.enc_additional_stroke_data(5.0, 215, 800, 9))
    assert (add.power_w, add.stroke_count) == (215, 9)
    gs = parsers.parse_general_status(syn.enc_general_status(10.5, 2, 118))
    assert gs.stroke_state == spec.StrokeState.DRIVING and gs.drag_factor == 118
    st = parsers.parse_additional_status_1(syn.enc_additional_status_1(10.5, 24))
    assert st.stroke_rate_spm == 24 and st.heart_rate_bpm is None


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        parsers.parse_stroke_data(bytes(19))


def test_force_curve_assembly_and_loss():
    curve = syn.bell_curve(0.4, n=40)
    chunks = syn.enc_force_curve(curve)
    assert len(chunks) == 5 and all(len(c) <= 20 for c in chunks)
    asm = parsers.ForceCurveAssembler()
    got = [asm.feed(c) for c in chunks]
    assert got[:-1] == [None] * 4 and list(got[-1]) == curve
    # Une notification perdue : courbe abandonnee, la suivante est bien lue.
    asm = parsers.ForceCurveAssembler()
    assert all(asm.feed(c) is None for c in chunks[:2] + chunks[3:])
    assert [asm.feed(c) for c in chunks][-1] is not None


def test_multiplexed_split():
    char_id, data = parsers.split_multiplexed(bytes([0x35]) + bytes(20))
    assert char_id == spec.STROKE_DATA and len(data) == 20


def test_uuid():
    assert spec.char_uuid(0x35) == "ce060035-43e5-11e4-916c-0800200c9a66"


# --- Journal, session, CSV ----------------------------------------------------------------------


def make_report(**kw):
    p = RowerParams(duration_s=70.0, period_jitter=0.08, **kw)
    t, xy, score, _ = generate_rower(p)
    return analyze_keypoints(t, xy, score, Config())


def test_event_log_and_csv_roundtrip(tmp_path):
    report = make_report()
    events = syn.make_events(report, curve=syn.bell_curve(0.4))
    path = tmp_path / "pm5.jsonl"
    with path.open("w") as fh:
        write_event_header(fh, "PM5 430000000")
        for t, c, d in events:
            write_event(fh, t, c, d)
    session = load_pm5(path)
    assert len(session.strokes) == len(report.strokes) + 3
    assert session.strokes[0].power_w == 180 and session.has_force_curves
    assert session.drag_factor == 115
    assert len(read_events(path)) == len(events)
    csv = session_to_csv(session, tmp_path / "pm5.csv")
    again = load_pm5(csv)
    assert len(again.strokes) == len(session.strokes)
    assert np.allclose(again.strokes[2].force_curve, session.strokes[2].force_curve)


def test_unknown_format_rejected(tmp_path):
    (tmp_path / "x.txt").write_text("hello")
    with pytest.raises(ValueError):
        load_pm5(tmp_path / "x.txt")


# --- Courbe de force ------------------------------------------------------------------------------


def test_force_features():
    cfg = Pm5Config()
    good = curve_features(np.array(syn.bell_curve(0.4)), cfg)
    early = curve_features(np.array(syn.bell_curve(0.08)), cfg)
    double = curve_features(np.array(syn.bell_curve(0.5, humps=2)), cfg)
    assert 0.3 < good.peak_pos < 0.5 and good.n_peaks == 1
    assert early.peak_pos < 0.2
    assert double.n_peaks == 2
    assert curve_features(np.array([1.0, 2.0]), cfg) is None


# --- Couplage video / PM5 --------------------------------------------------


def couple(report, **kw):
    events = syn.make_events(report, **{k: v for k, v in kw.items() if k != "lag_arg"})
    session = session_from_events(events)
    return couple_pm5(report, session, Config(), lag=kw.get("lag_arg")), session


def test_recovers_lag_and_offset_from_irregular_rhythm():
    report = make_report()
    analysis, _ = couple(report, lag=3, offset_s=12.3, curve=syn.bell_curve(0.4))
    assert analysis.lag == 3 and not analysis.ambiguous
    assert analysis.phase == "finish" and analysis.phase_source == "status"
    assert analysis.offset_s == pytest.approx(12.3, abs=0.1)
    assert analysis.offset_jitter_s < 0.1
    assert analysis.n_matched >= len(report.strokes) - 1


def test_phase_catch_is_inferred_from_status():
    report = make_report()
    events = syn.make_events(report, phase="catch", offset_s=30.0)
    session = session_from_events(events)
    assert infer_record_phase(session, 0.3) == "catch"


def test_validation_metrics_and_implied_torso():
    report = make_report()
    analysis, _ = couple(report, lag=2, drive_bias_s=0.04)
    v = analysis.validation
    assert v["period_mae_ms"] < 80
    assert v["drive_bias_ms"] == pytest.approx(40, abs=40)
    assert v["drive_std_ms"] < 100
    assert v["implied_torso_m"] == pytest.approx(0.5, abs=0.05)
    assert v["implied_torso_cv_pct"] < 15


def test_regular_rhythm_is_flagged_ambiguous():
    p = RowerParams(duration_s=60.0)  # aucune variation : tous les decalages se valent
    t, xy, score, _ = generate_rower(p)
    report = analyze_keypoints(t, xy, score, Config())
    analysis, _ = couple(report, lag=2)
    assert analysis.ambiguous
    assert any("ambigu" in w for w in analysis.warnings)


def test_forced_lag_overrides_search():
    report = make_report()
    analysis, _ = couple(report, lag=3, lag_arg=3)
    assert analysis.lag == 3 and not analysis.ambiguous


def test_force_findings_flag_early_peak_and_multipeak():
    report = make_report()
    early, _ = couple(report, curve=syn.bell_curve(0.06))
    assert "force_peak_early" in {f.code for f in early.findings}
    multi, _ = couple(report, curve=syn.bell_curve(0.5, humps=2))
    assert "force_multipeak" in {f.code for f in multi.findings}
    clean, _ = couple(report, curve=syn.bell_curve(0.4))
    assert not clean.findings


def test_no_curves_warns():
    report = make_report()
    analysis, _ = couple(report)
    assert any("courbe de force" in w for w in analysis.warnings)


def test_too_few_strokes_is_handled():
    report = make_report()
    session = session_from_events(syn.make_events(report)[:5])
    analysis = couple_pm5(report, session, Config())
    assert analysis.n_matched == 0 and analysis.warnings


# --- Enregistreur (faux client BLE) ---------------------------------------------


class FakeClient:
    def __init__(self, script):
        self.script, self.handlers, self.writes = script, {}, []

    async def write_gatt_char(self, uuid, data, response=True):
        self.writes.append((uuid, bytes(data)))

    async def start_notify(self, uuid, handler):
        self.handlers[uuid] = handler

    async def stop_notify(self, uuid):
        self.handlers.pop(uuid, None)

    def push(self):
        for char_id, data in self.script:
            self.handlers[spec.char_uuid(char_id)](None, bytearray(data))


def test_recorder_writes_event_log(tmp_path):
    raw = syn.enc_stroke_data(10.0, 50.0, 1.3, 0.8, 1.6, 9.0, 400.0, 220.0, 280.0, 5)
    client = FakeClient([(spec.STROKE_DATA, raw), (spec.GENERAL_STATUS,
                                                    syn.enc_general_status(10.0, 2, 120))])
    out = tmp_path / "rec.jsonl"

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(record_events(client, out, stop=stop, clock=lambda: 42.0))
        await asyncio.sleep(0.05)
        client.push()
        await asyncio.sleep(0.05)
        stop.set()
        return await task

    assert asyncio.run(run()) == 2
    assert client.writes[0] == (spec.char_uuid(spec.STATUS_SAMPLE_RATE),
                                bytes([spec.SAMPLE_RATE_100MS]))
    session = load_pm5(out)
    assert session.strokes[0].stroke_count == 5 and session.drag_factor == 120
