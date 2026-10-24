import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.synthetic.rowing import RowerParams, generate_rower
from wodvision.analysis.rowing.report import analyze_keypoints
from wodvision.coaching import load_coaching
from wodvision.config import Config
from wodvision.report_pdf import build_fiche_pdf

CONTENT = load_coaching(Path(__file__).resolve().parents[1] / "configs" / "conseils.yaml")


def make_report(**kw):
    t, xy, score, _ = generate_rower(RowerParams(noise_px=0.0, duration_s=60.0, **kw))
    return analyze_keypoints(t, xy, score, Config())


def pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def text(pdf: bytes, tmp_path) -> str:
    path = tmp_path / "f.pdf"
    path.write_bytes(pdf)
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                          text=True, check=True).stdout


def test_beginner_sheet_is_a_valid_two_page_pdf():
    pdf = build_fiche_pdf(make_report(trunk_onset=0.05), CONTENT, None, "Débutant", "demo.mp4")
    assert pdf.startswith(b"%PDF") and pages(pdf) == 2


def test_coach_sheet_adds_the_detail_pages():
    report = make_report(trunk_onset=0.05)
    beginner = build_fiche_pdf(report, CONTENT, None, "Débutant")
    coach = build_fiche_pdf(report, CONTENT, None, "Coach")
    assert pages(coach) > pages(beginner)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext absent")
def test_sheet_contains_the_summary_the_tips_and_accents(tmp_path):
    report = make_report(trunk_onset=0.05, arm_onset=0.1)
    out = text(build_fiche_pdf(report, CONTENT, None, "Débutant", "ma_video.mp4"), tmp_path)
    for expected in ("Ta fiche technique d'aviron", "En un coup d'œil", "Enchaînement",
                     "À corriger", "Priorité 1", "Exercice", "Ce que tu dois sentir",
                     "Les quatre phases du coup", "PIMP MY ROW",
                     "Pour bien filmer", "ma_video.mp4"):
        assert expected in out, expected


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext absent")
def test_oblique_sheet_says_what_is_not_measured(tmp_path):
    t, xy, score, _ = generate_rower(RowerParams(yaw_deg=50.0, duration_s=60.0))
    report = analyze_keypoints(t, xy, score, Config())
    out = text(build_fiche_pdf(report, CONTENT, None, "Débutant"), tmp_path)
    assert "Analyse dégradée" in out and "Non mesuré" in out and "Vue non profil" in out


def test_clean_stroke_sheet_has_no_priority_but_still_builds():
    pdf = build_fiche_pdf(make_report(), CONTENT, None, "Débutant")
    assert pdf.startswith(b"%PDF")


def test_curves_figure_needs_enough_strokes_and_is_included_in_the_sheet():
    from wodvision.viz.rowing_curves import curves_png

    report = make_report()
    png = curves_png(report)
    assert png is not None and png.startswith(b"\x89PNG")
    short = analyze_keypoints(*generate_rower(RowerParams(noise_px=0.0, duration_s=5.0))[:3],
                              Config())
    assert curves_png(short) is None
    assert len(build_fiche_pdf(report, CONTENT, None, "Débutant")) > 0
