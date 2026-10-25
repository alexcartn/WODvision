"""Fiche PDF recapitulative : ce qu'il faut retenir, priorites avec conseils, images cles.

Deux niveaux comme dans l'application : "Débutant" (deux pages : l'essentiel, les images, le
schema des phases, le glossaire et les conseils de tournage) et "Coach" (ajoute le detail par
coup et les seuils). La video ne peut pas etre integree a un PDF : la fiche en renvoie les
images cles ; la video annotee reste un fichier a part.

Bibliotheque : reportlab (licence BSD). Police DejaVu (fournie avec matplotlib) pour les accents.
"""

from __future__ import annotations

import datetime as dt
import io
import os
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from wodvision.analysis.rowing.ghost import build_ghost
from wodvision.coaching import CoachingContent, summarize
from wodvision.config import GhostConfig
from wodvision.viz.rowing_curves import curves_png, force_png, sequence_png
from wodvision.viz.rowing_diagram import default_phases_diagram

STATUS_COLOR = {"ok": "#2f8f5b", "warn": "#e08a1e", "bad": "#d64545", "na": "#9aa0ad"}
STATUS_WORD = {"ok": "Bien", "warn": "À surveiller", "bad": "À corriger", "na": "Non mesuré"}
INK, MUTED, PAPER = "#1d2330", "#5d6475", "#f3f5f8"
_FONTS_READY = False


def _register_fonts() -> None:
    global _FONTS_READY
    if _FONTS_READY:
        return
    fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(fonts, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(fonts, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
    _FONTS_READY = True


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="DejaVu", fontSize=9.5, leading=13.5,
                          textColor=colors.HexColor(INK))
    return {
        "base": base,
        "brand": ParagraphStyle("brand", parent=base, fontName="DejaVu-Bold", fontSize=26,
                                leading=30, textColor=colors.HexColor("#e5602b")),
        "title": ParagraphStyle("title", parent=base, fontName="DejaVu-Bold", fontSize=21,
                                leading=25),
        "sub": ParagraphStyle("sub", parent=base, textColor=colors.HexColor(MUTED), fontSize=9),
        "h2": ParagraphStyle("h2", parent=base, fontName="DejaVu-Bold", fontSize=13, leading=17,
                             spaceBefore=10, spaceAfter=5),
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=11,
                                textColor=colors.HexColor(MUTED)),
        "tiny": ParagraphStyle("tiny", parent=base, fontSize=6.8, leading=8.5),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11, alignment=1),
        "num": ParagraphStyle("num", parent=base, fontName="DejaVu-Bold", fontSize=15, leading=19,
                              alignment=1),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8, leading=10.5,
                                  textColor=colors.HexColor(MUTED)),
    }


class _Dot(Flowable):
    """Pastille de statut coloree."""

    def __init__(self, color: str, radius: float = 7.0) -> None:
        super().__init__()
        self.color, self.r = color, radius
        self.width = self.height = 2 * radius

    def draw(self) -> None:
        self.canv.setFillColor(colors.HexColor(self.color))
        self.canv.circle(self.r, self.r, self.r, stroke=0, fill=1)


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def _img(data: bytes, width: float, height: float | None = None) -> Image:
    """Image a la largeur donnee ; si `height` est fixee, la largeur est reduite au besoin."""
    w, h = ImageReader(io.BytesIO(data)).getSize()
    if height is not None:
        width = min(width, height * w / h)
        return Image(io.BytesIO(data), width=width, height=width * h / w)
    return Image(io.BytesIO(data), width=width, height=width * h / w)


def _on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(1.6 * cm, 1.0 * cm, "Pimp my row · fiche technique d'aviron · conseils "
                      "indicatifs, à confirmer avec ton coach")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _criteria_table(summary, s) -> Table:
    n = len(summary.criteria)
    rows = [
        [_Dot(STATUS_COLOR[c.status]) for c in summary.criteria],
        [_p(c.label, s["cell"]) for c in summary.criteria],
        [Paragraph(f'<font color="{STATUS_COLOR[c.status]}"><b>{STATUS_WORD[c.status]}</b>'
                   f'{" (indicatif)" if c.indicative and c.status != "na" else ""}</font>',
                   s["cell"]) for c in summary.criteria],
    ]
    t = Table(rows, colWidths=[(A4[0] - 3.2 * cm) / n] * n)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PAPER)),
                           ("TOPPADDING", (0, 0), (-1, 0), 8),
                           ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
    return t


def _numbers_table(report, s) -> Table:
    target = float(np.nanmedian([m.target_recovery_ratio for m in report.strokes])) \
        if report.strokes else float("nan")
    cells = [
        (f"{report.n_strokes}", "coups analysés"),
        (f"{report.stroke_rate_spm:.0f}", "coups par minute"),
        (f"{report.recovery_drive_ratio:.1f} ×", f"retour / poussée (repère à ta cadence : "
                                                 f"{target:.1f})"),
    ]
    t = Table([[Paragraph(v, s["num"]) for v, _ in cells], [_p(t, s["cell"]) for _, t in cells]],
              colWidths=[(A4[0] - 3.2 * cm) / 3] * 3)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d5d9e2")),
                           ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e6e9ef"))]))
    return t


def _priority_block(i: int, p, s, coach: bool) -> Table:
    color = STATUS_COLOR["bad" if p.severity >= 0.6 and p.reliability == "reliable" else "warn"]
    tag = " (indicatif : vue non profil)" if p.reliability == "indicative" else ""
    parts = [
        Paragraph(f"<b>Priorité {i} : {escape(p.tip.title)}</b>{tag}", ParagraphStyle(
            "pt", parent=s["base"], fontSize=11, leading=15)),
        Spacer(1, 3),
        Paragraph(escape(p.tip.why), s["base"]),
        Spacer(1, 2),
        Paragraph(f"<b>Ce que tu dois sentir :</b> {escape(p.tip.feeling)}", s["base"]),
        Spacer(1, 2),
        Paragraph(f"<b>Exercice :</b> {escape(p.tip.drill)}", s["base"]),
    ]
    if coach:
        parts += [Spacer(1, 2), _p(p.metric_line, s["small"])]
    t = Table([["", parts]], colWidths=[0.25 * cm, A4[0] - 3.2 * cm - 0.25 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(color)),
                           ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d5d9e2")),
                           ("LEFTPADDING", (1, 0), (1, 0), 9), ("TOPPADDING", (1, 0), (1, 0), 7),
                           ("BOTTOMPADDING", (1, 0), (1, 0), 8)]))
    return t


def _highlights_section(hl, s) -> list:
    if hl is None:
        return []
    similar = hl.meta["worst"].get("similar", False)
    keys = (("worst", "Un de tes coups comparé à la référence"),) if similar else (
        ("best", "Ton meilleur coup"), ("worst", "Coup à améliorer"))
    out = [Paragraph("Tes images clés", s["h2"]),
           _p("Vert : toi. Jaune : posture de référence. La flèche blanche montre l'articulation "
              "qui s'écarte le plus et où elle devrait aller. La vidéo annotée, avec ses "
              "ralentis, est un fichier à part.", s["small"]), Spacer(1, 4)]
    width = (A4[0] - 3.2 * cm) / 2 - 0.2 * cm
    height = 4.9 * cm
    for key, label in keys:
        m = hl.meta[key]
        cap = f"{label} (coup n°{m['stroke']})"
        row = [[_img(hl.images[f"{key}_catch"], width, height),
                _img(hl.images[f"{key}_finish"], width, height)],
               [_p(f"{cap} : au catch" + (f", écart surtout au {m['catch_joint']}"
                                          if m["catch_joint"] else ""), s["caption"]),
                _p("Au finish" + (f", écart surtout au {m['finish_joint']}"
                                  if m["finish_joint"] else ""), s["caption"])]]
        out.append(KeepTogether(Table(row, colWidths=[width + 0.2 * cm] * 2,
                                      style=[("VALIGN", (0, 0), (-1, -1), "TOP")])))
        out.append(Spacer(1, 6))
    return out


def _coach_appendix(report, s) -> list:
    out = [PageBreak(), Paragraph("Détail pour le coach", s["h2"]), Paragraph(
        "Mesures par coup (repère machine : le rail sert d'horizontale, indépendant du roulis de "
        "caméra).", s["small"]), Spacer(1, 4)]
    head = ["#", "c/min", "drive s", "retour s", "ratio", "genou catch", "genou finish",
            "buste catch", "buste finish", "tibia catch"]
    rows = [head] + [[str(m.index + 1), f"{m.stroke_rate_spm:.0f}", f"{m.drive_time_s:.2f}",
                      f"{m.recovery_time_s:.2f}", f"{m.recovery_drive_ratio:.2f}",
                      f"{m.knee_angle_catch:.0f}°", f"{m.knee_angle_finish:.0f}°",
                      f"{m.trunk_lean_catch:.0f}°", f"{m.trunk_lean_finish:.0f}°",
                      f"{m.shin_angle_catch:.0f}°"] for m in report.strokes]
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
                           ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PAPER)),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e4eb"))]))
    out += [t, Spacer(1, 8), Paragraph("Constats et seuils", s["h2"])]
    if not report.findings:
        out.append(_p("Aucun constat avec les seuils actuels.", s["base"]))
    for f in report.findings:
        out.append(_p(f"{f.title}{' (indicatif)' if f.reliability == 'indicative' else ''} : "
                      f"{f.metric} médian = {f.median_value:.2f} (seuil {f.comparator} "
                      f"{f.threshold:g}), {f.n_flagged}/{f.n_valid} coups.", s["small"]))
    out += [Spacer(1, 6), _p(
        "Repères : ratio retour/poussée de Kleshnev 2005 (Concept2, 5 rameuses), angles de "
        "genou et de buste de Penichet-Tomas et al. 2025 (13 rameuses d'élite), consignes Concept2 "
        "pour le reste (docs/references_ergometre.md). Les autres seuils sont des hypothèses ; la "
        "posture de référence est un modèle, pas une norme.",
        s["small"])]
    return out


def build_fiche_pdf(
    report,
    content: CoachingContent,
    highlights=None,
    level: str = "Débutant",
    source_name: str = "",
    athlete: str = "",
) -> bytes:
    """Fiche PDF (octets) pour une analyse. `level` : "Débutant" ou "Coach"."""
    _register_fonts()
    s = _styles()
    coach = level == "Coach"
    summary = summarize(report, content, max_priorities=3 if coach else 2)

    doc_title = "Pimp my row : fiche technique d'aviron" + (f" : {athlete}" if athlete else "")
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                          topMargin=1.5 * cm, bottomMargin=1.7 * cm,
                          title=doc_title,
                          author="Pimp my row")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_on_page)])

    sub = f"{dt.date.today():%d/%m/%Y} · {report.mode_label}" + (f" · {source_name}"
                                                                if source_name else "")
    de = "d'" if athlete[:1].lower() in "aeiouyhéèêâîô" else "de "
    heading = (f"Fiche technique d'aviron {de}{athlete}" if athlete
               else "Ta fiche technique d'aviron")
    story = [Paragraph("PIMP MY ROW", s["brand"]), Paragraph(escape(heading), s["title"]),
             Spacer(1, 2),
             _p(sub, s["sub"]), Spacer(1, 8),
             Paragraph("En un coup d'œil", s["h2"]), _criteria_table(summary, s), Spacer(1, 6),
             _numbers_table(report, s), Spacer(1, 6)]
    for note in summary.notes:
        story.append(_p(note, s["small"]))
    if summary.strengths:
        t = Table([[Paragraph(f"<b>Point fort :</b> {escape(summary.strengths[0])}", s["base"])]],
                  colWidths=[A4[0] - 3.2 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e5f3ec")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 9),
                               ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        story += [Spacer(1, 4), t]
    story.append(Paragraph("Tes priorités", s["h2"]))
    if summary.priorities:
        for i, p in enumerate(summary.priorities, 1):
            story += [_priority_block(i, p, s, coach), Spacer(1, 6)]
    else:
        story.append(_p("Rien à corriger en priorité avec les seuils actuels.", s["base"]))

    story.append(KeepTogether([Paragraph("Les quatre phases du coup", s["h2"]),
                               _img(default_phases_diagram(), A4[0] - 3.2 * cm, 4.2 * cm)]))
    story.append(PageBreak())
    story += _highlights_section(highlights, s)
    ghost = build_ghost(report, GhostConfig())
    full_width = A4[0] - 3.2 * cm
    seq = sequence_png(report, ghost)
    if seq is not None:
        story.append(KeepTogether([
            Paragraph("Ton enchaînement et ta régularité", s["h2"]),
            _p("Gauche : à quel moment de la poussée chaque segment se déplie (jambes en bleu, "
               "dos en vert, bras en orange). Les jambes doivent partir en premier. Droite : plus "
               "la barre est courte, plus tu répètes le même geste.", s["small"]),
            Spacer(1, 3), _img(seq, full_width, 6.2 * cm)]))
    curves = curves_png(report, ghost)
    if curves is not None:
        story += [PageBreak(), Paragraph("Ton geste, coup après coup", s["h2"]), _p(
            f"Chaque trait gris est un de tes {len(report.stroke_events)} coups, la courbe bleue "
            "est la moyenne et la bande l'écart-type : plus la bande est fine, plus tu répètes le "
            "même geste. Fond bleu : la poussée. Fond orange : le retour."
            + (" Pointillé jaune : la posture de référence." if ghost is not None else "")
            + " Repère jaune sur la vitesse : position habituelle du pic (74 à 76 % de la "
              "poussée sur Concept2, Kleshnev 2005).",
            s["small"]), Spacer(1, 3), _img(curves, full_width, 15.5 * cm)]
    force = force_png(report)
    if force is not None:
        story += [Spacer(1, 6), KeepTogether([
            Paragraph("Ta courbe de force (PM5)", s["h2"]),
            _p("Forme de la force pendant la poussée, chaque coup ramené à son pic. Une bosse "
               "arrondie, un peu avant le milieu, est le profil visé.", s["small"]),
            Spacer(1, 3), _img(force, full_width, 6.2 * cm)])]
    if coach:
        story += _coach_appendix(report, s)
    doc.build(story)
    return buf.getvalue()
