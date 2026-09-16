# -*- coding: utf-8 -*-
"""PSR şablonunu doldurup teslim edilebilir .docx üretir.

Şartname (son sayfa) kuralları:
  Yazı tipi Aptos 12 / başlık Aptos 14, satır aralığı 1,15, iki yana yaslı,
  kenar boşlukları üst 2,8 cm — alt/sağ/sol 2,5 cm, kapak + içindekiler + sayfa
  numarası zorunlu, kapak ve içindekiler hariç en fazla 10 sayfa.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT,
                            WD_TAB_LEADER)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import psr_content as C

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "2026_Sağlıkta_Yapay_Zeka_PSR-Üniversite_ve_Üzeri_lt7Hv (1).docx"
OUT = ROOT / "PatHUgen_Proje_Sunus_Raporu.docx"

FONT = "Aptos"
BODY_PT = 12
HEAD_PT = 14
W = qn


# --------------------------------------------------------------------- yardımcılar
def set_run(run, *, size=BODY_PT, bold=False, italic=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    # Doğu Asya / karmaşık yazı için de aynı yazı tipini zorla.
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(W("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(W(attr), FONT)
    return run


def fmt_para(p, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=3, spacing=1.15,
             keep_next=False):
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    pf.keep_with_next = keep_next
    return p


def add_para(doc, text, **kw):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=kw.pop("size", BODY_PT), bold=kw.pop("bold", False),
            italic=kw.pop("italic", False))
    return fmt_para(p, **kw)


def add_heading(doc, text, level):
    p = doc.add_paragraph()
    set_run(p.add_run(text), size=HEAD_PT, bold=True, color=RGBColor(0, 0, 0))
    return fmt_para(p, align=WD_ALIGN_PARAGRAPH.LEFT,
                    before=6 if level == 1 else 4, after=1, keep_next=True)


def _set_cell_borders(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(W("w:val"), "single")
        el.set(W("w:sz"), "4")
        el.set(W("w:color"), "9A9A9A")
        borders.append(el)
    tc_pr.append(borders)


def _shade(cell, hexcolor):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(W("w:val"), "clear")
    shd.set(W("w:fill"), hexcolor)
    tc_pr.append(shd)


def _tight_cell_margins(table):
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, w in (("top", 0), ("bottom", 0), ("left", 60), ("right", 60)):
        el = OxmlElement(f"w:{edge}")
        el.set(W("w:w"), str(w))
        el.set(W("w:type"), "dxa")
        mar.append(el)
    tbl_pr.append(mar)


def add_table(doc, spec):
    rows, widths = spec["rows"], spec["widths"]
    size = spec.get("fontsize", 9.5)
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = t.cell(r_i, c_i)
            cell.width = Cm(widths[c_i])
            p = cell.paragraphs[0]
            set_run(p.add_run(str(val)), size=size, bold=(r_i == 0))
            fmt_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=0, spacing=1.0)
            _set_cell_borders(cell)
            if r_i == 0:
                _shade(cell, "E8EDF3")
    _tight_cell_margins(t)
    # Başlık satırı sayfalar arasında tekrarlansın.
    tr_pr = t.rows[0]._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(W("w:val"), "true")
    tr_pr.append(hdr)
    cap = doc.add_paragraph()
    set_run(cap.add_run(spec["caption"]), size=8, italic=True)
    fmt_para(cap, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=4, spacing=1.0)
    return t


def add_page_field(paragraph):
    """Footer'a canlı PAGE alanı ekler."""
    run = paragraph.add_run()
    for tag, attrs, text in (("w:fldChar", {"w:fldCharType": "begin"}, None),
                             ("w:instrText", {"xml:space": "preserve"}, " PAGE "),
                             ("w:fldChar", {"w:fldCharType": "end"}, None)):
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(W(k) if k.startswith("w:") else qn(k), v)
        if text:
            el.text = text
        run._element.append(el)
    set_run(run, size=10)


def clear_body_after_cover(doc):
    body = doc.element.body
    keep = doc.paragraphs[0]._p
    for child in list(body):
        if child is keep or child.tag == W("w:sectPr"):
            continue
        body.remove(child)


def fill_cover(doc):
    """Kapaktaki 'Proje Adı / Takım Adı / Takım ID / Başvuru ID' alanlarını doldurur.

    Değer, etiketin biçimini (beyaz, kalın) taşıyan son run kopyalanarak eklenir;
    aksi hâlde bazı alanlarda değer varsayılan koyu renkle yazılıyor."""
    import copy
    p0 = doc.paragraphs[0]._p
    for inner in p0.iter(W("w:p")):
        texts = list(inner.iter(W("w:t")))
        if not texts:
            continue
        label = "".join(t.text or "" for t in texts).strip().replace("\xa0", "")
        if label not in C.COVER:
            continue
        runs = inner.findall(W("w:r"))
        styled = next((r for r in reversed(runs)
                       if r.find(W("w:rPr")) is not None
                       and r.find(W("w:rPr")).find(W("w:color")) is not None), runs[-1])
        new_run = copy.deepcopy(styled)
        for t in new_run.findall(W("w:t")):
            new_run.remove(t)
        t_el = OxmlElement("w:t")
        t_el.set(qn("xml:space"), "preserve")
        t_el.text = " " + C.COVER[label]
        new_run.append(t_el)
        runs[-1].addnext(new_run)


# --------------------------------------------------------------------- ana akış
def main():
    doc = Document(str(TEMPLATE))
    fill_cover(doc)
    clear_body_after_cover(doc)

    sec = doc.sections[0]
    sec.top_margin, sec.bottom_margin = Cm(2.8), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(2.5), Cm(2.5)
    sec.different_first_page_header_footer = True  # kapakta sayfa numarası yok
    # Sayfa numarası kapakta 0'dan başlar → içindekiler 1, gövde 2'den devam eder.
    # (Word ve LibreOffice'te aynı sonucu vermesi için açıkça belirtilir.)
    pg = OxmlElement("w:pgNumType")
    pg.set(W("w:start"), "0")
    sec._sectPr.append(pg)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(3)
    normal.element.rPr.rFonts.set(W("w:eastAsia"), FONT)
    normal.element.rPr.rFonts.set(W("w:cs"), FONT)

    footer_p = sec.footer.paragraphs[0]
    fmt_para(footer_p, align=WD_ALIGN_PARAGRAPH.CENTER, after=0, spacing=1.0)
    add_page_field(footer_p)

    # ---- İÇİNDEKİLER (kapaktan sonra, sayfa sayısına dâhil değil) -------------
    doc.paragraphs[0].add_run().add_break(WD_BREAK.PAGE)
    h = add_heading(doc, "İÇİNDEKİLER", 1)
    h.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    right_edge = Cm(21.0 - 2.5 - 2.5)
    for title, page in C.TOC:
        p = doc.add_paragraph()
        indent = title[:2].strip().isdigit() and "." in title[:4] and title[1] != "."
        set_run(p.add_run(title), size=11, bold=not indent)
        p.add_run("\t")
        set_run(p.runs[-1], size=11)
        set_run(p.add_run(str(page)), size=11)
        fmt_para(p, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=3, spacing=1.0)
        if indent:
            p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.tab_stops.add_tab_stop(
            right_edge, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)

    # ---- GÖVDE ---------------------------------------------------------------
    body_start = doc.add_paragraph()
    body_start.add_run().add_break(WD_BREAK.PAGE)
    fmt_para(body_start, after=0, spacing=1.0)

    for kind, payload in C.BODY:
        if kind == "h1":
            add_heading(doc, payload, 1)
        elif kind == "h2":
            add_heading(doc, payload, 2)
        elif kind == "p":
            add_para(doc, payload)
        elif kind == "table":
            add_table(doc, payload)

    add_heading(doc, "6. REFERANSLAR", 1)
    for i, ref in enumerate(C.REFERENCES, start=1):
        p = doc.add_paragraph()
        set_run(p.add_run(f"[{i}]\t"), size=9)
        set_run(p.add_run(ref), size=9)
        fmt_para(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, spacing=1.0)
        p.paragraph_format.left_indent = Cm(0.9)
        p.paragraph_format.first_line_indent = Cm(-0.9)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(0.9), WD_TAB_ALIGNMENT.LEFT)

    doc.save(str(OUT))
    print(f"yazıldı: {OUT}")


if __name__ == "__main__":
    main()
