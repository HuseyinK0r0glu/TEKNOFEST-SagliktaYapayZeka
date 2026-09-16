"""OTF (CFF) -> TTF (glyf) donusturucu.

Chrome'un Skia PDF arka ucu CFF tabanli OpenType fontlari PDF'e Type 3 olarak
gomuyor; Type 3 bircok baski RIP'inde ikinci sinif vatandas. Ayni konturlari
quadratic'e cevirip TrueType olarak gomdurdugumuzde font PDF'e normal bir
CID TrueType olarak giriyor.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

MAX_ERR = 0.6  # units-per-em cinsinden donusum toleransi


def otf_to_ttf(src: Path, dst: Path) -> None:
    font = TTFont(str(src))
    glyph_set = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder = font.getGlyphOrder()
    glyf.glyphs = {}
    for name in glyf.glyphOrder:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, MAX_ERR, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()

    font["glyf"] = glyf
    font["loca"] = newTable("loca")

    head = font["head"]
    head.indexToLocFormat = 0
    head.glyphDataFormat = 0

    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(
        (len(g.components) for g in glyf.glyphs.values() if g.isComposite()), default=0
    )

    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = glyf.glyphOrder

    for drop in ("CFF ", "VORG"):
        if drop in font:
            del font[drop]

    font.sfntVersion = "\000\001\000\000"
    dst.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(dst))
    print(f"{src.name} -> {dst.name}  ({dst.stat().st_size/1024:.0f} KB, {len(glyf.glyphOrder)} glif)")


if __name__ == "__main__":
    for pair in sys.argv[1:]:
        s, d = pair.split("::")
        otf_to_ttf(Path(s), Path(d))
