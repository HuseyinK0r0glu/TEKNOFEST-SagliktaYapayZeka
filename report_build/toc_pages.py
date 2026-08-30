# -*- coding: utf-8 -*-
"""Üretilen PDF'ten her başlığın gerçek sayfa numarasını çıkarır ve TOC'u günceller."""
from __future__ import annotations
import re, subprocess, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psr_content as C

PDF = next(Path("/tmp/claude-1000/-home-huseyink-Desktop-TEKNOFEST/"
                "c398d6bd-6ea3-4e42-b66e-e6eb8b13a05d/scratchpad/pdfout").glob("*.pdf"))
OFFSET = 1  # basılı numara = fiziksel sayfa - 1 (kapak 0'dan başlar)


def norm(t):
    t = unicodedata.normalize("NFKD", t)
    return re.sub(r"[^0-9a-zA-Z]+", "", t).lower()


pages = []
total = int(re.search(r"Pages:\s+(\d+)",
            subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout).group(1))
for i in range(1, total + 1):
    txt = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(PDF), "-"],
                         capture_output=True, text=True).stdout
    pages.append(norm(txt))

new_toc, missing = [], []
for title, _old in C.TOC:
    key = norm(title)[:40]
    found = None
    for i, ptxt in enumerate(pages[2:], start=3):   # gövde 3. fiziksel sayfada başlar
        if key in ptxt:
            found = i - OFFSET
            break
    if found is None:
        missing.append(title)
        found = _old
    new_toc.append((title, found))

for t, pg in new_toc:
    print(f"  {pg:>3}  {t}")
if missing:
    print("BULUNAMADI:", missing)

src = (HERE / "psr_content.py").read_text(encoding="utf-8")
block = "TOC = [\n" + "".join(
    f'    ("{t}", {pg}),\n' for t, pg in new_toc) + "]\n"
src = re.sub(r"TOC = \[.*?\n\]\n", block, src, count=1, flags=re.S)
(HERE / "psr_content.py").write_text(src, encoding="utf-8")
print("TOC güncellendi")
