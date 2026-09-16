"""Masthead fotoğrafını A0 baskı çözünürlüğüne (300 ppi) yükseltir.

Afişte `assets/nebula-300ppi.jpg` 841 mm genişlikte `cover` olarak basılıyor.
Kaynak dosya 4792 px, yani 145 ppi — baskıda yumuşak çıkıyor. Burada Lanczos
ile 300 ppi'nin gerektirdiği 9933 px'e çıkarıp mikro kontrastı unsharp mask ile
geri veriyoruz; RIP'in kendi bilinear interpolasyonuna bırakmaktan belirgin
şekilde daha keskin.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

PAGE_WIDTH_MM = 841.0
TARGET_PPI = 300
TARGET_WIDTH = round(PAGE_WIDTH_MM / 25.4 * TARGET_PPI)  # 9933 px

SRC = Path(__file__).resolve().parent.parent / "assets" / "nebula.jpg"
DST = SRC.with_name("nebula-300ppi.jpg")


def main() -> None:
    src = Image.open(SRC).convert("RGB")
    height = round(TARGET_WIDTH * src.height / src.width)

    out = src.resize((TARGET_WIDTH, height), Image.LANCZOS)
    out = out.filter(ImageFilter.UnsharpMask(radius=1.8, percent=60, threshold=1))
    out.save(DST, quality=95, subsampling=0, optimize=True, dpi=(TARGET_PPI, TARGET_PPI))

    print(f"{SRC.name} {src.size} -> {DST.name} {out.size} @ {TARGET_PPI} ppi")


if __name__ == "__main__":
    main()
