#!/bin/bash
SCR=/tmp/claude-1000/-home-huseyink-Desktop-TEKNOFEST/c398d6bd-6ea3-4e42-b66e-e6eb8b13a05d/scratchpad
rm -rf "$SCR/pdfout"; mkdir -p "$SCR/pdfout"
timeout 300 libreoffice --headless --convert-to pdf --outdir "$SCR/pdfout" \
  "/home/huseyink/Desktop/TEKNOFEST/PatHUgen_Proje_Sunus_Raporu.docx" >/dev/null 2>&1
PDF=$(ls "$SCR/pdfout"/*.pdf)
TOTAL=$(pdfinfo "$PDF" | awk '/^Pages/{print $2}')
echo "TOPLAM SAYFA: $TOTAL  (kapak 1 + içindekiler + gövde)"
for i in $(seq 1 $TOTAL); do
  printf "  s%-3s %s\n" "$i" "$(pdftotext -f $i -l $i "$PDF" - 2>/dev/null | tr '\n' ' ' | cut -c1-90)"
done
