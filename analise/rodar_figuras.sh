#!/usr/bin/env bash
# rodar_figuras.sh — gera todas as figuras de dados do capítulo, na ordem do texto, com o radar da Cemig.
# Uso: nohup bash analise/rodar_figuras.sh > logs/figuras_main.out 2>&1 &
# Acompanhar: tail -f logs/figuras.log    Terminou: "DEVOLVER" em logs/figuras_main.out
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs; PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/figuras.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
{
echo "== figuras $(date '+%Y-%m-%d %H:%M') | git $(git describe --tags --always --dirty 2>/dev/null)"
$PY -u analise/figuras_capitulo.py
echo "== fim $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | tee "$LOG"
cp "$LOG" resultados/analise/figuras_capitulo/
Z=logs/figuras_$C.zip
(cd resultados/analise && zip -qr "../../$Z" figuras_capitulo v8_eventos)
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
