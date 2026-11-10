#!/usr/bin/env bash
# rodar_v7.sh — análises da revisão da v6. Uso: nohup bash analise/rodar_v7.sh > logs/v7_main.out 2>&1 &
# Acompanhar: tail -f logs/v7.log   Terminou: aparece "DEVOLVER" em logs/v7_main.out
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs resultados/analise/v7
PY=${PYTHON:-python}
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
C=$(date +%Y%m%d_%H%M); LOG=logs/v7.log
{
echo "== v7 $(date '+%Y-%m-%d %H:%M') | git $(git describe --tags --always --dirty 2>/dev/null)"
$PY -u analise/v7_analises.py
echo; echo "######## evento em quadros"; $PY -u analise/eventos_quadros.py --horas 5
echo; echo "== fim $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | tee "$LOG"
cp "$LOG" resultados/analise/v7/
Z=logs/v7_$C.zip; (cd resultados/analise && zip -qr "../../$Z" v7)
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
