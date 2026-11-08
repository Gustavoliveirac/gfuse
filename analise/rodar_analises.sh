#!/usr/bin/env bash
# rodar_analises.sh — roda a análise exploratória, as tabelas de resultados e a
# comparação com os produtos, e junta tudo num zip para devolver.
#
# Uso (de dentro da pasta do gfuse3, com o disco Elements montado):
#   bash analise/rodar_analises.sh                  tudo
#   bash analise/rodar_analises.sh --sem-produtos   pula a comparação com os produtos
# Devolver: o zip indicado na última linha (logs/analises_AAAAMMDD_HHMM.zip)
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs resultados/analise/figuras
C=$(date +%Y%m%d_%H%M)
LOG=logs/analises.log
PY=${PYTHON:-python}
{
echo "== analises $(date '+%Y-%m-%d %H:%M') | git $(git describe --tags --always --dirty 2>/dev/null)"
echo; echo "######## exploratoria"; $PY -u analise/exploratoria.py
echo; echo "######## resultados";   $PY -u analise/resultados.py
if [ "$1" != "--sem-produtos" ]; then
  echo; echo "######## produtos";   $PY -u analise/produtos.py --aggs 60 30
fi
echo; echo "== fim $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | tee "$LOG"
cp "$LOG" resultados/analise/analises.log
Z=logs/analises_$C.zip
(cd resultados && zip -qr "../$Z" analise -x 'analise/produtos_pontos_*.parquet')
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
