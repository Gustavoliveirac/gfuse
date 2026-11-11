#!/usr/bin/env bash
# rodar_cem0.sh — preenche os zeros do CEMADEN, reavalia com os modelos já treinados e compara.
# Uso: nohup bash analise/rodar_cem0.sh > logs/cem0_main.out 2>&1 &
# Acompanhar: tail -f logs/cem0.log    Terminou: "DEVOLVER" em logs/cem0_main.out
# Para olhar só o diagnóstico e a figura antes da reavaliação (rápido):
#   python analise/cemaden_zeros.py diagnostico; python analise/cemaden_zeros.py preencher
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs resultados/analise/cem0
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/cem0.log
{
echo "== cem0 $(date '+%Y-%m-%d %H:%M') | git $(git describe --tags --always --dirty 2>/dev/null)"
if [ "$1" = "--continuar" ]; then
  for e in avaliar resumir sem_ana; do $PY -u analise/cemaden_zeros.py $e; done
else
  rm -rf processados_cem0 resultados_cem0
  $PY -u analise/cemaden_zeros.py tudo
fi
echo "== fim $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | tee -a "$LOG"
cp "$LOG" resultados/analise/cem0/
Z=logs/cem0_$C.zip; (cd resultados/analise && zip -qr "../../$Z" cem0)
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
