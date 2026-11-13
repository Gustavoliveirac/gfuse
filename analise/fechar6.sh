#!/usr/bin/env bash
# fechar6.sh — figuras 3.4 e 3.5 refeitas. Uso: nohup bash analise/fechar6.sh > logs/fechar6_main.out 2>&1 &
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar6.log
$PY -u analise/fechamento.py --etapas qc_bascula qc_mapas 2>&1 | tee "$LOG"
Z=logs/fechar6_$C.zip
zip -qj "$Z" resultados/analise/fechamento/qc_bascula_* resultados/analise/fechamento/qc_mapas* "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
