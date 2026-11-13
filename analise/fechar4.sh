#!/usr/bin/env bash
# fechar4.sh — totais dos meses invalidados, mapa do domínio com o INMET e tabelas de 10 min com o GFUSE.
# Uso: nohup bash analise/fechar4.sh > logs/fechar4_main.out 2>&1 &   Terminou: "DEVOLVER" (cerca de 30 min)
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar4.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
{
$PY -u analise/fechamento.py --etapas qc_mes
$PY -u analise/figuras_capitulo.py --etapas dominio
echo "######## resumo (consolida o GFUSE de 10 min já avaliado na base original)"
$PY -u resumo.py
} 2>&1 | tee "$LOG"
Z=logs/fechar4_$C.zip
zip -qj "$Z" resultados/analise/fechamento/qc_limite_mensal.csv resultados/analise/figuras_capitulo/fig3_2_dominio.* \
  resultados/metricas_10.csv resultados/holdout_10.csv resultados/nucleos_10.csv resultados/metricas_fold_10.csv "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
