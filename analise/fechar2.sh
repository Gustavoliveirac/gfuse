#!/usr/bin/env bash
# fechar2.sh — refaz o que falhou no fechamento e junta os resultados da reavaliação do CEMADEN.
# Antes: copie inventario_estacoes.csv (hidromet) para logs/.
# Uso: nohup bash analise/fechar2.sh > logs/fechar2_main.out 2>&1 &   Terminou: "DEVOLVER"
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar2.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
{
$PY -u analise/fechamento.py --etapas qc_mes qc_mapas nucleos_prod
$PY -u analise/figuras_capitulo.py --etapas dominio
$PY -u analise/eventos_quadros.py --stem 20211203_190021 --horas 5 --saida evento_2021_12_03
$PY -u analise/eventos_quadros.py --stem 20220221_190020 --horas 5 --saida evento_2022_02_21
} 2>&1 | tee "$LOG"
Z=logs/fechar2_$C.zip
zip -qj "$Z" resultados/analise/fechamento/qc_limite_mensal.csv resultados/analise/fechamento/qc_mapas* \
  resultados/analise/fechamento/nucleos_produtos_60.csv resultados/analise/figuras_capitulo/fig3_2_dominio.* \
  resultados/analise/v7/evento_202*.* "$LOG"
(cd resultados_cem0 && zip -q "../$Z" metricas_*.csv holdout_*.csv nucleos_*.csv brier_*.csv 2>/dev/null)
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
