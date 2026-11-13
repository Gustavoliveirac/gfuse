#!/usr/bin/env bash
# fechar5.sh — figuras do controle de qualidade refeitas e mapa do domínio com a contagem por papel.
# Uso: nohup bash analise/fechar5.sh > logs/fechar5_main.out 2>&1 &   Terminou: "DEVOLVER" (cerca de 15 min)
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar5.log
{
$PY -u analise/fechamento.py --etapas qc_bascula qc_mapas
$PY -u analise/figuras_capitulo.py --etapas dominio
} 2>&1 | tee "$LOG"
Z=logs/fechar5_$C.zip
zip -qj "$Z" resultados/analise/fechamento/qc_bascula_* resultados/analise/fechamento/qc_mapas* \
  resultados/analise/figuras_capitulo/fig3_2_dominio.* "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
