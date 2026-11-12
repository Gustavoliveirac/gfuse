#!/usr/bin/env bash
# refazer_eventos.sh — refaz só os eventos escolhidos (2021_1 e 2022_0) e o mapa do domínio.
# Uso: nohup bash analise/refazer_eventos.sh > logs/eventos_main.out 2>&1 &   Terminou: "DEVOLVER"
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/eventos.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
{
$PY -u analise/eventos_quadros.py --stem 20211203_190021 --horas 5 --saida evento_2021_12_03
$PY -u analise/eventos_quadros.py --stem 20220221_190020 --horas 5 --saida evento_2022_02_21
$PY -u analise/figuras_capitulo.py --etapas dominio
} 2>&1 | tee "$LOG"
Z=logs/eventos_$C.zip
zip -qj "$Z" resultados/analise/v7/evento_2021_12_03.* resultados/analise/v7/evento_2022_02_21.* \
   resultados/analise/figuras_capitulo/fig3_2_dominio.* "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
