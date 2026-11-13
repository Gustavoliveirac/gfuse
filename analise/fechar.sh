#!/usr/bin/env bash
# fechar.sh — análises e figuras pendentes para fechar o capítulo.
# Antes: copie o inventario_estacoes.csv do hidromet para logs/ (para o INMET no mapa).
# Uso: nohup bash analise/fechar.sh > logs/fechar_main.out 2>&1 &   Terminou: "DEVOLVER"
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
{
$PY -u analise/fechamento.py
$PY -u analise/figuras_capitulo.py --etapas dominio tolerancia
$PY -u analise/eventos_quadros.py --stem 20211203_190021 --horas 5 --saida evento_2021_12_03
$PY -u analise/eventos_quadros.py --stem 20220221_190020 --horas 5 --saida evento_2022_02_21
} 2>&1 | tee "$LOG"
Z=logs/fechar_$C.zip
zip -qj "$Z" resultados/analise/fechamento/* resultados/analise/figuras_capitulo/fig3_2_dominio.* \
  resultados/analise/figuras_capitulo/fig3_9_tolerancia.* resultados/analise/v7/evento_202*.* "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
