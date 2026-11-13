#!/usr/bin/env bash
# fechar3.sh — refaz o mapa do domínio (INMET e legendas no topo), o painel de mapas do controle de
# qualidade e os dois eventos com as fontes maiores. Requer logs/inventario_estacoes.csv.
# Uso: nohup bash analise/fechar3.sh > logs/fechar3_main.out 2>&1 &   Terminou: "DEVOLVER"
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}; C=$(date +%Y%m%d_%H%M); LOG=logs/fechar3.log
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
[ -f logs/inventario_estacoes.csv ] || echo "ATENÇÃO: falta logs/inventario_estacoes.csv (INMET fica fora do mapa)"
{
$PY -u analise/figuras_capitulo.py --etapas dominio
$PY -u analise/fechamento.py --etapas qc_mapas
$PY -u analise/eventos_quadros.py --stem 20211203_190021 --horas 5 --saida evento_2021_12_03
$PY -u analise/eventos_quadros.py --stem 20220221_190020 --horas 5 --saida evento_2022_02_21
} 2>&1 | tee "$LOG"
Z=logs/fechar3_$C.zip
zip -qj "$Z" resultados/analise/figuras_capitulo/fig3_2_dominio.* resultados/analise/fechamento/qc_mapas* \
  resultados/analise/v7/evento_202*.* "$LOG"
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
