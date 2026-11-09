#!/usr/bin/env bash
# rodar_v6.sh — análises e figuras para a v6 do capítulo, num só comando.
#  1. consolida os pares atuais (resumo.py), incluindo o que a v15 já terminou
#  2. exploratória (completude, correlação, descartes, defasagem do RRQPE)
#  3. tabelas por janela, núcleos por tolerância e desempenho por distância
#  4. comparação com os produtos onde cada um existe (dados resgatados no /backup)
#  5. figuras no padrão do hidromet (domínio, limite, tolerância, eventos, confiabilidade)
# Uso (pasta do gfuse3):  nohup bash analise/rodar_v6.sh > logs/v6_main.out 2>&1 &
#   --sem-resumo  pula a etapa 1 (usa os CSVs consolidados que já existem)
# Devolver: o zip da última linha de logs/v6_main.out (logs/v6_AAAAMMDD_HHMM.zip)
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs resultados/analise/figuras
PY=${PYTHON:-python}
[ -f logs/produtos_dirs.sh ] && source logs/produtos_dirs.sh
C=$(date +%Y%m%d_%H%M); LOG=logs/v6.log
{
echo "== v6 $(date '+%Y-%m-%d %H:%M') | git $(git describe --tags --always --dirty 2>/dev/null)"
if [ "$1" != "--sem-resumo" ]; then echo; echo "######## 1 resumo"; $PY -u resumo.py; fi
echo; echo "######## 2 exploratoria"; $PY -u analise/exploratoria.py --etapas completude correlacao descartes defasagem
echo; echo "######## 3 resultados";   $PY -u analise/resultados.py
echo; echo "######## 4 produtos";     $PY -u analise/produtos.py --aggs 60 30
echo; echo "######## 5 figuras";      $PY -u analise/figuras.py
echo; echo "== fim $(date '+%Y-%m-%d %H:%M')"
} 2>&1 | tee "$LOG"
cp "$LOG" resultados/analise/v6.log
Z=logs/v6_$C.zip
(cd resultados && zip -qr "../$Z" analise -x 'analise/produtos_pontos_*.parquet' && \
 zip -qj "../$Z" metricas_*.csv metricas_fold_*.csv holdout_*.csv nucleos_*.csv brier_*.csv calibracao_*.csv 2>/dev/null)
echo "DEVOLVER: $Z ($(du -h "$Z" | cut -f1))"
