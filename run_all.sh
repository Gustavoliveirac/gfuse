#!/usr/bin/env bash
# run_all.sh — roda o gfuse3 de ponta a ponta e gera resultados/envio_gfuse3.zip
#
#   bash run_all.sh rapido      # valida o código: fold 0, poucas épocas, subamostra
#   bash run_all.sh completo    # k-fold temporal k=5
#
# Escopo por variável de ambiente (opcional):
#   AGGS="60" ARMS="base rec" bash run_all.sh completo
set -u
MODO="${1:-rapido}"
PY="${PYTHON:-python}"
AGGS="${AGGS:-60 30 10}"
ARMS="${ARMS:-base hurdle krig_res s1 s2 s1_mw s2rk}"
cd "$(dirname "$0")"
mkdir -p resultados
LOG=resultados/master.log

{
echo "== gfuse3 | modo $MODO | aggs [$AGGS] | braços [$ARMS] | $(date '+%F %T') =="

echo "== etapa 1: base de dados =="
"$PY" dados.py || { echo "FALHA em dados.py"; exit 1; }

echo "== etapa 2: vetor de movimento (GOES) =="
"$PY" movimento.py || echo "AVISO: movimento falhou — hist/rec ficam sem advecção"

echo "== etapa 3: diagnósticos do satélite (RRQPE, paralaxe, regra do seco) =="
"$PY" diag_rrqpe.py || echo "AVISO: diag_rrqpe falhou"
"$PY" diag_satelite.py || echo "AVISO: diag_satelite falhou"

if [[ " $ARMS " == *"mw"* ]]; then
  echo "== etapa 3b: MWprecipitation =="
  "$PY" mw_prep.py || echo "[mw] extração falhou — mw_di será pulado"
fi

if [ "$MODO" = "completo" ]; then FOLDS="0 1 2 3 4"; else FOLDS="0"; fi

echo "== etapa 4: treino e avaliação =="
for agg in $AGGS; do
  for arm in $ARMS; do
    for f in $FOLDS; do
      echo "---- m$agg $arm fold$f ----"
      "$PY" treino.py --agg "$agg" --arm "$arm" --fold "$f" --modo "$MODO" \
        || { echo "PULANDO m$agg $arm fold$f (treino falhou)"; continue; }
      "$PY" avalia.py --agg "$agg" --arm "$arm" --fold "$f" --modo "$MODO" \
        || echo "AVISO: avaliação falhou em m$agg $arm fold$f"
    done
  done
done

echo "== etapa 5: mapas dos eventos mais intensos =="
"$PY" mapas_eventos.py --modo "$MODO" --aggs $AGGS --arms $ARMS || echo "AVISO: mapas falharam"

echo "== etapa 5b: advecção pura (krigagem e s2 em t0, projetados sem estações nem GOES) =="
"$PY" adveccao.py $([ "$MODO" = "rapido" ] && echo "--passo 20") || echo "AVISO: adveccao falhou"

echo "== etapa 6: consolidação =="
"$PY" resumo.py --aggs $AGGS
echo "== fim | $(date '+%F %T') =="
} 2>&1 | tee -a "$LOG"

"$PY" resumo.py --aggs $AGGS >/dev/null 2>&1   # re-zipar já com o master.log
echo
echo ">>> ENVIAR PARA ANÁLISE: resultados/envio_gfuse3.zip <<<"
