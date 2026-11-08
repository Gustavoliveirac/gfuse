#!/usr/bin/env bash
# baixar_reiniciar.sh — mata o download dos produtos se ainda estiver rodando,
# apaga arquivos possivelmente cortados no momento da interrupção, registra os
# erros dos logs anteriores, conta o que já foi baixado e reinicia o download.
#
# Uso (de dentro da pasta do gfuse3):
#   bash analise/baixar_reiniciar.sh                 diagnostica e reinicia
#   bash analise/baixar_reiniciar.sh --sem-reiniciar só diagnostica
# Devolver: logs/baixar_diagnostico.log
cd "$(dirname "$0")/.." || exit 1
DADOS=/media/gustavo/Elements/imerg/imerg_data
LOG=logs/baixar_diagnostico.log
mkdir -p logs/baixar_anterior
{
echo "== $(date '+%Y-%m-%d %H:%M')"
echo "== processos antes"
pgrep -af "baixar_produtos.py" || echo "nenhum"
if pgrep -f "baixar_produtos.py" >/dev/null; then
  pkill -f "baixar_produtos.py"; sleep 5
  pkill -9 -f "baixar_produtos.py" 2>/dev/null; sleep 1
fi
echo "== processos depois"
pgrep -af "baixar_produtos.py" || echo "nenhum"

echo "== disco"
if mountpoint -q /media/gustavo/Elements; then
  df -h /media/gustavo/Elements | tail -1
  if touch "$DADOS/.teste_escrita" 2>/dev/null; then echo "escrita ok"; rm -f "$DADOS/.teste_escrita"
  else echo "ERRO: sem permissão de escrita em $DADOS"; fi
else
  echo "ERRO: /media/gustavo/Elements não está montado"
fi

echo "== arquivos gravados nos últimos 5 min (podem ter sido cortados; serão apagados)"
for d in merge_hourly pdir_now gsmap; do
  find "$DADOS/$d" -type f \( -name '*.npz' -o -name '*.part' -o -name '*.tmp' -o -name '*.grib2' \) -mmin -5 -print -delete 2>/dev/null
done

for p in merge pdir gsmap; do
  f=logs/baixar_$p.log
  [ -f "$f" ] || continue
  cp "$f" "logs/baixar_anterior/baixar_${p}_$(date +%Y%m%d_%H%M).log"
  echo "== $f: linhas com erro (agrupadas)"
  grep -iE "erro|error|traceback|exception|falh|timeout|refused" "$f" | sed -E 's/[0-9]{8,}/N/g' | sort | uniq -c | sort -rn | head -15
  echo "== $f: progresso"
  grep -E "\.\.\. [0-9]+/|FIM|sem arquivo" "$f" | tail -5
  echo "== $f: últimas 15 linhas"
  tail -15 "$f"
done
[ -f logs/baixar_main.out ] && { echo "== baixar_main.out"; tail -20 logs/baixar_main.out; }

echo "== o que ainda falta"
python analise/baixar_produtos.py --dry-run --produto merge pdir gsmap
} > "$LOG" 2>&1

if [ "$1" != "--sem-reiniciar" ]; then
  nohup python analise/baixar_produtos.py > logs/baixar_main.out 2>&1 &
  echo "== download reiniciado (pid $!)" >> "$LOG"
fi
cat "$LOG"
echo "DEVOLVER: $LOG"
