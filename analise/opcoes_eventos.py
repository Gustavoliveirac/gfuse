#!/usr/bin/env python
"""
opcoes_eventos.py — escolhe eventos intensos em 2020, 2021 e 2024 e desenha cada um
em horas consecutivas (eventos_quadros.py), para escolher a figura do capítulo.

Critério: hora com o maior acumulado numa estação de 10 min, confirmado por outra
estação a até 20 km com ao menos 10 mm na mesma hora (evita erro de registro), com
pelo menos 3 dias entre eventos escolhidos. Padrão: 3 eventos por ano.

Uso:  source logs/produtos_dirs.sh; python analise/opcoes_eventos.py [--por-ano 3]
Saída: resultados/analise/v8_eventos/opcao_AAAA_k.png e opcoes_eventos.csv
"""
import argparse, subprocess, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import config3 as K
import cemig
import comum as C
from exploratoria import base, janela

SAI = RAIZ / "resultados" / "analise" / "v8_eventos"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--por-ano", type=int, default=3)
    ap.add_argument("--anos", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024]); a = ap.parse_args()
    SAI.mkdir(parents=True, exist_ok=True)
    b = base(); A, ok = janela(b, 60)
    i10 = b["i10"]; px, py = b["px"][i10], b["py"][i10]
    dentro = b["dentro"][i10]
    d_km = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :]) * K.KM_PER_PX
    viz = (d_km > 0) & (d_km <= 20)
    linhas = []
    for ano in a.anos:
        idx = np.flatnonzero(ok & (b["t"].year == ano))
        cand = []
        for i in idx:
            v = np.where(dentro, A[i], np.nan)
            if not np.isfinite(v).any():
                continue
            j = int(np.nanargmax(v)); mx = v[j]
            if mx < 20:
                continue
            conf = np.nanmax(np.where(viz[j], A[i], np.nan)) if viz[j].any() else np.nan
            if np.isfinite(conf) and conf >= 10:
                rad = sum(cemig.disponivel(b["t"][i] + pd.Timedelta(hours=3 + h)) for h in (-2, -1, 0, 1, 2))
                cand.append((rad >= 4, mx, i, conf, int((np.nan_to_num(v) >= 10).sum()), rad))
        cand.sort(reverse=True)   # primeiro os eventos com radar em ao menos 4 das 5 horas
        esc = []
        for tem_rad, mx, i, conf, n10, rad in cand:
            if all(abs((b["t"][i] - b["t"][e[1]]).total_seconds()) > 3 * 86400 for e in esc):
                esc.append((mx, i, conf, n10, rad))
            if len(esc) >= a.por_ano:
                break
        for k, (mx, i, conf, n10, rad) in enumerate(esc):
            stem = b["stems"][i]; nome = f"opcao_{ano}_{k}"
            print(f"{nome}: {stem} (BRT {b['t'][i]}), máx {mx:.1f} mm/h, vizinha {conf:.1f} mm, {n10} estações com ≥ 10 mm, radar em {rad} de 5 h", flush=True)
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "eventos_quadros.py"), "--stem", stem,
                                "--horas", "5", "--saida", nome], cwd=RAIZ, capture_output=True, text=True)
            ok_fig = (RAIZ / "resultados" / "analise" / "v7" / f"{nome}.png").exists()
            if ok_fig:
                shutil.move(str(RAIZ / "resultados" / "analise" / "v7" / f"{nome}.png"), SAI / f"{nome}.png")
            else:
                print(r.stdout[-800:], r.stderr[-800:])
            linhas.append(dict(opcao=nome, goes_stem=stem, hora_brt=b["t"][i], max_mm_h=round(mx, 1),
                               vizinha_mm=round(conf, 1), estacoes_10mm=n10, horas_com_radar=rad, figura=ok_fig))
    pd.DataFrame(linhas).to_csv(SAI / "opcoes_eventos.csv", index=False)
    idx = cemig.indice()
    if idx:
        t = pd.Series(sorted(idx))
        print("radar Cemig, horas com arquivo por ano:", t.dt.year.value_counts().sort_index().to_dict())
    print("FIM")


if __name__ == "__main__":
    main()
