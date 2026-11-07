#!/usr/bin/env python
"""
baixar_produtos.py — completa os produtos de comparação para TODOS os stems do
gfuse3 (a série contínua dos períodos chuvosos), e não só para as horas da
base antiga por eventos, que foi o que o gfuse2 baixou.

Reaproveita os baixadores do gfuse2 (download_merge.py e download_extras.py
em ~/Documentos/gfuse2), que gravam no mesmo formato e nas mesmas pastas do
disco Elements e pulam o que já existe. Por isso pode ser interrompido e
rodado de novo sem perda.

Produtos e necessidade (calculada dos stems de processados/estacoes.npz)
  merge   MERGE/CPTEC horário, uma hora UTC por stem de hora cheia (~90 kB/h)
  pdir    PDIR-Now horário (~1,5 MB/h baixado, recortado e apagado)
  gsmap   GSMaP NRT, Gauge_NRT e NOW horários (~2 MB/h por variante)
  hydro   Hidroestimador do INPE, um arquivo por stem de 10 min (~4,35 MB cada,
          algumas centenas de GB baixados no total; fica fora do padrão)
O IMERG Early já está completo no disco e não é baixado aqui.

Uso (de dentro da pasta do gfuse3, com o disco Elements montado em
/media/gustavo/Elements):
  python analise/baixar_produtos.py --dry-run          só conta o que falta
  python analise/baixar_produtos.py                    merge, pdir e gsmap em paralelo
  python analise/baixar_produtos.py --produto hydro    hidroestimador (demorado)

Saída: logs/baixar_produtos.log (log único; cada produto também deixa o seu
log em logs/baixar_<produto>.log enquanto roda).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
LOGS = RAIZ / "logs"
GFUSE2 = Path(os.environ.get("GFUSE2_DIR", Path.home() / "Documentos" / "gfuse2"))
NPZ = RAIZ / "processados" / "estacoes.npz"
PADRAO = ["merge", "pdir", "gsmap"]


def stems_utc():
    z = np.load(NPZ, allow_pickle=False)
    s = pd.Series(z["stems"].astype(str))
    return pd.to_datetime(s, format="%Y%m%d_%H%M%S")


def horas_fim_utc(t):
    """Hora UTC de fim da janela de 60 min, para os stems de hora cheia
    (mesma tolerância de 120 s do comum.stems_alinhados)."""
    r = t.dt.round("30min")
    ok = ((t - r).abs().dt.total_seconds() <= 120) & (r.dt.minute == 0)
    return sorted(set(r[ok]))


def passos_10min(t):
    return sorted(set(t.dt.round("10min")))


def importar_gfuse2():
    if not (GFUSE2 / "download_extras.py").exists():
        sys.exit(f"não achei {GFUSE2}/download_extras.py (defina GFUSE2_DIR)")
    sys.path.insert(0, str(GFUSE2))
    import download_extras as DX
    import download_merge as DM
    return DX, DM


def falta(lista, caminho):
    return [x for x in lista if not caminho(x).exists()]


def baixar_merge(DM, horas, dry):
    print(f"[merge] {len(horas)} horas sem arquivo")
    if dry or not horas:
        return
    ok = err = 0
    origem = []
    for i, h in enumerate(horas):
        pares, dest = DM.urls_for(h)
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = None
        for src, url in pares:
            if DM.fetch(url, dest):
                got = src
                break
        if got:
            ok += 1
            origem.append(dict(end_utc=h, arquivo=dest.name, fonte=got))
        else:
            err += 1
        if i % 200 == 0:
            print(f"  ... {i}/{len(horas)} (ok={ok} erros={err})", flush=True)
            DX_indice(DM.MERGE_DIR, origem); origem = []
    DX_indice(DM.MERGE_DIR, origem)
    print(f"[merge] FIM: baixados={ok} erros={err}")


def DX_indice(base, origem):
    if not origem:
        return
    idx = Path(base) / "indice_origem.csv"
    novo = pd.DataFrame(origem)
    if idx.exists():
        novo = pd.concat([pd.read_csv(idx), novo], ignore_index=True)
    novo.to_csv(idx, index=False)


def um_produto(produto, dry, kinds):
    DX, DM = importar_gfuse2()
    t = stems_utc()
    print(f"[{produto}] stems do gfuse3: {len(t):,} ({t.min()} a {t.max()} UTC)")
    horas = horas_fim_utc(t)
    um = pd.Timedelta(hours=1)
    if produto == "merge":
        h = falta(horas, lambda x: DM.urls_for(x)[1])
        baixar_merge(DM, h, dry)
    elif produto == "pdir":
        cam = lambda x: (DX.PDIR_DIR / f"{(x-um).year}" / f"{(x-um).month:02d}"
                         / f"{(x-um).day:02d}" / f"pdir_{x-um:%Y%m%d%H}.npz")
        h = falta(horas, cam)
        print(f"[pdir] {len(h)} de {len(horas)} horas sem arquivo")
        if h and not dry:
            DX.baixa_pdir(h, False, False)
    elif produto == "gsmap":
        for k in kinds:
            cam = lambda x, k=k: (DX.GSMAP_DIR / k / f"{(x-um).year}" / f"{(x-um).month:02d}"
                                  / f"{(x-um).day:02d}" / f"gsmap_{k}_{x-um:%Y%m%d%H}.npz")
            h = falta(horas, cam)
            print(f"[gsmap_{k}] {len(h)} de {len(horas)} horas sem arquivo")
            if h and not dry:
                DX.baixa_gsmap(h, [k], False, False)
    elif produto == "hydro":
        passos = passos_10min(t)
        cam = lambda x: (DX.HYDRO_DIR / f"{x.year}" / f"{x.month:02d}" / f"{x.day:02d}"
                         / f"hydro_{x:%Y%m%d%H%M}.npz")
        p = falta(passos, cam)
        print(f"[hydro] {len(p)} de {len(passos)} passos sem arquivo "
              f"(~{len(p) * 4.35 / 1024:.0f} GB de download)")
        if p and not dry:
            DX.baixa_hydro(p, False, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--produto", nargs="+", default=PADRAO,
                    choices=["merge", "pdir", "gsmap", "hydro"])
    ap.add_argument("--gsmap-kinds", nargs="+", default=["nrt", "gauge", "now"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--filho", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.filho or len(a.produto) == 1 or a.dry_run:
        for p in a.produto:
            um_produto(p, a.dry_run, a.gsmap_kinds)
        return

    # vários produtos: um processo por produto, em paralelo, e um log único no fim
    LOGS.mkdir(exist_ok=True)
    inicio = dt.datetime.now()
    procs = {}
    for p in a.produto:
        log = open(LOGS / f"baixar_{p}.log", "w")
        cmd = [sys.executable, "-u", __file__, "--filho", "--produto", p,
               "--gsmap-kinds", *a.gsmap_kinds]
        procs[p] = (subprocess.Popen(cmd, cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT), log)
        print(f"[baixar] {p} iniciado (log em logs/baixar_{p}.log)", flush=True)
    for p, (pr, log) in procs.items():
        pr.wait(); log.close()
        print(f"[baixar] {p} terminou com código {pr.returncode}", flush=True)
    with open(LOGS / "baixar_produtos.log", "w") as out:
        out.write(f"baixar_produtos.py | início {inicio:%Y-%m-%d %H:%M} | "
                  f"fim {dt.datetime.now():%Y-%m-%d %H:%M}\n")
        for p in a.produto:
            out.write(f"\n===== {p} =====\n")
            out.write((LOGS / f"baixar_{p}.log").read_text(errors="replace"))
    print("DEVOLVER: logs/baixar_produtos.log")


if __name__ == "__main__":
    main()
