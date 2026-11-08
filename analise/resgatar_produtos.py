#!/usr/bin/env python
"""
resgatar_produtos.py — copia os produtos de comparação do disco Elements (com
erros de leitura) para um disco saudável, lendo arquivo a arquivo e seguindo
em frente quando um arquivo ou pasta não pode ser lido.

  merge, pdir, gsmap   cópia dos arquivos como estão (grib2 e npz pequenos),
                       com grade.npz e indice_origem.csv
  imerg                só as meias-horas que o gfuse3 usa (janelas de 30 e 60
                       min dos seus instantes); de cada HDF5 lê apenas o recorte
                       do domínio e grava um npz pequeno (chave mmh, mm/h),
                       no mesmo formato dos demais produtos

Pode ser interrompido e rodado de novo: o que já foi salvo é pulado, e as
falhas de leitura são tentadas outra vez. Ao final, os caminhos novos ficam em
logs/produtos_dirs.sh (para usar com `source`).

Uso (de dentro da pasta do gfuse3, com o Elements montado):
  nohup python analise/resgatar_produtos.py --destino /dados/gfuse_produtos > logs/resgate_main.out 2>&1 &
Devolver: logs/resgate.log
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
ORIGEM = Path("/media/gustavo/Elements/imerg/imerg_data")
LOG = RAIZ / "logs" / "resgate.log"
FALHAS = RAIZ / "logs" / "resgate_falhas.csv"
IMERG_RE = re.compile(r"3B-HHR-E[._]MS[._]MRG[._]3IMERG[._](\d{8})-S(\d{6})-E\d{6}")
_falhas = []


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def falha(tipo, caminho, erro):
    _falhas.append(dict(tipo=tipo, caminho=str(caminho), erro=str(erro)[:200]))


def lista(d):
    """Conteúdo de uma pasta; None se não puder ser lida."""
    for k in range(3):
        try:
            return sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as e:
            err = e
            time.sleep(2)
    falha("pasta", d, err)
    return None


def copia_arvore(src, dst, nome):
    n_ok = n_pula = n_erro = 0
    pilha = [src]
    while pilha:
        d = pilha.pop()
        ents = lista(d)
        if ents is None:
            n_erro += 1
            continue
        for e in ents:
            try:
                if e.is_dir(follow_symlinks=False):
                    pilha.append(Path(e.path))
                    continue
                alvo = dst / Path(e.path).relative_to(src)
                tam = e.stat().st_size
                if alvo.exists() and alvo.stat().st_size == tam:
                    n_pula += 1
                    continue
                alvo.parent.mkdir(parents=True, exist_ok=True)
                tmp = alvo.with_suffix(alvo.suffix + ".part")
                shutil.copyfile(e.path, tmp)
                tmp.replace(alvo)
                n_ok += 1
            except OSError as err:
                falha(nome, e.path, err)
                n_erro += 1
            if (n_ok + n_pula + n_erro) % 2000 == 0:
                log(f"  [{nome}] copiados {n_ok}, já existiam {n_pula}, erros {n_erro}")
    log(f"[{nome}] FIM: copiados {n_ok}, já existiam {n_pula}, erros {n_erro}")


def meias_horas_necessarias():
    z = np.load(RAIZ / "processados" / "estacoes.npz", allow_pickle=False)
    t = pd.to_datetime(pd.Series(z["stems"].astype(str)), format="%Y%m%d_%H%M%S")
    r = t.dt.round("30min")
    ok = (t - r).abs().dt.total_seconds() <= 120
    fim = r[ok]
    ini = set(fim - pd.Timedelta(minutes=30)) | set(fim[fim.dt.minute == 0] - pd.Timedelta(minutes=60))
    return sorted(ini)


def resgata_imerg(dst):
    import h5py
    import comum as C
    glat, glon = C.grade_latlon()
    m = 0.15
    la0, la1, lo0, lo1 = glat.min() - m, glat.max() + m, glon.min() - m, glon.max() + m
    precisa = meias_horas_necessarias()
    log(f"[imerg] {len(precisa):,} meias-horas necessárias")
    por_dia = {}
    for s in precisa:
        por_dia.setdefault(s.normalize(), []).append(s)
    src = ORIGEM / "early"
    grade_ok = (dst / "grade.npz").exists()
    n_ok = n_pula = n_erro = n_falta = 0
    cache_mes = {}
    for k, (dia, lista_s) in enumerate(sorted(por_dia.items())):
        faltam = [s for s in lista_s if not (dst / f"{s.year}" / f"{s.month:02d}" / f"{s.day:02d}"
                                            / f"imerg_early_{s:%Y%m%d%H%M}.npz").exists()]
        n_pula += len(lista_s) - len(faltam)
        if not faltam:
            continue
        # a pasta pode ser early/AAAA/MM ou early/AAAA/MM/DD
        chave = (dia.year, dia.month)
        if chave not in cache_mes:
            arqs = {}
            base = src / f"{dia.year}" / f"{dia.month:02d}"
            ents = lista(base) or []
            pastas = [e for e in ents if e.is_dir()] or []
            for e in ents:
                if e.is_file():
                    mm = IMERG_RE.search(e.name)
                    if mm:
                        arqs[pd.Timestamp(f"{mm.group(1)} {mm.group(2)[:2]}:{mm.group(2)[2:4]}")] = e.path
            for p in pastas:
                for e in lista(p) or []:
                    mm = IMERG_RE.search(e.name)
                    if mm:
                        arqs[pd.Timestamp(f"{mm.group(1)} {mm.group(2)[:2]}:{mm.group(2)[2:4]}")] = e.path
            cache_mes = {chave: arqs}
        arqs = cache_mes[chave]
        for s in faltam:
            p = arqs.get(s)
            if p is None:
                n_falta += 1
                continue
            alvo = dst / f"{s.year}" / f"{s.month:02d}" / f"{s.day:02d}" / f"imerg_early_{s:%Y%m%d%H%M}.npz"
            try:
                with h5py.File(p, "r") as f:
                    lat = f["Grid/lat"][:]; lon = f["Grid/lon"][:]
                    ia = np.where((lat >= la0) & (lat <= la1))[0]
                    io = np.where((lon >= lo0) & (lon <= lo1))[0]
                    arr = f["Grid/precipitation"][0, io[0]:io[-1] + 1, ia[0]:ia[-1] + 1].astype(np.float32).T
                arr = np.where(arr < 0, np.nan, arr)
                alvo.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(alvo, mmh=arr)
                if not grade_ok:
                    np.savez(dst / "grade.npz", lat=lat[ia], lon=lon[io]); grade_ok = True
                n_ok += 1
            except Exception as err:
                falha("imerg", p, err)
                n_erro += 1
        if k % 30 == 0:
            log(f"  [imerg] dia {dia:%Y-%m-%d}: salvos {n_ok}, já existiam {n_pula}, "
                f"sem arquivo {n_falta}, erros {n_erro}")
    log(f"[imerg] FIM: salvos {n_ok}, já existiam {n_pula}, sem arquivo no disco {n_falta}, erros {n_erro}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="/dados/gfuse_produtos")
    ap.add_argument("--produtos", nargs="+", default=["merge", "pdir", "gsmap", "imerg"])
    a = ap.parse_args()
    dst = Path(a.destino)
    dst.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    log(f"=== resgate para {dst} | livre no destino: {shutil.disk_usage(dst).free / 1e9:.0f} GB")
    mapa = {"merge": ("merge_hourly", "MERGE_DIR"), "pdir": ("pdir_now", "PDIR_DIR"),
            "gsmap": ("gsmap", "GSMAP_DIR")}
    for p in a.produtos:
        if p in mapa:
            copia_arvore(ORIGEM / mapa[p][0], dst / mapa[p][0], p)
        elif p == "imerg":
            resgata_imerg(dst / "imerg_early_npz")
    if _falhas:
        pd.DataFrame(_falhas).to_csv(FALHAS, index=False)
        log(f"{len(_falhas)} falhas de leitura em {FALHAS} (rode de novo para tentar outra vez)")
    with open(RAIZ / "logs" / "produtos_dirs.sh", "w") as f:
        f.write(f"export MERGE_DIR={dst}/merge_hourly\nexport PDIR_DIR={dst}/pdir_now\n"
                f"export GSMAP_DIR={dst}/gsmap\nexport IMERG_NPZ_DIR={dst}/imerg_early_npz\n")
    log("caminhos novos em logs/produtos_dirs.sh")


if __name__ == "__main__":
    main()
