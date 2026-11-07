#!/usr/bin/env python
"""
mw_prep.py — extrai o MWprecipitation (micro-ondas passivas puro) dos HDF5 do
IMERG Early e grava um npz por meia hora COM sobrevoo no domínio, em
dados/mw/. É o rótulo denso do pré-treino da variante mw_di. Latência do
IMERG é irrelevante aqui: o MW só entra no TREINO.

Campos aceitos (na ordem): Grid/MWprecipitation (V07), Grid/HQprecipitation
(V06). Valor em mm/h; carimbo = INÍCIO da meia hora (UTC). Só grava quando a
fração de pixels finitos no recorte é >= --min-cobertura (o resto é meia hora
sem sobrevoo, que não ensina nada).

Uso:
  python mw_prep.py                       # período coberto pelos stems GOES
  python mw_prep.py --limite 200          # teste
  python mw_prep.py --inicio 2021-10-01 --fim 2022-04-30
"""
from __future__ import annotations

import argparse
import os
import re

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # disco NTFS/exFAT

import numpy as np
import pandas as pd

from config3 import BH_LAT, BH_LON, GOES_DIR, IMERG_DIR, MW_DIR

MEIA_LARGURA = 0.75          # graus para cada lado do centro (grade 64x64 ~ ±0.6)
# No V07 o MWprecipitation fica no subgrupo Grid/Intermediate; no V06 o
# equivalente é Grid/HQprecipitation. Se nenhum desses caminhos existir, o
# arquivo é varrido atrás de qualquer dataset com esses nomes.
CAMPOS = ["Grid/Intermediate/MWprecipitation", "Grid/MWprecipitation",
          "Grid/HQprecipitation", "Grid/Intermediate/HQprecipitation"]
RE_NOME = re.compile(r"3B-HHR.*?(\d{8})-S(\d{6})")


def periodo_goes():
    stems = sorted(p.stem for p in GOES_DIR.glob("*.npy"))
    if not stems:
        return None, None
    t0 = pd.to_datetime(stems[0], format="%Y%m%d_%H%M%S")
    t1 = pd.to_datetime(stems[-1], format="%Y%m%d_%H%M%S")
    return t0.floor("D"), t1.ceil("D")


def indices_recorte(lat, lon):
    ia = np.where((lat >= BH_LAT - MEIA_LARGURA) & (lat <= BH_LAT + MEIA_LARGURA))[0]
    io = np.where((lon >= BH_LON - MEIA_LARGURA) & (lon <= BH_LON + MEIA_LARGURA))[0]
    return ia, io


def achar_campo(h5):
    for campo in CAMPOS:
        if campo in h5:
            return campo
    achados = []
    h5.visit(lambda nome: achados.append(nome)
             if nome.split("/")[-1] in ("MWprecipitation", "HQprecipitation")
             else None)
    return achados[0] if achados else None


def listar_datasets(h5):
    nomes = []
    h5.visit(lambda n: nomes.append(n) if hasattr(h5[n], "shape") else None)
    return nomes


def ler_mw(h5, campo, ia, io, n_lat, n_lon):
    """Lê só o recorte do domínio (não carrega o campo global)."""
    ds = h5[campo]
    a0, a1, o0, o1 = ia[0], ia[-1] + 1, io[0], io[-1] + 1
    forma = ds.shape[-2:]
    if forma == (n_lon, n_lat):                     # (lon, lat)
        rec = (ds[0, o0:o1, a0:a1] if ds.ndim == 3 else ds[o0:o1, a0:a1]).T
    else:                                            # (lat, lon)
        rec = ds[0, a0:a1, o0:o1] if ds.ndim == 3 else ds[a0:a1, o0:o1]
    rec = np.asarray(rec, np.float32)
    return np.where(rec < 0, np.nan, rec)


def main():
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default=None)
    ap.add_argument("--fim", default=None)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--min-cobertura", type=float, default=0.05,
                    help="fração mínima de pixels finitos no recorte p/ gravar")
    args = ap.parse_args()

    t0, t1 = periodo_goes()
    if args.inicio:
        t0 = pd.Timestamp(args.inicio)
    if args.fim:
        t1 = pd.Timestamp(args.fim) + pd.Timedelta(days=1)
    if t0 is None:
        raise SystemExit(f"sem stems em {GOES_DIR} e sem --inicio/--fim")
    print(f"[mw] período {t0:%Y-%m-%d} a {t1:%Y-%m-%d} | fonte {IMERG_DIR}")
    if not IMERG_DIR.exists():
        raise SystemExit(f"IMERG_DIR não existe: {IMERG_DIR} — ajuste no "
                         f"config3.py ou exporte IMERG_DIR")

    arquivos = []
    for p in IMERG_DIR.rglob("3B-HHR*"):
        if p.suffix.lower() not in (".hdf5", ".h5", ".rt-h5"):
            continue
        m = RE_NOME.search(p.name)
        if not m:
            continue
        ts = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
        if t0 <= ts <= t1:
            arquivos.append((ts, p))
    arquivos.sort()
    if args.limite:
        arquivos = arquivos[:args.limite]
    print(f"[mw] {len(arquivos)} meias horas no período")

    ia = io = None
    campo_usado = None
    gravados = pulados = erros = 0
    log = []
    for k, (ts, p) in enumerate(arquivos):
        saida = MW_DIR / f"mw_{ts:%Y%m%d%H%M}.npz"
        if saida.exists():
            pulados += 1
            continue
        try:
            with h5py.File(p, "r") as h5:
                if ia is None:
                    lat = np.asarray(h5["Grid/lat"][...], float)
                    lon = np.asarray(h5["Grid/lon"][...], float)
                    ia, io = indices_recorte(lat, lon)
                    np.savez(MW_DIR / "grade.npz",
                             lat=lat[ia[0]:ia[-1] + 1].astype(np.float32),
                             lon=lon[io[0]:io[-1] + 1].astype(np.float32))
                    n_lat, n_lon = len(lat), len(lon)
                campo = achar_campo(h5)
                if campo is None:
                    rec = None
                    if erros == 0:
                        print("  datasets presentes no arquivo: "
                              + ", ".join(listar_datasets(h5)))
                else:
                    if campo_usado is None:
                        campo_usado = campo
                        print(f"[mw] campo lido: {campo}")
                    rec = ler_mw(h5, campo, ia, io, n_lat, n_lon)
        except Exception as e:
            erros += 1
            log.append(f"{p.name}: {type(e).__name__}: {e}")
            if erros <= 3:
                print(f"  ERRO {log[-1]}")
            if erros == 50 and gravados == 0:
                print("[mw] 50 erros seguidos sem nenhum arquivo lido, abortando")
                break
            continue
        if rec is None:
            erros += 1
            log.append(f"{p.name}: campo MW/HQ ausente")
            if erros <= 3:
                print(f"  ERRO {log[-1]}")
            if erros == 50 and gravados == 0:
                print("[mw] 50 erros seguidos sem nenhum arquivo lido, abortando")
                break
            continue
        frac = float(np.isfinite(rec).mean())
        if frac < args.min_cobertura:
            pulados += 1
            continue
        np.savez_compressed(saida, mmh=rec.astype(np.float32),
                            frac=np.float32(frac))
        gravados += 1
        if k % 500 == 0:
            print(f"  ... {k}/{len(arquivos)} (gravados {gravados})")

    txt = (f"[mw] gravados {gravados} | sem sobrevoo/já existiam {pulados} | "
           f"erros {erros} -> {MW_DIR}")
    print(txt)
    with open(MW_DIR / "mw_prep_log.txt", "w") as f:
        f.write(txt + "\n" + "\n".join(log[:200]))


if __name__ == "__main__":
    main()
