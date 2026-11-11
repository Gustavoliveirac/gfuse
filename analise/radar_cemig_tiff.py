#!/usr/bin/env python3
"""Converte os arquivos horarios do radar da Cemig (Mateus Leme, MG) em GeoTIFF para o QGIS e faz o inventario.

Arquivos de entrada: elefante-analysisAAMMDDhhmmss.RN#####, GRIB1 com a chuva acumulada em 1 hora, grade de 720 x 720
pontos num quadrado de ~500 km em torno do radar (lat -22,181 a -17,677, lon -46,854 a -42,083), em mm inteiros.

Orientacao e horario, conferidos em 08/11/2026 contra 4347 pluviometros nas 400 horas mais chuvosas de todo o periodo
(correlacao 0,44, contra 0,16 da segunda melhor opcao; ainda nao conferidos ano a ano):
  - as linhas do arquivo vao de SUL para NORTE (o cabecalho declara o contrario no modo de varredura);
  - o valor e a chuva da hora que TERMINA no instante do nome do arquivo, em UTC (o cabecalho declara de 0 a 1 h depois).
  Ex.: elefante-analysis211217150000 = chuva de 14:00 a 15:00 UTC (11:00 a 12:00 de Brasilia) de 17/12/2021.

Saida (GeoTIFF float32, EPSG:4326, mm, sem dado = -9999, compressao deflate, linha 0 no norte):
  SAIDA/AAAA/MM/DD/cemig_AAAAMMDD_HH00_utc.tif     (data e hora do FIM do acumulado, em UTC)

Requer Python 3.9+ com numpy e pillow (pip install numpy pillow). Nao usa GDAL.

Uso:
  python radar_cemig_tiff.py --dados PASTA_DOS_ARQUIVOS --saida PASTA_DE_SAIDA
  python radar_cemig_tiff.py --dados dados --saida saida --arquivo elefante-analysis211217150000.RN10928
  python radar_cemig_tiff.py --dados dados --saida saida --de 2021-12-01 --ate 2021-12-31
  python radar_cemig_tiff.py --dados dados --saida saida --inventario
        so o inventario, sem GeoTIFF: SAIDA/inventario_arquivos.csv (um por arquivo), SAIDA/inventario_mensal.csv e
        SAIDA/lacunas.csv (periodos de 6 h ou mais sem arquivo valido). CSV com ; e virgula decimal (Excel em pt-BR)

A pasta de dados pode ter subpastas (por ano, por mes, extraidas de zip ou 7z). Arquivos vazios, truncados ou de outro
formato ficam de fora, com o motivo na tela e no inventario. Se houver dois arquivos para a mesma hora, vale o maior.
Arquivos ja convertidos sao pulados (use --refazer para gerar de novo).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

UTC = timezone.utc
NOME = re.compile(r"elefante-analysis(\d{12})\.RN(\d+)$")
LINHAS_DE_SUL_PARA_NORTE = True        # conferido em 08/11/2026
DESLOCAMENTO_H = -1                    # fim do acumulado (UTC) = instante do arquivo + P2 + DESLOCAMENTO_H


# ------------------------------------------------------------------ leitura do GRIB1
def _s24(b):
    v = int.from_bytes(b, "big")
    return v - (1 << 24) if v & 0x800000 else v        # os cantos estao em complemento de dois


def _ibm(b):
    s = -1 if b[0] & 0x80 else 1
    return s * int.from_bytes(b[1:4], "big") * 16.0 ** ((b[0] & 0x7F) - 70)


def ler(caminho, so_cabecalho=False):
    """(info, matriz nj x ni na ordem gravada, NaN fora do alcance do radar)."""
    dados = Path(caminho).read_bytes()
    if dados[:2] == b"\x1f\x8b":
        dados = gzip.decompress(dados)
    if len(dados) < 8 or dados[:4] != b"GRIB" or dados[7] != 1:
        raise ValueError(f"nao e GRIB1 ({len(dados)} bytes, inicio {dados[:8].hex() or 'vazio'})")
    total = int.from_bytes(dados[4:7], "big")
    if len(dados) < total:
        raise ValueError(f"truncado ({len(dados)} de {total} bytes)")
    p = 8
    lp = int.from_bytes(dados[p:p + 3], "big")
    pds = dados[p:p + lp]
    p += lp
    tem_gds, tem_bms = bool(pds[7] & 0x80), bool(pds[7] & 0x40)
    info = dict(ano=(pds[24] - 1) * 100 + pds[12], mes=pds[13], dia=pds[14], hora=pds[15], minuto=pds[16],
                p1=pds[18], p2=pds[19], parametro=pds[8], bitmap=tem_bms)
    dd = int.from_bytes(pds[26:28], "big")
    dd = -(dd & 0x7FFF) if dd & 0x8000 else dd
    if not tem_gds:
        raise ValueError("sem secao de grade")
    lg = int.from_bytes(dados[p:p + 3], "big")
    g = dados[p:p + lg]
    p += lg
    info.update(ni=int.from_bytes(g[6:8], "big"), nj=int.from_bytes(g[8:10], "big"),
                la1=_s24(g[10:13]) / 1000, lo1=_s24(g[13:16]) / 1000,
                la2=_s24(g[17:20]) / 1000, lo2=_s24(g[20:23]) / 1000)
    n = info["ni"] * info["nj"]
    mask = None
    if tem_bms:
        lb = int.from_bytes(dados[p:p + 3], "big")
        b = dados[p:p + lb]
        p += lb
        mask = np.unpackbits(np.frombuffer(b[6:], np.uint8))[:n].astype(bool)
    lbd = int.from_bytes(dados[p:p + 3], "big")
    bd = dados[p:p + lbd]
    if len(bd) < 11:
        raise ValueError("secao de dados incompleta")
    if bd[3] & 0xC0:
        raise ValueError(f"empacotamento nao simples (flag {bd[3]:#04x})")
    nbits = bd[10]
    e = int.from_bytes(bd[4:6], "big")
    e = -(e & 0x7FFF) if e & 0x8000 else e
    r = _ibm(bd[6:10])
    info.update(nbits=nbits, escala_binaria=e, escala_decimal=dd, referencia=r)
    if so_cabecalho:
        return info, None
    nv = int(mask.sum()) if mask is not None else n
    if nbits == 0:
        v = np.full(nv, r)
    else:
        raw = np.unpackbits(np.frombuffer(bd[11:], np.uint8))[:nv * nbits]
        if raw.size < nv * nbits:
            raise ValueError("valores incompletos")
        v = (r + raw.reshape(nv, nbits).dot(1 << np.arange(nbits - 1, -1, -1, dtype=np.int64)) * 2.0 ** e) / 10.0 ** dd
    if e >= 0 and dd == 0:
        v = np.round(v)                 # passo inteiro: tira o resto do valor de referencia (0,004 nos arquivos)
    out = np.full(n, np.nan, "f4")
    if mask is not None:
        out[mask] = v
    else:
        out[:] = v
    return info, out.reshape(info["nj"], info["ni"])


def fim_utc(info):
    t = datetime(info["ano"], info["mes"], info["dia"], info["hora"], info["minuto"], tzinfo=UTC)
    return t + timedelta(hours=info["p2"] + DESLOCAMENTO_H)


def caixa(info):
    return (max(info["la1"], info["la2"]), min(info["la1"], info["la2"]),
            min(info["lo1"], info["lo2"]), max(info["lo1"], info["lo2"]))


# ------------------------------------------------------------------ GeoTIFF sem GDAL
def salvar_tiff(arq, a, oeste, norte, dx, dy, descricao):
    from PIL import Image, TiffImagePlugin
    v = np.where(np.isfinite(a), a, -9999).astype("<f4")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(dx), float(dy), 0.0)
    ifd.tagtype[33550] = 12                                   # ModelPixelScale
    ifd[33922] = (0.0, 0.0, 0.0, float(oeste), float(norte), 0.0)
    ifd.tagtype[33922] = 12                                   # ModelTiepoint
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    ifd.tagtype[34735] = 3                                    # GeoKeyDirectory: geografico, PixelIsArea, EPSG:4326
    ifd[42113] = "-9999"
    ifd.tagtype[42113] = 2                                    # GDAL_NODATA
    ifd[270] = descricao
    arq.parent.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_name(arq.name + ".tmp")
    Image.fromarray(v, "F").save(tmp, format="TIFF", tiffinfo=ifd, compression="tiff_adobe_deflate")
    os.replace(tmp, arq)


# ------------------------------------------------------------------ principal
def listar(dados, arquivo):
    if arquivo:
        f = Path(arquivo)
        if f.is_file():
            return [f]
        achados = [x for x in Path(dados).rglob(f.name) if x.is_file()]
        if not achados:
            sys.exit(f"arquivo {arquivo} nao encontrado em {dados}")
        return achados
    return sorted((f for f in Path(dados).rglob("*") if f.is_file() and NOME.search(f.name)), key=lambda f: f.name)


def num(x, casas=2):
    return "" if x is None else f"{x:.{casas}f}".replace(".", ",")


def inventario(fs, saida):
    saida.mkdir(parents=True, exist_ok=True)
    validas, mensal = set(), defaultdict(lambda: Counter())
    maximo_mes = defaultdict(float)
    t0 = time.time()
    with open(saida / "inventario_arquivos.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter=";")
        w.writerow(["arquivo", "pasta", "tamanho_bytes", "fim_utc", "bits", "escala_binaria", "escala_decimal",
                    "referencia", "bitmap", "pixels_validos", "pixels_1mm_ou_mais", "maximo_mm", "media_mm", "erro"])
        for i, f in enumerate(fs, 1):
            pasta = f.parent.name
            try:
                info, a = ler(f)
                h = fim_utc(info)
                ok = np.isfinite(a)
                vals = a[ok]
                n1 = int((vals >= 1).sum())
                mx = float(vals.max()) if vals.size else None
                w.writerow([f.name, pasta, f.stat().st_size, f"{h:%Y-%m-%d %H:%M}", info["nbits"],
                            info["escala_binaria"], info["escala_decimal"], num(info["referencia"], 4),
                            int(info["bitmap"]), int(ok.sum()), n1, num(mx, 1),
                            num(float(vals.mean()) if vals.size else None, 4), ""])
                m = f"{h:%Y-%m}"
                mensal[m]["arquivos"] += 1
                mensal[m]["com_chuva"] += n1 > 0
                mensal[m]["sem_pixel_valido"] += vals.size == 0
                mensal[m]["tudo_zero"] += vals.size > 0 and mx == 0
                mensal[m]["pixels_validos"] += int(ok.sum())
                maximo_mes[m] = max(maximo_mes[m], mx or 0)
                validas.add(h)
            except Exception as e:                       # noqa: BLE001
                m = NOME.search(f.name)
                mes = f"20{m.group(1)[:2]}-{m.group(1)[2:4]}" if m else "?"
                mensal[mes]["com_erro"] += 1
                w.writerow([f.name, pasta, f.stat().st_size, "", "", "", "", "", "", "", "", "", "", str(e)])
            if i % 2000 == 0:
                print(f"  inventario: {i} de {len(fs)} ({time.time() - t0:.0f} s)", flush=True)
    with open(saida / "inventario_mensal.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter=";")
        w.writerow(["mes", "horas_no_mes", "horas_com_arquivo_valido", "arquivos_validos", "com_erro", "com_chuva_1mm",
                    "tudo_zero", "sem_pixel_valido", "media_pixels_validos", "maximo_mm"])
        for m in sorted(mensal):
            c = mensal[m]
            if m == "?":
                w.writerow([m, "", "", "", c["com_erro"], "", "", "", "", ""])
                continue
            a, mm = int(m[:4]), int(m[5:])
            ini = datetime(a, mm, 1, tzinfo=UTC)
            fim = datetime(a + (mm == 12), mm % 12 + 1, 1, tzinfo=UTC)
            horas = int((fim - ini).total_seconds() // 3600)
            com = sum(1 for h in validas if ini < h <= fim)
            w.writerow([m, horas, com, c["arquivos"], c["com_erro"], c["com_chuva"], c["tudo_zero"],
                        c["sem_pixel_valido"], c["pixels_validos"] // max(c["arquivos"], 1), num(maximo_mes[m], 1)])
    hs = sorted(validas)
    with open(saida / "lacunas.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter=";")
        w.writerow(["ultimo_valido_utc", "proximo_valido_utc", "horas_sem_arquivo_valido"])
        for a, b in zip(hs, hs[1:]):
            falta = int((b - a).total_seconds() // 3600) - 1
            if falta >= 6:
                w.writerow([f"{a:%Y-%m-%d %H:%M}", f"{b:%Y-%m-%d %H:%M}", falta])
    print(f"inventario em {saida}: inventario_arquivos.csv, inventario_mensal.csv e lacunas.csv "
          f"({len(validas)} horas com arquivo valido"
          + (f", de {hs[0]:%d/%m/%Y %H:%M} a {hs[-1]:%d/%m/%Y %H:%M} UTC)" if hs else ")"))


def converter(fs, saida, de, ate, refazer):
    d0 = datetime.fromisoformat(de).replace(tzinfo=UTC) if de else None
    d1 = datetime.fromisoformat(ate).replace(tzinfo=UTC) + timedelta(days=1) if ate else None
    escolhido, ruins = {}, 0
    for f in fs:
        try:
            info, _ = ler(f, so_cabecalho=True)
        except Exception as e:                           # noqa: BLE001
            ruins += 1
            print(f"  {f.name}: {e}")
            continue
        h = fim_utc(info)
        if (d0 and h <= d0) or (d1 and h > d1):
            continue
        if h not in escolhido or f.stat().st_size > escolhido[h].stat().st_size:
            escolhido[h] = f
    n = pulados = 0
    t0 = time.time()
    for h in sorted(escolhido):
        f = escolhido[h]
        arq = saida / f"{h:%Y}" / f"{h:%m}" / f"{h:%d}" / f"cemig_{h:%Y%m%d_%H}00_utc.tif"
        if arq.exists() and not refazer:
            pulados += 1
            continue
        try:
            info, a = ler(f)
        except Exception as e:                           # noqa: BLE001
            ruins += 1
            print(f"  {f.name}: {e}")
            continue
        if LINHAS_DE_SUL_PARA_NORTE:
            a = a[::-1]
        ln, ls, lw, le = caixa(info)
        ini = h - timedelta(hours=1)
        salvar_tiff(arq, a, lw, ln, (le - lw) / a.shape[1], (ln - ls) / a.shape[0],
                    f"Radar Cemig Mateus Leme, chuva (mm) de {ini:%Y-%m-%d %H:%M} a {h:%H:%M} UTC, arquivo {f.name}")
        n += 1
        if n % 1000 == 0:
            print(f"  {n} GeoTIFF ({time.time() - t0:.0f} s)", flush=True)
    print(f"fim: {n} GeoTIFF novos, {pulados} ja existiam, {ruins} arquivos que nao abriram; pasta {saida}")


def main():
    ap = argparse.ArgumentParser(description="Radar da Cemig (Mateus Leme) para GeoTIFF e inventario.")
    ap.add_argument("--dados", required=True, help="pasta com os arquivos elefante-analysis* (subpastas valem)")
    ap.add_argument("--saida", required=True, help="pasta de saida")
    ap.add_argument("--arquivo", help="converter so este arquivo (nome ou caminho)")
    ap.add_argument("--de", help="primeiro dia (AAAA-MM-DD, UTC)")
    ap.add_argument("--ate", help="ultimo dia (AAAA-MM-DD, UTC)")
    ap.add_argument("--inventario", action="store_true", help="so o inventario em CSV, sem GeoTIFF")
    ap.add_argument("--refazer", action="store_true", help="gerar de novo os GeoTIFF que ja existem")
    a = ap.parse_args()
    fs = listar(a.dados, a.arquivo)
    print(f"{len(fs)} arquivos em {a.dados}")
    if a.inventario:
        inventario(fs, Path(a.saida))
    else:
        converter(fs, Path(a.saida), a.de, a.ate, a.refazer)


if __name__ == "__main__":
    main()
