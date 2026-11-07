#!/usr/bin/env python
"""
mapa_metodologia.py  figura da seção de metodologia.

Mapa principal: todas as estações do metadados.csv na região, coloridas pela fonte
(ANA, CEMADEN, INMET, SUDECAP) e com símbolo pelo tipo (pluviômetro ou estação
meteorológica), junto com o recorte do GOES.
Coluna lateral: legenda em cima e mapa de localização de menor zoom embaixo.
Recorte do GOES com 64 x 64 px de 2 km centrado em BH (geometria do common.py).

Mapa base: ruas do OpenStreetMap (tiles XYZ baixados com requests, colados com
Pillow e desenhados com imshow, com cache local). Sem cartopy nem contextily.

Uso:
  python mapa_metodologia.py 2>&1 | tee mapa_metodologia.log

Saídas (pasta saida_mapa_metodologia, ao lado do script):
  mapa_metodologia.png          figura (300 dpi)
  resumo_estacoes_mapa.csv      estações por fonte e tipo, total e dentro do recorte
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# =============================================================================
# CONFIGURAÇÃO (tudo aqui, sem variáveis de ambiente)
# =============================================================================
try:
    from config import GFUSE_ROOT                      # usa o do projeto se existir
except Exception:
    GFUSE_ROOT = "/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse"

METADADOS = Path(GFUSE_ROOT) / "processados" / "metadados.csv"
AQUI = Path(__file__).resolve().parent
SAIDA = AQUI / "saida_mapa_metodologia"
CACHE_TILES = AQUI / "cache_tiles_osm"

# geometria do recorte do GOES (igual ao common.py)
BH_LAT, BH_LON = -19.9068, -43.9765
IMG_SIZE, KM_PER_PX, R_EARTH = 64, 2.0, 6371.0

# mapa base
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "hidromet-gfuse/1.0 (pesquisa academica UFMG, mapa de metodologia)"
ZOOM_REGIAO = 10
# mapa de localização: caixa lon0, lon1, lat0, lat1 e zoom
CAIXA_LOCAL, ZOOM_LOCAL = (-53.0, -39.0, -24.5, -14.0), 6
MARGEM_REGIAO = 0.10           # graus
CLAREAR_BASE = 0.35                                    # 0 = original, 1 = branco

# aparência
COR_FONTE = {"ANA": "#0072B2", "CEMADEN": "#D55E00",
             "INMET": "#009E73", "SUDECAP": "#CC79A7"}
NOME_FONTE = {"ANA": "ANA", "CEMADEN": "CEMADEN", "INMET": "INMET",
              "SUDECAP": "Sudecap", "SUDECAP_METEO": "Sudecap"}
DPI = 300


# =============================================================================
# Geometria do recorte
# =============================================================================
def latlon_to_px(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dy = -(lat - BH_LAT) * np.pi / 180 * R_EARTH
    dx = (lon - BH_LON) * np.pi / 180 * R_EARTH * np.cos(np.radians(BH_LAT))
    return dx / KM_PER_PX + IMG_SIZE / 2, dy / KM_PER_PX + IMG_SIZE / 2


def px_to_latlon(px, py):
    px, py = np.asarray(px, float), np.asarray(py, float)
    dx, dy = (px - IMG_SIZE / 2) * KM_PER_PX, (py - IMG_SIZE / 2) * KM_PER_PX
    return (BH_LAT - dy / R_EARTH * 180 / np.pi,
            BH_LON + dx / (R_EARTH * np.cos(np.radians(BH_LAT))) * 180 / np.pi)


def cantos_recorte():
    """Cantos (lat, lon) do recorte, do canto superior esquerdo em sentido horário."""
    px = [0, IMG_SIZE, IMG_SIZE, 0]
    py = [0, 0, IMG_SIZE, IMG_SIZE]
    lat, lon = px_to_latlon(px, py)
    return np.asarray(lat), np.asarray(lon)


# =============================================================================
# Mapa base OSM (tiles XYZ)
# =============================================================================
def lonlat_to_world(lon, lat, z):
    n = 256 * 2 ** z
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    s = np.sin(np.radians(np.asarray(lat, float)))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return x, y


def world_to_lonlat(x, y, z):
    n = 256 * 2 ** z
    lon = np.asarray(x, float) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.asarray(y, float) / n))))
    return lon, lat


_sessao = requests.Session()
_sessao.headers.update({"User-Agent": USER_AGENT})
_falhas = {"n": 0}


def baixar_tile(z, x, y):
    f = CACHE_TILES / str(z) / str(x) / f"{y}.png"
    if f.exists():
        return Image.open(f).convert("RGB")
    try:
        r = _sessao.get(TILE_URL.format(z=z, x=x, y=y), timeout=20)
        r.raise_for_status()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
        time.sleep(0.15)                               # educado com o servidor
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        _falhas["n"] += 1
        if _falhas["n"] <= 3:
            print(f"  aviso: tile {z}/{x}/{y} não baixado ({type(e).__name__})")
        return None


def monta_base(lon0, lon1, lat0, lat1, z):
    """Cola os tiles que cobrem a caixa. Retorna imagem e extent em coords de mundo."""
    xa, ya = lonlat_to_world(lon0, lat1, z)
    xb, yb = lonlat_to_world(lon1, lat0, z)
    tx0, tx1 = int(xa // 256), int(xb // 256)
    ty0, ty1 = int(ya // 256), int(yb // 256)
    mosaico = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (225, 225, 225))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            t = baixar_tile(z, tx, ty)
            if t is not None:
                mosaico.paste(t, ((tx - tx0) * 256, (ty - ty0) * 256))
    img = np.asarray(mosaico, float) / 255.0
    img = img * (1 - CLAREAR_BASE) + CLAREAR_BASE
    extent = (tx0 * 256, (tx1 + 1) * 256, (ty1 + 1) * 256, ty0 * 256)
    return img, extent, (xa, xb, ya, yb)


def rotulo_grau(v, eixo, casas=2):
    letra = ("O" if v < 0 else "L") if eixo == "lon" else ("S" if v < 0 else "N")
    return f"{abs(v):.{casas}f}".replace(".", ",") + f"°{letra}"


def configura_eixos(ax, caixa, z, passo):
    lon0, lon1, lat0, lat1 = caixa
    xa, ya = lonlat_to_world(lon0, lat1, z)
    xb, yb = lonlat_to_world(lon1, lat0, z)
    ax.set_xlim(xa, xb)
    ax.set_ylim(yb, ya)
    ax.set_aspect("equal")
    lons = np.arange(np.ceil(lon0 / passo) * passo, lon1, passo)
    lats = np.arange(np.ceil(lat0 / passo) * passo, lat1, passo)
    ax.set_xticks(lonlat_to_world(lons, np.full_like(lons, lat0), z)[0])
    casas = 0 if passo >= 1 else 2
    ax.set_xticklabels([rotulo_grau(v, "lon", casas) for v in lons], fontsize=8)
    ax.set_yticks(lonlat_to_world(np.full_like(lats, lon0), lats, z)[1])
    ax.set_yticklabels([rotulo_grau(v, "lat", casas) for v in lats], fontsize=8)
    ax.tick_params(length=3)


# =============================================================================
# Dados
# =============================================================================
def carrega(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for c in ("Latitude", "Longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    df["Fonte"] = df["Fonte"].str.strip().str.upper().replace({"CEMADEM": "CEMADEN"})  # grafia no csv
    df["Fonte_mapa"] = df["Fonte"].replace({"SUDECAP_METEO": "SUDECAP"})
    df["meteo"] = df["Tipo"].str.lower().str.startswith("meteo")
    px, py = latlon_to_px(df["Latitude"].values, df["Longitude"].values)
    df["no_recorte"] = (px >= 0) & (px < IMG_SIZE) & (py >= 0) & (py < IMG_SIZE)
    return df


def resumo(df):
    t = (df.groupby(["Fonte", "Tipo"])
           .agg(n_total=("Codigo", "size"), n_no_recorte=("no_recorte", "sum"))
           .reset_index())
    tot = pd.DataFrame([{"Fonte": "TOTAL", "Tipo": "", "n_total": len(df),
                         "n_no_recorte": int(df["no_recorte"].sum())}])
    return pd.concat([t, tot], ignore_index=True)


# =============================================================================
# Figura
# =============================================================================
def desenha_estacoes(ax, df, z):
    for meteo, marcador, tam, borda in ((False, "o", 26, 0.5), (True, "D", 70, 1.0)):
        for fonte, cor in COR_FONTE.items():
            d = df[(df["Fonte_mapa"] == fonte) & (df["meteo"] == meteo)]
            if d.empty:
                continue
            x, y = lonlat_to_world(d["Longitude"].values, d["Latitude"].values, z)
            ax.scatter(x, y, s=tam, marker=marcador, c=cor,
                       edgecolors="black" if meteo else "white",
                       linewidths=borda, zorder=4 if meteo else 3)


def desenha_recorte(ax, z, lw=1.8, preenche=True):
    lat, lon = cantos_recorte()
    x, y = lonlat_to_world(lon, lat, z)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=preenche,
                         facecolor="#222222", alpha=0.10 if preenche else 1.0,
                         edgecolor="none", zorder=2))
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=False,
                         edgecolor="#222222", linewidth=lw, linestyle="--", zorder=5))
    return x, y


def painel(ax, df, caixa, z, passo, titulo, mostrar_rotulo_recorte):
    lon0, lon1, lat0, lat1 = caixa
    img, extent, _ = monta_base(lon0, lon1, lat0, lat1, z)
    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    x, y = desenha_recorte(ax, z)
    if mostrar_rotulo_recorte:
        ax.text(x.mean(), y.min() - 6, "Recorte do GOES", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222222", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    desenha_estacoes(ax, df, z)
    configura_eixos(ax, caixa, z, passo)
    if titulo:
        ax.set_title(titulo, fontsize=11, loc="left")
    ax.text(0.99, 0.01, "© colaboradores do OpenStreetMap", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, zorder=7,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8))


def painel_localizacao(ax, caixa_a):
    """Mapa de menor zoom com a área do mapa principal destacada."""
    z = ZOOM_LOCAL
    img, extent, _ = monta_base(*CAIXA_LOCAL, z)
    ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
    lon0, lon1, lat0, lat1 = caixa_a
    x, y = lonlat_to_world([lon0, lon1, lon1, lon0], [lat1, lat1, lat0, lat0], z)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=True, facecolor="#d62728",
                         alpha=0.25, edgecolor="none", zorder=2))
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fill=False, edgecolor="#d62728",
                         linewidth=1.8, zorder=3))
    configura_eixos(ax, CAIXA_LOCAL, z, 4)
    ax.text(0.99, 0.01, "© colaboradores do OpenStreetMap", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, zorder=7,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8))


def painel_legenda(ax, df):
    ax.axis("off")
    h_fonte = []
    for fonte, cor in COR_FONTE.items():
        n = int((df["Fonte_mapa"] == fonte).sum())
        h_fonte.append(Line2D([], [], marker="s", ls="", ms=9, mfc=cor, mec="none",
                              label=f"{NOME_FONTE[fonte]} ({n})"))
    n_pl, n_me = int((~df["meteo"]).sum()), int(df["meteo"].sum())
    h_tipo = [Line2D([], [], marker="o", ls="", ms=7, mfc="#888888", mec="white",
                     label=f"Pluviômetro ({n_pl})"),
              Line2D([], [], marker="D", ls="", ms=8, mfc="#888888", mec="black",
                     label=f"Estação meteorológica ({n_me})")]
    h_area = [Line2D([], [], ls="--", lw=1.8, color="#222222", label="Recorte do GOES"),
              Line2D([], [], marker="s", ls="", ms=10, mfc="#d62728", mec="#d62728",
                     alpha=0.5, label="Área do mapa principal")]
    kw = dict(frameon=False, fontsize=9.5, title_fontsize=10, alignment="left")
    l1 = ax.legend(handles=h_fonte, title="Fonte dos dados", loc="upper left",
                   bbox_to_anchor=(0.0, 1.0), **kw)
    ax.add_artist(l1)
    l2 = ax.legend(handles=h_tipo, title="Tipo de estação", loc="upper left",
                   bbox_to_anchor=(0.0, 0.64), **kw)
    ax.add_artist(l2)
    ax.legend(handles=h_area, title="Áreas", loc="upper left",
              bbox_to_anchor=(0.0, 0.40), **kw)


def figura(df, destino):
    lat_c, lon_c = cantos_recorte()
    # mapa principal: todas as estações e o recorte
    caixa_a = (min(df["Longitude"].min(), lon_c.min()) - MARGEM_REGIAO,
               max(df["Longitude"].max(), lon_c.max()) + MARGEM_REGIAO,
               min(df["Latitude"].min(), lat_c.min()) - MARGEM_REGIAO,
               max(df["Latitude"].max(), lat_c.max()) + MARGEM_REGIAO)
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 0.6], height_ratios=[1, 1],
                          left=0.05, right=0.99, top=0.97, bottom=0.05,
                          wspace=0.12, hspace=0.12)
    axa = fig.add_subplot(gs[:, 0])
    axl, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    painel(axa, df, caixa_a, ZOOM_REGIAO, 0.5, "", True)
    painel_legenda(axl, df)
    painel_localizacao(axc, caixa_a)
    fig.savefig(destino, dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadados", default=str(METADADOS),
                    help="caminho do metadados.csv (padrão: GFUSE_ROOT/processados)")
    args = ap.parse_args()

    path = Path(args.metadados)
    if not path.exists() and Path("metadados.csv").exists():
        path = Path("metadados.csv")
    print(f"metadados: {path}")
    df = carrega(path)

    SAIDA.mkdir(parents=True, exist_ok=True)
    r = resumo(df)
    r.to_csv(SAIDA / "resumo_estacoes_mapa.csv", sep=";", index=False)
    print(r.to_string(index=False))

    destino = SAIDA / "mapa_metodologia.png"
    figura(df, destino)
    if _falhas["n"]:
        print(f"ATENÇÃO: {_falhas['n']} tiles não baixados; o mapa base está incompleto. "
              f"Confira a internet e rode de novo (os tiles já baixados ficam em cache).")
    print(f"figura: {destino}")


if __name__ == "__main__":
    main()
