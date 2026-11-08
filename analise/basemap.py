"""Mapa base com ruas do OpenStreetMap (tiles XYZ), no padrão do hidromet.

Tiles de https://tile.openstreetmap.org/{z}/{x}/{y}.png baixados com requests
(User-Agent identificado), com cache local, colados com Pillow e desenhados
com imshow no matplotlib, em coordenadas lon/lat (extent em graus).
"""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "cache_tiles"
UA = {"User-Agent": "gfuse3-tese-UFMG/1.0 (pesquisa academica; contato via GitHub Gustavoliveirac)"}


def _xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def _tile(z, x, y):
    import requests
    from PIL import Image
    f = CACHE / str(z) / str(x) / f"{y}.png"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers=UA, timeout=30)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(0.2)
    return Image.open(f).convert("RGB")


def desenha(ax, lon0, lon1, lat0, lat1, z=10, alpha=0.55):
    """Cola os tiles que cobrem a caixa e desenha com imshow (extent em graus).
    Em caso de falha de rede, segue sem mapa base."""
    from PIL import Image
    try:
        x0, y0 = _xy(lat1, lon0, z)
        x1, y1 = _xy(lat0, lon1, z)
        tx = range(int(x0), int(x1) + 1)
        ty = range(int(y0), int(y1) + 1)
        img = Image.new("RGB", (256 * len(tx), 256 * len(ty)))
        for i, x in enumerate(tx):
            for j, y in enumerate(ty):
                img.paste(_tile(z, x, y), (256 * i, 256 * j))
        w_lon, n_lat = _lonlat(tx[0], ty[0], z)
        e_lon, s_lat = _lonlat(tx[-1] + 1, ty[-1] + 1, z)
        ax.imshow(np.asarray(img), extent=(w_lon, e_lon, s_lat, n_lat), alpha=alpha,
                  zorder=0, interpolation="bilinear")
    except Exception as e:                                   # sem rede: sem mapa
        print(f"  [basemap] sem tiles ({e})")
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / math.cos(math.radians((lat0 + lat1) / 2)))
    ax.text(0.995, 0.005, "© OpenStreetMap contributors", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="0.3", zorder=10)
