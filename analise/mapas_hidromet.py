"""Mapas de chuva no padrão do hidromet.

Tudo em Web Mercator (EPSG:3857), com extent em metros. Ruas do OpenStreetMap em
tiles XYZ (requests com User-Agent identificado, cache local, colados com Pillow,
desenhados com imshow). Campos de satélite em mm/h numa escala única, com rampa
de transparência de 0,1 a 1 mm/h e alpha geral 0,6; suavização por convolução
gaussiana normalizada e interpolação bilinear para a grade de saída.
Pluviômetros como círculos com a escala de "chuva agora" do webgis.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, grey_closing

CACHE = Path(__file__).resolve().parent.parent / "cache_tiles"
UA = {"User-Agent": "gfuse3-tese-UFMG/1.0 (pesquisa academica; github.com/Gustavoliveirac/gfuse)"}
_LL2M = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_M2LL = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
R_TERRA = 6378137.0

# ---------------------------------------------------------------- escalas
CAMPO = [(0.1, "#30123b", 0.0), (1, "#392a73", 1.0), (2, "#4143a7", 1), (3, "#4559cb", 1), (4, "#4771e9", 1),
         (6, "#3e9bfe", 1), (8, "#22c5e2", 1), (10, "#1ae4b6", 1), (12.5, "#55fa76", 1), (15, "#a4fc3c", 1),
         (17.5, "#d9e436", 1), (20, "#faba39", 1), (22.5, "#fb7e21", 1), (25, "#e4450a", 1),
         (27.5, "#b91e02", 1), (30, "#7a0403", 1), (40, "#c51b8a", 1), (50, "#7a0177", 1), (60, "#3f007d", 1)]
EST = [(0.2, "#deebf7"), (1, "#9ecae1"), (10, "#4292c6"), (20, "#08519c"), (30, "#a1009c")]
SEM_CHUVA, SEM_DADO = "#bdbdbd", "#777777"
ALPHA_CAMPO = 0.6


def rgba_campo(v):
    """mm/h -> RGBA, interpolado linearmente; < 0,1 transparente."""
    v = np.asarray(v, float)
    xs = np.array([c[0] for c in CAMPO])
    cores = np.array([to_rgb(c[1]) for c in CAMPO])
    alf = np.array([c[2] for c in CAMPO])
    vv = np.clip(np.nan_to_num(v, nan=0.0), xs[0], xs[-1])
    out = np.zeros(v.shape + (4,))
    for k in range(3):
        out[..., k] = np.interp(vv, xs, cores[:, k])
    out[..., 3] = np.interp(vv, xs, alf) * ALPHA_CAMPO
    out[..., 3][~(np.nan_to_num(v, nan=0) >= 0.1)] = 0
    return out


def cor_estacao(mm):
    if mm is None or not np.isfinite(mm):
        return SEM_DADO, 0.5
    if mm < 0.2:
        return SEM_CHUVA, 1.0
    xs = np.array([e[0] for e in EST]); cs = np.array([to_rgb(e[1]) for e in EST])
    m = min(mm, xs[-1])
    return tuple(np.interp(m, xs, cs[:, k]) for k in range(3)), 1.0


# ---------------------------------------------------------------- tiles
def _tile(z, x, y):
    import requests
    from PIL import Image
    f = CACHE / str(z) / str(x) / f"{y}.png"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://tile.openstreetmap.org/{z}/{x}/{y}.png", headers=UA, timeout=30)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(0.15)
    return Image.open(f).convert("RGB")


def _zoom(x0, x1, max_tiles=36):
    for z in range(16, 3, -1):
        n = (x1 - x0) / (2 * math.pi * R_TERRA / 2 ** z)
        if (n + 1) ** 2 <= max_tiles:
            return z
    return 4


def mapa_base(ax, x0, x1, y0, y1):
    from PIL import Image
    z = _zoom(x0, x1)
    tam = 2 * math.pi * R_TERRA / 2 ** z
    org = math.pi * R_TERRA
    tx0, tx1 = int((x0 + org) // tam), int((x1 + org) // tam)
    ty0, ty1 = int((org - y1) // tam), int((org - y0) // tam)
    try:
        img = Image.new("RGB", (256 * (tx1 - tx0 + 1), 256 * (ty1 - ty0 + 1)))
        for i, x in enumerate(range(tx0, tx1 + 1)):
            for j, y in enumerate(range(ty0, ty1 + 1)):
                img.paste(_tile(z, x, y), (256 * i, 256 * j))
        ext = (tx0 * tam - org, (tx1 + 1) * tam - org, org - (ty1 + 1) * tam, org - ty0 * tam)
        ax.imshow(np.asarray(img), extent=ext, zorder=0, interpolation="bilinear")
    except Exception as e:
        print(f"  [mapa base] sem tiles ({str(e)[:80]})")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.text(0.995, 0.005, "© OpenStreetMap", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="0.25", zorder=20)
    _marcas(ax, x0, x1, y0, y1)


def _marcas(ax, x0, x1, y0, y1):
    """Marcas discretas de lon/lat nas bordas."""
    lo0, la0 = _M2LL.transform(x0, y0); lo1, la1 = _M2LL.transform(x1, y1)
    passo = 0.25 if lo1 - lo0 < 2 else 0.5
    los = np.arange(math.ceil(lo0 / passo) * passo, lo1, passo)
    las = np.arange(math.ceil(la0 / passo) * passo, la1, passo)
    ax.set_xticks([_LL2M.transform(l, la0)[0] for l in los]); ax.set_xticklabels([f"{l:.2f}°" for l in los], fontsize=6)
    ax.set_yticks([_LL2M.transform(lo0, l)[1] for l in las]); ax.set_yticklabels([f"{l:.2f}°" for l in las], fontsize=6)
    ax.tick_params(length=2, direction="in")


def caixa(lat, lon, folga_km=5):
    x, y = _LL2M.transform(np.asarray(lon, float), np.asarray(lat, float))
    f = folga_km * 1000 / math.cos(math.radians(float(np.nanmean(lat))))
    return np.nanmin(x) - f, np.nanmax(x) + f, np.nanmin(y) - f, np.nanmax(y) + f


# ---------------------------------------------------------------- campos
def suaviza(campo, sigma):
    c = np.asarray(campo, float)
    m = np.isfinite(c)
    num = gaussian_filter(np.where(m, c, 0), sigma)
    den = gaussian_filter(m.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-3] = np.nan
    return out


def desenha_campo(ax, campo, lat2d, lon2d, tipo="goes", n_saida=400):
    """campo em mm/h sobre lat2d/lon2d (grade qualquer); tipo goes ou imerg/merge."""
    c = np.asarray(campo, float).copy()
    c[(c < 0) | (c > 500)] = np.nan
    if tipo == "goes":
        c = np.nan_to_num(c, nan=0.0)
        c = grey_closing(c, size=3)
        c = suaviza(c, 0.7)
    elif tipo != "modelo":                 # campos das redes já são contínuos
        c = suaviza(c, 0.8)
    x, y = _LL2M.transform(np.asarray(lon2d, float).ravel(), np.asarray(lat2d, float).ravel())
    x0, x1, y0, y1 = ax.get_xlim() + ax.get_ylim()
    gx, gy = np.meshgrid(np.linspace(x0, x1, n_saida), np.linspace(y1, y0, n_saida))
    f = LinearNDInterpolator(np.c_[x, y], c.ravel())
    img = rgba_campo(f(gx, gy))
    ax.imshow(img, extent=(x0, x1, y0, y1), zorder=2, interpolation="bilinear")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def desenha_estacoes(ax, lat, lon, mm, rotulos=False, raio_pt=5.5):
    x, y = _LL2M.transform(np.asarray(lon, float), np.asarray(lat, float))
    for xi, yi, v in zip(x, y, mm):
        cor, a = cor_estacao(v)
        ax.scatter(xi, yi, s=raio_pt ** 2, color=cor, alpha=a, edgecolors="white", linewidths=0.5, zorder=5)
        if rotulos and v is not None and np.isfinite(v) and v >= 0.2:
            ax.annotate(f"{v:.1f}" if v < 10 else f"{v:.0f}", (xi, yi), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, zorder=6)


def marca(ax, lat, lon, **kw):
    x, y = _LL2M.transform(np.asarray(lon, float), np.asarray(lat, float))
    ax.scatter(x, y, zorder=7, **kw)


def contorno(ax, lat, lon, **kw):
    x, y = _LL2M.transform(np.asarray(lon, float), np.asarray(lat, float))
    ax.plot(x, y, zorder=4, **kw)


def circulo_km(ax, lat, lon, r_km, **kw):
    import matplotlib.pyplot as plt
    x, y = _LL2M.transform(lon, lat)
    ax.add_patch(plt.Circle((x, y), r_km * 1000 / math.cos(math.radians(lat)), fill=False, zorder=6, **kw))


def legendas(fig, ax_campo=None, ax_est=None):
    """Colorbar do campo (estimativa de satélite) e das estações (pluviômetro)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if ax_campo is not None:
        xs = np.linspace(0, 60, 601)
        cm = LinearSegmentedColormap.from_list("campo", [(x / 60, tuple(rgba_campo(x)[:3]) if x >= 1 else
                                                          tuple(rgba_campo(1)[:3])) for x in [0, 1, 2, 3, 4, 6, 8, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 40, 50, 60]])
        cb = fig.colorbar(ScalarMappable(Normalize(0, 60), cm), cax=ax_campo, orientation="horizontal")
        cb.set_ticks([0, 10, 20, 30, 40, 50, 60]); cb.set_label("Chuva (mm/h), estimativa de satélite", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    if ax_est is not None:
        pos = [0.2, 1, 10, 20, 30]
        cm = LinearSegmentedColormap.from_list("est", [(i / 4, to_rgb(e[1])) for i, e in enumerate(EST)])
        cb = fig.colorbar(ScalarMappable(Normalize(0, 4), cm), cax=ax_est, orientation="horizontal")
        cb.set_ticks(range(5)); cb.set_ticklabels(["0,2", "1", "10", "20", "30+"])
        cb.set_label("Chuva no pluviômetro (mm em 1 h)", fontsize=7); cb.ax.tick_params(labelsize=6)
