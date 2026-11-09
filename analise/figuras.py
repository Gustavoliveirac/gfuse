#!/usr/bin/env python
"""
figuras.py — figuras do capítulo no padrão de mapas do hidromet.

  dominio     Figura 3.1, domínio com estações por fonte e papel na avaliação
  limite      um mapa por registro removido pelo limite físico (RRQPE e IMERG
              em painéis separados, mesma escala, estações por cima)
  tolerancia  caso real de núcleo com os raios de 4 km (Figura 3.5)
  eventos     piores eventos em cada fonte disponível (RRQPE, IMERG, MERGE,
              PDIR-Now, GSMaP), 1 h, estações por cima (Figura 4.5)
  confiab     diagrama de confiabilidade a partir de calibracao_60.csv

Cada mapa vem com um CSV (estação, lat, lon, chuva observada e valor de cada
fonte no pixel da estação). Janela de 1 h terminando no instante, em UTC.
Saída: resultados/analise/figuras/ e resultados/analise/mapas_csv/.

Uso:  source logs/produtos_dirs.sh; python analise/figuras.py [--etapas ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

import config3 as K                                     # noqa: E402
import comum as C                                       # noqa: E402
import mapas_hidromet as M                              # noqa: E402
from exploratoria import base, janela                   # noqa: E402

SAI = RAIZ / "resultados" / "analise"
FIG = SAI / "figuras"
CSV = SAI / "mapas_csv"
NOME_PAPEL = {"holdout": "validação espacial", "alvo": "avaliação", "contexto": "entrada", "fora": "fora do recorte"}
COR_PAPEL = {"holdout": "#d62728", "alvo": "#ff7f0e", "contexto": "#1f77b4", "fora": "#9e9e9e"}
MARC = {"CEMADEN": "o", "SUDECAP": "s", "ANA": "^"}
NOMES_FONTE = {"rrqpe": "GOES-16 RRQPE", "imerg_early": "IMERG Early", "merge": "MERGE CPTEC",
               "pdir_now": "PDIR-Now", "gsmap_nrt": "GSMaP NRT", "gsmap_gauge": "GSMaP Gauge",
               "gsmap_now": "GSMaP NOW"}


def latlon_fonte(f):
    la = getattr(f, "lat", None)
    lo = getattr(f, "lon", None)
    if la is None:
        la, lo = f._lat, f._lon
    la, lo = np.asarray(la, float), np.asarray(lo, float)
    if la.ndim == 1:
        lo, la = np.meshgrid(lo, la)
    return la, lo


def rrqpe_hora(b, it):
    """Média das 6 imagens do RRQPE (mm/h) na hora que termina no instante it."""
    cs = []
    for k in range(6):
        try:
            cs.append(np.load(K.GOES_DIR / f"{b['stems'][it - k]}.npy", mmap_mode="r")[K.RRQPE_CH].astype(float))
        except Exception:
            pass
    return np.nanmean(cs, 0) if cs else None


def obs_hora(b, it):
    A, _ = janela(b, 60)
    v = np.full(len(b["cod"]), np.nan)
    v[b["i10"]] = A[it]
    v[b["iH"]] = b["RH"][it]
    return v


def painel_fontes(b, it, titulo_arq, fs, destaque=None, extras=None):
    """Figura com um painel por fonte (1 h), estações por cima; devolve o CSV."""
    t_brt = b["t"][it]
    t_utc = t_brt + pd.Timedelta(hours=3)
    rot = f"{(t_utc - pd.Timedelta(hours=1)):%d/%m/%Y %H:%M} a {t_utc:%H:%M} UTC"
    obs = obs_hora(b, it)
    glat, glon = C.grade_latlon()
    campos = {"rrqpe": (rrqpe_hora(b, it), glat, glon, "goes")}
    for n, f in fs.items():
        try:
            c = f.window_mm(t_brt, 60)
            if c is not None:
                la, lo = latlon_fonte(f)
                campos[n] = (np.asarray(c, float), la, lo, "imerg")
        except Exception:
            pass
    campos = {k: v for k, v in campos.items() if v[0] is not None} or {"estações": (None, None, None, None)}
    n = len(campos)
    ncol = min(n, 4); nlin = int(np.ceil(n / ncol))
    x0, x1, y0, y1 = M.caixa(glat, glon, folga_km=3)
    fig = plt.figure(figsize=(4.2 * ncol, 4.4 * nlin + 0.8))
    linhas = dict(cod=b["cod"], lat=b["lat"], lon=b["lon"], obs_mm_1h=obs)
    for k, (nome, (c, la, lo, tipo)) in enumerate(campos.items()):
        ax = fig.add_axes([0.02 + (k % ncol) / ncol, 0.12 + (nlin - 1 - k // ncol) / nlin * 0.86,
                           0.96 / ncol - 0.02, 0.86 / nlin - 0.04])
        M.mapa_base(ax, x0, x1, y0, y1)
        if c is not None:
            M.desenha_campo(ax, c, la, lo, tipo=tipo)
        M.contorno(ax, *contorno_dom(), color="k", lw=0.8)
        M.desenha_estacoes(ax, b["lat"], b["lon"], obs)
        if destaque is not None:
            M.marca(ax, b["lat"][destaque], b["lon"][destaque], marker="X", s=90, color="m",
                    edgecolors="white", linewidths=0.6)
        if extras:
            extras(ax)
        ax.set_title(f"{NOMES_FONTE.get(nome, nome)}, {rot}", fontsize=8)
        try:
            assert c is not None
            vals = np.array([float(c[np.unravel_index(np.nanargmin((la - a) ** 2 + (lo - o) ** 2), la.shape)])
                             for a, o in zip(b["lat"], b["lon"])])
        except Exception:
            vals = np.full(len(b["cod"]), np.nan)
        linhas[f"{nome}_mmh"] = vals
    M.legendas(fig, fig.add_axes([0.08, 0.05, 0.38, 0.018]), fig.add_axes([0.56, 0.05, 0.38, 0.018]))
    fig.savefig(FIG / f"{titulo_arq}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    d = pd.DataFrame(linhas)
    d.to_csv(CSV / f"{titulo_arq}.csv", index=False)
    return d


def contorno_dom(recuo_km=0.0):
    G = K.IMG_SIZE
    r = recuo_km / K.KM_PER_PX
    xs = np.array([-0.5 + r, G - 0.5 - r, G - 0.5 - r, -0.5 + r, -0.5 + r])
    ys = np.array([-0.5 + r, -0.5 + r, G - 0.5 - r, G - 0.5 - r, -0.5 + r])
    la, lo = C.px_para_latlon(xs, ys)
    return la, lo


def fontes_disponiveis():
    try:
        import produtos
        return produtos.fontes(60)
    except Exception as e:
        print(f"  produtos indisponíveis ({e})")
        return {}


# ---------------------------------------------------------------- etapas
def dominio(b, fs=None):
    x0, x1, y0, y1 = M.caixa(b["lat"], b["lon"], folga_km=6)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    M.mapa_base(ax, x0, x1, y0, y1)
    M.contorno(ax, *contorno_dom(), color="k", lw=1.6)
    M.contorno(ax, *contorno_dom(K.MARGEM_KM), color="k", lw=0.8, ls="--")
    for papel in ("fora", "contexto", "alvo", "holdout"):
        for fonte, mk in MARC.items():
            for tipo in (0, 1):
                m = (b["papel"] == papel) & (b["fonte"] == fonte) & (b["tipo"] == tipo)
                if m.any():
                    M.marca(ax, b["lat"][m], b["lon"][m], marker=mk if tipo == 0 else "v",
                            s=46 if papel == "holdout" else 30, color=COR_PAPEL[papel], edgecolors="k",
                            linewidths=0.4, label=f"{fonte}{' horária' if tipo else ''}, {NOME_PAPEL[papel]} ({m.sum()})")
    M.marca(ax, K.BH_LAT, K.BH_LON, marker="+", s=120, color="k")
    ax.plot([], [], color="k", lw=1.6, label="domínio (128 km)")
    ax.plot([], [], color="k", lw=0.8, ls="--", label=f"faixa de {K.MARGEM_KM:g} km fora das métricas")
    ax.legend(fontsize=6.5, loc="lower left", ncol=2, framealpha=0.92)
    fig.savefig(FIG / "fig3_1_dominio.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    print("  fig3_1_dominio.png")


def limite(b, fs):
    f = SAI / "registros_acima_limite.csv"
    if not f.exists():
        print("  falta registros_acima_limite.csv (rode a exploratória, etapa descartes)")
        return
    regs = pd.read_csv(f, parse_dates=["date"])
    for k, r in regs.iterrows():
        dt = np.abs((b["t"] - r["date"]).total_seconds())
        it = int(np.argmin(dt))
        if dt[it] > 1800:
            continue
        i = int(np.flatnonzero(b["cod"] == r["cod"])[0])

        def extra(ax, i=i, r=r):
            x, y = M._LL2M.transform(b["lon"][i], b["lat"][i])
            ax.annotate(f"{r['mm']:.1f} mm em 10 min (removido)", (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=7, color="m", zorder=8)
        painel_fontes(b, it, f"limite_{k:02d}_{r['cod']}", {k2: v for k2, v in fs.items() if k2 == "imerg_early"},
                      destaque=i, extras=extra)
    print(f"  {len(regs)} mapas de registros acima do limite")


def tolerancia(b, fs):
    A, ok = janela(b, 60)
    alvo = b["papel"][b["i10"]] == "alvo"
    cand = np.where(ok[:, None] & alvo[None, :], A, np.nan)
    it = int(np.nanargmax(np.nanmax(cand, 1)))
    obs = obs_hora(b, it)

    def extra(ax):
        for j in np.flatnonzero(np.nan_to_num(obs) >= 20):
            M.circulo_km(ax, b["lat"][j], b["lon"][j], 4, color="m", lw=1.0)
    painel_fontes(b, it, "fig3_5_tolerancia", {}, extras=extra)
    print(f"  caso {b['stems'][it]}, máximo {np.nanmax(obs):.1f} mm")


def eventos(b, fs, n=4):
    f = SAI / "eventos_produtos_60.csv"
    if f.exists():
        stems = pd.read_csv(f)["goes_stem"].astype(str).tolist()[:n]
    else:
        A, ok = janela(b, 60)
        mx = np.where(ok, np.nanmax(A, 1), -1)
        stems, usados = [], []
        for i in np.argsort(mx)[::-1]:
            if all(abs((b["t"][i] - b["t"][u]).total_seconds()) > 6 * 3600 for u in usados):
                usados.append(i); stems.append(b["stems"][i])
            if len(stems) >= n:
                break
    idx = {s: i for i, s in enumerate(b["stems"])}
    for k, s in enumerate(stems):
        if s in idx:
            painel_fontes(b, idx[s], f"fig4_5_evento_{k}", fs)
    print(f"  eventos: {stems}")


def confiab(b, fs=None):
    f = RAIZ / "resultados" / "calibracao_60.csv"
    if not f.exists():
        print("  falta calibracao_60.csv")
        return
    d = pd.read_csv(f)
    d = d[d["tau"] == 0.2]
    nomes = {"s2rk": "GFUSE", "s1_rk": "satélite corrigido", "s1": "estimador por satélite",
             "base": "fusão em um estágio"}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8)
    for br, nome in nomes.items():
        x = d[d["braco"] == br]
        if len(x):
            ax.plot(x["p_medio"], x["freq_obs"], "o-", ms=4, label=nome)
    ax.set_xlabel("probabilidade de chuva estimada"); ax.set_ylabel("frequência observada de chuva")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_4_confiabilidade.png", dpi=200); plt.close(fig)
    d.to_csv(SAI / "confiabilidade_60.csv", index=False)


ETAPAS = dict(dominio=dominio, limite=limite, tolerancia=tolerancia, eventos=eventos, confiab=confiab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", nargs="+", default=list(ETAPAS))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); CSV.mkdir(parents=True, exist_ok=True)
    b = base()
    fs = fontes_disponiveis() if {"limite", "eventos"} & set(a.etapas) else {}
    for e in a.etapas:
        print(f"\n===== {e} =====", flush=True)
        try:
            ETAPAS[e](b, fs)
        except Exception:
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
