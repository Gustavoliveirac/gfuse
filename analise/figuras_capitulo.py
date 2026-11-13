#!/usr/bin/env python
"""
figuras_capitulo.py — gera de novo as figuras de dados do capítulo, na ordem do texto,
em resultados/analise/figuras_capitulo/ (PNG e CSV de cada uma).

  fig3_2_dominio      estações com a forma indicando o papel e a cor indicando a rede
  fig3_4_completude   fração diária de registros por estação, blocos por rede (usa a base
                      com o CEMADEN preenchido, processados_cem0, quando ela existir)
  fig3_5_correlacao   correlação entre pares de estações x distância, com e sem defasagem
  fig3_9_tolerancia   núcleo real (confirmado por vizinha) com os raios de 4 km
  fig4_3_defasagem    correlação RRQPE x pluviômetro, com o zero na imagem iniciada 10 min
                      antes do fim da janela da estação (a imagem de 10:10 com a chuva de
                      10:10 a 10:20, que é quando ela fica disponível)
  eventos             opções de eventos com o radar da Cemig (opcoes_eventos.py)

Uso:  source logs/produtos_dirs.sh; python analise/figuras_capitulo.py [--etapas ...]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402

import config3 as K                                     # noqa: E402
import mapas_hidromet as H                              # noqa: E402

SAI = RAIZ / "resultados" / "analise" / "figuras_capitulo"
AN = RAIZ / "resultados" / "analise"
REDE = {"CEMADEM": "CEMADEN"}
COR_PAPEL = {"holdout": "#d62728", "alvo": "#ff7f0e", "contexto": "#1f77b4", "fora": "#8c8c8c"}
NOME_PAPEL = {"holdout": "validação espacial", "alvo": "avaliação", "contexto": "entrada", "fora": "fora do recorte"}
SIMB_REDE = {"ANA": "^", "CEMADEN": "o", "SUDECAP": "s", "INMET": "D"}


def base():
    from exploratoria import base as b0
    return b0()


def dominio(b):
    """Símbolo pela rede e cor pelo papel."""
    from figuras import contorno_dom
    x0, x1, y0, y1 = H.caixa(b["lat"], b["lon"], folga_km=6)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    H.mapa_base(ax, x0, x1, y0, y1)
    H.contorno(ax, *contorno_dom(), color="k", lw=1.6)
    redes = np.array([REDE.get(f, f) for f in b["fonte"]])
    for papel, cor in COR_PAPEL.items():
        for rede, mk in SIMB_REDE.items():
            m = (b["papel"] == papel) & (redes == rede)
            if m.any():
                H.marca(ax, b["lat"][m], b["lon"][m], marker=mk, s=44 if papel == "holdout" else 30, color=cor,
                        edgecolors="k", linewidths=0.5)
    inv = Path(__import__("os").environ.get("INVENTARIO", RAIZ / "logs" / "inventario_estacoes.csv"))
    if not inv.exists():
        inv = Path(__file__).resolve().parent / "inventario_estacoes.csv"   # cópia que vem com a entrega
    if inv.exists():                       # INMET a partir do inventário do hidromet
        iv = pd.read_csv(inv, sep=";", decimal=",")
        iv.columns = [c.lstrip("\ufeff") for c in iv.columns]
        iv = iv[(iv["fonte"].str.lower() == "inmet") & iv["situacao"].str.startswith("com dado")]
        for dentro, papel in ((True, "contexto"), (False, "fora")):
            m = iv["dentro_do_recorte"] == dentro
            if m.any():
                H.marca(ax, iv.loc[m, "lat"].values, iv.loc[m, "lon"].values, marker="D", s=32, color=COR_PAPEL[papel],
                        edgecolors="k", linewidths=0.5)
        print(f"  INMET: {len(iv)} estações com dado")
    else:
        print(f"  sem o inventário ({inv}); INMET fora da figura")
    H.marca(ax, K.BH_LAT, K.BH_LON, marker="+", s=120, color="k")
    cont = {p: int((b["papel"] == p).sum()) for p in COR_PAPEL}
    l1 = [Line2D([], [], ls="", marker="o", color=c, markeredgecolor="k", label=f"{NOME_PAPEL[p]} ({cont[p]})") for p, c in COR_PAPEL.items()]
    l2 = [Line2D([], [], ls="", marker=mk, color="0.6", markeredgecolor="k", label=r) for r, mk in SIMB_REDE.items()]
    leg = ax.legend(handles=l1, title="papel", fontsize=7, title_fontsize=7, loc="upper left", framealpha=0.92)
    ax.add_artist(leg)
    ax.legend(handles=l2 + [Line2D([], [], color="k", lw=1.6, label="domínio (128 km)")], title="rede",
              fontsize=7, title_fontsize=7, loc="upper right", framealpha=0.92)
    fig.savefig(SAI / "fig3_2_dominio.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(dict(cod=b["cod"], rede=redes, papel=b["papel"], lat=b["lat"], lon=b["lon"])).to_csv(
        SAI / "fig3_2_dominio.csv", index=False)


def completude(b):
    arq = RAIZ / "processados_cem0" / "estacoes.npz"
    z = dict(np.load(arq if arq.exists() else K.EST_NPZ, allow_pickle=False))
    print(f"  base: {'CEMADEN preenchido' if arq.exists() else 'original'}")
    i10 = np.flatnonzero(z["tipo"] == 0)
    redes = np.array([REDE.get(f, f) for f in z["fonte"].astype(str)])[i10]
    t = pd.to_datetime(z["t_brt"]).normalize()
    m = pd.DataFrame(np.isfinite(z["R10"][:, :, -1]).astype(float)).groupby(t.values).mean().T
    ordem = sorted(range(len(i10)), key=lambda k: (redes[k], -m.iloc[k].mean()))
    m = m.iloc[ordem]; rr = redes[ordem]
    fig, ax = plt.subplots(figsize=(12, 7.5))
    im = ax.imshow(m.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    xs = pd.to_datetime(m.columns)
    marc = [k for k in range(len(xs)) if xs[k].day == 1 and xs[k].month in (1, 4, 7, 10)]
    ax.set_xticks(marc); ax.set_xticklabels([xs[k].strftime("%m/%Y") for k in marc], rotation=90, fontsize=7)
    lim = 0
    for rede in pd.unique(rr):
        n = int((rr == rede).sum())
        if lim:
            ax.axhline(lim - 0.5, color="k", lw=0.8)
        ax.text(-0.01, 1 - (lim + n / 2) / len(rr), f"{rede}\n({n})", transform=ax.transAxes, ha="right",
                va="center", fontsize=8)
        lim += n
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.03); cb.set_label("fração dos registros de 10 min disponíveis no dia")
    fig.tight_layout(); fig.savefig(SAI / "fig3_4_completude.png", dpi=200); plt.close(fig)
    m.to_csv(SAI / "fig3_4_completude.csv")
    print(m.mean(1).groupby(rr).mean().round(3).to_string())


def correlacao(b):
    import v7_analises as V
    V.SAI.mkdir(parents=True, exist_ok=True)
    V.correlacao_lag(b)
    for ext in ("png", "csv"):
        shutil.copy(V.SAI / f"correlacao_lag_10min.{ext}", SAI / f"fig3_5_correlacao.{ext}")


def tolerancia(b):
    import figuras as Fg
    Fg.FIG = SAI; Fg.CSV = SAI
    Fg.tolerancia(b, {})
    (SAI / "fig3_5_tolerancia.png").rename(SAI / "fig3_9_tolerancia.png")
    (SAI / "fig3_5_tolerancia.csv").rename(SAI / "fig3_9_tolerancia.csv")


def defasagem(b):
    f = AN / "rrqpe_defasagem_curva.csv"
    if not f.exists():
        from exploratoria import defasagem as dfs, SAI as S0
        dfs(b)
    d = pd.read_csv(f)
    d["deslocamento_min"] = 10 - d["rrqpe_antes_min"]      # 0 = imagem iniciada 10 min antes do fim da janela
    d = d.sort_values("deslocamento_min")
    d.to_csv(SAI / "fig4_3_defasagem.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(d["deslocamento_min"], d["r"], "o-", color="#08519c")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("deslocamento da imagem (min); 0 = imagem disponível ao fim da janela da estação")
    ax.set_ylabel("correlação, 10 min"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(SAI / "fig4_3_defasagem.png", dpi=200); plt.close(fig)
    print(d[["deslocamento_min", "r", "n"]].to_string(index=False))


def eventos(b):
    subprocess.run([sys.executable, "-u", str(Path(__file__).parent / "opcoes_eventos.py")], cwd=RAIZ)


ETAPAS = dict(dominio=dominio, completude=completude, correlacao=correlacao, tolerancia=tolerancia,
              defasagem=defasagem, eventos=eventos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", nargs="+", default=list(ETAPAS))
    a = ap.parse_args()
    SAI.mkdir(parents=True, exist_ok=True)
    b = base()
    for e in a.etapas:
        print(f"\n===== {e} =====", flush=True)
        try:
            ETAPAS[e](b)
        except Exception:
            traceback.print_exc(file=sys.stdout)
    print("FIM")


if __name__ == "__main__":
    main()
