#!/usr/bin/env python
"""
v7_analises.py — análises pedidas na revisão da v6 (somente leitura dos dados).

  gsmap_hora     correlação de cada produto com a estação deslocando o produto de
                 -3 a +3 h (testa desalinhamento de horário, em especial do GSMaP NOW)
  por_rede       comparação com os produtos separada por rede da estação de avaliação
                 (SUDECAP não deve entrar no MERGE; ANA e CEMADEN podem entrar)
  correlacao_lag correlação entre pares até 100 km com todos os instantes (sem
                 condicionar à chuva) e máxima correlação com defasagem de até 30 min
  completude     figura nova, eixo x em dias, blocos por rede com rótulo
  tempo          tempo de uma inferência completa (satélite, correção e GFUSE) por
                 instante, em GPU e em CPU, sem contar o carregamento dos modelos

Saída: resultados/analise/v7/ (CSVs e figuras) e logs/v7_analises.log via o runner.
Uso:  source logs/produtos_dirs.sh; python analise/v7_analises.py [--etapas ...]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

import config3 as K                                     # noqa: E402
from exploratoria import base, janela                   # noqa: E402

SAI = RAIZ / "resultados" / "analise" / "v7"
AN = RAIZ / "resultados" / "analise"
REDE = {"CEMADEM": "CEMADEN"}


def _pares60():
    import produtos as Pr
    d = Pr.pares(60, "s2rk")
    pts = pd.read_parquet(AN / "produtos_pontos_60.parquet")
    return d.merge(pts, on=["goes_stem", "px", "py"], how="left")


def _rede(b, d):
    chave = {(round(float(x), 3), round(float(y), 3)): REDE.get(f, f) for x, y, f in zip(b["px"], b["py"], b["fonte"])}
    return [chave.get((round(float(x), 3), round(float(y), 3)), "?") for x, y in zip(d["px"], d["py"])]


def gsmap_hora(b):
    d = _pares60()
    d = d[d["lado"].isin(["alvo", "holdout"])]
    d["t"] = pd.to_datetime(d["goes_stem"].str[:15], format="%Y%m%d_%H%M%S").dt.round("h")
    prods = [c for c in ("imerg_early", "merge", "pdir_now", "gsmap_now", "gsmap_nrt", "gsmap_gauge") if c in d]
    lin = []
    for p in prods:
        tab = d[["t", "px", "py", p]].dropna()
        for k in range(-3, 4):
            x = d[["t", "px", "py", "obs"]].merge(tab.assign(t=tab["t"] - pd.Timedelta(hours=k)), on=["t", "px", "py"])
            m = (x["obs"] > 0.2) | (x[p] > 0.2)
            lin.append(dict(produto=p, deslocamento_h=k, r=np.corrcoef(x.loc[m, "obs"], x.loc[m, p])[0, 1], n=int(m.sum())))
    t = pd.DataFrame(lin)
    t.to_csv(SAI / "produtos_deslocamento_horario.csv", index=False)
    print(t.pivot(index="deslocamento_h", columns="produto", values="r").round(3).to_string())
    print("deslocamento positivo: produto tomado k horas depois do instante da estação")


def por_rede(b):
    import produtos as Pr
    d = _pares60()
    d["rede"] = _rede(b, d)
    prods = [c for c in ("imerg_early", "merge", "pdir_now", "gsmap_nrt", "gsmap_gauge") if c in d]
    comum = d.dropna(subset=prods)
    comum = comum[comum["lado"].isin(["alvo", "holdout"])]
    pcol = next((c for c in comum.columns if c.startswith("p0.2") and c.endswith("_w0")), None)
    lin = []
    for rede, x in comum.groupby("rede"):
        obs = x["obs"].values
        lin.append(dict(rede=rede, produto="GFUSE", **Pr.metricas(x["mm_w0"].values, obs, x[pcol].values if pcol else None,
                                                               x["thr_chuva"].values if pcol else None)))
        for ref, nome in (("sk", "krigagem simples"), ("idw", "IDW")):
            if f"{ref}_w0" in x:
                lin.append(dict(rede=rede, produto=nome, **Pr.metricas(x[f"{ref}_w0"].values, obs)))
        for p in prods:
            lin.append(dict(rede=rede, produto=p, **Pr.metricas(x[p].values, obs)))
    t = pd.DataFrame(lin)
    t.to_csv(SAI / "produtos_por_rede_60.csv", index=False)
    print(t.pivot(index="produto", columns="rede", values="r").round(3).to_string())
    print(t.pivot(index="produto", columns="rede", values="csi").round(3).to_string())


def correlacao_lag(b, dmax=100.0, lags=range(0, 4)):
    A, _ = janela(b, 10)
    i10 = b["i10"]; px, py = b["px"][i10], b["py"][i10]
    lin = []
    n = A.shape[1]
    for a in range(n):
        for c in range(a + 1, n):
            d = float(np.hypot(px[a] - px[c], py[a] - py[c]) * K.KM_PER_PX)
            if d > dmax:
                continue
            res = dict(cod_a=b["cod"][i10[a]], cod_b=b["cod"][i10[c]], dist_km=d)
            for L in lags:
                for s, (x, y) in (("+", (A[L:, a], A[:len(A) - L, c])), ("-", (A[:len(A) - L, a], A[L:, c]))):
                    if L == 0 and s == "-":
                        continue
                    m = np.isfinite(x) & np.isfinite(y)
                    r = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 500 and x[m].std() > 0 and y[m].std() > 0 else np.nan
                    res[f"r_lag{s}{10 * L}"] = r
            lin.append(res)
    t = pd.DataFrame(lin)
    cols = [c for c in t.columns if c.startswith("r_lag")]
    t["r_max"] = t[cols].max(1); t["lag_do_max"] = t[cols].idxmax(1)
    t.to_csv(SAI / "correlacao_lag_10min.csv", index=False)
    g = t.groupby(pd.cut(t.dist_km, np.arange(0, dmax + 5, 5)))[["r_lag+0", "r_max"]].median()
    print(g.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(t.dist_km, t["r_lag+0"], s=3, alpha=0.2, color="0.5", label="mesmo instante")
    ax.plot([i.mid for i in g.index], g["r_lag+0"], "o-", color="C0", label="mediana, mesmo instante")
    ax.plot([i.mid for i in g.index], g["r_max"], "s-", color="C3", label="mediana, maior r com defasagem até 30 min")
    ax.set_xlabel("distância (km)"); ax.set_ylabel("correlação dos acumulados de 10 min (todos os instantes)")
    ax.set_xlim(0, dmax); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(SAI / "correlacao_lag_10min.png", dpi=170); plt.close(fig)


def completude(b):
    dia = b["t"].normalize()
    lin = []
    redes = np.array([REDE.get(f, f) for f in b["fonte"]])
    for j, i in enumerate(b["i10"]):
        f = pd.Series(np.isfinite(b["R10"][:, j, -1]).astype(float)).groupby(dia).mean()
        lin.append(f.rename(b["cod"][i]))
    m = pd.concat(lin, axis=1).T
    ordem = sorted(range(len(b["i10"])), key=lambda k: (redes[b["i10"][k]], -m.iloc[k].mean()))
    m = m.iloc[ordem]
    rr = redes[b["i10"]][ordem]
    m.to_csv(SAI / "completude_dia.csv")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    xs = pd.to_datetime(m.columns)
    marcas = [k for k in range(len(xs)) if xs[k].day == 1 and xs[k].month in (1, 4, 7, 10)]
    ax.set_xticks(marcas); ax.set_xticklabels([xs[k].strftime("%b/%Y") for k in marcas], rotation=90, fontsize=7)
    lim = 0
    for rede in pd.unique(rr):
        n = int((rr == rede).sum())
        ax.axhline(lim - 0.5, color="w", lw=1.5)
        ax.text(-0.01, (lim + n / 2) / len(rr), f"{rede} ({n})", transform=ax.transAxes, ha="right", va="center",
                fontsize=8)
        lim += n
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("fração dos registros de 10 min disponíveis no dia")
    ax.set_title("Disponibilidade diária dos pluviômetros de 10 minutos (dias sem imagem do satélite não aparecem)", fontsize=9)
    fig.tight_layout(); fig.savefig(SAI / "completude_dia.png", dpi=170); plt.close(fig)
    print(m.mean(1).groupby(rr).describe()[["count", "mean", "min", "max"]].round(3).to_string())


def tempo(b, n=30):
    import torch
    import campos_gfuse as CG
    import comum as C
    import modelo as Mo
    from estacoes import Base
    from config3 import PROC, RES
    base_ = Base()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    mov = Mo.Movimento(); vario = C.carregar_vario(60, PROC)
    stems = C.stems_alinhados(list(base_.stems), 60)
    folds = C.folds_temporais(stems)
    amostra = list(folds[0]["val"])[:n]
    lin = []
    for disp in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        dev = torch.device(disp)
        CG.campos_modelos(amostra[0], 0, base_, holdout, vario, mov, dev)      # aquecimento e carga
        ts = []
        for s in amostra:
            t0 = time.perf_counter()
            CG.campos_modelos(s, 0, base_, holdout, vario, mov, dev)
            if disp == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        lin.append(dict(dispositivo=disp, n=len(ts), mediana_s=np.median(ts), p90_s=np.percentile(ts, 90)))
    t = pd.DataFrame(lin)
    t.to_csv(SAI / "tempo_inferencia.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("inclui leitura dos 6 quadros, montagem das entradas, as redes e as krigagens; "
          "as redes são recarregadas a cada instante, então o tempo é um limite superior")


ETAPAS = dict(gsmap_hora=gsmap_hora, por_rede=por_rede, correlacao_lag=correlacao_lag, completude=completude,
              tempo=tempo)


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


if __name__ == "__main__":
    main()
