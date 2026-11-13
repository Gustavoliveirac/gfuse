#!/usr/bin/env python
"""
fechamento.py — análises pendentes dos comentários do capítulo.

  qc_mes         estação e mês dos registros invalidados pelo limite mensal (regra 1)
  qc_bascula     gráficos com dados reais de báscula travada na escala horária (regra 16)
                 e em 10 min (regra 2): o dia do caso com mais registros sinalizados
  qc_mapas       painel 2 x 2 de mapas dos casos acima do limite sem confirmação das
                 vizinhas em 10 min (5), 1 h (8), 6 h (11) e 24 h (14)
  nucleos_prod   CSI de núcleos de 20 e 30 mm dos produtos de satélite, no ponto, nos
                 instantes com todos os produtos, separado em avaliação e validação espacial
Saída: resultados/analise/fechamento/ (CSV e PNG).
Uso:  source logs/produtos_dirs.sh; python analise/fechamento.py [--etapas ...]
"""
from __future__ import annotations

import argparse
import re
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

import config3 as K                                     # noqa: E402
import mapas_hidromet as H                              # noqa: E402
from exploratoria import base, _ler_serie              # noqa: E402

SAI = RAIZ / "resultados" / "analise" / "fechamento"
ESCALA = {5: ("10 min", 1), 8: ("1 h", 6), 11: ("6 h", 36), 14: ("24 h", 144)}


def _tem(motivo, cod):
    return motivo.fillna("").astype(str).apply(lambda m: str(cod) in re.findall(r"\d+", m))


def _series(b):
    for c, f, tp in zip(b["cod"], b["fonte"], b["tipo"]):
        if tp != 0:
            continue
        s = _ler_serie(f, c)
        if s is not None and "motivo" in s.columns:
            yield c, f, s


def qc_mes(b):
    lin = []
    for c, f, s in _series(b):
        m = s[_tem(s["motivo"], 1)]
        if len(m):
            for mes, n in m.groupby(m["date"].dt.to_period("M")).size().items():
                lin.append(dict(cod=c, rede=f, mes=str(mes), registros=int(n)))
    d = pd.DataFrame(lin).sort_values("registros", ascending=False)
    d.to_csv(SAI / "qc_limite_mensal.csv", index=False)
    print(d.head(10).to_string(index=False))
    print(f"total {d['registros'].sum()} registros em {len(d)} estação-mês")


def qc_bascula(b):
    for cod_regra, nome in ((16, "horaria"), (2, "10min")):
        melhor = None
        for c, f, s in _series(b):
            m = _tem(s["motivo"], cod_regra)
            if m.any():
                dia = s.loc[m, "date"].dt.floor("D").value_counts()
                if melhor is None or dia.iloc[0] > melhor[3]:
                    melhor = (c, f, dia.index[0], dia.iloc[0])
        if melhor is None:
            print(f"  regra {cod_regra}: nenhum caso"); continue
        c, f, dia, n = melhor
        s = _ler_serie(f, c)
        x = s[(s["date"] >= dia - pd.Timedelta(hours=12)) & (s["date"] < dia + pd.Timedelta(hours=36))].copy()
        x["sinal"] = _tem(x["motivo"], cod_regra)
        x.to_csv(SAI / f"qc_bascula_{nome}.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(x["date"], x["rainfall"], width=0.006, color="#4292c6", label="registro válido")
        ax.bar(x.loc[x["sinal"], "date"], x.loc[x["sinal"], "rainfall"], width=0.006, color="#d62728",
               label="sinalizado (báscula travada)")
        ax.set_ylabel("chuva em 10 min (mm)"); ax.legend(fontsize=8)
        ax.set_title(f"Estação {c} ({f}), {dia:%d/%m/%Y}: {n} registros sinalizados", fontsize=10)
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(SAI / f"qc_bascula_{nome}.png", dpi=200); plt.close(fig)
        print(f"  regra {cod_regra}: {c} ({f}) em {dia:%Y-%m-%d}, {n} registros")


def qc_mapas(b):
    i10 = b["i10"]
    t_all = b["t"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    glat, glon = __import__("comum").grade_latlon()
    x0, x1, y0, y1 = H.caixa(glat, glon, folga_km=25)
    lin = []
    for ax, (regra, (rot, nwin)) in zip(axs.ravel(), ESCALA.items()):
        H.mapa_base(ax, x0, x1, y0, y1)
        caso = None
        for c, f, s in _series(b):
            m = _tem(s["motivo"], regra)
            if m.any():
                k = s.loc[m, "rainfall"].idxmax()
                if caso is None or s.loc[k, "rainfall"] > caso[3]:
                    caso = (c, f, s.loc[k, "date"], float(s.loc[k, "rainfall"]))
        if caso is None:
            ax.set_title(f"{rot}: nenhum caso"); continue
        c, f, t0, v = caso
        it = int(np.argmin(np.abs((t_all - t0).total_seconds())))
        R = b["R10"][max(0, it - nwin + 1):it + 1, :, -1]
        obs = np.where(np.isfinite(R).all(0), R.sum(0), np.nan) if nwin > 1 else R[-1]
        lat, lon = b["lat"][i10], b["lon"][i10]
        H.desenha_estacoes(ax, lat, lon, obs, rotulos=True)
        j = int(np.flatnonzero(b["cod"][i10] == c)[0]) if (b["cod"][i10] == c).any() else None
        if j is not None:
            H.marca(ax, lat[j], lon[j], marker="X", s=140, color="m", edgecolors="white", linewidths=0.8)
        ax.set_title(f"{rot}: {c} ({f}), {t0:%d/%m/%Y %H:%M}, {v:.1f} mm", fontsize=10)
        lin.append(dict(escala=rot, cod=c, rede=f, data=t0, valor_mm=v))
    fig.tight_layout(); fig.savefig(SAI / "qc_mapas_nao_confirmados.png", dpi=170); plt.close(fig)
    pd.DataFrame(lin).to_csv(SAI / "qc_mapas_nao_confirmados.csv", index=False)
    print(pd.DataFrame(lin).to_string(index=False))


def nucleos_prod(b):
    import produtos as Pr
    d = Pr.pares(60, "s2rk")
    pts = pd.read_parquet(RAIZ / "resultados" / "analise" / "produtos_pontos_60.parquet")
    d = d.merge(pts, on=["goes_stem", "px", "py"], how="left")
    prods = [c for c in ("imerg_early", "merge", "pdir_now", "gsmap_nrt", "gsmap_gauge") if c in d]
    d = d.dropna(subset=prods)
    lin = []
    for lado, x in d.groupby("lado"):
        o = x["obs"].values
        cols = [("GFUSE", "mm_w0"), ("krigagem simples", "sk_w0"), ("IDW", "idw_w0"), ("RRQPE", "rrq_w0")] + [(p, p) for p in prods]
        for nome, c in cols:
            if c not in x:
                continue
            e = x[c].values
            for tau in (20, 30):
                h = ((e >= tau) & (o >= tau)).sum(); m_ = ((e < tau) & (o >= tau)).sum(); fa = ((e >= tau) & (o < tau)).sum()
                lin.append(dict(lado={"alvo": "avaliação", "holdout": "validação espacial"}.get(lado, lado),
                                produto=nome, limiar_mm=tau, csi=h / max(h + m_ + fa, 1), acertos=int(h),
                                perdidos=int(m_), alarmes=int(fa)))
    t = pd.DataFrame(lin)
    t.to_csv(SAI / "nucleos_produtos_60.csv", index=False)
    print(t.pivot_table(index="produto", columns=["lado", "limiar_mm"], values="csi").round(3).to_string())
    print("CSI no ponto, sem tolerância espacial, nos instantes com todos os produtos")


ETAPAS = dict(qc_mes=qc_mes, qc_bascula=qc_bascula, qc_mapas=qc_mapas, nucleos_prod=nucleos_prod)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--etapas", nargs="+", default=list(ETAPAS))
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
