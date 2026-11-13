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
        m = s[_tem(s["motivo"], 1) & s["rainfall"].notna()]      # só linhas com valor (a grade tem linhas vazias)
        if len(m):
            for mes, n in m.groupby(m["date"].dt.to_period("M")).size().items():
                tot = s.loc[s["date"].dt.to_period("M") == mes, "rainfall"].sum()
                lin.append(dict(cod=c, rede=f, mes=str(mes), registros=int(n), total_mes_mm=round(float(tot), 1)))
    d = pd.DataFrame(lin).sort_values("registros", ascending=False)
    d.to_csv(SAI / "qc_limite_mensal.csv", index=False)
    print(d.head(10).to_string(index=False))
    print(f"total {d['registros'].sum()} registros em {len(d)} estação-mês")


def qc_bascula(b):
    for cod_regra, nome in ((16, "horaria"), (2, "10min")):
        melhor = None
        for c, f, s in _series(b):
            m = _tem(s["motivo"], cod_regra) & s["rainfall"].notna()
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
    """Para cada escala, o registro sinalizado com maior acumulado na escala, e o acumulado das vizinhas
    na mesma janela, lidos das séries brutas (inclusive o valor invalidado da estação sinalizada)."""
    import comum as C
    series = {c: (f, s.assign(rainfall=s["rainfall"].fillna(0)).set_index("date")["rainfall"]) for c, f, s in _series(b)}
    brutas = {c: s for c, f, s in _series(b)}
    pos = {c: (la, lo) for c, la, lo in zip(b["cod"], b["lat"], b["lon"])}
    fig, axs = plt.subplots(2, 2, figsize=(13, 13))
    glat, glon = C.grade_latlon()
    x0, x1, y0, y1 = H.caixa(glat, glon, folga_km=25)
    lin = []
    for ax, (regra, (rot, nwin)) in zip(axs.ravel(), ESCALA.items()):
        H.mapa_base(ax, x0, x1, y0, y1)
        caso = None
        for c, s in brutas.items():
            m = _tem(s["motivo"], regra) & s["rainfall"].notna()
            if not m.any():
                continue
            acc = series[c][1].rolling(nwin, min_periods=1).sum()
            for t0 in s.loc[m, "date"]:
                v = float(acc.get(t0, np.nan))
                if np.isfinite(v) and (caso is None or v > caso[2]):
                    caso = (c, t0, v)
        if caso is None:
            ax.set_title(f"{rot}: nenhum caso"); continue
        c, t0, v = caso
        ini = t0 - pd.Timedelta(minutes=10 * nwin)
        vals, la, lo = [], [], []
        for cc, (f, ser) in series.items():
            if cc == c or cc not in pos:
                continue
            w = ser[(ser.index > ini) & (ser.index <= t0)]
            vals.append(float(w.sum()) if len(w) else np.nan); la.append(pos[cc][0]); lo.append(pos[cc][1])
        H.desenha_estacoes(ax, np.array(la), np.array(lo), np.array(vals), rotulos=True)
        H.marca(ax, pos[c][0], pos[c][1], marker="X", s=160, color="m", edgecolors="white", linewidths=0.8)
        xx, yy = H._LL2M.transform(pos[c][1], pos[c][0])
        ax.annotate(f"{v:.1f} mm", (xx, yy), xytext=(8, 8), textcoords="offset points", fontsize=11, color="m", weight="bold")
        ax.set_title(f"{rot}: {c}, {ini:%d/%m/%Y %H:%M} a {t0:%H:%M}, {v:.1f} mm", fontsize=10)
        lin.append(dict(escala=rot, cod=c, inicio=ini, fim=t0, acumulado_mm=v))
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
        def pcol(tau):
            for c in x.columns:
                if c.startswith("p") and c.endswith("_w0"):
                    try:
                        if float(c[1:-3]) == tau:
                            return c
                    except ValueError:
                        pass
        for nome, c in cols:
            if c not in x:
                continue
            for tau in (20, 30):
                pc = pcol(tau) if nome == "GFUSE" else None
                if pc is not None:          # GFUSE: probabilidade de passar do limiar, no corte de maior CSI
                    melhor = None
                    for corte in np.arange(0.05, 0.96, 0.05):
                        e_ = x[pc].values >= corte
                        h = (e_ & (o >= tau)).sum(); m_ = (~e_ & (o >= tau)).sum(); fa = (e_ & (o < tau)).sum()
                        if melhor is None or h / max(h + m_ + fa, 1) > melhor[0]:
                            melhor = (h / max(h + m_ + fa, 1), h, m_, fa)
                    _, h, m_, fa = melhor
                else:
                    e = x[c].values
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
