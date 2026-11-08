#!/usr/bin/env python
"""
resultados.py — tabelas e figuras de desempenho por janela (10, 30 e 60 min),
a partir dos CSVs consolidados pelo resumo.py em resultados/.

Para cada janela grava em resultados/analise/
  tabela_{agg}.csv        produtos x métricas (todos os instantes, estações-alvo)
  tabela_holdout_{agg}.csv  o mesmo nas estações nunca vistas
  folds_{agg}.csv         média, mínimo e máximo de cada métrica entre os 5 blocos
  calibracao_{agg}.csv    decomposição do Brier por produto
  figuras/desempenho_{agg}.png  r e CSI por produto com a faixa entre blocos
Cada CSV traz as colunas versao_codigo e arquivo_origem.

Uso:  python analise/resultados.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

RES = RAIZ / "resultados"
SAI = RES / "analise"
FIG = SAI / "figuras"

NOMES = {  # código -> nome no texto (ordem de apresentação)
    "s2rk": "GFUSE (dois estágios)",
    "s1_rk": "satélite corrigido",
    "s1": "estimador por satélite",
    "base": "fusão em um estágio",
    "sk": "krigagem simples",
    "ok": "krigagem ordinária",
    "krig": "krigagem ordinária",
    "idw": "IDW",
    "rrq": "RRQPE",
    "pers": "referência de teto",
}
METR = ["n_pares", "r", "bias", "ratio", "pod0.2", "far0.2", "csi0.2_w0"]


def versao():
    try:
        g = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=RAIZ,
                           capture_output=True, text=True).stdout.strip()
    except Exception:
        g = "?"
    import re
    v = re.search(r'^VERSAO\s*=\s*"([^"]+)"', (RAIZ / "config3.py").read_text(), re.M)
    return f"{v.group(1) if v else '?'} / {g}"


def tabela(arq, agg, ver):
    if not arq.exists():
        print(f"  falta {arq.name}")
        return None
    d = pd.read_csv(arq)
    d = d[d["subconjunto"] == "todos"] if "subconjunto" in d else d
    nuc = [c for c in d.columns if c.startswith("nuc") and c.endswith("_csi")]
    cols = [c for c in METR + nuc if c in d.columns]
    d = d[d["produto"].isin(NOMES)].copy()
    d["ordem"] = d["produto"].map(list(NOMES).index)
    d = d.sort_values("ordem")
    d.insert(1, "nome", d["produto"].map(NOMES))
    out = d[["produto", "nome"] + cols].copy()
    out["versao_codigo"] = ver
    out["arquivo_origem"] = arq.name
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ver = versao()
    print(f"versão do código: {ver}")
    for agg in (10, 30, 60):
        print(f"\n===== janela de {agg} min =====")
        t = tabela(RES / f"metricas_{agg}.csv", agg, ver)
        if t is not None:
            t.to_csv(SAI / f"tabela_{agg}.csv", index=False)
            print(t.drop(columns=["versao_codigo", "arquivo_origem"]).round(3).to_string(index=False))
        h = tabela(RES / f"holdout_{agg}.csv", agg, ver)
        if h is not None:
            h.to_csv(SAI / f"tabela_holdout_{agg}.csv", index=False)
            print("holdout:")
            print(h[["produto", "r", "csi0.2_w0"]].round(3).to_string(index=False))
        f = RES / f"metricas_fold_{agg}.csv"
        fo = None
        if f.exists():
            d = pd.read_csv(f)
            d = d[d["braco"].isin(NOMES)]
            met = [c for c in ("r", "csi0.2_w0") + tuple(c for c in d.columns if c.endswith("_csi"))
                   if c in d.columns]
            fo = d.groupby("braco")[met].agg(["mean", "min", "max"])
            fo.columns = [f"{a}_{b}" for a, b in fo.columns]
            fo = fo.reset_index()
            fo["versao_codigo"] = ver
            fo.to_csv(SAI / f"folds_{agg}.csv", index=False)
            print("faixa entre blocos:")
            print(fo[["braco", "r_mean", "r_min", "r_max"]].round(3).to_string(index=False))
        b = RES / f"brier_{agg}.csv"
        if b.exists():
            bb = pd.read_csv(b)
            bb = bb[bb["braco"].isin(NOMES)]
            bb["versao_codigo"] = ver
            bb.to_csv(SAI / f"calibracao_{agg}.csv", index=False)
        if t is not None:
            fig, axs = plt.subplots(1, 2, figsize=(11, 4))
            for ax, col, nome in zip(axs, ("r", "csi0.2_w0"), ("r (mm)", "CSI de ocorrência (> 0,2 mm)")):
                tt = t.dropna(subset=[col])
                y = np.arange(len(tt))
                ax.barh(y, tt[col], color="C0")
                if fo is not None and f"{col}_min" in fo:
                    for k, p in enumerate(tt["produto"]):
                        m = fo[fo["braco"] == p]
                        if len(m):
                            ax.plot([m[f"{col}_min"].iloc[0], m[f"{col}_max"].iloc[0]], [k, k], color="k", lw=2)
                ax.set_yticks(y); ax.set_yticklabels(tt["nome"], fontsize=8); ax.invert_yaxis()
                ax.set_xlabel(nome); ax.grid(axis="x", alpha=0.3)
            fig.suptitle(f"janela de {agg} min (barra preta: mínimo e máximo entre os 5 blocos)", fontsize=10)
            fig.tight_layout(); fig.savefig(FIG / f"desempenho_{agg}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
