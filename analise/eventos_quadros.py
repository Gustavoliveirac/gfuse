#!/usr/bin/env python
"""
eventos_quadros.py — um evento importante em horas consecutivas (colunas) e cada
fonte numa linha: krigagem das estações de entrada, GFUSE, satélite corrigido,
RRQPE, IMERG Early e MERGE. Padrão de mapas do hidromet; estações de avaliação
como círculos e de validação espacial como triângulos, coloridas pela chuva
observada na hora. Modelos do bloco em que o evento está no teste.

Uso:  source logs/produtos_dirs.sh; python analise/eventos_quadros.py [--stem AAAAMMDD_HHMMSS] [--horas 5]
Sem --stem, usa o primeiro evento de resultados/analise/eventos_produtos_60.csv.
Saída: resultados/analise/v7/evento_quadros.png e evento_quadros.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import comum as C, modelo as Mo, mapas_hidromet as H
import campos_gfuse as CG
from config3 import PROC, RES
from estacoes import Base
from exploratoria import base as base_expl
from figuras import obs_hora

SAI = RES / "analise" / "v7"
LINHAS = [("sk", "Krigagem das estações de entrada"), ("s2rk", "GFUSE"), ("s1_rk", "Satélite corrigido"),
          ("rrqpe", "GOES-16 RRQPE"), ("imerg_early", "IMERG Early"), ("merge", "MERGE CPTEC")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem"); ap.add_argument("--horas", type=int, default=5)
    a = ap.parse_args()
    SAI.mkdir(parents=True, exist_ok=True)
    stem = a.stem or pd.read_csv(RES / "analise" / "eventos_produtos_60.csv")["goes_stem"].astype(str).iloc[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = Base(); be = base_expl()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    mov = Mo.Movimento(); vario = C.carregar_vario(60, PROC)
    stems = C.stems_alinhados(list(bb.stems), 60); folds = C.folds_temporais(stems)
    idx = {s: i for i, s in enumerate(be["stems"])}
    t_ev = C.stem_para_brt(stem)
    alvo_t = [t_ev + pd.Timedelta(hours=h) for h in range(-(a.horas // 2), a.horas - a.horas // 2)]
    sel = [min(stems, key=lambda s: abs((C.stem_para_brt(s) - t).total_seconds())) for t in alvo_t]
    import produtos
    fs = produtos.fontes(60)
    glat, glon = C.grade_latlon()
    x0, x1, y0, y1 = H.caixa(glat, glon, folga_km=2)
    fig = plt.figure(figsize=(3.3 * len(sel), 3.4 * len(LINHAS) + 0.8))
    linhas_csv = []
    papel = be["papel"]
    for j, s in enumerate(sel):
        f = CG.fold_de(s, folds)
        campos, _, _ = CG.campos_modelos(s, f, bb, holdout, vario, mov, dev)
        tb = C.stem_para_brt(s)
        for n in ("imerg_early", "merge"):
            if n in fs:
                try:
                    c = fs[n].window_mm(tb.floor("30min") if n == "imerg_early" else tb, 60)
                    if c is not None:
                        lat_ = getattr(fs[n], "lat", None); lon_ = getattr(fs[n], "lon", None)
                        if lat_ is None:
                            lat_, lon_ = fs[n]._lat, fs[n]._lon
                        la, lo = np.meshgrid(np.asarray(lat_), np.asarray(lon_), indexing="ij")
                        campos[n] = (np.asarray(c, float), la, lo)
                except Exception:
                    pass
        obs = obs_hora(be, idx[s])
        t_utc = tb + pd.Timedelta(hours=3)
        for i, (nome, rot) in enumerate(LINHAS):
            ax = fig.add_axes([0.03 + j / len(sel) * 0.96, 0.1 + (len(LINHAS) - 1 - i) / len(LINHAS) * 0.88,
                               0.96 / len(sel) - 0.01, 0.88 / len(LINHAS) - 0.02])
            H.mapa_base(ax, x0, x1, y0, y1); ax.set_xticks([]); ax.set_yticks([])
            v = campos.get(nome)
            if v is not None:
                if isinstance(v, tuple):
                    H.desenha_campo(ax, v[0], v[1], v[2], tipo="imerg")
                else:
                    H.desenha_campo(ax, v, glat, glon, tipo="goes" if nome == "rrqpe" else "modelo")
            for pap, mk in (("alvo", "o"), ("holdout", "^")):
                m = papel == pap
                xs, ys = H._LL2M.transform(be["lon"][m], be["lat"][m])
                for xx, yy, val in zip(xs, ys, obs[m]):
                    cor, al = H.cor_estacao(val)
                    ax.scatter(xx, yy, marker=mk, s=26, color=cor, alpha=al, edgecolors="k", linewidths=0.4, zorder=5)
            if i == 0:
                ax.set_title(f"{(t_utc - pd.Timedelta(hours=1)):%d/%m %H:%M} a {t_utc:%H:%M} UTC", fontsize=8)
            if j == 0:
                ax.set_ylabel(rot, fontsize=8)
        linhas_csv.append(dict(goes_stem=s, bloco=f, max_obs=float(np.nanmax(obs))))
    H.legendas(fig, fig.add_axes([0.1, 0.04, 0.35, 0.012]), fig.add_axes([0.55, 0.04, 0.35, 0.012]))
    fig.text(0.5, 0.012, "círculos: estações de avaliação; triângulos: estações de validação espacial", ha="center", fontsize=8)
    fig.savefig(SAI / "evento_quadros.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(linhas_csv).to_csv(SAI / "evento_quadros.csv", index=False)
    print("FIM", sel)


if __name__ == "__main__":
    main()
