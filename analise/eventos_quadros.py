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
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import comum as C, modelo as Mo, mapas_hidromet as H
H.FS = 3.0
import campos_gfuse as CG
import cemig
import functools
CG.carregar_rede = functools.lru_cache(maxsize=None)(CG.carregar_rede)
_campos_orig = CG.campos_modelos


def _campos_cache(stem, fold, *a, **k):
    """Guarda os campos dos modelos de cada instante; refazer a figura não roda o modelo de novo."""
    pasta = RES / "analise" / "v8_eventos" / "cache"; pasta.mkdir(parents=True, exist_ok=True)
    f = pasta / f"{stem}.npz"
    if f.exists():
        z = np.load(f)
        return {n: z[n] for n in z.files}, None, None
    campos, probs, inf = _campos_orig(stem, fold, *a, **k)
    np.savez_compressed(f, **{n: np.asarray(v) for n, v in campos.items() if not isinstance(v, tuple)})
    return campos, probs, inf


CG.campos_modelos = _campos_cache
from config3 import PROC, RES
from estacoes import Base
from exploratoria import base as base_expl
from figuras import obs_hora

SAI = RES / "analise" / "v7"
LINHAS = [("imerg_early", "IMERG Early"), ("rrqpe", "GOES-16 RRQPE"), ("sk", "Krigagem das estações"),
          ("s1_rk", "Satélite corrigido"), ("s2rk", "GFUSE"), ("merge", "MERGE CPTEC"), ("cemig", "Radar Cemig")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stem"); ap.add_argument("--horas", type=int, default=5)
    ap.add_argument("--saida", default="evento_quadros")
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
    fig = plt.figure(figsize=(4.2 * len(sel), 4.2 * len(LINHAS) + 2.2))
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
        rad = cemig.hora(tb + pd.Timedelta(hours=3))
        if rad is not None:
            campos["cemig"] = rad
        obs = obs_hora(be, idx[s])
        t_utc = tb + pd.Timedelta(hours=3)
        for i, (nome, rot) in enumerate(LINHAS):
            ax = fig.add_axes([0.07 + j / len(sel) * 0.92, 0.075 + (len(LINHAS) - 1 - i) / len(LINHAS) * 0.9,
                               0.92 / len(sel) - 0.004, 0.9 / len(LINHAS) - 0.004])
            H.mapa_base(ax, x0, x1, y0, y1); ax.set_xticks([]); ax.set_yticks([])
            v = campos.get(nome)
            if v is not None:
                if isinstance(v, tuple):
                    H.desenha_campo(ax, v[0], v[1], v[2], tipo="modelo" if nome == "cemig" else "imerg")
                else:
                    H.desenha_campo(ax, v, glat, glon, tipo="goes" if nome == "rrqpe" else "modelo")
            else:
                ax.text(0.5, 0.5, "sem dado", transform=ax.transAxes, ha="center", va="center", fontsize=22, color="0.3")
            for pap, mk in (("alvo", "o"), ("holdout", "^")):
                m = papel == pap
                xs, ys = H._LL2M.transform(be["lon"][m], be["lat"][m])
                for xx, yy, val in zip(xs, ys, obs[m]):
                    cor, al = H.cor_estacao(val)
                    ax.scatter(xx, yy, marker=mk, s=26, color=cor, alpha=al, edgecolors="k", linewidths=0.4, zorder=5)
            if i == 0:
                ax.set_title(f"{(t_utc - pd.Timedelta(hours=1)):%d/%m %H:%M} a {t_utc:%H:%M} UTC", fontsize=22)
            if j == 0:
                ax.set_ylabel(rot, fontsize=22)
        linhas_csv.append(dict(goes_stem=s, bloco=f, max_obs=float(np.nanmax(obs)), cemig="cemig" in campos))
    H.legendas(fig, fig.add_axes([0.07, 0.05, 0.3, 0.01]), fig.add_axes([0.42, 0.05, 0.3, 0.01]))
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color="0.6", markeredgecolor="k", markersize=16, label="avaliação"),
                        Line2D([], [], ls="", marker="^", color="0.6", markeredgecolor="k", markersize=16, label="validação espacial")],
               loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=20, title="estações", title_fontsize=20, frameon=False)
    fig.savefig(SAI / f"{a.saida}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(linhas_csv).to_csv(SAI / f"{a.saida}.csv", index=False)
    print("FIM", sel)


if __name__ == "__main__":
    main()
