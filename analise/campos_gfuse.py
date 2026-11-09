#!/usr/bin/env python
"""
campos_gfuse.py — roda os modelos treinados nos piores eventos de 60 min e
desenha os campos no padrão do hidromet, ao lado do RRQPE e do IMERG.

Para cada evento usa os checkpoints do bloco em que o instante está no teste
(nenhum modelo viu o evento). Painéis, todos em mm na hora e com as estações
de todos os papéis por cima:
  krigagem simples das estações de entrada, estimador por satélite (s1),
  satélite corrigido (s1_rk), GFUSE (s2rk), fusão em um estágio (base),
  RRQPE e IMERG Early.
Nos produtos das redes, a chuva aparece só onde a probabilidade passa do
limiar escolhido no treinamento (o mesmo produto das métricas).

Eventos: os de resultados/analise/eventos_produtos_60.csv, se existir (os
mesmos da comparação com os produtos); senão, os N maiores do teste.
Saída: resultados/analise/figuras/fig4_5b_gfuse_{k}.png,
       resultados/analise/mapas_csv/fig4_5b_gfuse_{k}.csv (valores no pixel
       de cada estação) e resultados/analise/campos_gfuse/{stem}.npz.

Uso:  source logs/produtos_dirs.sh; python analise/campos_gfuse.py [--n 4] [--cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import torch                                            # noqa: E402

import comum as C                                       # noqa: E402
import modelo as Mo                                     # noqa: E402
import mapas_hidromet as H                              # noqa: E402
from config3 import ARM_FLAGS, EDGES, IMG_SIZE, PROC, RES  # noqa: E402
from estacoes import Base                               # noqa: E402
from mapas_eventos import carregar_rede, escolher_eventos, thr_rk  # noqa: E402

SAI = RES / "analise"
FIG, CSV, NPZ = SAI / "figuras", SAI / "mapas_csv", SAI / "campos_gfuse"
AGG = 60
BRACOS = ["s1", "s2rk", "base"]
TITULO = {"sk": "Krigagem simples das estações", "s1": "Estimador por satélite", "s1_rk": "Satélite corrigido",
          "s2rk": "GFUSE", "base": "Fusão em um estágio", "rrqpe": "GOES-16 RRQPE", "imerg_early": "IMERG Early"}


def mascara(mm, p, thr):
    return np.where(np.asarray(p) >= thr, np.asarray(mm) / np.maximum(p, 1e-6), 0.0)


def fold_de(stem, folds):
    for f, d in enumerate(folds):
        if stem in set(d["val"]):
            return f
    return None


@torch.no_grad()
def campos_modelos(stem, fold, base, holdout, vario, mov, device):
    redes = {a: carregar_rede(a, AGG, fold, device) for a in BRACOS}
    redes = {a: r for a, r in redes.items() if r is not None}
    ds_ref = Mo.DatasetGfuse([], base, AGG, ARM_FLAGS["base"], holdout, vario, treino=False, movimento=mov)
    inf = ds_ref.info(stem)
    w = inf["w"]
    cx = np.r_[inf["ctx"], inf["ext"]]
    out = {"sk": C.krigagem_simples(base.px[cx], base.py[cx], w[cx], vario)[0],
           "rrqpe": ds_ref.amostra(stem)["rrqpe_mm"].numpy()}
    probs = {}
    for a, (net, s1, thr) in redes.items():
        ds = Mo.DatasetGfuse([], base, AGG, ARM_FLAGS[a], holdout, vario, treino=False, movimento=mov)
        x = ds.amostra(stem)
        mm, pt = Mo.decodificar(Mo.prever(net, x["cubo"][None].to(device), x["estat"][None].to(device),
                                          ARM_FLAGS[a], AGG, s1, ctx=x["ctx"][None], n_ctx=x["n_ctx"][None]), AGG)
        p0 = pt[EDGES[AGG][0]][0]
        out[a] = mascara(mm[0], p0, thr); probs[a] = np.asarray(p0)
        if ARM_FLAGS[a]["s1"]:
            lam, _ = C.pesos_krigagem(base.px[cx], base.py[cx], vario)
            mm2 = C.krigagem_residuo(mm[0], base.px[cx], base.py[cx], w[cx], vario, 0.0, None, lam)
            p2 = C.krigagem_residuo(p0, base.px[cx], base.py[cx], (w[cx] > EDGES[AGG][0]).astype(float),
                                    vario, 0.0, 1.0, lam)
            out[f"{a}_rk"] = mascara(mm2, p2, thr_rk(a, AGG, fold, thr)); probs[f"{a}_rk"] = np.asarray(p2)
    return out, probs, inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    for p in (FIG, CSV, NPZ):
        p.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    base = Base()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    mov = Mo.Movimento()
    vario = C.carregar_vario(AGG, PROC)
    stems_al = C.stems_alinhados(list(base.stems), AGG)
    folds = C.folds_temporais(stems_al)
    f_ev = SAI / "eventos_produtos_60.csv"
    if f_ev.exists():
        stems = pd.read_csv(f_ev)["goes_stem"].astype(str).tolist()[:a.n]
    else:
        stems = [e[1] for e in escolher_eventos(base, AGG, list(range(len(folds))), a.n)]
    imerg = None
    try:
        import produtos
        imerg = produtos.fontes(AGG).get("imerg_early")
    except Exception as e:
        print(f"  IMERG indisponível ({e})")
    glat, glon = C.grade_latlon()
    lat_e, lon_e = C.px_para_latlon(base.px, base.py)
    tabela = []
    for k, stem in enumerate(stems):
        fold = fold_de(stem, folds)
        if fold is None:
            print(f"  {stem}: fora dos blocos de teste, pulado"); continue
        campos, probs, inf = campos_modelos(stem, fold, base, holdout, vario, mov, device)
        np.savez_compressed(NPZ / f"{stem}.npz", **{f"mm_{n}": v for n, v in campos.items()},
                            **{f"p_{n}": v for n, v in probs.items()})
        t_brt = C.stem_para_brt(stem)
        if imerg is not None:
            try:
                c = imerg.window_mm(t_brt, AGG)
                if c is not None:
                    la, lo = np.meshgrid(imerg._lat, imerg._lon, indexing="ij")
                    campos["imerg_early"] = (np.asarray(c, float), la, lo)
            except Exception:
                pass
        ordem = [n for n in ("sk", "s1", "s1_rk", "s2rk", "base", "rrqpe", "imerg_early") if n in campos]
        ncol = 4; nlin = int(np.ceil(len(ordem) / ncol))
        x0, x1, y0, y1 = H.caixa(glat, glon, folga_km=3)
        fig = plt.figure(figsize=(4.2 * ncol, 4.4 * nlin + 0.8))
        w = inf["w"]
        t_utc = t_brt + pd.Timedelta(hours=3)
        rot = f"{(t_utc - pd.Timedelta(hours=1)):%d/%m/%Y %H:%M} a {t_utc:%H:%M} UTC"
        linhas = dict(cod=base.cod, lat=lat_e, lon=lon_e, obs_mm_1h=w)
        for j, nome in enumerate(ordem):
            ax = fig.add_axes([0.02 + (j % ncol) / ncol, 0.12 + (nlin - 1 - j // ncol) / nlin * 0.86,
                               0.96 / ncol - 0.02, 0.86 / nlin - 0.04])
            H.mapa_base(ax, x0, x1, y0, y1)
            v = campos[nome]
            if isinstance(v, tuple):
                c, la, lo = v; tipo = "imerg"
            else:
                c, la, lo = v, glat, glon; tipo = "goes" if nome == "rrqpe" else "modelo"
            H.desenha_campo(ax, c, la, lo, tipo=tipo)
            H.desenha_estacoes(ax, lat_e, lon_e, w)
            ax.set_title(f"{TITULO[nome]}, {rot}", fontsize=8)
            try:
                if tipo == "imerg":
                    vals = np.array([float(c[np.unravel_index(np.argmin((la - p) ** 2 + (lo - q) ** 2), la.shape)])
                                     for p, q in zip(lat_e, lon_e)])
                else:
                    ix = np.clip(np.round(base.px).astype(int), 0, IMG_SIZE - 1)
                    iy = np.clip(np.round(base.py).astype(int), 0, IMG_SIZE - 1)
                    dentro = (base.px >= -0.5) & (base.px < IMG_SIZE - 0.5) & (base.py >= -0.5) & (base.py < IMG_SIZE - 0.5)
                    vals = np.where(dentro, np.asarray(c)[iy, ix], np.nan)
            except Exception:
                vals = np.full(len(base.cod), np.nan)
            linhas[f"{nome}_mm"] = vals
        H.legendas(fig, fig.add_axes([0.08, 0.05, 0.38, 0.018]), fig.add_axes([0.56, 0.05, 0.38, 0.018]))
        arq = f"fig4_5b_gfuse_{k}"
        fig.savefig(FIG / f"{arq}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
        pd.DataFrame(linhas).to_csv(CSV / f"{arq}.csv", index=False)
        tabela.append(dict(ordem=k, goes_stem=stem, hora_utc=t_utc, bloco=fold, max_obs_mm=float(np.nanmax(w)),
                           paineis=" ".join(ordem), arquivo=f"{arq}.png"))
        print(f"  {arq}.png  ({stem}, bloco {fold})", flush=True)
    pd.DataFrame(tabela).to_csv(SAI / "campos_gfuse_eventos.csv", index=False)
    print("FIM")


if __name__ == "__main__":
    main()
