#!/usr/bin/env python
"""
produtos.py — compara o GFUSE e as referências com os produtos externos
(IMERG Early, MERGE horário, PDIR-Now, GSMaP NOW, NRT e Gauge) apenas nos
instantes em que todos os produtos da janela estão disponíveis, e desenha os
piores eventos em cada fonte.

Lê os pares do GFUSE (resultados/pares/m{agg}/s2rk_f*.parquet; se faltar, usa
base) e do satélite corrigido (s1_rk_f*), e os produtos pelos leitores do
gfuse2 (~/Documentos/gfuse2/sources*.py), amostrados no ponto de cada estação.
Janelas: 30 min (só IMERG) e 60 min (todos). Os valores amostrados ficam em
cache em resultados/analise/produtos_pontos_{agg}.parquet; rodar de novo só
completa o que falta.

Saída em resultados/analise/
  produtos_{agg}.csv           métricas por produto no subconjunto comum
  produtos_cobertura_{agg}.csv instantes disponíveis por produto e em comum
  figuras/eventos_produtos_{agg}_{k}.png  painéis dos piores eventos

Uso:  python analise/produtos.py [--aggs 60 30] [--eventos 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
GFUSE2 = Path(os.environ.get("GFUSE2_DIR", Path.home() / "Documentos" / "gfuse2"))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import BoundaryNorm             # noqa: E402

import config3 as K                                     # noqa: E402
import comum as C                                       # noqa: E402

SAI = RAIZ / "resultados" / "analise"
FIG = SAI / "figuras"
NIVEIS = [0, 0.2, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80]
CMAP = plt.get_cmap("turbo", len(NIVEIS) - 1)
NORM = BoundaryNorm(NIVEIS, CMAP.N)


def fontes(agg):
    sys.path.insert(1, str(GFUSE2))
    from sources import ImergEarly, MergeHourly
    out = {"imerg_early": ImergEarly(imerg_dir=K.IMERG_DIR, verbose=False)}
    if agg == 60:
        from sources_extra import GsmapHourly, PdirNow
        out["merge"] = MergeHourly(verbose=False)
        out["pdir_now"] = PdirNow(verbose=False)
        for k in ("now", "nrt", "gauge"):
            out[f"gsmap_{k}"] = GsmapHourly(k, verbose=False)
    return out


def pares(agg, braco):
    fs = sorted((RAIZ / "resultados" / "pares" / f"m{agg}").glob(f"{braco}_f[0-9].parquet"))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d["lado"].isin(["alvo", "holdout"])]
    if "margem" in d:
        d = d[~d["margem"].astype(bool)]
    return d


def amostra(agg, d, fs):
    cache = SAI / f"produtos_pontos_{agg}.parquet"
    pts = d[["goes_stem", "px", "py"]].drop_duplicates().reset_index(drop=True)
    if cache.exists():
        old = pd.read_parquet(cache)
        pts = pts.merge(old, on=["goes_stem", "px", "py"], how="left")
    for nome in fs:
        if nome not in pts:
            pts[nome] = np.nan
    stems = pts["goes_stem"].unique()
    print(f"  {len(stems):,} instantes a amostrar")
    lat, lon = C.px_para_latlon(pts["px"].values, pts["py"].values)
    pts["lat"], pts["lon"] = lat, lon
    grupos = pts.groupby("goes_stem").indices
    for k, s in enumerate(stems):
        idx = grupos[s]
        t0 = C.stem_para_brt(s)
        for nome, f in fs.items():
            if pts.loc[idx, nome].notna().all():
                continue
            try:
                campo = f.window_mm(t0, agg)
            except Exception:
                campo = None
            if campo is None:
                continue
            pts.loc[idx, nome] = f.at_points(campo, pts.loc[idx, "lat"].values, pts.loc[idx, "lon"].values)
        if k % 500 == 0:
            print(f"    ... {k}/{len(stems)}", flush=True)
            pts.drop(columns=["lat", "lon"]).to_parquet(cache, index=False)
    pts = pts.drop(columns=["lat", "lon"])
    pts.to_parquet(cache, index=False)
    return pts


def metricas(pred, obs, prob=None, thr=None):
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    occ = (prob[ok] >= thr[ok]) if prob is not None else pred > 0.2
    o = obs > 0.2
    h, m, f = (occ & o).sum(), (~occ & o).sum(), (occ & ~o).sum()
    q = C.metricas_quantidade(pred, obs)
    r = q.get("r", np.nan) if isinstance(q, dict) else np.nan
    return dict(n_pares=int(ok.sum()), r=r, razao_volume=pred.sum() / max(obs.sum(), 1e-9),
                pod=h / max(h + m, 1), far=f / max(h + f, 1), csi=h / max(h + m + f, 1))


def avaliar(agg, n_ev):
    d = pares(agg, "s2rk")
    nome_g = "GFUSE (dois estágios)"
    if d is None:
        d = pares(agg, "base"); nome_g = "fusão em um estágio"
    if d is None:
        print(f"  sem pares para m{agg}")
        return
    rk = pares(agg, "s1_rk")
    fs = fontes(agg)
    pts = amostra(agg, d, fs)
    d = d.merge(pts, on=["goes_stem", "px", "py"], how="left")
    if rk is not None:
        d = d.merge(rk[["goes_stem", "px", "py", "mm_w0"]].rename(columns={"mm_w0": "s1_rk_mm"}),
                    on=["goes_stem", "px", "py"], how="left")
    cob = [dict(produto=n, instantes=int(d.groupby("goes_stem")[n].apply(lambda s: s.notna().any()).sum()))
           for n in fs]
    comum = d.dropna(subset=list(fs))
    cob.append(dict(produto="todos em comum", instantes=int(comum["goes_stem"].nunique())))
    pd.DataFrame(cob).to_csv(SAI / f"produtos_cobertura_{agg}.csv", index=False)
    print(pd.DataFrame(cob).to_string(index=False))
    if comum.empty:
        print("  nenhum instante com todos os produtos")
        return
    pcol = next((c for c in comum.columns if c.startswith("p0.2") and c.endswith("_w0")), None)
    lin = []
    for lado in ("alvo", "holdout"):
        x = comum[comum["lado"] == lado]
        obs = x["obs"].values
        lin.append(dict(lado=lado, produto=nome_g, **metricas(
            x["mm_w0"].values, obs, x[pcol].values if pcol else None,
            x["thr_chuva"].values if pcol and "thr_chuva" in x else None)))
        if "s1_rk_mm" in x:
            lin.append(dict(lado=lado, produto="satélite corrigido", **metricas(x["s1_rk_mm"].values, obs)))
        for ref, nome in (("sk", "krigagem simples"), ("idw", "IDW"), ("rrq", "RRQPE")):
            if f"{ref}_w0" in x:
                lin.append(dict(lado=lado, produto=nome, **metricas(x[f"{ref}_w0"].values, obs)))
        for n in fs:
            lin.append(dict(lado=lado, produto=n, **metricas(x[n].values, obs)))
    t = pd.DataFrame(lin)
    t["instantes_comuns"] = comum["goes_stem"].nunique()
    t.to_csv(SAI / f"produtos_{agg}.csv", index=False)
    print(t.round(3).to_string(index=False))
    eventos(agg, comum, fs, n_ev)


def eventos(agg, comum, fs, n_ev):
    z = np.load(RAIZ / "processados" / "estacoes.npz", allow_pickle=False)
    stems = z["stems"].astype(str)
    idx = {s: i for i, s in enumerate(stems)}
    t_all = pd.to_datetime(z["t_brt"])
    px, py, tipo = z["px"], z["py"], z["tipo"]
    i10, iH = np.flatnonzero(tipo == 0), np.flatnonzero(tipo == 1)
    R10, RH = z["R10"], z["RH"]
    mx = comum.groupby("goes_stem")["obs"].max().sort_values(ascending=False)
    esc = []
    for s in mx.index:
        t = C.stem_para_brt(s)
        if all(abs((t - C.stem_para_brt(e)).total_seconds()) > 6 * 3600 for e in esc):
            esc.append(s)
        if len(esc) >= n_ev:
            break
    n_col = 6
    passo = pd.Timedelta(minutes=agg)
    for k, s in enumerate(esc):
        t_fim = C.stem_para_brt(s)
        tempos = [t_fim - passo * (n_col - 1 - j) for j in range(n_col)]
        linhas = ["estações", "RRQPE"] + list(fs)
        fig, axs = plt.subplots(len(linhas), n_col, figsize=(2.0 * n_col, 2.0 * len(linhas)))
        for j, tt in enumerate(tempos):
            dt = np.abs((t_all - tt).total_seconds())
            it = int(np.argmin(dt)) if dt.min() < 300 else None
            obs = np.full(len(px), np.nan)
            if it is not None:
                kk = agg // 10
                v = R10[it, :, -kk:]
                obs[i10] = np.where(np.isfinite(v).all(1), v.sum(1), np.nan)
                if agg == 60:
                    obs[iH] = RH[it]
            ok = np.isfinite(obs)
            campos = {}
            if ok.sum() >= 3:
                try:
                    vario = C.carregar_vario(agg, RAIZ / "processados")
                    campos["estações"] = C.krigagem_simples(px[ok], py[ok], obs[ok], vario)[0]
                except Exception:
                    pass
            if it is not None:
                try:
                    rr = []
                    for q in range(agg // 10):
                        sq = stems[it - q]
                        rr.append(np.load(K.GOES_DIR / f"{sq}.npy", mmap_mode="r")[K.RRQPE_CH] / 6.0)
                    campos["RRQPE"] = np.sum(rr, 0)
                except Exception:
                    pass
            for n, f in fs.items():
                try:
                    c = f.window_mm(tt, agg)
                    campos[n] = f.to_grid64(c) if c is not None else None
                except Exception:
                    campos[n] = None
            for i, nome in enumerate(linhas):
                ax = axs[i, j]
                c = campos.get(nome)
                if c is not None:
                    ax.imshow(c, cmap=CMAP, norm=NORM, origin="upper", interpolation="nearest")
                else:
                    ax.set_facecolor("0.85")
                ax.scatter(px[ok], py[ok], c=obs[ok], cmap=CMAP, norm=NORM, s=7, edgecolor="k", lw=0.3)
                ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(tt.strftime("%H:%M"), fontsize=8)
                if j == 0:
                    ax.set_ylabel(nome, fontsize=8)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
        fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, label=f"mm / {agg} min", ticks=NIVEIS)
        fig.suptitle(f"Evento {t_fim:%Y-%m-%d} (BRT), janelas consecutivas de {agg} min; "
                     f"pontos = pluviômetros", fontsize=9)
        fig.savefig(FIG / f"eventos_produtos_{agg}_{k}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(dict(goes_stem=esc, max_obs_mm=[mx[s] for s in esc])).to_csv(
        SAI / f"eventos_produtos_{agg}.csv", index=False)
    print(f"  eventos desenhados: {esc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggs", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--eventos", type=int, default=4)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for agg in a.aggs:
        print(f"\n===== janela de {agg} min =====", flush=True)
        try:
            avaliar(agg, a.eventos)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
