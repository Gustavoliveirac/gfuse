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


def _dirs():
    """Lê logs/produtos_dirs.sh (gerado pelo resgate) e exporta os caminhos novos."""
    f = RAIZ / "logs" / "produtos_dirs.sh"
    if f.exists():
        for lin in f.read_text().splitlines():
            if lin.startswith("export ") and "=" in lin:
                k, v = lin[7:].split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class ImergNpz:
    """IMERG Early recortado pelo resgate (npz com mmh por meia-hora, carimbo = início)."""

    def __init__(self, base):
        self.dir = Path(base)
        g = np.load(self.dir / "grade.npz")
        self._lat, self._lon = g["lat"].astype(float), g["lon"].astype(float)

    def rate(self, start_utc):
        t = pd.Timestamp(start_utc).round("30min")
        p = self.dir / f"{t.year}" / f"{t.month:02d}" / f"{t.day:02d}" / f"imerg_early_{t:%Y%m%d%H%M}.npz"
        if not p.exists():
            return None
        try:
            a = np.load(p)["mmh"].astype(np.float32)
        except Exception:
            return None
        return np.where((a < 0) | (a > 500), np.nan, a)

    def window_mm(self, t0_brt, agg_min):
        t0 = pd.Timestamp(t0_brt).round("30min")
        if agg_min not in (30, 60) or abs((pd.Timestamp(t0_brt) - t0).total_seconds()) > 120:
            return None
        t0u = t0 + pd.Timedelta(hours=3)
        rs = [self.rate(t0u - pd.Timedelta(minutes=m)) for m in range(agg_min, 0, -30)]
        if any(r is None for r in rs):
            return None
        return np.nanmean(np.stack(rs), 0) * (agg_min / 60.0)

    def at_points(self, field, lat, lon):
        da, do = float(np.median(np.diff(self._lat))), float(np.median(np.diff(self._lon)))
        ia = np.clip(np.round((np.asarray(lat, float) - self._lat[0]) / da).astype(int), 0, len(self._lat) - 1)
        io = np.clip(np.round((np.asarray(lon, float) - self._lon[0]) / do).astype(int), 0, len(self._lon) - 1)
        return field[ia, io]

    def to_grid64(self, field):
        glat, glon = C.grade_latlon()
        return self.at_points(field, glat.ravel(), glon.ravel()).reshape(K.IMG_SIZE, K.IMG_SIZE)


def fontes(agg):
    """Cada fonte é opcional: a que não abrir fica de fora, sem interromper as outras."""
    _dirs()
    sys.path.insert(1, str(GFUSE2))
    out = {}

    def tenta(nome, f):
        try:
            out[nome] = f()
            print(f"  fonte {nome}: ok")
        except Exception as e:
            print(f"  fonte {nome}: indisponível ({str(e)[:120]})")

    if os.environ.get("IMERG_NPZ_DIR") and (Path(os.environ["IMERG_NPZ_DIR"]) / "grade.npz").exists():
        tenta("imerg_early", lambda: ImergNpz(os.environ["IMERG_NPZ_DIR"]))
    else:
        def _im():
            from sources import ImergEarly
            return ImergEarly(imerg_dir=K.IMERG_DIR, verbose=False)
        tenta("imerg_early", _im)
    if agg == 60:
        def _mg():
            from sources import MergeHourly
            return MergeHourly(merge_dir=os.environ.get("MERGE_DIR"), verbose=False)
        tenta("merge", _mg)

        def _pd():
            from sources_extra import PdirNow
            p = PdirNow(base_dir=os.environ.get("PDIR_DIR"), verbose=False)
            assert p.lat is not None, "sem grade.npz"
            return p
        tenta("pdir_now", _pd)
        for k in ("now", "nrt", "gauge"):
            def _gs(k=k):
                from sources_extra import GsmapHourly
                g = GsmapHourly(k, base_dir=os.environ.get("GSMAP_DIR"), verbose=False)
                assert g.lat is not None, "sem grade.npz"
                return g
            tenta(f"gsmap_{k}", _gs)
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
    if not fs:
        print("  nenhuma fonte disponível")
        return
    cob = [dict(produto=n, instantes=int(d.groupby("goes_stem")[n].apply(lambda s: s.notna().any()).sum()))
           for n in fs]
    comum = d.dropna(subset=list(fs))
    cob.append(dict(produto="todos em comum", instantes=int(comum["goes_stem"].nunique())))
    pd.DataFrame(cob).to_csv(SAI / f"produtos_cobertura_{agg}.csv", index=False)
    print(pd.DataFrame(cob).to_string(index=False))
    pcol = next((c for c in d.columns if c.startswith("p0.2") and c.endswith("_w0")), None)

    def bloco_metricas(x, rotulo):
        lin = []
        for lado in ("alvo", "holdout"):
            y = x[x["lado"] == lado]
            if y.empty:
                continue
            obs = y["obs"].values
            base = dict(subconjunto=rotulo, lado=lado, instantes=int(y["goes_stem"].nunique()))
            lin.append(dict(**base, produto=nome_g, **metricas(
                y["mm_w0"].values, obs, y[pcol].values if pcol else None,
                y["thr_chuva"].values if pcol and "thr_chuva" in y else None)))
            if "s1_rk_mm" in y:
                lin.append(dict(**base, produto="satélite corrigido", **metricas(y["s1_rk_mm"].values, obs)))
            for ref, nome in (("sk", "krigagem simples"), ("idw", "IDW"), ("rrq", "RRQPE")):
                if f"{ref}_w0" in y:
                    lin.append(dict(**base, produto=nome, **metricas(y[f"{ref}_w0"].values, obs)))
            for n in fs:
                if y[n].notna().any():
                    lin.append(dict(**base, produto=n, **metricas(y[n].values, obs)))
        return lin

    lin = []
    for n in fs:                      # cada produto onde ele existe, com o GFUSE nos mesmos pontos
        x = d[d[n].notna()]
        if len(x):
            lin += [r for r in bloco_metricas(x, f"onde há {n}") if r["produto"] in
                    (n, nome_g, "satélite corrigido", "krigagem simples", "IDW", "RRQPE")]
    if len(comum):
        lin += bloco_metricas(comum, "todos em comum")
    t = pd.DataFrame(lin)
    t.to_csv(SAI / f"produtos_{agg}.csv", index=False)
    print(t.round(3).to_string(index=False))
    base_ev = comum if len(comum) else d.dropna(subset=[n for n in fs if n != "imerg_early"] or list(fs), how="all")
    eventos(agg, base_ev, fs, n_ev)


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
