#!/usr/bin/env python
"""
exploratoria.py — análise exploratória dos dados e figuras do método.

Somente leitura de processados/ e resultados/; grava em resultados/analise/.

Etapas (cada uma independente; uma falha não interrompe as outras)
  completude     fração de instantes com leitura por estação e por mês
  correlacao     correlação entre pares de estações x distância (10, 30, 60 min)
                 e ajuste r(d) = c·exp(-d/d0), distância de decorrelação d0
  descartes      registros removidos por regra do QA/QC (coluna motivo) e pelo
                 limite físico, por estação e no total
  limite         painéis de cada registro removido pelo limite físico (vizinhas,
                 Tb 10,3 µm, RRQPE e IMERG Early no instante)
  mapa           mapa do domínio com as estações por fonte e papel (Figura 3.1)
  tolerancia     ilustração da tolerância espacial num caso real (Figura 3.4)
  defasagem      correlação entre o RRQPE em t-k e o pluviômetro em t (k em min)

Uso:  python analise/exploratoria.py [--etapas completude correlacao ...]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

import config3 as K                                     # noqa: E402
import comum as C                                       # noqa: E402
import basemap                                          # noqa: E402

SAI = RAIZ / "resultados" / "analise"
FIG = SAI / "figuras"
GFUSE2 = Path(os.environ.get("GFUSE2_DIR", Path.home() / "Documentos" / "gfuse2"))
MARC = {"CEMADEN": "o", "SUDECAP": "s", "ANA": "^"}
COR_PAPEL = {"holdout": "#d62728", "alvo": "#ff7f0e", "contexto": "#1f77b4", "fora": "#7f7f7f"}


# ------------------------------------------------------------------ base
def base():
    z = np.load(RAIZ / "processados" / "estacoes.npz", allow_pickle=False)
    b = dict(stems=z["stems"].astype(str), t=pd.to_datetime(z["t_brt"]),
             cod=z["cod"].astype(str), tipo=z["tipo"].astype(int),
             fonte=z["fonte"].astype(str), px=z["px"], py=z["py"], R10=z["R10"], RH=z["RH"])
    G = K.IMG_SIZE
    b["dentro"] = (b["px"] >= -0.5) & (b["px"] < G - 0.5) & (b["py"] >= -0.5) & (b["py"] < G - 0.5)
    hold = set()
    hf = RAIZ / "resultados" / "holdout.csv"
    if hf.exists():
        hold = set(pd.read_csv(hf).iloc[:, 0].astype(str))
    alvo = np.array([(zlib.crc32(f"{c}|42".encode()) & 1) == 0 for c in b["cod"]])
    b["papel"] = np.where(~b["dentro"], "fora", np.where(np.isin(b["cod"], list(hold)), "holdout",
                          np.where(alvo, "alvo", "contexto")))
    b["lat"], b["lon"] = C.px_para_latlon(b["px"], b["py"])
    b["i10"] = np.flatnonzero(b["tipo"] == 0)
    b["iH"] = np.flatnonzero(b["tipo"] == 1)
    return b


def janela(b, agg):
    """Acumulados (T, n10) das estações de 10 min na janela agg e máscara de stems válidos."""
    R = b["R10"]
    k = agg // 10
    A = R[:, :, -k:].sum(2) if k > 1 else R[:, :, -1]
    A = np.where(np.isfinite(R[:, :, -k:]).all(2), A, np.nan) if k > 1 else A
    alin = set(C.stems_alinhados(b["stems"], agg))
    ok = np.array([s in alin for s in b["stems"]])
    return A, ok


def caixa(b, folga=0.08):
    glat, glon = C.grade_latlon()
    lat0, lat1 = min(glat.min(), b["lat"].min()), max(glat.max(), b["lat"].max())
    lon0, lon1 = min(glon.min(), b["lon"].min()), max(glon.max(), b["lon"].max())
    return lon0 - folga, lon1 + folga, lat0 - folga, lat1 + folga


def contorno_dominio(ax, recuo_km=0.0, **kw):
    G = K.IMG_SIZE
    r = recuo_km / K.KM_PER_PX
    xs = np.array([-0.5 + r, G - 0.5 - r, G - 0.5 - r, -0.5 + r, -0.5 + r])
    ys = np.array([-0.5 + r, -0.5 + r, G - 0.5 - r, G - 0.5 - r, -0.5 + r])
    la, lo = C.px_para_latlon(xs, ys)
    ax.plot(lo, la, **kw)


# ------------------------------------------------------------------ etapas
def completude(b):
    t = b["t"]
    mes = t.to_period("M").astype(str)
    lin, por_mes = [], []
    for j, i in enumerate(b["i10"]):
        f = np.isfinite(b["R10"][:, j, -1])
        lin.append(dict(cod=b["cod"][i], fonte=b["fonte"][i], tipo="10 min", papel=b["papel"][i],
                        frac=float(f.mean())))
        por_mes.append(pd.Series(f).groupby(mes).mean().rename(b["cod"][i]))
    h_ok = np.array([s[11:13] == "00" or True for s in b["stems"]])
    for j, i in enumerate(b["iH"]):
        f = np.isfinite(b["RH"][:, j])
        lin.append(dict(cod=b["cod"][i], fonte=b["fonte"][i], tipo="horária", papel=b["papel"][i],
                        frac=float(f[h_ok].mean())))
    d = pd.DataFrame(lin).sort_values(["tipo", "fonte", "frac"])
    d.to_csv(SAI / "completude_estacoes.csv", index=False)
    m = pd.concat(por_mes, axis=1).T
    m.to_csv(SAI / "completude_mes.csv")
    print(d.groupby(["tipo", "fonte", "papel"]).frac.describe()[["count", "mean", "min"]].round(3).to_string())
    ordem = d[d.tipo == "10 min"].sort_values(["fonte", "frac"]).cod
    m = m.loc[[c for c in ordem if c in m.index]]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(0, m.shape[1], 3)); ax.set_xticklabels(m.columns[::3], rotation=90, fontsize=7)
    ax.set_yticks([]); ax.set_ylabel("pluviômetros de 10 min (agrupados por fonte)")
    plt.colorbar(im, ax=ax, label="fração de instantes com leitura")
    fig.tight_layout(); fig.savefig(FIG / "completude_mes.png", dpi=150); plt.close(fig)


def correlacao(b):
    i10 = b["i10"]
    px, py = b["px"][i10], b["py"][i10]
    res = []
    for agg in (10, 30, 60):
        A, ok = janela(b, agg)
        A = A[ok]
        lin = []
        n = A.shape[1]
        fin = np.isfinite(A)
        for a in range(n):
            for c in range(a + 1, n):
                m = fin[:, a] & fin[:, c]
                if m.sum() < 100:
                    continue
                x, y = A[m, a], A[m, c]
                mo = (x > 0.2) | (y > 0.2)
                if mo.sum() < 50:
                    continue
                r = np.corrcoef(x[mo], y[mo])[0, 1]
                ix, iy = x > 0.2, y > 0.2
                phi = np.corrcoef(ix, iy)[0, 1] if ix.std() > 0 and iy.std() > 0 else np.nan
                d = float(np.hypot(px[a] - px[c], py[a] - py[c]) * K.KM_PER_PX)
                lin.append(dict(agg=agg, cod_a=b["cod"][i10[a]], cod_b=b["cod"][i10[c]], dist_km=d,
                                n_chuva=int(mo.sum()), r_mm=r, phi_ocorrencia=phi))
        dd = pd.DataFrame(lin)
        dd.to_csv(SAI / f"correlacao_pares_{agg}.csv", index=False)
        for col in ("r_mm", "phi_ocorrencia"):
            v = dd.dropna(subset=[col])
            v = v[v[col] > 0.01]
            if len(v) < 20:
                continue
            best = (np.inf, None, None)
            for d0 in np.arange(1, 150, 0.5):
                e = np.exp(-v.dist_km.values / d0)
                c0 = np.clip((e @ v[col].values) / (e @ e), 0, 1)
                sse = ((v[col].values - c0 * e) ** 2).sum()
                if sse < best[0]:
                    best = (sse, c0, d0)
            res.append(dict(agg=agg, medida=col, c0=round(best[1], 3), d0_km=best[2],
                            d_r05_km=round(best[2] * np.log(best[1] / 0.5), 1) if best[1] > 0.5 else 0.0,
                            n_pares=len(v)))
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, col, nome in zip(axs, ("r_mm", "phi_ocorrencia"), ("correlação em mm", "correlação da ocorrência")):
            ax.scatter(dd.dist_km, dd[col], s=3, alpha=0.25, color="0.4")
            bins = np.arange(0, dd.dist_km.max() + 5, 5)
            g = dd.groupby(pd.cut(dd.dist_km, bins))[col].median()
            ax.plot([iv.mid for iv in g.index], g.values, "o-", color="C3", ms=3, label="mediana a cada 5 km")
            f = [r for r in res if r["agg"] == agg and r["medida"] == col]
            if f:
                xx = np.linspace(0, dd.dist_km.max(), 200)
                ax.plot(xx, f[0]["c0"] * np.exp(-xx / f[0]["d0_km"]), "-", color="C0",
                        label=f"c·exp(-d/d0), d0 = {f[0]['d0_km']:.0f} km")
            ax.set_xlabel("distância entre estações (km)"); ax.set_ylabel(nome); ax.set_ylim(-0.2, 1)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"janela de {agg} min"); fig.tight_layout()
        fig.savefig(FIG / f"correlacao_distancia_{agg}.png", dpi=150); plt.close(fig)
    r = pd.DataFrame(res)
    r.to_csv(SAI / "correlacao_ajuste.csv", index=False)
    print(r.to_string(index=False))


def _ler_serie(fonte, cod):
    try:
        from dados import PASTA_FONTE
    except Exception:
        PASTA_FONTE = {}
    arq = K.EST_DIR / PASTA_FONTE.get(fonte, fonte) / f"{cod}.csv"
    if not arq.exists():
        return None
    s = pd.read_csv(arq, sep=";", usecols=lambda c: c in ("date", "rainfall", "status_final", "motivo"))
    s["date"] = pd.to_datetime(s["date"])
    s["rainfall"] = pd.to_numeric(s["rainfall"], errors="coerce")
    return s


def descartes(b):
    lim10, limh = 50.0, 150.0
    lin, regs, motivos = [], [], {}
    for i in range(len(b["cod"])):
        cod, fonte, tipo = b["cod"][i], b["fonte"][i], b["tipo"][i]
        s = _ler_serie(fonte, cod)
        if s is None:
            continue
        st = pd.to_numeric(s.get("status_final", pd.Series(0, index=s.index)), errors="coerce").fillna(0)
        inval = (st > K.QAQC_STATUS_MAX) & s["rainfall"].notna()
        n_m = {}
        if "motivo" in s.columns:
            for txt in s.loc[inval, "motivo"].fillna("sem_motivo").astype(str):
                for m in re.split(r"[;,|]", txt):
                    m = m.strip() or "sem_motivo"
                    n_m[m] = n_m.get(m, 0) + 1
        lim = lim10 if tipo == 0 else limh
        acima = (~inval) & (s["rainfall"] > lim)
        for _, r in s[acima].iterrows():
            regs.append(dict(cod=cod, fonte=fonte, tipo="10 min" if tipo == 0 else "horária",
                             date=r["date"], mm=r["rainfall"]))
        lin.append(dict(cod=cod, fonte=fonte, tipo="10 min" if tipo == 0 else "horária",
                        n_registros=int(s["rainfall"].notna().sum()), n_invalidos=int(inval.sum()),
                        n_acima_limite=int(acima.sum()), **{f"motivo:{k}": v for k, v in n_m.items()}))
        for k, v in n_m.items():
            motivos[k] = motivos.get(k, 0) + v
    d = pd.DataFrame(lin).fillna(0)
    d.to_csv(SAI / "descartes_por_estacao.csv", index=False)
    tot = pd.DataFrame(sorted(motivos.items(), key=lambda x: -x[1]), columns=["regra", "registros"])
    tot.loc[len(tot)] = ["limite físico (50 mm/10 min, 150 mm/h)", int(d.n_acima_limite.sum())]
    tot["fracao_do_total"] = tot.registros / max(int(d.n_registros.sum()), 1)
    tot.to_csv(SAI / "descartes_por_regra.csv", index=False)
    pd.DataFrame(regs).to_csv(SAI / "registros_acima_limite.csv", index=False)
    print(tot.to_string(index=False))
    print(f"registros acima do limite: {len(regs)}")


def limite(b):
    f = SAI / "registros_acima_limite.csv"
    if not f.exists():
        descartes(b)
    regs = pd.read_csv(f, parse_dates=["date"])
    if regs.empty:
        print("nenhum registro acima do limite")
        return
    imerg = None
    try:
        sys.path.insert(0, str(GFUSE2))
        from sources import ImergEarly
        imerg = ImergEarly(imerg_dir=K.IMERG_DIR, verbose=False)
    except Exception as e:
        print(f"  IMERG indisponível ({e})")
    glat, glon = C.grade_latlon()
    ext = (glon.min(), glon.max(), glat.min(), glat.max())
    lin = []
    for k, r in regs.head(40).iterrows():
        dt = np.abs((b["t"] - r["date"]).total_seconds())
        it = int(np.argmin(dt))
        if dt[it] > 1800:
            lin.append(dict(cod=r["cod"], date=r["date"], mm=r["mm"], obs="sem imagem próxima"))
            continue
        stem = b["stems"][it]
        i = int(np.flatnonzero(b["cod"] == r["cod"])[0])
        agg = 10 if r["tipo"] == "10 min" else 60
        A, _ = janela(b, agg)
        viz = A[it]
        d_km = np.hypot(b["px"][b["i10"]] - b["px"][i], b["py"][b["i10"]] - b["py"][i]) * K.KM_PER_PX
        perto = (d_km < 15) & np.isfinite(viz) & (d_km > 0)
        try:
            arr = np.load(K.GOES_DIR / f"{stem}.npy").astype(np.float32)
            tb = C.rad_para_tb(arr[5], 5)
            rr = arr[K.RRQPE_CH]
        except Exception:
            tb = rr = np.full((K.IMG_SIZE, K.IMG_SIZE), np.nan)
        ix, iy = int(round(np.clip(b["px"][i], 0, 63))), int(round(np.clip(b["py"][i], 0, 63)))
        im_f = None
        if imerg is not None:
            try:
                f = imerg.window_mm(b["t"][it], 30)
                im_f = imerg.to_grid64(f) if f is not None else None
            except Exception:
                im_f = None
        lin.append(dict(cod=r["cod"], fonte=r["fonte"], date=r["date"], mm=r["mm"], stem=stem,
                        n_vizinhas_15km=int(perto.sum()),
                        max_vizinha_15km=float(np.nanmax(viz[perto])) if perto.any() else np.nan,
                        tb_min_pixel_K=float(np.nanmin(tb[max(iy-2,0):iy+3, max(ix-2,0):ix+3])),
                        rrqpe_pixel_mmh=float(rr[iy, ix]),
                        imerg_pixel_mm30=float(im_f[iy, ix]) if im_f is not None else np.nan))
        fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
        lon0, lon1, lat0, lat1 = b["lon"][i] - 0.25, b["lon"][i] + 0.25, b["lat"][i] - 0.25, b["lat"][i] + 0.25
        ax = axs[0]
        basemap.desenha(ax, lon0, lon1, lat0, lat1, z=11)
        ok = np.isfinite(viz)
        sc = ax.scatter(b["lon"][b["i10"]][ok], b["lat"][b["i10"]][ok], c=viz[ok], cmap="turbo",
                        vmin=0, vmax=max(10, np.nanmax(viz) if ok.any() else 10), s=40, edgecolor="k", zorder=3)
        ax.scatter(b["lon"][i], b["lat"][i], marker="X", s=140, color="m", zorder=4)
        plt.colorbar(sc, ax=ax, fraction=0.046, label=f"mm em {agg} min")
        ax.set_title(f"{r['cod']} registrou {r['mm']:.1f} mm", fontsize=9)
        for ax, campo, nome, cm, vmin, vmax in ((axs[1], tb, "Tb 10,3 µm (K)", "gray_r", 190, 300),
                                                (axs[2], rr, "RRQPE (mm/h)", "turbo", 0, 40),
                                                (axs[3], im_f, "IMERG Early (mm/30 min)", "turbo", 0, 20)):
            if campo is None:
                ax.text(0.5, 0.5, "indisponível", ha="center", transform=ax.transAxes)
            else:
                m = ax.imshow(campo, extent=ext, origin="upper", cmap=cm, vmin=vmin, vmax=vmax)
                plt.colorbar(m, ax=ax, fraction=0.046)
            ax.scatter(b["lon"][i], b["lat"][i], marker="X", s=100, color="m")
            ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1); ax.set_title(nome, fontsize=9)
        fig.suptitle(f"{r['date']} (BRT), imagem {stem}", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"limite_{k:02d}_{r['cod']}.png", dpi=130); plt.close(fig)
    pd.DataFrame(lin).to_csv(SAI / "registros_acima_limite_contexto.csv", index=False)
    print(pd.DataFrame(lin).round(2).to_string(index=False))


def mapa(b):
    lon0, lon1, lat0, lat1 = caixa(b)
    fig, ax = plt.subplots(figsize=(9, 9))
    basemap.desenha(ax, lon0, lon1, lat0, lat1, z=10)
    contorno_dominio(ax, color="k", lw=1.5, label="domínio (128 km)")
    contorno_dominio(ax, recuo_km=K.MARGEM_KM, color="k", lw=0.8, ls="--", label=f"margem de {K.MARGEM_KM:g} km")
    for papel, cor in COR_PAPEL.items():
        for fonte, mk in MARC.items():
            for tipo, nome_t in ((0, ""), (1, " horária")):
                m = (b["papel"] == papel) & (b["fonte"] == fonte) & (b["tipo"] == tipo)
                if not m.any():
                    continue
                ax.scatter(b["lon"][m], b["lat"][m], marker=mk if tipo == 0 else "v", s=34 if papel != "holdout" else 60,
                           color=cor, edgecolor="k", lw=0.5, zorder=3, label=f"{fonte}{nome_t}, {papel} ({m.sum()})")
    try:
        meta = pd.read_csv(K.META_CSV, sep=";")
        cl = {c.lower(): c for c in meta.columns}
        la = next(cl[c] for c in cl if c.startswith("lat")); lo = next(cl[c] for c in cl if c.startswith("lon"))
        tp = next((cl[c] for c in cl if c == "tipo"), None)
        if tp:
            mm = meta[meta[tp].astype(str).str.contains("meteor", case=False)]
            ax.scatter(mm[lo], mm[la], marker="*", s=90, color="gold", edgecolor="k", zorder=4,
                       label=f"meteorológica ({len(mm)})")
    except Exception as e:
        print(f"  sem estações meteorológicas no mapa ({e})")
    ax.scatter(K.BH_LON, K.BH_LAT, marker="+", s=120, color="k", zorder=5)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2, framealpha=0.9)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(FIG / "fig3_1_mapa_dominio.png", dpi=200); plt.close(fig)
    pd.DataFrame(dict(cod=b["cod"], fonte=b["fonte"], tipo=np.where(b["tipo"] == 0, "10 min", "horária"),
                      papel=b["papel"], lat=b["lat"], lon=b["lon"])).to_csv(SAI / "estacoes_papel.csv", index=False)


def tolerancia(b):
    A, ok = janela(b, 60)
    i10 = b["i10"]
    alvo = b["papel"][i10] == "alvo"
    cand = np.where(ok[:, None] & alvo[None, :], A, np.nan)
    it = int(np.nanargmax(np.nanmax(cand, 1)))
    obs = A[it]
    ctx = np.isfinite(obs) & np.isin(b["papel"][i10], ["contexto", "fora"])
    vario = C.carregar_vario(60, RAIZ / "processados")
    sk, _, _ = C.krigagem_simples(b["px"][i10][ctx], b["py"][i10][ctx], obs[ctx], vario)
    glat, glon = C.grade_latlon()
    lon0, lon1, lat0, lat1 = glon.min(), glon.max(), glat.min(), glat.max()
    fig, ax = plt.subplots(figsize=(8, 8))
    basemap.desenha(ax, lon0, lon1, lat0, lat1, z=10, alpha=0.5)
    m = ax.imshow(np.ma.masked_less(sk, 1), extent=(lon0, lon1, lat0, lat1), origin="upper", cmap="turbo",
                  vmin=0, vmax=max(30, np.nanmax(obs)), alpha=0.6, zorder=1)
    okk = np.isfinite(obs)
    ax.scatter(b["lon"][i10][okk], b["lat"][i10][okk], c=obs[okk], cmap="turbo", vmin=0,
               vmax=max(30, np.nanmax(obs)), s=35, edgecolor="k", zorder=3)
    raio_lat = 4 / 111.2
    for j in np.flatnonzero(okk & (obs >= 20)):
        ax.add_patch(plt.Circle((b["lon"][i10][j], b["lat"][i10][j]), raio_lat / np.cos(np.radians(K.BH_LAT)),
                                fill=False, color="m", lw=1.2, zorder=4))
    plt.colorbar(m, ax=ax, fraction=0.046, label="mm em 60 min")
    ax.set_title(f"{b['t'][it]} (BRT), círculos de 4 km nas estações com ≥ 20 mm", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig3_4_tolerancia.png", dpi=200); plt.close(fig)
    print(f"caso: {b['stems'][it]}, máximo {np.nanmax(obs):.1f} mm")


def defasagem(b, n_amostra=4000, ks=range(-3, 7)):
    A, _ = janela(b, 10)
    i10 = b["i10"]
    dentro = b["dentro"][i10]
    ix = np.clip(np.round(b["px"][i10]).astype(int), 0, 63)
    iy = np.clip(np.round(b["py"][i10]).astype(int), 0, 63)
    chuva = np.nanmax(np.where(dentro[None, :], A, np.nan), 1) > 0.2
    rng = np.random.default_rng(0)
    cand = np.flatnonzero(chuva)
    sel = rng.choice(cand, min(n_amostra, len(cand)), replace=False)
    t = b["t"].values
    cache = {}

    def rr(k):
        if k not in cache:
            try:
                cache[k] = np.load(K.GOES_DIR / f"{b['stems'][k]}.npy", mmap_mode="r")[K.RRQPE_CH][iy, ix] / 6.0
            except Exception:
                cache[k] = None
        return cache[k]

    acc = {k: ([], []) for k in ks}
    for i in sel:
        g = A[i]
        for k in ks:
            j = i - k
            if j < 0 or j >= len(t):
                continue
            if abs((t[i] - t[j]) / np.timedelta64(1, "m") - 10 * k) > 3:
                continue
            v = rr(j)
            if v is None:
                continue
            m = dentro & np.isfinite(g) & np.isfinite(v) & ((g > 0) | (v > 0))
            acc[k][0].append(g[m]); acc[k][1].append(np.asarray(v)[m])
        if len(cache) > 20000:
            cache.clear()
    lin = []
    for k, (gg, vv) in acc.items():
        if not gg:
            continue
        g, v = np.concatenate(gg), np.concatenate(vv)
        lin.append(dict(rrqpe_antes_min=10 * k, r=np.corrcoef(g, v)[0, 1], n=len(g)))
    d = pd.DataFrame(lin)
    d.to_csv(SAI / "rrqpe_defasagem_curva.csv", index=False)
    print(d.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.rrqpe_antes_min, d.r, "o-")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("RRQPE tomado k minutos antes do pluviômetro"); ax.set_ylabel("correlação (10 min)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(FIG / "rrqpe_defasagem_curva.png", dpi=150); plt.close(fig)


ETAPAS = dict(completude=completude, correlacao=correlacao, descartes=descartes, limite=limite,
              mapa=mapa, tolerancia=tolerancia, defasagem=defasagem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapas", nargs="+", default=list(ETAPAS))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    b = base()
    print(f"base: {len(b['stems']):,} instantes, {len(b['cod'])} estações")
    for e in a.etapas:
        print(f"\n===== {e} =====", flush=True)
        try:
            ETAPAS[e](b)
        except Exception:
            print(f"ERRO em {e}:")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
