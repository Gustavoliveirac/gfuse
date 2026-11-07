#!/usr/bin/env python
"""
resumo.py — consolida os pares de todos os braços/folds em CSVs e num
resumo em texto, e compacta tudo em resultados/envio_gfuse3.zip (o único
arquivo a enviar de volta). Sem métrica composta: só r, POD, FAR e CSI.

Por janela (m10/m30/m60):
  metricas_{agg}.csv        por produto: quantidade (r, r_log, R2, viés,
                            razão de volume) e, para cada limiar, POD/FAR/CSI
                            com decisão P >= 0,5 (ou campo > limiar nos
                            determinísticos) no pixel e com tolerância 3x3,
                            CSI com o melhor limiar de probabilidade, núcleos
                            com tolerância de 4 km e a ocorrência sem borda
                            (diagnóstico, com a fração de estações excluídas)
  escada_{agg}.csv          POD/FAR/CSI por produto x limiar x tolerância
  nucleos_{agg}.csv         núcleos a 4, 8 e 12 km
  metricas_fold_{agg}.csv   r e CSI por fold
  holdout_{agg}.csv         métricas nas estações nunca vistas
  calibracao_{agg}.csv, brier_{agg}.csv, skill_dist_{agg}.csv,
  ternario_{agg}.csv, intensidade_dist_{agg}.csv
  coerencia_10.csv          diferença média entre P(t) e P(t-10) advectado,
                            células por quadro e fração com antecessor
  resumo_{agg}.txt
Produtos: os braços (valor esperado; decisão P >= 0,5), <braço>_det (mapa
entregue: chuva condicional onde P >= 0,5, zero no resto), <braço>_filtro
(m10) e as referências idw, ok
(krigagem ordinária), sk (krigagem simples em mm), rrq (RRQPE) e pers
(persistência da própria estação, só como referência). Estações a menos de
MARGEM_KM da borda ficam fora de todas as métricas.
  area_{agg}.csv            área prevista com chuva x fração de estações com
                            chuva, stems sem chuva, borda x interior
"""
from __future__ import annotations

import argparse
import gc
import re
import zipfile

import numpy as np
import pandas as pd

import comum as C
import verif as V
from config3 import (AGGS, C_DUVIDA, CUSTO_FA, CUSTO_MISS, EDGES, KM_PER_PX,
                     NUCLEOS, PARES_DIR, PROC, RAIOS_KM, RES, TOL_WINS, VERSAO)

DIST_BINS = [0, 2, 5, 10, 20, 40, np.inf]
PROB_BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
SD_BINS = [-np.inf, -8, -4, -2, 0, 2, 4, 8, np.inf]
BASELINES = ("idw", "ok", "sk", "rrq")
REFERENCIAS = ("idw", "ok", "sk", "rrq", "pers")
RAIO_REF_KM = 4


# ------------------------------------------------------------------
# carga
# ------------------------------------------------------------------
def _arquivos_braco(pasta, arm):
    padrao = re.compile(rf"^{re.escape(arm)}_f\d+\.parquet$")
    return sorted(f for f in pasta.glob("*.parquet") if padrao.match(f.name))


def bracos_disponiveis(agg):
    pasta = PARES_DIR / f"m{agg}"
    if not pasta.exists():
        return []
    nomes = {f.stem.rsplit("_f", 1)[0] for f in pasta.glob("*.parquet")
             if f.name.count(".") == 1}
    # base primeiro (é dela que saem as referências)
    return sorted(nomes, key=lambda a: (a != "base", a))


def carregar_braco(agg, arm):
    """Pares de UM braço (todos os folds), em float32, sem a faixa da borda,
    com as colunas de vizinhança. Carregar um braço por vez mantém a memória
    sob controle com a base completa (~1 milhão de pares por fold no m10)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    partes = []
    for f in _arquivos_braco(PARES_DIR / f"m{agg}", arm):
        t = pq.read_table(f)
        # float64 -> float32 ainda no pyarrow, antes de virar DataFrame
        esquema = pa.schema([pa.field(c.name, pa.float32()) if c.type == pa.float64()
                             else c for c in t.schema])
        d = t.cast(esquema).to_pandas()
        del t
        if "versao" not in d.columns or d["versao"].iloc[0] != VERSAO:
            print(f"  [resumo] ignorando {f.name} (versão antiga)")
            continue
        if "margem" in d.columns:                 # borda fora das métricas
            d = d[~d["margem"].astype(bool)]
        d = d.drop(columns=[c for c in ("versao", "braco", "agg", "margem")
                            if c in d.columns])
        for c in d.columns:
            if d[c].dtype == np.float64:
                d[c] = d[c].astype(np.float32)
        d["goes_stem"] = d["goes_stem"].astype("category")
        partes.append(d.reset_index(drop=True))
    if not partes:
        return None
    d = pd.concat(partes, ignore_index=True)
    del partes
    d["goes_stem"] = d["goes_stem"].astype(str).astype("category")
    return adicionar_vizinhanca(d)


def carregar_aux(agg):
    """Tabelas pequenas por stem (.areas, .coer, .area) de todos os braços."""
    pasta = PARES_DIR / f"m{agg}"
    areas, coer, area = {}, {}, {}
    if not pasta.exists():
        return areas, coer, area
    for f in sorted(pasta.glob("*.parquet")):
        partes = f.name.split(".")
        if len(partes) > 2:
            arm = partes[0].rsplit("_f", 1)[0]
            alvo_d = {"areas": areas, "coer": coer, "area": area}.get(partes[1])
            if alvo_d is not None:
                alvo_d.setdefault(arm, []).append(pd.read_parquet(f))
    junta = lambda dd: {a: pd.concat(v, ignore_index=True) for a, v in dd.items()}
    return junta(areas), junta(coer), junta(area)


def adicionar_vizinhanca(d):
    """Máximo observado entre estações verificadoras (mesmo stem e lado) em
    cada raio dos núcleos -> colunas ov_r{px}."""
    raios = sorted({int(round(r / KM_PER_PX)) for r in RAIOS_KM})
    out = {r: np.full(len(d), np.nan, np.float32) for r in raios}
    for _, idx in d.groupby(["goes_stem", "lado"], observed=True).indices.items():
        px = d["px"].values[idx]; py = d["py"].values[idx]
        ob = d["obs"].values[idx]
        for r in raios:
            out[r][idx] = V.max_obs_vizinhas(px, py, ob, r)
    for r in raios:
        d[f"ov_r{r}"] = out[r]
    return d


# ------------------------------------------------------------------
# decisões e métricas
# ------------------------------------------------------------------
def _col_campo(produto):
    """Prefixo da coluna de um produto determinístico (referências e o
    <braço>_det, chuva condicional onde P >= 0,5), ou None."""
    if produto in BASELINES:
        return produto
    if produto.endswith("_det"):
        return "det"
    return None


def decisao(df, produto, tau, win):
    """(pred, obs, válido) com P >= 0,5 (modelos) ou campo > tau."""
    obs = df["obs"].values > tau
    if _col_campo(produto):
        col = f"{_col_campo(produto)}_w{win}"
        if col not in df.columns:
            return None
        v = df[col].values
        return v > tau, obs, np.isfinite(v)
    if produto == "pers":
        if win != 0:
            return None
        v = df["pers"].values
        return v > tau, obs, np.isfinite(v)
    col = f"p{tau:g}_w{win}"
    if col not in df.columns:
        return None
    v = df[col].values
    # ocorrência: limiar do braço escolhido no treino; demais limiares: 0,5
    thr = (df["thr_chuva"].values if tau == EDGES[10][0] and "thr_chuva" in df.columns
           else 0.5)
    return v >= thr, obs, np.isfinite(v)


def campo_mm(df, produto, win=0):
    if produto == "pers":
        return df["pers"].values
    nome = _col_campo(produto) or "mm"
    return df[f"{nome}_w{win}"].values


def _prob(df, produto, tau, win, fn="max"):
    if _col_campo(produto):
        c = _col_campo(produto)
        col = f"{c}_min1" if fn == "min" else f"{c}_w{win}"
        if col not in df.columns:
            return None
        v = df[col].values
        return (v > tau if tau == EDGES[10][0] else v >= tau).astype(float)
    col = f"pmin{tau:g}_w1" if fn == "min" else f"p{tau:g}_w{win}"
    return df[col].values if col in df.columns else None


def nucleo_cont(df, produto, tau, raio_km):
    r_px = int(round(raio_km / KM_PER_PX))
    pv = _prob(df, produto, tau, r_px)
    if pv is None or produto == "pers":
        return None
    return V.cont_nucleo(pv, df["obs"].values, df[f"ov_r{r_px}"].values, tau)


def escada(df, produto, agg):
    linhas = []
    for tau in EDGES[agg]:
        for win in TOL_WINS:
            dc = decisao(df, produto, tau, win)
            if dc is None:
                continue
            pred, obs, m = dc
            row = dict(produto=produto, tau=tau, win=win,
                       **C.contingencia(pred[m], obs[m]))
            p = _prob(df, produto, tau, win)
            if p is not None and not _col_campo(produto) and produto != "pers":
                o = obs[m]; t = V.GRADE_THR[:, None]; pr = p[m][None] >= t
                h = (pr & o).sum(1); mi = (~pr & o).sum(1); f = (pr & ~o).sum(1)
                row["CSI_best"], row["thr_best"], _ = V.csi_melhor(
                    h.astype(float), mi.astype(float), f.astype(float))
            linhas.append(row)
    return pd.DataFrame(linhas)


def larga(df, produto, agg):
    out = dict(produto=produto, n_pares=len(df))
    q = C.metricas_quantidade(campo_mm(df, produto), df["obs"].values)
    out.update({k: q[k] for k in ("r", "R2mm", "bias", "ratio", "MAE")})
    for tau in EDGES[agg]:
        for win in (0, 1):
            dc = decisao(df, produto, tau, win)
            if dc is None:
                continue
            pred, obs, m = dc
            c = C.contingencia(pred[m], obs[m])
            out[f"csi{tau:g}_w{win}"] = c["CSI"]
            if win == 0:
                out[f"pod{tau:g}"] = c["POD"]; out[f"far{tau:g}"] = c["FAR"]
    for tau in NUCLEOS[agg]:
        c = nucleo_cont(df, produto, tau, RAIO_REF_KM)
        if c is None:
            continue
        h, m, f = c
        csi, thr, n_ev = V.csi_melhor(h, m, f)
        k = int(np.argmin(np.abs(V.GRADE_THR - (thr if np.isfinite(thr) else 0.5))))
        out[f"nuc{tau:g}_csi"] = csi
        out[f"nuc{tau:g}_pod"] = h[k] / (h[k] + m[k]) if h[k] + m[k] else np.nan
        out[f"nuc{tau:g}_far"] = f[k] / (h[k] + f[k]) if h[k] + f[k] else np.nan
        out[f"nuc{tau:g}_thr"] = thr
        out[f"nuc{tau:g}_n_ev"] = n_ev
    pmax = _prob(df, produto, EDGES[agg][0], 1)
    pmin = _prob(df, produto, EDGES[agg][0], 1, "min")
    if pmax is not None and pmin is not None and produto != "pers":
        h, m, f = V.cont_ocorrencia(pmax, pmin, df["obs"].values)
        csi, thr, _ = V.csi_melhor(h, m, f)
        k = int(np.argmin(np.abs(V.GRADE_THR - (thr if np.isfinite(thr) else 0.5))))
        out["csi_sem_borda"] = csi
        t = V.GRADE_THR[k]
        out["frac_excluida_borda"] = float(((pmin < t) & (pmax >= t)).mean())
    return out


def calibracao(df, braco, agg):
    linhas = []
    for tau in [EDGES[agg][0]] + NUCLEOS[agg]:
        col = f"p{tau:g}_w0"
        if col not in df.columns:
            continue
        p = df[col].values
        obs = (df["obs"].values > tau).astype(float)
        m = np.isfinite(p)
        idx = np.digitize(p[m], PROB_BINS) - 1
        for k in range(len(PROB_BINS) - 1):
            sel = idx == k
            linhas.append(dict(braco=braco, tau=tau,
                               faixa=f"{PROB_BINS[k]:.1f}-{min(PROB_BINS[k+1], 1):.1f}",
                               n=int(sel.sum()),
                               p_medio=float(p[m][sel].mean()) if sel.any() else np.nan,
                               freq_obs=float(obs[m][sel].mean()) if sel.any() else np.nan))
    return pd.DataFrame(linhas)


def skill_dist(df, produto, agg):
    linhas = []
    d = df["dist_ctx"].values
    idx = np.digitize(d, DIST_BINS) - 1
    for k in range(len(DIST_BINS) - 1):
        sel = (idx == k) & np.isfinite(d)
        sub = df[sel]
        row = dict(produto=produto,
                   faixa_km=f"{DIST_BINS[k]:g}-{DIST_BINS[k+1]:g}",
                   n=int(sel.sum()), n_umido=int((sub["obs"] > EDGES[agg][0]).sum()))
        dc = decisao(sub, produto, EDGES[agg][0], 0) if len(sub) else None
        if dc is not None:
            c = C.contingencia(dc[0][dc[2]], dc[1][dc[2]])
            row.update(POD=c["POD"], FAR=c["FAR"], CSI=c["CSI"])
        row["r"] = (C.metricas_quantidade(campo_mm(sub, produto),
                                          sub["obs"].values)["r"] if len(sub) else np.nan)
        linhas.append(row)
    return pd.DataFrame(linhas)


def tabela_nucleos(df, produto, agg):
    linhas = []
    for r_km in RAIOS_KM:
        for tau in NUCLEOS[agg]:
            c = nucleo_cont(df, produto, tau, r_km)
            if c is None:
                continue
            h, m, f = c
            csi, thr, n_ev = V.csi_melhor(h, m, f)
            k = int(np.argmin(np.abs(V.GRADE_THR - (thr if np.isfinite(thr) else 0.5))))
            k5 = int(np.argmin(np.abs(V.GRADE_THR - 0.5)))
            den5 = h[k5] + m[k5] + f[k5]
            linhas.append(dict(
                produto=produto, tau_mm=tau, tau_mm_h=tau * 60 / agg,
                raio_km=r_km, n_ev=n_ev, CSI_best=csi, thr_best=thr,
                POD_best=h[k] / (h[k] + m[k]) if h[k] + m[k] else np.nan,
                FAR_best=f[k] / (h[k] + f[k]) if h[k] + f[k] else np.nan,
                CSI_p05=h[k5] / den5 if den5 else np.nan,
                POD_p05=h[k5] / (h[k5] + m[k5]) if h[k5] + m[k5] else np.nan,
                FAR_p05=f[k5] / (h[k5] + f[k5]) if h[k5] + f[k5] else np.nan))
    return pd.DataFrame(linhas)


def tabela_ternario(df, areas, braco, agg):
    linhas = []
    umido = df["obs"].values > EDGES[agg][0]
    for c in C_DUVIDA:
        for r_km in RAIOS_KM:
            col = f"ter_c{c:g}_R{r_km}"
            if col not in df.columns:
                continue
            t = df[col].values
            chuv = t == V.CHUVOSO; duv = t == V.DUVIDA
            seco = (t == V.SECO) | (t == V.ORFAO)
            custo = (np.where(chuv & ~umido, CUSTO_FA, 0.0)
                     + np.where(seco & umido, CUSTO_MISS, 0.0)
                     + np.where(duv, c, 0.0))
            row = dict(braco=braco, c=c, R_km=r_km, n=len(t),
                       n_umido=int(umido.sum()),
                       POD_chuvoso=chuv[umido].mean() if umido.any() else np.nan,
                       POD_chuvoso_duvida=(chuv | duv)[umido].mean() if umido.any() else np.nan,
                       FAR_chuvoso=(~umido[chuv]).mean() if chuv.any() else np.nan,
                       frac_estacoes_duvida=duv.mean(),
                       frac_chuva_em_orfao=(t[umido] == V.ORFAO).mean() if umido.any() else np.nan,
                       custo_medio=custo.mean())
            if areas is not None and len(areas):
                a = areas[(areas["c"] == c) & (areas["R_km"] == r_km)]
                row.update(area_chuvoso=a["frac_chuvoso"].mean(),
                           area_duvida=a["frac_duvida"].mean(),
                           area_orfao=a["frac_orfao"].mean())
            linhas.append(row)
    return pd.DataFrame(linhas)


def tabela_brier(df, braco, agg):
    col = f"p{EDGES[agg][0]:g}_w0"
    if col not in df.columns:
        return pd.DataFrame()
    p = df[col].values
    o = (df["obs"].values > EDGES[agg][0]).astype(float)
    m = np.isfinite(p); p, o = p[m], o[m]
    if not len(p) or set(np.unique(p)) <= {0.0, 1.0}:
        return pd.DataFrame()             # campo determinístico (rec 30/60)
    bins = np.linspace(0, 1.0001, 11)
    k = np.digitize(p, bins) - 1
    ob = o.mean(); rel = res = 0.0
    for j in range(10):
        s = k == j
        if s.any():
            rel += s.sum() * (p[s].mean() - o[s].mean()) ** 2
            res += s.sum() * (o[s].mean() - ob) ** 2
    n = len(p)
    faixas = [0, 0.1, 0.3, 0.7, 0.9, 1.0001]
    nit = {f"frac_p_{faixas[i]:g}_{min(faixas[i+1], 1):g}":
           float(((p >= faixas[i]) & (p < faixas[i + 1])).mean())
           for i in range(len(faixas) - 1)}
    return pd.DataFrame([dict(braco=braco, n=n, freq_obs=ob,
                              brier=float(((p - o) ** 2).mean()),
                              confiabilidade=rel / n, resolucao=res / n,
                              incerteza=ob * (1 - ob), **nit)])


def tabela_intensidade(df, braco, agg):
    if "sd_nucleo_km" not in df.columns:
        return pd.DataFrame()
    d = df[(df["obs"] > EDGES[agg][0]) & np.isfinite(df["sd_nucleo_km"])]
    k = np.digitize(d["sd_nucleo_km"].values, SD_BINS) - 1
    tau_n = NUCLEOS[agg][0]
    linhas = []
    for j in range(len(SD_BINS) - 1):
        s = d[k == j]
        linhas.append(dict(braco=braco,
                           faixa_km=f"{SD_BINS[j]:g} a {SD_BINS[j+1]:g}",
                           n=len(s),
                           obs_media=s["obs"].mean() if len(s) else np.nan,
                           obs_mediana=s["obs"].median() if len(s) else np.nan,
                           prev_media=s["mm_w0"].mean() if len(s) else np.nan,
                           frac_obs_nucleo=(s["obs"] >= tau_n).mean() if len(s) else np.nan))
    return pd.DataFrame(linhas)


def tabela_area(area):
    """Área prevista com chuva (P >= 0,5) contra a fração de estações com
    chuva, separando stems com e sem chuva nas estações, e borda x interior."""
    linhas = []
    for arm, a in area.items():
        a = a[a["n_estacoes"] > 0]
        com = a[a["frac_estacoes_chuva"] > 0]; sem = a[a["frac_estacoes_chuva"] == 0]
        linhas.append(dict(
            braco=arm, n_stems=len(a),
            frac_prevista_media=a["frac_prevista"].mean(),
            frac_estacoes_media=a["frac_estacoes_chuva"].mean(),
            razao_area=a["frac_prevista"].sum() / max(a["frac_estacoes_chuva"].sum(), 1e-9),
            frac_prevista_stems_sem_chuva=sem["frac_prevista"].mean() if len(sem) else np.nan,
            frac_stems_sem_chuva_com_alarme=(sem["frac_prevista"] > 0.05).mean() if len(sem) else np.nan,
            razao_area_stems_com_chuva=(com["frac_prevista"].sum()
                                        / max(com["frac_estacoes_chuva"].sum(), 1e-9)) if len(com) else np.nan,
            frac_prevista_borda=a["frac_prevista_borda"].mean(),
            frac_prevista_interior=a["frac_prevista_interior"].mean()))
    return pd.DataFrame(linhas)


def tabela_coerencia(coer):
    linhas = []
    for arm, c in coer.items():
        linhas.append(dict(braco=arm, n_stems=len(c),
                           diferenca_media_P=c["flicker"].mean(),
                           diferenca_mediana_P=c["flicker"].median(),
                           celulas_por_quadro=c["n_celulas"].mean(),
                           frac_com_antecessor=c["frac_com_antecessor"].mean(),
                           celulas_removidas_por_quadro=c["n_removidas"].mean()))
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------
def consolidar_agg(agg):
    arms = bracos_disponiveis(agg)
    if not arms:
        print(f"[resumo] m{agg}: sem pares")
        return None
    areas, coer, area = carregar_aux(agg)
    SUB = {"todos": lambda d: d,
           "chuvosos": lambda d: d[d["stem_chuvoso"].astype(bool)]
           if "stem_chuvoso" in d.columns else d.iloc[:0]}
    esc, met, metf, cal, sd, hld = [], [], [], [], [], []
    nuc, ter, bri, ints = [], [], [], []
    resumo_ev = None
    refs_feitas = False

    def processar(produto, d, h, braco_aux):
        esc.append(escada(d, produto, agg))
        if produto != "pers":
            sd.append(skill_dist(d, produto, agg))
        if braco_aux:
            cal.append(calibracao(d, produto, agg))
            ter.append(tabela_ternario(d, areas.get(produto), produto, agg))
            bri.append(tabela_brier(d, produto, agg))
            ints.append(tabela_intensidade(d, produto, agg))
        for nome, f in SUB.items():
            fd = f(d)
            if len(fd):
                met.append(dict(subconjunto=nome, **larga(fd, produto, agg)))
                if produto != "pers":
                    nuc.append(tabela_nucleos(fd, produto, agg).assign(subconjunto=nome))
            if h is not None and len(h):
                fh = f(h)
                if len(fh):
                    hld.append(dict(subconjunto=nome, **larga(fh, produto, agg)))
        if braco_aux is not None and produto not in REFERENCIAS:
            for fold, df_f in d.groupby("fold"):
                lf = larga(df_f, produto, agg)
                metf.append(dict(braco=produto, fold=int(fold), n_pares=len(df_f),
                                 **{k: v for k, v in lf.items()
                                    if k in ("r", "ratio") or k.startswith(("csi", "nuc"))
                                    and not k.endswith(("_thr", "_n_ev"))}))

    for arm in arms:
        dd = carregar_braco(agg, arm)
        if dd is None:
            continue
        print(f"  [resumo] m{agg} {arm}: {len(dd):,} pares")
        eh_alvo = (dd["lado"] == "alvo").values
        h = dd[~eh_alvo].reset_index(drop=True)
        d = dd[eh_alvo].reset_index(drop=True)
        del dd
        gc.collect()
        processar(arm, d, h, braco_aux=not arm.endswith("_filtro"))
        if "det_w0" in d.columns:
            processar(f"{arm}_det", d, h, braco_aux=None)
        if not refs_feitas and "idw_w0" in d.columns:
            for p in REFERENCIAS:
                processar(p, d, h, braco_aux=None)
            refs_feitas = True
            resumo_ev = dict(n=len(d), n_ch=int(d["stem_chuvoso"].sum())
                             if "stem_chuvoso" in d.columns else 0,
                             ev={t: int((d["obs"] > t).sum()) for t in EDGES[agg]})
        del d, h
        gc.collect()

    junta = lambda L: (pd.concat([x for x in L if len(x)], ignore_index=True)
                       if any(len(x) for x in L) else pd.DataFrame())
    met = pd.DataFrame(met); hld = pd.DataFrame(hld); metf = pd.DataFrame(metf)
    tabelas = {f"metricas_{agg}": met, f"escada_{agg}": junta(esc),
               f"metricas_fold_{agg}": metf, f"holdout_{agg}": hld,
               f"calibracao_{agg}": junta(cal), f"skill_dist_{agg}": junta(sd),
               f"nucleos_{agg}": junta(nuc), f"ternario_{agg}": junta(ter),
               f"brier_{agg}": junta(bri), f"intensidade_dist_{agg}": junta(ints)}
    if coer:
        tabelas[f"coerencia_{agg}"] = tabela_coerencia(coer)
    if area:
        tabelas[f"area_{agg}"] = tabela_area(area)
    for nome, t in tabelas.items():
        if len(t):
            t.to_csv(RES / f"{nome}.csv", index=False)

    tau0 = EDGES[agg][0]
    ev = resumo_ev or dict(n=0, n_ch=0, ev={})
    cols = (["produto", "r", "ratio", f"pod{tau0:g}", f"far{tau0:g}",
             f"csi{tau0:g}_w0", "csi1_w0"]
            + [c for t in NUCLEOS[agg] for c in
               (f"nuc{t:g}_pod", f"nuc{t:g}_far", f"nuc{t:g}_csi")] + ["n_pares"])
    fmt = lambda t: C.fmt_tabela(t[[c for c in cols if c in t.columns]])
    txt = [f"===== m{agg} =====",
           f"pares alvo: {ev['n']:,} ({ev['n_ch']:,} em stems com chuva) | eventos: "
           + "  ".join(f">{t:g}mm:{n}" for t, n in ev["ev"].items()),
           ""]
    for nome, rot in (("todos", "todos os stems (operação contínua)"),
                      ("chuvosos", "só stems com chuva em alguma estação")):
        if len(met) and (met["subconjunto"] == nome).any():
            txt += [f"--- alvos, {rot}; chuva com o limiar de P escolhido no treino "
                    f"de cada braço; núcleos a 4 km com o melhor limiar ---",
                    fmt(met[met["subconjunto"] == nome]), ""]
    if len(hld):
        txt += ["--- holdout espacial (estações nunca vistas), todos os stems ---",
                fmt(hld[hld["subconjunto"] == "todos"]), ""]
    if f"area_{agg}" in tabelas:
        txt += ["--- área prevista com chuva (P>=0,5) x fração de estações com chuva ---",
                C.fmt_tabela(tabelas[f"area_{agg}"]), ""]
    if f"coerencia_{agg}" in tabelas:
        txt += ["--- coerência entre quadros de 10 min ---",
                C.fmt_tabela(tabelas[f"coerencia_{agg}"]), ""]
    if len(metf) and "r" in metf.columns:
        txt += ["--- r por fold ---",
                metf.pivot_table(index="braco", columns="fold",
                                 values="r").round(3).to_string(), ""]
    with open(RES / f"resumo_{agg}.txt", "w") as f:
        f.write("\n".join(txt))
    print(f"[resumo] m{agg}: {len(met)} produtos, csv gravados")
    return "\n".join(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggs", nargs="*", type=int, default=AGGS)
    args = ap.parse_args()
    blocos = [t for t in (consolidar_agg(a) for a in args.aggs) if t]
    with open(RES / "resumo.txt", "w") as f:
        f.write("\n\n".join(blocos) if blocos else "sem resultados")
    z = RES / "envio_gfuse3.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for pad in ("*.csv", "*.txt", "*.log"):
            for f in sorted(RES.glob(pad)):
                zf.write(f, f.name)
        for f in sorted(PROC.glob("variograma_*.csv")):
            zf.write(f, f.name)
        for f in sorted((RES / "mapas").glob("*.png")):
            zf.write(f, f"mapas/{f.name}")
        mv = PROC / "movimento.csv"
        if mv.exists():
            d = pd.read_csv(mv)
            resumo_mv = pd.DataFrame([dict(
                n=len(d), frac_valido=d["valido"].mean(),
                vel_mediana_kmh=float(np.hypot(d.loc[d.valido, "u"],
                                               d.loc[d.valido, "v"]).median()
                                      * KM_PER_PX * 6),
                corr_mediana=d["corr"].median())])
            zf.writestr("movimento_resumo.csv", resumo_mv.to_csv(index=False))
    print(f"[resumo] pronto -> {z}")
    print("ENVIAR PARA ANÁLISE: resultados/envio_gfuse3.zip")


if __name__ == "__main__":
    main()
