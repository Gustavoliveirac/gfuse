#!/usr/bin/env python
"""
dados.py — monta a base do gfuse3 a partir do zero:

  1. estacoes.npz      matrizes tempo x estação (ver estacoes.py): chuva das
                       estações de 10 min nos 6 passos de cada stem
                       (tolerância 12 min) e das horárias na hora que termina
                       no stem (tolerância 31 min). Só estações pluviométricas.
                       Estações fora do recorte de 64 px ficam na base, mas só
                       como contexto (nunca alvo nem holdout).
  2. goes_stats.json   média/desvio dos 8 canais brutos + 3 diferenças Tb
  3. variograma_XX.csv variograma exponencial (log1p mm) por janela
  4. holdout.csv       20% das estações de 10 min de dentro do recorte
  5. folds_XX.csv      k-fold temporal (blocos, folga de 6 h) por janela

Uso:  python dados.py            (só o que falta)
      python dados.py --forcar   (refaz tudo)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import comum as C
from config3 import (AGGS, DIFF_PARES, EDGES, EST_DIR, EST_NPZ, ESTACOES_EXCLUIDAS,
                     GOES_DIR, LIMITE_10MIN_MM, LIMITE_HORA_MM, MEDIAS_F,
                     QAQC_STATUS_MAX,
                     HOLDOUT_FRAC, HOLDOUT_SEED, K_FOLDS, META_CSV, N_CH_RAW,
                     N_FRAMES, PROC, RES, STATS_F, STEP_MIN)

PASTA_FONTE = {"CEMADEN": "CEMADEM"}   # nome da pasta difere do rótulo
LIMITES_CHUVA = (0.0, 500.0)


def stems_goes():
    """Stems do GOES em ordem temporal, um por passo de 10 min."""
    stems = []
    for f in sorted(GOES_DIR.glob("*.npy")):
        try:
            pd.to_datetime(f.stem, format="%Y%m%d_%H%M%S")
            stems.append(f.stem)
        except Exception:
            pass
    ref = pd.DataFrame({"goes_stem": stems,
                        "t": [C.stem_para_brt(s).round("10min") for s in stems]})
    ref = ref.sort_values("goes_stem").drop_duplicates("t").sort_values("t")
    return ref.reset_index(drop=True)


def tipo_id(tipo, freq):
    t = str(tipo).strip().lower()
    if t.startswith("met"):
        return None                      # meteorológicas ficam fora (só chuva)
    return 0 if "10" in str(freq) else 1


DESCARTES = []
SEM_STATUS = []


def ler_serie(fonte, cod, limite=LIMITES_CHUVA[1]):
    arq = EST_DIR / PASTA_FONTE.get(fonte, fonte) / f"{cod}.csv"
    if not arq.exists():
        return None
    try:
        stn = pd.read_csv(arq, sep=";", usecols=lambda c: c in (
            "date", "rainfall", "status_final", "motivo"))
        stn["date"] = pd.to_datetime(stn["date"])
    except Exception as e:
        print(f"  ERRO {cod}: {e}")
        return None
    stn["rainfall"] = pd.to_numeric(stn["rainfall"], errors="coerce")
    n_inv = 0
    if "status_final" in stn.columns:
        st = pd.to_numeric(stn["status_final"], errors="coerce").fillna(0)
        inval = st > QAQC_STATUS_MAX
        n_inv = int((inval & stn["rainfall"].notna()).sum())
        stn.loc[inval, "rainfall"] = np.nan
    else:
        SEM_STATUS.append(cod)
    n_lim = int((stn["rainfall"] > limite).sum())
    DESCARTES.append(dict(station_code=cod, fonte=fonte,
                          n_registros=int(stn["rainfall"].notna().sum()) + n_inv,
                          n_invalido_qaqc=n_inv, n_acima_limite_fisico=n_lim,
                          max_valido_mm=float(stn.loc[stn["rainfall"] <= limite,
                                                      "rainfall"].max())))
    lo = LIMITES_CHUVA[0]
    fora = (stn["rainfall"] > limite).sum()
    stn["rainfall"] = stn["rainfall"].where((stn["rainfall"] >= lo)
                                            & (stn["rainfall"] <= limite))
    if fora:
        print(f"    {cod}: {int(fora)} registro(s) acima de {limite:g} mm descartado(s)")
    if n_inv:
        print(f"    {cod}: {n_inv} registro(s) inválido(s) pelo QA/QC descartado(s)")
    stn = (stn.dropna(subset=["rainfall"]).sort_values("date")
              .drop_duplicates("date").reset_index(drop=True))
    return stn if len(stn) else None


def alinhar(stn, t_ref, tol):
    """Valor da estação mais próximo de cada instante de t_ref (NaN se não há
    leitura dentro da tolerância)."""
    a = pd.DataFrame({"alvo": pd.DatetimeIndex(t_ref).astype("datetime64[ns]"),
                      "k": np.arange(len(t_ref))}).sort_values("alvo")
    stn = stn.assign(date=stn["date"].astype("datetime64[ns]"))
    m = pd.merge_asof(a, stn.rename(columns={"date": "obs"}), left_on="alvo",
                      right_on="obs", tolerance=tol, direction="nearest")
    out = np.full(len(t_ref), np.nan, np.float32)
    out[m["k"].values] = m["rainfall"].values
    return out


def montar_base(ref):
    t = pd.DatetimeIndex(ref["t"])
    meta = pd.read_csv(META_CSV, sep=";")
    meta["Codigo"] = meta["Codigo"].astype(str).str.strip()
    cods, tipos, fontes, pxs, pys, r10, rh = [], [], [], [], [], [], []
    for _, row in meta.iterrows():
        tid = tipo_id(row["Tipo"], row["Frequencia"])
        tag = f"{row['Codigo']:>15}  {str(row['Fonte']):<14}"
        if tid is None:
            continue
        if row["Codigo"] in ESTACOES_EXCLUIDAS:
            print(f"  {tag} EXCLUÍDA ({ESTACOES_EXCLUIDAS[row['Codigo']]})")
            continue
        stn = ler_serie(str(row["Fonte"]).strip(), row["Codigo"],
                        LIMITE_10MIN_MM if tid == 0 else LIMITE_HORA_MM)
        if stn is None:
            print(f"  {tag} sem dados"); continue
        px, py = C.latlon_para_px(float(row["Latitude"]), float(row["Longitude"]))
        if tid == 0:
            m = np.stack([alinhar(stn, t - pd.Timedelta(minutes=STEP_MIN * (N_FRAMES - 1 - i)),
                                  pd.Timedelta("12min")) for i in range(N_FRAMES)], 1)
            n = int(np.isfinite(m[:, -1]).sum())
            if n == 0:
                print(f"  {tag} sem leituras no período"); continue
            r10.append(m)
        else:
            v = alinhar(stn, t, pd.Timedelta("31min"))
            n = int(np.isfinite(v).sum())
            if n == 0:
                print(f"  {tag} sem leituras no período"); continue
            rh.append(v)
        dentro = -0.5 <= px < 63.5 and -0.5 <= py < 63.5
        print(f"  {tag} {n:>8,} stems com leitura"
              + ("" if dentro else "  (fora do recorte: só contexto)"))
        cods.append(row["Codigo"]); tipos.append(tid); fontes.append(str(row["Fonte"]))
        pxs.append(px); pys.append(py)
    if not cods:
        raise RuntimeError("nenhuma estação alinhada — confira EST_DIR")
    tipos = np.array(tipos, np.int8)
    ordem = np.r_[np.flatnonzero(tipos == 0), np.flatnonzero(tipos == 1)]
    R10 = (np.stack(r10, 1) if r10 else np.zeros((len(t), 0, N_FRAMES), np.float32))
    RH = (np.stack(rh, 1) if rh else np.zeros((len(t), 0), np.float32))
    dsc = pd.DataFrame(DESCARTES)
    dsc.to_csv(RES / "dados_descartes.csv", index=False)
    print(f"  descartes: {int(dsc['n_invalido_qaqc'].sum()):,} registros inválidos "
          f"pelo QA/QC e {int(dsc['n_acima_limite_fisico'].sum()):,} acima do "
          f"limite físico -> resultados/dados_descartes.csv")
    if SEM_STATUS:
        print(f"  AVISO: {len(SEM_STATUS)} séries sem a coluna status_final (QA/QC "
              f"não aplicado nelas): {', '.join(SEM_STATUS[:10])}")
    np.savez(EST_NPZ, stems=ref["goes_stem"].values.astype("U15"),
             t_brt=t.values.astype("datetime64[ns]"),
             cod=np.array(cods)[ordem].astype("U20"), tipo=tipos[ordem],
             fonte=np.array(fontes)[ordem].astype("U20"),
             px=np.array(pxs, np.float32)[ordem], py=np.array(pys, np.float32)[ordem],
             R10=R10.astype(np.float32), RH=RH.astype(np.float32),
             origem=np.array(str(EST_DIR)))
    dentro = [(-0.5 <= x < 63.5 and -0.5 <= y < 63.5) for x, y in zip(pxs, pys)]
    res = pd.DataFrame({"fonte": fontes, "tipo": tipos, "dentro": dentro})
    print("  estações por fonte (dentro / fora do recorte):")
    for (f, tp), g in res.groupby(["fonte", "tipo"]):
        print(f"    {f:<14} {'10 min' if tp == 0 else 'horária':<8} "
              f"{int(g['dentro'].sum()):>4} / {int((~g['dentro']).sum()):>4}")
    print(f"  base: {len(t):,} stems x {len(cods)} estações -> {EST_NPZ}")


def medias_classes(base):
    """Média observada da chuva dentro de cada compartimento do histograma,
    por janela (estações de dentro do recorte). Substitui o ponto médio fixo,
    que superestimava o volume (a chuva se concentra no início de cada
    compartimento)."""
    out = {}
    for agg in AGGS:
        W = base.janelas(agg)[:, base.dentro]
        v = W[np.isfinite(W)]
        edges = np.asarray(EDGES[agg])
        k = (v[:, None] > edges[None]).sum(1)
        out[str(agg)] = [0.0] + [float(v[k == j].mean()) if (k == j).any()
                                 else float(edges[j - 1]) for j in range(1, len(edges) + 1)]
        print(f"  médias por compartimento m{agg}: "
              + " ".join(f"{m:.2f}" for m in out[str(agg)]))
    with open(MEDIAS_F, "w") as f:
        json.dump(out, f)


def calcular_stats(stems, n=300, seed=0):
    rng = np.random.default_rng(seed)
    escolha = (stems if len(stems) <= n
               else [stems[i] for i in rng.choice(len(stems), n, replace=False)])
    n_tot = N_CH_RAW + len(DIFF_PARES)
    valores = [[] for _ in range(n_tot)]
    for stem in escolha:
        try:
            arr = np.nan_to_num(np.load(GOES_DIR / f"{stem}.npy").astype(np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            continue
        for ch in range(N_CH_RAW):
            valores[ch].append(arr[ch].ravel())
        tb = {c: C.rad_para_tb(arr[c], c) for par in DIFF_PARES for c in par}
        for j, (a, b) in enumerate(DIFF_PARES):
            valores[N_CH_RAW + j].append(np.nan_to_num(tb[a] - tb[b], nan=0.0).ravel())
    mean = [float(np.concatenate(v).mean()) if v else 0.0 for v in valores]
    std = [float(max(np.concatenate(v).std(), 1e-3)) if v else 1.0 for v in valores]
    with open(STATS_F, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"  stats ({n_tot} canais) -> {STATS_F}")


def montar_holdout(base):
    m = (base.tipo == 0) & base.dentro
    codes = np.sort(base.cod[m])
    rng = np.random.default_rng(HOLDOUT_SEED)
    n = max(1, int(len(codes) * HOLDOUT_FRAC))
    hold = sorted(rng.choice(codes, n, replace=False).tolist())
    pd.Series(hold, name="station_code").to_csv(RES / "holdout.csv", index=False)
    print(f"  holdout espacial: {n}/{len(codes)} estações de 10 min dentro do recorte")


def montar_folds(stems):
    for agg in AGGS:
        al = C.stems_alinhados(stems, agg)
        linhas = [dict(fold=fd["fold"], papel="val", goes_stem=s)
                  for fd in C.folds_temporais(al, K_FOLDS) for s in fd["val"]]
        pd.DataFrame(linhas).to_csv(RES / f"folds_{agg}.csv", index=False)
        print(f"  folds m{agg}: {len(al)} stems, blocos de ~{len(al) // K_FOLDS}")


def montar_variogramas(base):
    for agg in AGGS:
        W = base.janelas(agg)
        al = set(C.stems_alinhados(list(base.stems), agg))
        linhas = []
        for i in np.flatnonzero([s in al for s in base.stems]):
            w = W[i]; v = np.isfinite(w) & base.dentro
            if v.sum() < 4 or np.nanmean(w[v]) <= 0.2:
                continue
            linhas.append(pd.DataFrame(dict(goes_stem=base.stems[i], obs_mm=w[v],
                                            px=base.px[v], py=base.py[v])))
        par = C.ajustar_variograma(pd.concat(linhas)) if linhas else None
        if par is None:
            print(f"  variograma m{agg}: pares insuficientes, usando padrão")
            continue
        pd.Series(par).to_csv(PROC / f"variograma_{agg}.csv", header=["valor"])
        print(f"  variograma m{agg}: nugget={par['nugget']:.3f} "
              f"sill={par['sill']:.3f} range={par['range']:.1f} km")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true")
    args = ap.parse_args()
    if not META_CSV.exists():
        raise FileNotFoundError(f"metadados.csv não encontrado em {META_CSV} — "
                                f"corrija META_CSV no topo do config3.py")
    if not EST_DIR.exists():
        raise FileNotFoundError(f"pasta de estações não encontrada em {EST_DIR} — "
                                f"corrija EST_DIR no topo do config3.py")
    ref = stems_goes()
    if not len(ref):
        raise FileNotFoundError(f"nenhum .npy em {GOES_DIR} — ajuste GOES_DIR")
    print(f"[dados] {len(ref)} stems em {GOES_DIR}")
    novo = args.forcar or not EST_NPZ.exists()
    if not novo:
        z = np.load(EST_NPZ)
        novo = len(z["stems"]) != len(ref)
        if novo:
            print("[dados] número de stems mudou, refazendo a base")
        elif np.isin(z["cod"], list(ESTACOES_EXCLUIDAS)).any():
            novo = True
            print("[dados] lista de estações excluídas mudou, refazendo a base")
        elif str(z["origem"]) != str(EST_DIR) if "origem" in z.files else True:
            novo = True
            print("[dados] base gerada de outra pasta de estações, refazendo")
    if novo:
        montar_base(ref)
    else:
        print(f"[dados] base existente ({EST_NPZ.name})")
    from estacoes import Base
    base = Base()
    stems = list(base.stems)
    if novo or not STATS_F.exists():
        calcular_stats(stems)
    if novo or not MEDIAS_F.exists():
        medias_classes(base)
    if novo or not (RES / "holdout.csv").exists():
        montar_holdout(base)
    if novo or not (RES / f"folds_{AGGS[0]}.csv").exists():
        montar_folds(stems)
    if novo or not (PROC / f"variograma_{AGGS[-1]}.csv").exists():
        montar_variogramas(base)
    print("[dados] pronto")


if __name__ == "__main__":
    main()
