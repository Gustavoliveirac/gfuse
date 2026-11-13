#!/usr/bin/env python
"""
cemaden_zeros.py — o CEMADEN só envia registro a cada 10 min quando chove e espaça
o envio quando está seco; a base do gfuse3 trata esses intervalos como falta de
dado. Este script preenche com zero os intervalos sem registro e refaz a avaliação
com os modelos já treinados (sem retreinar), para comparar as métricas.

Etapas (python analise/cemaden_zeros.py <etapa>):
  diagnostico  intervalo entre registros consecutivos e fração de zeros por rede,
               e primeiro registro de cada estação da ANA (série bruta x base)
  preencher    cria processados_cem0/estacoes.npz: nas estações do CEMADEN, um
               instante sem registro vira 0 mm quando está entre dois registros
               separados por no máximo 70 min (o envio horário do período seco);
               registros invalidados pelo QA/QC continuam sem dado. Refaz a figura
               de completude (diária, escala sequencial, blocos por rede)
  avaliar      roda avalia.py com os checkpoints existentes sobre a base preenchida,
               gravando em resultados_cem0/ (nada em resultados/ é alterado)
  resumir      consolida resultados_cem0/ e compara com resultados/
Saída principal: resultados/analise/cem0/ (CSVs, figura e log)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

import config3 as K                                     # noqa: E402

PROC0 = RAIZ / "processados_cem0"
RES0 = RAIZ / "resultados_cem0"
SAI = RAIZ / "resultados" / "analise" / "cem0"
GAP_MAX = pd.Timedelta(minutes=70)
REDE = {"CEMADEM": "CEMADEN"}


def _serie(fonte, cod):
    from exploratoria import _ler_serie
    return _ler_serie(fonte, cod)


def diagnostico():
    z = np.load(K.EST_NPZ, allow_pickle=False)
    cod, fonte, tipo = z["cod"].astype(str), z["fonte"].astype(str), z["tipo"]
    t_base = pd.to_datetime(z["t_brt"])
    lin = []
    for c, f, tp in zip(cod, fonte, tipo):
        if tp != 0:
            continue
        s = _serie(f, c)
        if s is None or s.empty:
            continue
        s = s.sort_values("date")
        anos = s.groupby(s["date"].dt.year)["rainfall"].apply(lambda x: x.notna().mean())
        s = s[s["rainfall"].notna()]
        if s.empty:
            continue
        dt = s["date"].diff().dt.total_seconds().div(60).dropna()
        lin.append(dict(cod=c, rede=REDE.get(f, f), n_registros=len(s), primeiro=s["date"].min(),
                        **{f"com_valor_{a}": round(v, 3) for a, v in anos.items()},
                        ultimo=s["date"].max(), gap_mediano_min=dt.median(),
                        frac_gap_10=float((dt <= 10.5).mean()), frac_gap_60=float(((dt > 50) & (dt <= 70)).mean()),
                        frac_gap_maior_70=float((dt > 70).mean()), frac_zero=float((s["rainfall"] == 0).mean())))
    d = pd.DataFrame(lin)
    d.to_csv(SAI / "diagnostico_registros.csv", index=False)
    print(d.groupby("rede")[["gap_mediano_min", "frac_gap_10", "frac_gap_60", "frac_gap_maior_70", "frac_zero"]]
          .median().round(3).to_string())
    print(f"\nbase do gfuse3: {t_base.min()} a {t_base.max()}")
    cols = [c for c in d.columns if c.startswith("com_valor_")]
    print("\nfração de registros com valor, por ano (mediana das estações):")
    print(d.groupby("rede")[cols].median().round(3).to_string())
    ana = d[d.rede == "ANA"][["cod", "primeiro", "ultimo", "n_registros"] + cols].sort_values("primeiro")
    print("\nANA, primeiro registro com valor e fração com valor por ano:")
    print(ana.to_string(index=False))


def preencher():
    z = dict(np.load(K.EST_NPZ, allow_pickle=False))
    cod, fonte, tipo = z["cod"].astype(str), z["fonte"].astype(str), z["tipo"]
    t = pd.to_datetime(z["t_brt"])
    R10 = z["R10"].copy()
    i10 = np.flatnonzero(tipo == 0)
    nf = R10.shape[2]
    lin = []
    for j, i in enumerate(i10):
        if REDE.get(fonte[i], fonte[i]) != "CEMADEN":
            continue
        s = _serie(fonte[i], cod[i])
        if s is None or s.empty:
            continue
        s = s.sort_values("date")
        inval_df = s
        s = s[s["rainfall"].notna()]                 # só os envios com valor; linhas vazias não são envio
        tr = s["date"].values.astype("datetime64[ns]")
        s_inv = inval_df
        inval = set(s_inv.loc[pd.to_numeric(s_inv.get("status_final", 0), errors="coerce").fillna(0) > K.QAQC_STATUS_MAX,
                              "date"].values.astype("datetime64[ns]"))
        antes = 0
        for k in range(nf):
            tau = (t - pd.to_timedelta((nf - 1 - k) * 10, unit="min")).values.astype("datetime64[ns]")
            nan = ~np.isfinite(R10[:, j, k])
            antes += int(nan.sum())
            p = np.searchsorted(tr, tau, side="left")
            ok = nan & (p > 0) & (p < len(tr))
            pa, pp = np.clip(p - 1, 0, len(tr) - 1), np.clip(p, 0, len(tr) - 1)
            gap = (tr[pp] - tr[pa]).astype("timedelta64[m]").astype(float)
            ok &= gap <= GAP_MAX.total_seconds() / 60
            ok &= ~np.isin(tau, list(inval)) if inval else True
            R10[ok, j, k] = 0.0
        lin.append(dict(cod=cod[i], nan_antes=antes, nan_depois=int((~np.isfinite(R10[:, j, :])).sum()),
                        completude_antes=float(np.isfinite(z["R10"][:, j, -1]).mean()),
                        completude_depois=float(np.isfinite(R10[:, j, -1]).mean())))
    z["R10"] = R10
    PROC0.mkdir(exist_ok=True)
    for f in K.PROC.iterdir():                      # demais arquivos processados, por ligação
        alvo = PROC0 / f.name
        if f.name != Path(K.EST_NPZ).name and not alvo.exists():
            alvo.symlink_to(f)
    np.savez(PROC0 / Path(K.EST_NPZ).name, **z)
    d = pd.DataFrame(lin)
    d.to_csv(SAI / "preenchimento_cemaden.csv", index=False)
    print(d[["completude_antes", "completude_depois"]].describe().round(3).to_string())
    figura_completude(z)


def figura_completude(z):
    tipo = z["tipo"]; fonte = z["fonte"].astype(str); i10 = np.flatnonzero(tipo == 0)
    t = pd.to_datetime(z["t_brt"]).normalize()
    redes = np.array([REDE.get(f, f) for f in fonte])[i10]
    m = pd.DataFrame(np.isfinite(z["R10"][:, :, -1]).astype(float)).groupby(t.values).mean().T
    ordem = sorted(range(len(i10)), key=lambda k: (redes[k], -m.iloc[k].mean()))
    m = m.iloc[ordem]; rr = redes[ordem]
    m.to_csv(SAI / "completude_dia_cem0.csv")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    xs = pd.to_datetime(m.columns)
    marc = [k for k in range(len(xs)) if xs[k].day == 1 and xs[k].month in (1, 4, 7, 10)]
    ax.set_xticks(marc); ax.set_xticklabels([xs[k].strftime("%m/%Y") for k in marc], rotation=90, fontsize=7)
    lim = 0
    for rede in pd.unique(rr):
        n = int((rr == rede).sum())
        ax.axhline(lim - 0.5, color="k", lw=0.8)
        ax.text(-0.01, 1 - (lim + n / 2) / len(rr), f"{rede}\n({n})", transform=ax.transAxes, ha="right",
                va="center", fontsize=8)
        lim += n
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.03); cb.set_label("fração dos registros de 10 min disponíveis no dia")
    fig.tight_layout(); fig.savefig(SAI / "completude_dia_cem0.png", dpi=170); plt.close(fig)
    print(m.mean(1).groupby(rr).mean().round(3).to_string())


def _roda(codigo, args):
    pre = ("import sys; sys.path.insert(0, '.'); from pathlib import Path; import config3 as K; "
           f"K.PROC = Path(r'{PROC0}'); K.EST_NPZ = K.PROC / Path(K.EST_NPZ).name; "
           f"K.RES = Path(r'{RES0}'); K.PARES_DIR = K.RES / 'pares'; ")
    return subprocess.run([sys.executable, "-u", "-c", pre + codigo] + args, cwd=RAIZ)


def avaliar():
    RES0.mkdir(exist_ok=True); (RES0 / "pares").mkdir(exist_ok=True)
    for f in K.RES.iterdir():
        if f.is_file() and f.suffix == ".csv" and not (RES0 / f.name).exists():
            shutil.copy(f, RES0 / f.name)
    import os
    janelas = [int(x) for x in os.environ.get("CEM0_JANELAS", "60 30 10").split()]
    for agg, bracos in ((60, ["s1", "s2rk"]), (30, ["s1", "s2rk"]), (10, ["s1", "s2rk"])):
        if agg not in janelas:
            continue
        for b in bracos:
            for fold in range(K.K_FOLDS):
                if not (K.CKPT_DIR / f"m{agg}" / f"{b}_f{fold}.pt").exists():
                    print(f"sem checkpoint m{agg} {b} f{fold}"); continue
                if (RES0 / "pares" / f"m{agg}" / f"{b}_f{fold}.parquet").exists():
                    continue
                print(f"---- m{agg} {b} fold{fold}", flush=True)
                _roda("sys.argv = ['avalia.py'] + sys.argv[1:]; import avalia; avalia.main()",
                      ["--agg", str(agg), "--arm", b, "--fold", str(fold), "--modo", "completo"])


def resumir():
    _roda("sys.argv = ['resumo.py']; import resumo; resumo.main()", [])
    lin = []
    for agg in (60, 30, 10):
        a, b = K.RES / f"metricas_{agg}.csv", RES0 / f"metricas_{agg}.csv"
        if not (a.exists() and b.exists()):
            continue
        x = pd.read_csv(a); y = pd.read_csv(b)
        x = x[x["subconjunto"] == "todos"].set_index("produto"); y = y[y["subconjunto"] == "todos"].set_index("produto")
        for p in ("s2rk", "s1_rk", "s1", "sk", "ok", "idw", "rrq", "pers"):
            if p in x.index and p in y.index:
                for m in ("r", "pod0.2", "far0.2", "csi0.2_w0"):
                    lin.append(dict(janela=agg, produto=p, metrica=m, original=x.loc[p, m], cemaden_com_zeros=y.loc[p, m]))
    d = pd.DataFrame(lin)
    d.to_csv(SAI / "comparacao_cemaden_zeros.csv", index=False)
    print(d.pivot_table(index=["janela", "produto"], columns="metrica", values="cemaden_com_zeros").round(3).to_string())
    print("\noriginal:")
    print(d.pivot_table(index=["janela", "produto"], columns="metrica", values="original").round(3).to_string())


def sem_ana():
    """Métricas nas estações de avaliação com e sem a ANA, a partir dos pares da base preenchida.
    A série da ANA tem um valor a cada 30 min na grade de 10 min do QA/QC (o envio é de 15 min),
    e o alinhamento com tolerância de 12 min repete esse valor em até três instantes de 10 min."""
    import produtos as Pr
    z = np.load(PROC0 / Path(K.EST_NPZ).name, allow_pickle=False)
    chave = {(round(float(x), 3), round(float(y), 3)): REDE.get(f, f) for x, y, f in
             zip(z["px"], z["py"], z["fonte"].astype(str))}
    lin = []
    for agg in (60, 30):
        for braco, cols in (("s2rk", [("GFUSE", "mm_w0")]), ("s1", [("estimador por satélite", "mm_w0")]),
                            ("s1_rk", [("satélite corrigido", "mm_w0")])):
            fs = sorted((RES0 / "pares" / f"m{agg}").glob(f"{braco}_f[0-9].parquet"))
            if not fs:
                continue
            d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
            d = d[d["lado"] == "alvo"]
            d["rede"] = [chave.get((round(float(x), 3), round(float(y), 3)), "?") for x, y in zip(d["px"], d["py"])]
            pcol = next((c for c in d.columns if c.startswith("p0.2") and c.endswith("_w0")), None)
            extra = [("krigagem simples", "sk_w0"), ("IDW", "idw_w0"), ("RRQPE", "rrq_w0")] if braco == "s2rk" else []
            for rotulo, sub in (("todas", d), ("sem ANA", d[d["rede"] != "ANA"]), ("só ANA", d[d["rede"] == "ANA"])):
                for nome, c in cols + extra:
                    if c not in sub:
                        continue
                    usa_p = pcol is not None and c == "mm_w0"
                    m = Pr.metricas(sub[c].values, sub["obs"].values, sub[pcol].values if usa_p else None,
                                    sub["thr_chuva"].values if usa_p and "thr_chuva" in sub else None)
                    lin.append(dict(janela=agg, estacoes=rotulo, produto=nome, **m))
    t = pd.DataFrame(lin)
    t.to_csv(SAI / "metricas_com_e_sem_ana.csv", index=False)
    print(t.pivot_table(index=["janela", "produto"], columns="estacoes", values="r").round(3).to_string())
    print(t.pivot_table(index=["janela", "produto"], columns="estacoes", values="csi").round(3).to_string())


if __name__ == "__main__":
    SAI.mkdir(parents=True, exist_ok=True)
    etapa = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    for e in (["diagnostico", "preencher", "avaliar", "resumir", "sem_ana"] if etapa == "tudo" else [etapa]):
        print(f"\n===== {e} =====", flush=True)
        globals()[e]()
