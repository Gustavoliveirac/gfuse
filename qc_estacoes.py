#!/usr/bin/env python
"""
qc_estacoes.py — controle de qualidade dos registros de 10 min com o
satélite e as estações vizinhas (parâmetros QC_* no config3.py).

Chuva suspeita (sai do treino e da avaliação): chuva > 0,2 mm em 10 min
com céu limpo, ou seja, Tb 10,3 µm mínima > QC_T0 num raio de QC_RAIO_PX
em todos os quadros de t-QC_PASSADO_MIN até t, RRQPE zero no mesmo raio e
período, e nenhuma estação a até QC_RAIO_PX com chuva em t. Só estações
dentro do recorte podem ser checadas (fora dele não há satélite).

Zero suspeito (só reportado): zero com Tb < 220 K no pixel, RRQPE >= 5 mm/h
e pelo menos 3 vizinhas a até QC_RAIO_PX com >= 2 mm em 10 min, o sinal
típico de pluviômetro entupido.

Saídas: processados/qc_estacoes.npz (lida pelo estacoes.Base) e
resultados/qc_estacoes.csv (taxas por estação; a coluna "candidata" marca
taxas acima de 10 vezes a mediana, para revisão manual).

Uso:  python qc_estacoes.py [--forcar]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import comum as C
from config3 import (GOES_DIR, QC_NPZ, QC_PASSADO_MIN, QC_RAIO_PX, QC_T0,
                     RES, RRQPE_CH)
from estacoes import Base

LIM = 0.2


def filtro(campo, w, fn):
    H, W = campo.shape
    viz = -np.inf if fn is np.max else np.inf
    p = np.pad(campo, w, constant_values=viz)
    lin = fn([p[:, w + d:w + d + W] for d in range(-w, w + 1)], 0)
    return fn([lin[w + d:w + d + H, :] for d in range(-w, w + 1)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true")
    args = ap.parse_args()
    base = Base(aplicar_qc=False)
    if QC_NPZ.exists() and not args.forcar:
        z = np.load(QC_NPZ)
        if len(z["stems"]) == base.T and np.array_equal(z["cod10"], base.cod[base.i0]):
            print(f"[qc] já existe {QC_NPZ.name}, pulando")
            return
    W10 = base.janelas(10)
    i0 = base.i0
    dentro0 = base.dentro[i0]
    xi = np.clip(np.round(base.px[i0]).astype(int), 0, 63)
    yi = np.clip(np.round(base.py[i0]).astype(int), 0, 63)
    d = np.hypot(base.px[i0][:, None] - base.px[i0][None],
                 base.py[i0][:, None] - base.py[i0][None])
    viz = (d <= QC_RAIO_PX) & ~np.eye(len(i0), dtype=bool)
    n_q = QC_PASSADO_MIN // 10 + 1
    Q = np.zeros((base.T, len(i0)), bool)
    zero_susp = np.zeros(len(i0)); n_reg = np.zeros(len(i0)); n_ch = np.zeros(len(i0))
    cache = {}

    def campos(k):
        if k not in cache:
            try:
                a = np.load(GOES_DIR / f"{base.stems[k]}.npy", mmap_mode="r")
                tb = np.nan_to_num(C.rad_para_tb(np.asarray(a[5], np.float32), 5), nan=300.0)
                rr = np.nan_to_num(np.asarray(a[RRQPE_CH], np.float32), nan=0.0)
                cache[k] = (tb, rr)
            except Exception:
                cache[k] = None
            if len(cache) > n_q + 2:
                cache.pop(next(iter(cache)))
        return cache[k]

    for k in range(base.T):
        obs = W10[k, i0]
        ok = np.isfinite(obs) & dentro0
        if not ok.any():
            continue
        chuva = ok & (np.nan_to_num(obs) > LIM)
        n_reg += ok; n_ch += chuva
        quadros = [base.deslocado(-10 * j)[k] for j in range(n_q)]
        if any(q < 0 for q in quadros):
            continue
        cs = [campos(q) for q in quadros]
        if any(c is None for c in cs):
            continue
        tbmin = np.min([c[0] for c in cs], 0)
        rrmax = np.max([c[1] for c in cs], 0)
        limpo = (filtro(tbmin, QC_RAIO_PX, np.min)[yi, xi] > QC_T0) & \
                (filtro(rrmax, QC_RAIO_PX, np.max)[yi, xi] <= 0)
        viz_chuva = (viz & chuva[None]).any(1)
        Q[k] = chuva & limpo & ~viz_chuva
        tb0, rr0 = cs[0]
        forte = ok & (np.nan_to_num(obs) >= 2.0)
        zero_susp += (ok & (np.nan_to_num(obs) == 0) & (tb0[yi, xi] < 220)
                      & (rr0[yi, xi] >= 5) & ((viz & forte[None]).sum(1) >= 3))
        if k % 20000 == 0:
            print(f"  ... {k}/{base.T}")

    np.savez_compressed(QC_NPZ, stems=base.stems, cod10=base.cod[i0], suspeito=Q)
    t = pd.DataFrame(dict(station_code=base.cod[i0], fonte=base.fonte[i0],
                          dentro=dentro0, n_registros=n_reg.astype(int),
                          n_chuva=n_ch.astype(int),
                          n_chuva_suspeita=Q.sum(0).astype(int),
                          n_zero_suspeito=zero_susp.astype(int)))
    t["taxa_chuva_suspeita"] = t["n_chuva_suspeita"] / t["n_chuva"].clip(lower=1)
    t["taxa_zero_suspeito"] = t["n_zero_suspeito"] / t["n_registros"].clip(lower=1)
    med = t.loc[t["dentro"], "taxa_chuva_suspeita"].median()
    t["candidata"] = t["dentro"] & (t["taxa_chuva_suspeita"] > 10 * max(med, 1e-4))
    t.sort_values("taxa_chuva_suspeita", ascending=False).to_csv(
        RES / "qc_estacoes.csv", index=False)
    print(f"[qc] chuva suspeita: {int(Q.sum()):,} registros de "
          f"{int(n_ch.sum()):,} com chuva ({Q.sum() / max(n_ch.sum(), 1):.2%}) | "
          f"mediana por estação {med:.2%} | {int(t['candidata'].sum())} estações "
          f"candidatas a revisão | zeros suspeitos {int(zero_susp.sum()):,}")


if __name__ == "__main__":
    main()
