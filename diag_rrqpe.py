#!/usr/bin/env python
"""
diag_rrqpe.py — mede, contra as estações de 10 min, as duas impressões das
visualizações: o RRQPE aparece DEPOIS da estação começar a registrar, e a
área de chuva do RRQPE é muito maior que a real.

Por evento (estação a estação; evento = chuva > 0,2 mm/10 min, separado
por 1 h seca, total >= 2 mm e pico >= 1 mm/10 min):
  defasagem da correlação cruzada entre a taxa da estação e o RRQPE no
  pixel (positiva = RRQPE atrasado), defasagem do início (primeiro RRQPE
  >= 0,5 mm/h menos primeira chuva na estação), defasagem do pico, razão
  de volume RRQPE/estação.

Por stem (área):
  fração das estações com chuva, fração dos pixels de estação com RRQPE
  >= 0,5 mm/h, fração do domínio com RRQPE >= 0,5 mm/h e com topo frio
  (Tb 10,3 µm < 235 K); contingência RRQPE x estação no pixel e com
  tolerância de 2 e 5 px (máximo do RRQPE na janela).

Saídas em resultados/: rrqpe_eventos.csv, rrqpe_defasagem_resumo.csv,
rrqpe_area.csv, rrqpe_contingencia.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import comum as C
from config3 import GOES_DIR, N_FRAMES, RES, RRQPE_CH
from estacoes import Base

LIM_EST = 0.2          # mm em 10 min
LIM_RRQ = 0.5          # mm/h
LAGS = range(-12, 13)  # passos de 10 min (+-2 h)
JANELAS = [0, 2, 5]


def max_janela(campo, w):
    if w == 0:
        return campo
    H, W = campo.shape
    p = np.pad(campo, w, constant_values=-np.inf)
    lin = np.max([p[:, w + d:w + d + W] for d in range(-w, w + 1)], 0)
    return np.max([lin[w + d:w + d + H, :] for d in range(-w, w + 1)], 0)


def main():
    base = Base()
    sel = np.flatnonzero((base.tipo == 0) & base.dentro)
    cods = list(base.cod[sel])
    fonte = dict(zip(base.cod, base.fonte))
    xi = np.clip(np.round(base.px[sel]).astype(int), 0, 63)
    yi = np.clip(np.round(base.py[sel]).astype(int), 0, 63)
    stems = list(base.stems)
    t_idx = base.t
    grade = pd.date_range(t_idx.min(), t_idx.max(), freq="10min")
    pos_t = {t: k for k, t in enumerate(grade)}
    T, S = len(grade), len(cods)
    print(f"[diag_rrqpe] {S} estações de 10 min dentro do recorte, "
          f"{len(stems)} stems, {T} passos na grade regular")

    G = np.full((T, S), np.nan, np.float32)
    k_reg = np.array([pos_t[t] for t in t_idx])
    j0 = np.searchsorted(base.i0, sel)              # coluna em R10
    G[k_reg] = base.R10[:, j0, N_FRAMES - 1]

    R = np.full((T, S), np.nan, np.float32)
    R_w = {w: np.full((T, S), np.nan, np.float32) for w in JANELAS}
    dom = np.full((T, 2), np.nan, np.float32)
    for k, s in enumerate(stems):
        p = GOES_DIR / f"{s}.npy"
        try:
            arr = np.load(p, mmap_mode="r")
            rr = np.nan_to_num(np.asarray(arr[RRQPE_CH], np.float32), nan=0.0)
            tb = C.rad_para_tb(np.asarray(arr[5], np.float32), 5)
        except Exception:
            continue
        i = k_reg[k]
        for w in JANELAS:
            R_w[w][i] = max_janela(rr, w)[yi, xi]
        R[i] = R_w[0][i]
        dom[i] = [(rr >= LIM_RRQ).mean(), np.nanmean(tb < 235.0)]
        if k % 10000 == 0:
            print(f"  ... {k}/{len(stems)}")

    # ---------------- eventos ----------------
    ev = []
    for j, cod in enumerate(cods):
        g = G[:, j]; r = R[:, j]
        mol = np.where(np.isfinite(g), g > LIM_EST, False)
        idx = np.flatnonzero(mol)
        if not len(idx):
            continue
        grupos = np.split(idx, np.flatnonzero(np.diff(idx) > 6) + 1)
        for gr in grupos:
            a, b = gr[0], gr[-1]
            tot = np.nansum(g[a:b + 1]); pico = np.nanmax(g[a:b + 1])
            if tot < 2.0 or pico < 1.0:
                continue
            i0, i1 = max(0, a - 12), min(T, b + 13)
            gg, rr = g[i0:i1] * 6.0, r[i0:i1]
            ok = np.isfinite(gg) & np.isfinite(rr)
            if ok.sum() < 8:
                continue
            cor = {}
            for L in LAGS:
                if L >= 0:
                    x, y = gg[:len(gg) - L], rr[L:]
                else:
                    x, y = gg[-L:], rr[:len(rr) + L]
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 8 and x[m].std() > 0 and y[m].std() > 0:
                    cor[L] = float(np.corrcoef(x[m], y[m])[0, 1])
            lag = max(cor, key=cor.get) if cor else np.nan
            r_on = np.flatnonzero(np.nan_to_num(r[max(0, a - 12):i1]) >= LIM_RRQ)
            onset = ((r_on[0] + max(0, a - 12) - a) * 10.0) if len(r_on) else np.nan
            ev.append(dict(
                station_code=cod, fonte=fonte.get(cod, ""),
                inicio=grade[a], hora=grade[a].hour, mes=grade[a].month,
                duracao_min=(b - a + 1) * 10, total_mm=float(tot),
                pico_mm10=float(pico),
                lag_xcorr_min=lag * 10.0 if np.isfinite(lag) else np.nan,
                r_max=cor.get(lag, np.nan) if cor else np.nan,
                r_lag0=cor.get(0, np.nan),
                defasagem_inicio_min=onset,
                rrqpe_detectou=bool(len(r_on)),
                defasagem_pico_min=(float(np.nanargmax(rr) - np.nanargmax(gg)) * 10.0
                                    if np.isfinite(rr).any() else np.nan),
                razao_volume=float(np.nansum(rr[ok]) / 6.0 / max(np.nansum(gg[ok]) / 6.0, 1e-6))))
    ev = pd.DataFrame(ev)
    ev.to_csv(RES / "rrqpe_eventos.csv", index=False)

    def resumo(d, grupo):
        return dict(grupo=grupo, n_eventos=len(d),
                    lag_xcorr_mediana_min=d["lag_xcorr_min"].median(),
                    lag_xcorr_p25=d["lag_xcorr_min"].quantile(.25),
                    lag_xcorr_p75=d["lag_xcorr_min"].quantile(.75),
                    frac_rrqpe_atrasado=(d["lag_xcorr_min"] > 0).mean(),
                    frac_rrqpe_adiantado=(d["lag_xcorr_min"] < 0).mean(),
                    r_lag0_mediana=d["r_lag0"].median(),
                    r_max_mediana=d["r_max"].median(),
                    defasagem_inicio_mediana_min=d["defasagem_inicio_min"].median(),
                    frac_rrqpe_nao_detectou=1 - d["rrqpe_detectou"].mean(),
                    defasagem_pico_mediana_min=d["defasagem_pico_min"].median(),
                    razao_volume_mediana=d["razao_volume"].median())
    linhas = [resumo(ev, "todas")]
    for f, d in ev.groupby("fonte"):
        linhas.append(resumo(d, f"fonte={f}"))
    for faixa, d in ev.groupby(pd.cut(ev["pico_mm10"], [1, 2, 5, 10, 100]), observed=True):
        linhas.append(resumo(d, f"pico_mm10={faixa}"))
    for faixa, d in ev.groupby(pd.cut(ev["hora"], [-1, 5, 11, 17, 23]), observed=True):
        linhas.append(resumo(d, f"hora_inicio={faixa}"))
    pd.DataFrame(linhas).to_csv(RES / "rrqpe_defasagem_resumo.csv", index=False)

    # ---------------- área e contingência ----------------
    ok_st = np.isfinite(G) & np.isfinite(R)
    n_ok = ok_st.sum(1)
    est_chuva = np.where(ok_st, G > LIM_EST, False).sum(1) / np.maximum(n_ok, 1)
    est_rrq = np.where(ok_st, R >= LIM_RRQ, False).sum(1) / np.maximum(n_ok, 1)
    a = pd.DataFrame(dict(n=n_ok, frac_est_chuva=est_chuva, frac_est_rrqpe=est_rrq,
                          frac_dom_rrqpe=dom[:, 0], frac_dom_frio=dom[:, 1]))
    a = a[(a["n"] >= 10) & ((a["frac_est_chuva"] > 0) | (a["frac_est_rrqpe"] > 0))]
    a["faixa_est_chuva"] = pd.cut(a["frac_est_chuva"],
                                  [-0.01, 0, 0.05, 0.1, 0.2, 0.4, 1.0])
    ar = a.groupby("faixa_est_chuva", observed=True).agg(
        n_stems=("n", "size"),
        frac_est_chuva=("frac_est_chuva", "median"),
        frac_est_rrqpe=("frac_est_rrqpe", "median"),
        frac_dom_rrqpe=("frac_dom_rrqpe", "median"),
        frac_dom_frio=("frac_dom_frio", "median")).reset_index()
    ar["faixa_est_chuva"] = ar["faixa_est_chuva"].astype(str)
    ar.to_csv(RES / "rrqpe_area.csv", index=False)

    cont = []
    obs = np.where(ok_st, G > LIM_EST, False)
    for w in JANELAS:
        Rw = R_w[w]
        m = ok_st & np.isfinite(Rw)
        c = C.contingencia(Rw[m] >= LIM_RRQ, obs[m])
        cont.append(dict(tolerancia_px=w, tolerancia_km=w * 2, **c))
    pd.DataFrame(cont).to_csv(RES / "rrqpe_contingencia.csv", index=False)

    r0 = linhas[0]
    print(f"[diag_rrqpe] {len(ev)} eventos | defasagem mediana da correlação "
          f"{r0['lag_xcorr_mediana_min']:.0f} min | início "
          f"{r0['defasagem_inicio_mediana_min']:.0f} min | RRQPE não detectou "
          f"{r0['frac_rrqpe_nao_detectou']:.0%} dos eventos")
    print(f"[diag_rrqpe] contingência no pixel: POD {cont[0]['POD']:.2f} "
          f"FAR {cont[0]['FAR']:.2f} | csv em {RES}")


if __name__ == "__main__":
    main()
