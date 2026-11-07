#!/usr/bin/env python
"""
diag_satelite.py — dois diagnósticos sem treino, que definem parâmetros do
modelo:

1. PARALAXE. O GOES-16 vê Belo Horizonte de lado (zenital ~42°), e um topo
   alto aparece deslocado para leste-sudeste da posição real. O teste
   desloca o campo do satélite (RRQPE >= 0,5 mm/h e topos frios em três
   limiares) de -8 a +8 px em x e y em relação às estações e mede POD, FAR
   e CSI em cada deslocamento. Se for paralaxe, o melhor deslocamento
   aponta para leste-sudeste, cresce com a altura do topo (limiares de Tb
   mais frios) e é o mesmo qualquer que seja o sentido do movimento da
   tempestade. Se fosse efeito de deslocamento da chuva no tempo, ele
   acompanharia o movimento. Por isso o teste é repetido separando stems
   com movimento para leste, para oeste e lentos.

2. REGRA DO SECO. Frequência de chuva observada nas estações por faixa de
   Tb 10,3 µm no pixel e no mínimo de uma vizinhança (3 e 6 px, que cobrem
   a paralaxe), e candidatas a regra "seco certo" (Tb mínima na vizinhança
   acima de T0 e RRQPE zero na vizinhança): fração de casos rotulados como
   seco, quantos desses tinham chuva na estação (violação) e quanto da
   chuva observada a regra apagaria. Lista também as estações que mais
   violam a regra de referência (podem ter problema de horário ou de dado).

Saídas em resultados/: paralaxe_teorica.csv, paralaxe_deslocamentos.csv,
paralaxe_resumo.csv, seco_tb_faixas.csv, seco_regras.csv,
seco_violacoes_estacao.csv

Uso:  python diag_satelite.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import comum as C
from config3 import (BH_LAT, BH_LON, GOES_DIR, KM_PER_PX, MOV_FILE, N_FRAMES,
                     RES, RRQPE_CH)
from estacoes import Base

LIM_EST = 0.2                      # mm em 10 min
D_MAX = 8                          # px de deslocamento testados
MARGEM_SHIFT = D_MAX               # estações a >= 8 px da borda no teste
PREDITORES = {"rrqpe>=0,5": None, "tb<=250": 250.0, "tb<=235": 235.0,
              "tb<=220": 220.0}
FAIXAS_TB = [0, 220, 235, 250, 260, 270, 280, 290, 400]
T0S = [255, 260, 265, 270, 275, 280]
RAIOS = [0, 3, 6]
REGRA_REF = (270, 6)               # T0, raio usados na lista por estação


def janela(campo, w, fn):
    if w == 0:
        return campo
    H, W = campo.shape
    viz = -np.inf if fn is np.max else np.inf
    p = np.pad(campo, w, constant_values=viz)
    lin = fn([p[:, w + d:w + d + W] for d in range(-w, w + 1)], 0)
    return fn([lin[w + d:w + d + H, :] for d in range(-w, w + 1)], 0)


def main():
    base = Base()
    W10 = base.janelas(10)
    sel = np.flatnonzero((base.tipo == 0) & base.dentro)
    xi = np.round(base.px[sel]).astype(int)
    yi = np.round(base.py[sel]).astype(int)
    inner = (xi >= MARGEM_SHIFT) & (xi < 64 - MARGEM_SHIFT) \
        & (yi >= MARGEM_SHIFT) & (yi < 64 - MARGEM_SHIFT)
    print(f"[diag_satelite] {len(sel)} estações de 10 min dentro do recorte "
          f"({int(inner.sum())} a >= {MARGEM_SHIFT} px da borda, usadas na paralaxe)")

    # ---------------- paralaxe teórica ----------------
    teo = []
    for h in (3, 6, 9, 12, 15):
        e, s = C.paralaxe_km(BH_LAT, BH_LON, h)
        teo.append(dict(altura_km=h, leste_km=e, sul_km=s,
                        total_km=float(np.hypot(e, s)),
                        dx_px=e / KM_PER_PX, dy_px=s / KM_PER_PX,
                        rumo_graus=float((np.degrees(np.arctan2(e, -s)) + 360) % 360)))
    pd.DataFrame(teo).to_csv(RES / "paralaxe_teorica.csv", index=False)

    # ---------------- preparação ----------------
    D = np.arange(-D_MAX, D_MAX + 1)
    DX, DY = np.meshgrid(D, D)
    DX, DY = DX.ravel(), DY.ravel()
    xi_in, yi_in = xi[inner], yi[inner]
    IX = xi_in[None] + DX[:, None]                  # (n_desl, S_in)
    IY = yi_in[None] + DY[:, None]
    grupos = ("todos", "para_leste", "para_oeste", "lento")
    acc = {(p, g): np.zeros((3, len(DX))) for p in PREDITORES for g in grupos}
    mov = {}
    if MOV_FILE.exists():
        d = pd.read_csv(MOV_FILE, usecols=["goes_stem", "u", "valido"])
        mov = {s: (u if v else 0.0) for s, u, v in
               zip(d["goes_stem"].astype(str), d["u"], d["valido"])}

    nb = len(FAIXAS_TB) - 1
    faixas = {r: np.zeros((2, nb)) for r in ["pixel"] + [f"min_r{r}" for r in RAIOS[1:]]}
    regras = {(t0, r): np.zeros(3) for t0 in T0S for r in RAIOS}   # n_seco, n_chuva_seco, n_chuva_tot
    cobertura = {(t0, r): [] for t0 in T0S for r in RAIOS}
    viol_est = np.zeros((2, len(sel)))              # n_seco_ref, n_chuva_seco_ref

    for k, s in enumerate(base.stems):
        obs = W10[k, sel]
        ok = np.isfinite(obs)
        if not ok.any():
            continue
        try:
            arr = np.load(GOES_DIR / f"{s}.npy", mmap_mode="r")
            tb = C.rad_para_tb(np.asarray(arr[5], np.float32), 5)
            rr = np.nan_to_num(np.asarray(arr[RRQPE_CH], np.float32), nan=0.0)
        except Exception:
            continue
        tb = np.nan_to_num(tb, nan=300.0)
        chuva = ok & (np.nan_to_num(obs) > LIM_EST)

        # ---- paralaxe: só stems com algum sinal ----
        o_in, ok_in = chuva[inner], ok[inner]
        u = mov.get(s, 0.0)
        g_mov = "para_leste" if u > 0.5 else ("para_oeste" if u < -0.5 else "lento")
        for p, lim in PREDITORES.items():
            F = (rr >= 0.5) if lim is None else (tb <= lim)
            if not F.any() and not o_in.any():
                continue
            P = F[IY, IX] & ok_in[None]
            h = (P & o_in[None]).sum(1); f = (P & ~o_in[None]).sum(1)
            m = (~P & o_in[None]).sum(1)
            for g in ("todos", g_mov):
                acc[(p, g)] += np.stack([h, m, f])

        # ---- regra do seco ----
        tb_pix = tb[yi, xi]
        b = np.clip(np.digitize(tb_pix, FAIXAS_TB) - 1, 0, nb - 1)
        np.add.at(faixas["pixel"][0], b[ok], 1)
        np.add.at(faixas["pixel"][1], b[chuva], 1)
        for r in RAIOS:
            tmin = janela(tb, r, np.min) if r else tb
            rmax = janela(rr, r, np.max) if r else rr
            tmin_s, rmax_s = tmin[yi, xi], rmax[yi, xi]
            if r:
                bb = np.clip(np.digitize(tmin_s, FAIXAS_TB) - 1, 0, nb - 1)
                np.add.at(faixas[f"min_r{r}"][0], bb[ok], 1)
                np.add.at(faixas[f"min_r{r}"][1], bb[chuva], 1)
            for t0 in T0S:
                seco = ok & (tmin_s > t0) & (rmax_s <= 0)
                regras[(t0, r)] += [seco.sum(), (seco & chuva).sum(), chuva.sum()]
                if k % 10 == 0:
                    cobertura[(t0, r)].append(float(((tmin > t0) & (rmax <= 0)).mean()))
                if (t0, r) == REGRA_REF:
                    viol_est[0] += seco; viol_est[1] += seco & chuva
        if k % 20000 == 0:
            print(f"  ... {k}/{base.T}")

    # ---------------- saídas: paralaxe ----------------
    linhas, resumo = [], []
    for (p, g), (h, m, f) in acc.items():
        den = h + m + f
        csi = np.where(den > 0, h / np.maximum(den, 1), np.nan)
        for j in range(len(DX)):
            linhas.append(dict(preditor=p, grupo_movimento=g, dx_px=int(DX[j]),
                               dy_px=int(DY[j]), n_ev=h[j] + m[j],
                               POD=h[j] / max(h[j] + m[j], 1),
                               FAR=f[j] / max(h[j] + f[j], 1), CSI=csi[j]))
        if not np.isfinite(csi).any():
            continue
        jb = int(np.nanargmax(csi)); j0 = int(np.flatnonzero((DX == 0) & (DY == 0))[0])
        # centro dos deslocamentos com CSI >= 99% do máximo (desempata platôs)
        topo = np.nan_to_num(csi) >= 0.99 * csi[jb]
        wq = np.nan_to_num(csi)[topo]
        cx = float((DX[topo] * wq).sum() / wq.sum()); cy = float((DY[topo] * wq).sum() / wq.sum())
        resumo.append(dict(preditor=p, grupo_movimento=g, n_ev=int(h[j0] + m[j0]),
                           CSI_sem_desloc=csi[j0], CSI_melhor=csi[jb],
                           dx_px=round(cx, 1), dy_px=round(cy, 1),
                           desloc_km=float(np.hypot(cx, cy) * KM_PER_PX),
                           rumo_graus=float((np.degrees(np.arctan2(cx, -cy)) + 360) % 360)
                           if np.hypot(cx, cy) > 0.25 else np.nan))
    pd.DataFrame(linhas).to_csv(RES / "paralaxe_deslocamentos.csv", index=False)
    rs = pd.DataFrame(resumo)
    rs.to_csv(RES / "paralaxe_resumo.csv", index=False)

    # ---------------- saídas: regra do seco ----------------
    fx = []
    for tipo, (n, nc) in faixas.items():
        for j in range(nb):
            fx.append(dict(tb=tipo, faixa_K=f"{FAIXAS_TB[j]}-{FAIXAS_TB[j + 1]}",
                           n=int(n[j]), freq_chuva=nc[j] / n[j] if n[j] else np.nan,
                           frac_das_chuvas=nc[j] / max(faixas[tipo][1].sum(), 1)))
    pd.DataFrame(fx).to_csv(RES / "seco_tb_faixas.csv", index=False)
    rg = []
    for (t0, r), (ns, ncs, nct) in regras.items():
        rg.append(dict(T0_K=t0, raio_px=r, raio_km=r * KM_PER_PX,
                       frac_casos_secos=ns / max(faixas["pixel"][0].sum(), 1),
                       violacao=ncs / ns if ns else np.nan,
                       frac_chuva_apagada=ncs / nct if nct else np.nan,
                       cobertura_dominio=np.mean(cobertura[(t0, r)]) if cobertura[(t0, r)] else np.nan))
    pd.DataFrame(rg).to_csv(RES / "seco_regras.csv", index=False)
    ve = pd.DataFrame(dict(station_code=base.cod[sel], fonte=base.fonte[sel],
                           n_seco=viol_est[0].astype(int),
                           n_chuva_no_seco=viol_est[1].astype(int)))
    ve["taxa"] = ve["n_chuva_no_seco"] / ve["n_seco"].clip(lower=1)
    ve.sort_values("taxa", ascending=False).to_csv(
        RES / "seco_violacoes_estacao.csv", index=False)

    # ---------------- log ----------------
    t12 = teo[3]
    print(f"[diag_satelite] paralaxe teórica para topo a 12 km: "
          f"{t12['total_km']:.1f} km no rumo {t12['rumo_graus']:.0f}° "
          f"({t12['dx_px']:.1f} px leste, {t12['dy_px']:.1f} px sul)")
    if len(rs):
        for _, r in rs[rs["grupo_movimento"] == "todos"].iterrows():
            print(f"  {r['preditor']:<11} melhor deslocamento dx={r['dx_px']:+.1f} "
                  f"dy={r['dy_px']:+.1f} px ({r['desloc_km']:.0f} km, rumo "
                  f"{r['rumo_graus']:.0f}°) | CSI {r['CSI_sem_desloc']:.3f} -> "
                  f"{r['CSI_melhor']:.3f}")
    ref = pd.DataFrame(rg).query("T0_K == @REGRA_REF[0] and raio_px == @REGRA_REF[1]")
    if len(ref):
        r = ref.iloc[0]
        print(f"[diag_satelite] regra do seco (Tb mín. em {REGRA_REF[1]} px > "
              f"{REGRA_REF[0]} K e RRQPE zero): cobre {r['cobertura_dominio']:.0%} "
              f"do domínio, violação {r['violacao']:.2%}, apagaria "
              f"{r['frac_chuva_apagada']:.1%} da chuva observada")
    print(f"[diag_satelite] csv em {RES}")


if __name__ == "__main__":
    main()
