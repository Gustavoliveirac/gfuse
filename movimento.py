#!/usr/bin/env python
"""
movimento.py — vetor de movimento (u, v) por stem, estimado entre o quadro
do GOES de t-10 min e o de t, no canal de 10,3 µm (C13).

Método: correlação cruzada por FFT de um campo que só tem valor nos topos
frios (max(0, 255 K - Tb), limitado a 40 K), para seguir os núcleos e não a
bigorna inteira nem a superfície. Busca até MOV_MAX_PX pixels por 10 min,
com ajuste sub-pixel por parábola. Usa o recorte de 128 px (GOES128_DIR)
quando existe, que dá mais área para a correlação; senão o de 64 px.

Como a estimativa de um único par de quadros é ruidosa, o vetor final de
cada stem é a mediana das estimativas válidas dos últimos 30 min (só
passado, então vale em tempo real). Sem estimativa válida em 60 min, o
vetor é zero e o stem fica marcado como não válido.

Unidades: px por 10 min; u positivo para leste (coluna), v positivo para
sul (linha), 1 px = 2 km.

Saída: processados/movimento.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import comum as C
from config3 import (GOES128_DIR, GOES_DIR, KM_PER_PX, MOV_FILE, MOV_MAX_PX,
                     MOV_TB_FRIO)

CANAL_C13 = 5


def campo_frio(arr2d):
    tb = C.rad_para_tb(np.asarray(arr2d, np.float32), CANAL_C13)
    f = np.clip(MOV_TB_FRIO - np.nan_to_num(tb, nan=400.0), 0.0, 40.0)
    return f.astype(np.float32)


def deslocamento(f0, f1, max_px=MOV_MAX_PX):
    """(dx, dy, corr) tal que f1(x) ~ f0(x - d). dx em colunas, dy em linhas."""
    H, W = f0.shape
    jan = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    a = (f0 - f0.mean()) * jan
    b = (f1 - f1.mean()) * jan
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na < 1e-6 or nb < 1e-6:
        return np.nan, np.nan, np.nan
    cc = np.real(np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)))
    cc = np.fft.fftshift(cc) / (na * nb)
    cy, cx = H // 2, W // 2
    sub = cc[cy - max_px:cy + max_px + 1, cx - max_px:cx + max_px + 1]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    corr = float(sub[iy, ix])

    def parabola(m1, c0, p1):
        den = m1 - 2 * c0 + p1
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m1 - p1) / den

    fx = (parabola(sub[iy, ix - 1], sub[iy, ix], sub[iy, ix + 1])
          if 0 < ix < sub.shape[1] - 1 else 0.0)
    fy = (parabola(sub[iy - 1, ix], sub[iy, ix], sub[iy + 1, ix])
          if 0 < iy < sub.shape[0] - 1 else 0.0)
    return ix - max_px + fx, iy - max_px + fy, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--min-corr", type=float, default=0.3)
    ap.add_argument("--min-frio", type=float, default=0.005,
                    help="fração mínima de pixels frios nos dois quadros")
    args = ap.parse_args()
    if MOV_FILE.exists() and not args.forcar:
        if "recorte" in pd.read_csv(MOV_FILE, nrows=1).columns:
            print(f"[movimento] já existe {MOV_FILE}, pulando")
            return
        print("[movimento] arquivo de versão anterior, recalculando")

    tem128 = GOES128_DIR.exists()
    if not tem128:
        print(f"[movimento] AVISO: {GOES128_DIR} não encontrado, usando só o "
              f"recorte de 64 px")
    stems = sorted(p.stem for p in GOES_DIR.glob("*.npy"))
    n128 = sum((GOES128_DIR / f"{s}.npy").exists() for s in stems) if tem128 else 0
    print(f"[movimento] {len(stems)} stems | {n128} com recorte de 128 px "
          f"({n128 / max(len(stems), 1):.0%}); os demais usam o de 64 px")
    if n128:                                  # confere o centro do 128 x 64
        s0 = next(s for s in stems[len(stems) // 2:] + stems
                  if (GOES128_DIR / f"{s}.npy").exists())
        try:
            dif = np.nanmax(np.abs(np.load(GOES128_DIR / f"{s0}.npy")[:, 32:96, 32:96]
                                   - np.load(GOES_DIR / f"{s0}.npy")))
            print(f"[movimento] conferência 128 x 64 ({s0}): diferença máxima "
                  f"{dif:.4g}" + ("" if dif < 1e-3 else
                                  "  ATENÇÃO: recortes não coincidem"))
        except Exception as e:
            print(f"[movimento] conferência 128 x 64 não feita: {e}")

    t_brt = {s: C.stem_para_brt(s).round("10min") for s in stems}
    por_t = {t: s for s, t in t_brt.items()}
    linhas = []
    cache = {}

    def frio(s, tam):
        """Campo frio do stem no recorte pedido (128 ou 64); None se faltar."""
        chave = (s, tam)
        if chave not in cache:
            p = (GOES128_DIR if tam == 128 else GOES_DIR) / f"{s}.npy"
            try:
                cache[chave] = (campo_frio(np.load(p, mmap_mode="r")[CANAL_C13])
                                if p.exists() else None)
            except Exception:
                cache[chave] = None
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[chave]

    motivos = dict(sem_quadro_anterior=0, sem_arquivo=0, pouco_frio=0,
                   corr_baixa=0, ok=0)
    for k, s in enumerate(stems):
        prev = por_t.get(t_brt[s] - pd.Timedelta(minutes=10))
        u = v = corr = np.nan
        frac = np.nan
        recorte = 0
        if prev is None:
            motivos["sem_quadro_anterior"] += 1
        else:
            f0 = f1 = None
            if tem128:
                f0, f1 = frio(prev, 128), frio(s, 128)
                recorte = 128
            if f0 is None or f1 is None:              # cai para o de 64 px
                f0, f1 = frio(prev, 64), frio(s, 64)
                recorte = 64
            if f0 is None or f1 is None:
                motivos["sem_arquivo"] += 1
            else:
                frac = float(min((f0 > 0).mean(), (f1 > 0).mean()))
                if frac < args.min_frio:
                    motivos["pouco_frio"] += 1
                else:
                    u, v, corr = deslocamento(f0, f1)
                    if np.isfinite(corr) and corr >= args.min_corr:
                        motivos["ok"] += 1
                    else:
                        motivos["corr_baixa"] += 1
        ok = bool(np.isfinite(corr) and corr >= args.min_corr)
        linhas.append(dict(goes_stem=s, t=t_brt[s], u_bruto=u, v_bruto=v,
                           corr=corr, frac_frio=frac, recorte=recorte,
                           bruto_ok=ok))
        if k % 5000 == 0:
            print(f"  ... {k}/{len(stems)}")
    print("[movimento] estimativas por par de quadros: " + ", ".join(
        f"{m} {n} ({n / max(len(stems), 1):.0%})" for m, n in motivos.items()))

    df = pd.DataFrame(linhas).set_index("t").sort_index()
    ok = df[df["bruto_ok"]]
    u_fin, v_fin, val = [], [], []
    for t in df.index:
        for janela in (30, 60):
            w = ok.loc[t - pd.Timedelta(minutes=janela):t]
            if len(w):
                u_fin.append(float(w["u_bruto"].median()))
                v_fin.append(float(w["v_bruto"].median()))
                val.append(True)
                break
        else:
            u_fin.append(0.0); v_fin.append(0.0); val.append(False)
    df["u"], df["v"], df["valido"] = u_fin, v_fin, val
    df = df.reset_index(drop=True)
    MOV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MOV_FILE, index=False)

    vel = np.hypot(df.loc[df["valido"], "u"], df.loc[df["valido"], "v"]) \
        * KM_PER_PX * 6
    rumo = (np.degrees(np.arctan2(df.loc[df["valido"], "u"],
                                  -df.loc[df["valido"], "v"])) + 360) % 360
    com_frio = df["frac_frio"] >= args.min_frio
    print(f"[movimento] válidos {df['valido'].mean():.1%} de todos os stems e "
          f"{df.loc[com_frio, 'valido'].mean():.1%} dos stems com topo frio")
    print(f"[movimento] velocidade "
          f"mediana {vel.median():.0f} km/h (p90 {vel.quantile(.9):.0f}) | "
          f"rumo de deslocamento mediano {rumo.median():.0f}° -> {MOV_FILE}")


if __name__ == "__main__":
    main()
