#!/usr/bin/env python
"""
comum.py — geometria da grade, tempo, métricas, interpolação (IDW e kriging
ordinário), conversão radiância->Tb e k-fold temporal. Sem torch, sem estado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config3 import (BH_LAT, BH_LON, GAP_HORAS, IMG_SIZE, K_FOLDS, KM_PER_PX,
                     N_FRAMES, NU_BANDAS, VARIO_PADRAO)

R_TERRA = 6371.0


# ============================================================
# Geometria e tempo
# ============================================================
def latlon_para_px(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dy = -(lat - BH_LAT) * np.pi / 180 * R_TERRA
    dx = (lon - BH_LON) * np.pi / 180 * R_TERRA * np.cos(np.radians(BH_LAT))
    return dx / KM_PER_PX + IMG_SIZE / 2, dy / KM_PER_PX + IMG_SIZE / 2


def px_para_latlon(px, py):
    px, py = np.asarray(px, float), np.asarray(py, float)
    dx, dy = (px - IMG_SIZE / 2) * KM_PER_PX, (py - IMG_SIZE / 2) * KM_PER_PX
    return (BH_LAT - dy / R_TERRA * 180 / np.pi,
            BH_LON + dx / (R_TERRA * np.cos(np.radians(BH_LAT))) * 180 / np.pi)


def grade_latlon():
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    return px_para_latlon(xs, ys)


def stem_para_brt(stem):
    """'YYYYmmdd_HHMMSS' (UTC do scan) -> Timestamp BRT (UTC-3, sem DST)."""
    return pd.to_datetime(stem, format="%Y%m%d_%H%M%S") - pd.Timedelta(hours=3)


def stems_alinhados(stems, agg_min, tol_s=120):
    """Stems cujo t0 fecha meia hora (30) ou hora cheia (60); 10 aceita todos."""
    out = []
    for s in stems:
        t = stem_para_brt(s)
        if agg_min == 10:
            out.append(s); continue
        r = t.round("30min")
        if abs((t - r).total_seconds()) > tol_s:
            continue
        if (agg_min == 30 and r.minute in (0, 30)) or (agg_min == 60 and r.minute == 0):
            out.append(s)
    return out


def folds_temporais(stems, k=K_FOLDS, gap_h=GAP_HORAS):
    """K blocos temporais contíguos. Devolve lista de dicts
    {fold, treino, val}; do treino saem os stems a menos de gap_h horas do
    bloco de validação."""
    ts = pd.Series({s: stem_para_brt(s) for s in stems}).sort_values()
    ordem = list(ts.index)
    lim = np.linspace(0, len(ordem), k + 1).astype(int)
    folds = []
    for f in range(k):
        val = ordem[lim[f]:lim[f + 1]]
        t_ini, t_fim = ts[val[0]], ts[val[-1]]
        gap = pd.Timedelta(hours=gap_h)
        tre = [s for s in ordem
               if not (t_ini - gap <= ts[s] <= t_fim + gap)]
        folds.append(dict(fold=f, treino=tre, val=list(val)))
    return folds


# ============================================================
# Radiância -> temperatura de brilho (Planck inversa aproximada)
# ============================================================
def rad_para_tb(rad, canal):
    """rad em mW/(m2 sr cm-1) da banda `canal` (0..6 = C08..C14) -> Tb em K.
    fk1/fk2 derivados do número de onda central nominal; a correção de banda
    (bc1/bc2) é omitida — o desvio é de poucos K e idêntico em todo arquivo."""
    nu = NU_BANDAS[canal]
    fk1 = 1.191042e-5 * nu ** 3
    fk2 = 1.4387752 * nu
    r = np.where(np.asarray(rad, np.float32) > 1e-6, rad, np.nan)
    return (fk2 / np.log1p(fk1 / r)).astype(np.float32)


# ============================================================
# Métricas
# ============================================================
def contingencia(pred_flag, obs_flag):
    p, o = np.asarray(pred_flag, bool), np.asarray(obs_flag, bool)
    h = float((p & o).sum()); m = float((~p & o).sum())
    f = float((p & ~o).sum()); c = float((~p & ~o).sum())
    n = h + m + f + c
    exp = ((h + m) * (h + f) + (c + m) * (c + f)) / n if n else np.nan
    return dict(n_ev=h + m,
                CSI=h / (h + m + f) if h + m + f else np.nan,
                POD=h / (h + m) if h + m else np.nan,
                FAR=f / (h + f) if h + f else np.nan,
                FBIAS=(h + f) / (h + m) if h + m else np.nan,
                HSS=(h + c - exp) / (n - exp) if n and n > exp else np.nan)


def metricas_quantidade(pred, obs, so_umidos=True):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    if so_umidos:
        ok &= obs > 0
    p, o = pred[ok], obs[ok]
    out = dict(n=int(ok.sum()))
    if len(o) < 3:
        return {**out, "bias": np.nan, "ratio": np.nan, "MAE": np.nan,
                "r": np.nan, "r_log": np.nan, "R2mm": np.nan}
    lp, lo = np.log1p(np.clip(p, 0, None)), np.log1p(o)
    r = lambda a, b: (float(np.corrcoef(a, b)[0, 1])
                      if a.std() > 0 and b.std() > 0 else np.nan)
    ss = ((o - o.mean()) ** 2).sum()
    return {**out, "bias": float((p - o).mean()),
            "ratio": float(p.sum() / o.sum()) if o.sum() > 0 else np.nan,
            "MAE": float(np.abs(p - o).mean()), "r": r(p, o), "r_log": r(lp, lo),
            "R2mm": float(1 - ((p - o) ** 2).sum() / ss) if ss > 1e-8 else np.nan}


def melhor_limiar_csi(prob, obs_flag, grade=np.arange(0.05, 0.96, 0.05)):
    melhor, thr = np.nan, np.nan
    prob = np.asarray(prob, float)
    for t in grade:
        c = contingencia(prob >= t, obs_flag)["CSI"]
        if np.isfinite(c) and (not np.isfinite(melhor) or c > melhor):
            melhor, thr = c, float(t)
    return melhor, thr


def amostra_com_tolerancia(campo, px, py, win=0, fn=np.max):
    """Amostra campo (H,W) em px/py; win>0 aplica fn (máximo por padrão) na
    janela (2w+1)²."""
    campo = np.asarray(campo, float)
    H, W = campo.shape
    xi = np.clip(np.round(np.asarray(px, float)).astype(int), 0, W - 1)
    yi = np.clip(np.round(np.asarray(py, float)).astype(int), 0, H - 1)
    if win == 0:
        return campo[yi, xi]
    out = np.empty(len(xi))
    for k, (x, y) in enumerate(zip(xi, yi)):
        out[k] = fn(campo[max(0, y - win):y + win + 1,
                          max(0, x - win):x + win + 1])
    return out


def brier(prob, obs_flag):
    p = np.clip(np.asarray(prob, float), 0, 1)
    o = np.asarray(obs_flag, bool).astype(float)
    ok = np.isfinite(p)
    return float(np.mean((p[ok] - o[ok]) ** 2)) if ok.any() else np.nan


# ============================================================
# Interpolação
# ============================================================
def idw(xy, val, grade_xy, potencia=2.0, eps=1e-3):
    xy, val, grade_xy = (np.asarray(a, float) for a in (xy, val, grade_xy))
    if len(val) == 0:
        return np.zeros(len(grade_xy))
    d = np.sqrt(((grade_xy[:, None] - xy[None]) ** 2).sum(-1)).clip(eps)
    w = 1.0 / d ** potencia
    return np.maximum((w * val[None]).sum(1) / w.sum(1), 0.0)


def ok_lote(xy_ctx, val, xy_alvo, vario, jitter=1e-6, com_var=False):
    """Kriging ordinário, uma fatoração para todos os alvos. val em log1p mm."""
    xy_ctx, val, xy_alvo = (np.asarray(a, float) for a in (xy_ctx, val, xy_alvo))
    n, m = len(val), len(xy_alvo)
    if n == 0:
        z = np.zeros(m)
        return (z, np.full(m, np.nan)) if com_var else z
    if n < 3:
        e = idw(xy_ctx, val, xy_alvo)
        return (e, np.full(m, np.nan)) if com_var else e
    cov = lambda d: vario["sill"] * np.exp(-d / vario["range"])
    dnn = np.sqrt(((xy_ctx[:, None] - xy_ctx[None]) ** 2).sum(-1))
    dgt = np.sqrt(((xy_alvo[:, None] - xy_ctx[None]) ** 2).sum(-1))
    K = np.empty((n + 1, n + 1))
    K[:n, :n] = cov(dnn) + np.eye(n) * (vario["nugget"] + jitter)
    K[:n, n] = 1.0; K[n, :n] = 1.0; K[n, n] = 0.0
    rhs = np.empty((n + 1, m)); rhs[:n] = cov(dgt).T; rhs[n] = 1.0
    try:
        lam = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(K, rhs, rcond=None)[0]
    est = lam[:n].T @ val
    if not com_var:
        return est
    var = (vario["sill"] + vario["nugget"]) - (lam * rhs).sum(0)
    return est, np.clip(var, 0.0, None)


def grade_km():
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    return np.stack([xs.ravel(), ys.ravel()], 1) * KM_PER_PX


def campos_interp(px_ctx, py_ctx, mm_ctx, vario):
    """Campos 64x64 a partir do contexto: (idw_mm, ok_mm, ok_log, peso_info).
    ok_mm segue a convenção do bench (expm1 do OK em log1p, piso 0.05).
    peso_info = 1 - var/(sill+nugget), ~1 junto de estação, ~0 sem dado."""
    G = IMG_SIZE
    if len(mm_ctx) == 0:
        z = np.zeros((G, G), np.float32)
        return z, z.copy(), z.copy(), z.copy()
    gxy = grade_km()
    xy = np.stack([np.asarray(px_ctx, float), np.asarray(py_ctx, float)], 1) * KM_PER_PX
    idw_mm = idw(xy, np.asarray(mm_ctx, float), gxy).reshape(G, G).astype(np.float32)
    if len(mm_ctx) >= 3 and np.max(mm_ctx) > 0:
        est, var = ok_lote(xy, np.log1p(np.clip(mm_ctx, 0, None)), gxy, vario,
                           com_var=True)
        ok_mm = np.maximum(np.expm1(est), 0.0)
        ok_mm[ok_mm < 0.05] = 0.0
        w = np.clip(1.0 - var / (vario["sill"] + vario["nugget"]), 0.0, 1.0)
        return (idw_mm, ok_mm.reshape(G, G).astype(np.float32),
                est.reshape(G, G).astype(np.float32),
                w.reshape(G, G).astype(np.float32))
    return idw_mm, idw_mm.copy(), np.log1p(idw_mm), np.ones((G, G), np.float32)


# ============================================================
# Janelas das estações e variograma
# ============================================================
def ajustar_variograma(win, max_stems=400, seed=0):
    """Variograma exponencial (log1p mm) por busca em grade — sem scipy.
    win: DataFrame com goes_stem, obs_mm, px, py. Devolve dict nugget/sill/range."""
    rng = np.random.default_rng(seed)
    medios = win.groupby("goes_stem")["obs_mm"].mean()
    stems = medios[medios > 0.2].sort_values(ascending=False).index.tolist()
    if len(stems) > max_stems:
        stems = [stems[i] for i in
                 rng.choice(len(stems), max_stems, replace=False)]
    bins = np.arange(0.0, 42.0, 2.0)
    soma = np.zeros(len(bins) - 1); cont = np.zeros(len(bins) - 1)
    for stem in stems:
        w = win[win["goes_stem"] == stem]
        if len(w) < 4:
            continue
        v = np.log1p(np.clip(w["obs_mm"].values, 0, None))
        x = w["px"].values * 2.0; y = w["py"].values * 2.0
        d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
        g = 0.5 * (v[:, None] - v[None]) ** 2
        iu = np.triu_indices(len(v), 1)
        db, gb = d[iu], g[iu]
        idx = np.digitize(db, bins) - 1
        m = (idx >= 0) & (idx < len(bins) - 1)
        np.add.at(soma, idx[m], gb[m]); np.add.at(cont, idx[m], 1)
    ok = cont > 30
    if ok.sum() < 4:
        return None
    h = 0.5 * (bins[:-1] + bins[1:])[ok]
    gamma = (soma / np.maximum(cont, 1))[ok]
    melhor, par = np.inf, None
    for nug in np.linspace(0.0, gamma[0], 9):
        for sill in np.linspace(max(gamma.max() - nug, 1e-3) * 0.5,
                                max(gamma.max() - nug, 1e-3) * 1.5, 9):
            for rng_ in np.linspace(3.0, 40.0, 16):
                mod = nug + sill * (1 - np.exp(-h / rng_))
                sse = float(((mod - gamma) ** 2).sum())
                if sse < melhor:
                    melhor, par = sse, dict(nugget=float(nug), sill=float(sill),
                                            range=float(rng_))
    return par


def carregar_vario(agg, proc_dir):
    f = proc_dir / f"variograma_{agg}.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0).iloc[:, 0].to_dict()
    return dict(VARIO_PADRAO[agg])


def fmt_tabela(df, floatfmt="{:.3f}"):
    with pd.option_context("display.width", 240, "display.max_columns", 80,
                           "display.float_format", floatfmt.format):
        return df.to_string(index=False)


# ============================================================
# Paralaxe do GOES-16
# ============================================================
GOES_LON = -75.2            # longitude do ponto subsatélite (GOES-16/East)
GOES_ALT = 35786.0          # km acima da superfície


def paralaxe_km(lat, lon, altura_km):
    """Deslocamento aparente (km para leste, km para sul) de um topo de nuvem
    a `altura_km` visto pelo GOES-16: a imagem coloca o topo mais longe do
    ponto subsatélite do que a posição real no solo. Para corrigir, o pixel
    deve ser movido no sentido oposto."""
    R = R_TERRA
    la, lo = np.radians(lat), np.radians(lon)
    sat = np.array([(R + GOES_ALT) * np.cos(np.radians(GOES_LON)),
                    (R + GOES_ALT) * np.sin(np.radians(GOES_LON)), 0.0])

    def xyz(phi, lam, r):
        return np.array([r * np.cos(phi) * np.cos(lam),
                         r * np.cos(phi) * np.sin(lam), r * np.sin(phi)])
    topo = xyz(la, lo, R + altura_km)
    d = topo - sat
    # interseção da linha satélite -> topo com a esfera de raio R
    a = d @ d; b = 2 * sat @ d; c = sat @ sat - R * R
    t = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if t < 1.0:                       # a reta cruza a Terra depois do topo
        t = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    p = sat + t * d
    la2 = np.arcsin(p[2] / R); lo2 = np.arctan2(p[1], p[0])
    leste = (lo2 - lo) * R * np.cos(la)
    sul = -(la2 - la) * R
    return float(leste), float(sul)


def pesos_krigagem(px, py, vario):
    """Pesos da krigagem simples (média zero) de cada pixel do recorte em
    relação às estações (px, py), e o peso de informação. Correlação
    exponencial com o alcance e a fração de pepita do variograma.
    Devolve (lam (n, 4096), peso_info (4096,)); n == 0 -> (None, zeros)."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    if len(px) == 0:
        return None, np.zeros(IMG_SIZE * IMG_SIZE, np.float32)
    xy = np.stack([px, py], 1) * KM_PER_PX
    gxy = grade_km()
    nf = vario["nugget"] / max(vario["nugget"] + vario["sill"], 1e-9)
    rho = lambda d: (1.0 - nf) * np.exp(-d / vario["range"])
    dnn = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    Cm = rho(dnn); np.fill_diagonal(Cm, 1.0)
    c = rho(np.sqrt(((gxy[:, None] - xy[None]) ** 2).sum(-1)))     # (Ng, n)
    try:
        lam = np.linalg.solve(Cm + np.eye(len(px)) * 1e-6, c.T)     # (n, Ng)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(Cm, c.T, rcond=None)[0]
    w = np.clip((lam * c.T).sum(0), 0.0, 1.0)
    return lam, w.astype(np.float32)


def krigagem_simples(px, py, mm, vario, limiar=0.2):
    """Krigagem simples com média zero, em mm (sem transformação), e
    krigagem de indicador (chove > limiar ou não) com os mesmos pesos.
    Longe das estações as duas voltam para zero (seco), em vez de voltar para
    a média das estações como o IDW e a krigagem ordinária.
    Devolve (sk_mm, ik_prob, peso_info), cada um 64x64; peso_info vai de ~1
    junto a uma estação a 0 longe de todas."""
    G = IMG_SIZE
    mm = np.asarray(mm, float)
    lam, w = pesos_krigagem(px, py, vario)
    if lam is None:
        z = np.zeros((G, G), np.float32)
        return z, z.copy(), z.copy()
    sk = np.clip(lam.T @ mm, 0.0, None)
    ik = np.clip(lam.T @ (mm > limiar).astype(float), 0.0, 1.0)
    return (sk.reshape(G, G).astype(np.float32), ik.reshape(G, G).astype(np.float32),
            w.reshape(G, G))


def krigagem_residuo(campo, px, py, obs, vario, minimo=None, maximo=None, lam=None):
    """Estágio 2 analítico: soma ao campo a krigagem simples (média zero) do
    erro do campo nas estações (obs - campo no pixel). Perto das estações o
    resultado passa a respeitar o valor medido; longe delas volta ao campo."""
    G = IMG_SIZE
    campo = np.asarray(campo, np.float32)
    if lam is None:
        lam, _ = pesos_krigagem(px, py, vario)
    if lam is None:
        return campo.copy()
    xi = np.clip(np.round(np.asarray(px)).astype(int), 0, G - 1)
    yi = np.clip(np.round(np.asarray(py)).astype(int), 0, G - 1)
    dentro = (np.asarray(px) >= -0.5) & (np.asarray(px) < G - 0.5) & \
             (np.asarray(py) >= -0.5) & (np.asarray(py) < G - 0.5)
    res = np.where(dentro, np.asarray(obs, float) - campo[yi, xi], 0.0)
    out = campo + (lam.T @ res).reshape(G, G).astype(np.float32)
    if minimo is not None or maximo is not None:
        out = np.clip(out, minimo, maximo)
    return out.astype(np.float32)
