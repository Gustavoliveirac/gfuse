#!/usr/bin/env python
"""
verif.py — regras de verificação do gfuse3 (numpy puro, sem torch):

  ocorrência sem borda   diagnóstico apenas: CSI de P(>0,2) ignorando
                         estações a até 2 km da fronteira prevista; a fração
                         de estações excluídas é reportada junto
  núcleos com tolerância acerto se a estação observou >= tau e o modelo previu
                         o núcleo dentro do raio; alarme falso se previu o
                         núcleo e nenhuma estação dentro do raio observou
                         >= tau/2; entre tau/2 e tau com previsão = neutro
  CSI "melhor"           limiar de probabilidade que maximiza o CSI numa
                         grade de 0,01 a 0,95 (modelos calibrados dão
                         probabilidades baixas para eventos raros)
  mapa ternário          regra de Chow com custos assimétricos + regra
                         espacial (dúvida só perto de pixel chuvoso)
"""
from __future__ import annotations

import numpy as np

from config3 import CUSTO_FA, CUSTO_MISS

GRADE_THR = np.round(np.r_[0.01, 0.02, 0.03, 0.04, np.arange(0.05, 0.951, 0.05)], 2)


# ------------------------------------------------------------------
# contagens por limiar de probabilidade (vetorizadas na grade)
# ------------------------------------------------------------------
def cont_ocorrencia(pmax, pmin, obs, tau=0.2, grade=GRADE_THR):
    """h, m, f por limiar, ignorando estações de borda. pmax/pmin = máximo e
    mínimo de P(>tau) na janela 3x3 da estação; a decisão usa o próprio
    pixel via pmax >= thr só quando a janela não é mista."""
    pmax, pmin = np.asarray(pmax, float), np.asarray(pmin, float)
    o = np.asarray(obs, float) > tau
    t = grade[:, None]
    borda = (pmin[None] < t) & (pmax[None] >= t)
    pred = pmin[None] >= t                       # janela toda acima
    val = ~borda & np.isfinite(pmax)[None]
    h = ((pred & o[None]) & val).sum(1)
    m = ((~pred & o[None]) & val).sum(1)
    f = ((pred & ~o[None]) & val).sum(1)
    return h.astype(float), m.astype(float), f.astype(float)


def cont_nucleo(pviz, obs, obs_viz, tau, grade=GRADE_THR):
    """h, m, f por limiar para um núcleo tau com tolerância espacial.
    pviz = máximo de P(>tau) no raio; obs_viz = máximo observado entre as
    estações verificadoras no raio (inclui a própria)."""
    pviz = np.asarray(pviz, float)
    o = np.asarray(obs, float) >= tau
    fraco = np.asarray(obs_viz, float) < tau / 2
    t = grade[:, None]
    pred = pviz[None] >= t
    val = np.isfinite(pviz)[None]
    h = (pred & o[None] & val).sum(1)
    m = (~pred & o[None] & val).sum(1)
    f = (pred & fraco[None] & val).sum(1)
    return h.astype(float), m.astype(float), f.astype(float)


def csi_melhor(h, m, f):
    if np.max(h + m) == 0:                 # sem eventos observados
        return np.nan, np.nan, 0.0
    den = h + m + f
    csi = np.where(den > 0, h / np.where(den > 0, den, 1), np.nan)
    if not np.isfinite(csi).any():
        return np.nan, np.nan, 0.0
    k = int(np.nanargmax(csi))
    return float(csi[k]), float(GRADE_THR[k]), float(h[k] + m[k])


# ------------------------------------------------------------------
# vizinhança entre estações
# ------------------------------------------------------------------
def max_obs_vizinhas(px, py, obs, raio_px):
    """Máximo observado entre as estações a até raio_px (inclui a própria)."""
    px, py, obs = (np.asarray(a, float) for a in (px, py, obs))
    if len(obs) == 0:
        return obs
    d = np.hypot(px[:, None] - px[None], py[:, None] - py[None])
    return np.where(d <= raio_px, obs[None], -np.inf).max(1)


# ------------------------------------------------------------------
# mapa ternário
# ------------------------------------------------------------------
SECO, DUVIDA, CHUVOSO, ORFAO = 0, 1, 2, 3


def dilatar(mask, r_px):
    """Dilatação binária por disco de raio r_px (sem scipy)."""
    H, W = mask.shape
    out = np.zeros_like(mask, bool)
    pad = np.pad(mask, r_px)
    for dy in range(-r_px, r_px + 1):
        for dx in range(-r_px, r_px + 1):
            if dx * dx + dy * dy <= r_px * r_px:
                out |= pad[r_px + dy:r_px + dy + H, r_px + dx:r_px + dx + W]
    return out


def ternario(p, c, raio_px):
    """Regra de Chow com custos CUSTO_MISS (dizer seco e chover) e CUSTO_FA
    (dizer chuva e não chover); dúvida custa c. Pixels de dúvida longe de
    qualquer pixel chuvoso viram ORFAO (tratados como seco)."""
    p = np.asarray(p, float)
    c_chuva = (1 - p) * CUSTO_FA
    c_seco = p * CUSTO_MISS
    chuv = (c_chuva <= c_seco) & (c_chuva <= c)
    seco = (c_seco < c_chuva) & (c_seco <= c)
    duv = ~chuv & ~seco
    perto = dilatar(chuv, raio_px) if chuv.any() else np.zeros_like(chuv)
    out = np.full(p.shape, SECO, np.int8)
    out[chuv] = CHUVOSO
    out[duv & perto] = DUVIDA
    out[duv & ~perto] = ORFAO
    return out


def dist_sinalizada(mask, px, py, km_por_px):
    """Distância (km) de cada estação à borda do núcleo: positiva dentro
    (profundidade), negativa fora. NaN se não há núcleo."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    if not mask.any():
        return np.full(len(px), np.nan)
    ys, xs = np.nonzero(mask)
    yn, xn = np.nonzero(~mask)
    xi = np.clip(np.round(px).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.round(py).astype(int), 0, mask.shape[0] - 1)
    dentro = mask[yi, xi]
    d_fora = np.hypot(px[:, None] - xs[None], py[:, None] - ys[None]).min(1)
    if len(xn):
        d_dentro = np.hypot(px[:, None] - xn[None], py[:, None] - yn[None]).min(1)
    else:
        d_dentro = np.full(len(px), 99.0)
    return np.where(dentro, d_dentro, -d_fora) * km_por_px
