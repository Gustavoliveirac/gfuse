#!/usr/bin/env python
"""
estacoes.py — base das estações em matrizes (tempo x estação), lida do
processados/estacoes.npz gerado pelo dados.py. Substitui o parquet longo,
que com a série contínua do período chuvoso passaria de 75 milhões de linhas.

Conteúdo do npz:
  stems   (T,)       stems do GOES em ordem temporal
  t_brt   (T,)       horário BRT arredondado a 10 min
  cod, tipo, fonte, px, py   por estação (tipo 0 = 10 min, 1 = horária)
  R10     (T, S0, 6) chuva das estações de 10 min nos 6 passos do stem
                     (passo 5 = janela que termina no último quadro)
  RH      (T, S1)    chuva das horárias na hora que termina no stem

dentro = estação dentro do recorte de 64 px. Só estações de dentro podem
ser alvo ou holdout; as de fora entram apenas como contexto no canal de
IDW, que respeita a distância real até a borda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config3 import EST_NPZ, ESTACOES_EXCLUIDAS, IMG_SIZE, N_FRAMES, QC_APLICAR, QC_NPZ


class Base:
    def __init__(self, arquivo=EST_NPZ, aplicar_qc=QC_APLICAR):
        z = np.load(arquivo, allow_pickle=False)
        self.stems = z["stems"].astype(str)
        self.t = pd.DatetimeIndex(z["t_brt"].astype("datetime64[ns]"))
        self.cod = z["cod"].astype(str)
        self.tipo = z["tipo"].astype(np.int8)
        self.fonte = z["fonte"].astype(str)
        self.px = z["px"].astype(np.float32)
        self.py = z["py"].astype(np.float32)
        self.R10 = z["R10"]
        self.RH = z["RH"]
        self.i0 = np.flatnonzero(self.tipo == 0)
        self.i1 = np.flatnonzero(self.tipo == 1)
        G = IMG_SIZE
        self.dentro = ((self.px >= -0.5) & (self.px < G - 0.5)
                       & (self.py >= -0.5) & (self.py < G - 0.5))
        self.pos = {s: i for i, s in enumerate(self.stems)}
        self._W = {}
        self._desl = {}
        # estações excluídas (config3.ESTACOES_EXCLUIDAS) viram dado ausente
        for j in np.flatnonzero(np.isin(self.cod, list(ESTACOES_EXCLUIDAS))):
            if self.tipo[j] == 0:
                self.R10[:, np.searchsorted(self.i0, j)] = np.nan
            else:
                self.RH[:, np.searchsorted(self.i1, j)] = np.nan
        self.n_qc = 0
        if aplicar_qc and QC_NPZ.exists():
            self._aplicar_qc()

    def _aplicar_qc(self):
        """Registros de chuva suspeita (qc_estacoes.py) viram dado ausente em
        todos os stems que os contêm (passo 5 do próprio stem e passos
        anteriores dos stems seguintes)."""
        z = np.load(QC_NPZ, allow_pickle=False)
        if len(z["stems"]) != self.T or not np.array_equal(z["cod10"], self.cod[self.i0]):
            print("[estacoes] AVISO: qc_estacoes.npz não corresponde à base, ignorado")
            return
        Q = z["suspeito"]                                   # (T, S0) passo 5
        self.n_qc = int(Q.sum())
        for i in range(N_FRAMES):
            k = self.deslocado(-(N_FRAMES - 1 - i) * 10)    # stem do registro
            ok = k >= 0
            m = np.zeros_like(Q)
            m[ok] = Q[k[ok]]
            self.R10[:, :, i][m] = np.nan

    @property
    def T(self):
        return len(self.stems)

    @property
    def S(self):
        return len(self.cod)

    def janelas(self, agg):
        """(T, S) acumulado da janela de agg min terminando no stem; NaN sem
        janela válida. 10 min: válida se completa ou soma 0 com >= 1 leitura;
        horárias só no 60, em stem de hora cheia."""
        if agg in self._W:
            return self._W[agg]
        A = max(1, agg // 10)
        W = np.full((self.T, self.S), np.nan, np.float32)
        x = self.R10[:, :, N_FRAMES - A:]
        n = np.isfinite(x).sum(-1)
        sm = np.nansum(x, -1)
        ok = (n == A) | ((n >= 1) & (sm == 0))
        W[:, self.i0] = np.where(ok, sm, np.nan)
        if agg == 60 and len(self.i1):
            hora = np.asarray(self.t.minute == 0)
            W[np.ix_(hora, self.i1)] = self.RH[hora]
        self._W[agg] = W
        return W

    def deslocado(self, minutos):
        """Índice do stem em t + minutos (ou -1), para todos os stems."""
        if minutos not in self._desl:
            alvo = self.t + pd.Timedelta(minutes=minutos)
            k = self.t.get_indexer(alvo)
            self._desl[minutos] = k.astype(np.int64)
        return self._desl[minutos]

    def chuvoso(self, agg, limiar=0.2):
        """(T,) choveu (> limiar) em alguma estação de dentro na janela."""
        W = self.janelas(agg)[:, self.dentro]
        return np.nan_to_num(W, nan=0.0).max(1) > limiar

    def tem_estacao(self, agg, holdout_mask=None):
        W = self.janelas(agg)
        m = self.dentro if holdout_mask is None else self.dentro & ~holdout_mask
        return np.isfinite(W[:, m]).any(1)
