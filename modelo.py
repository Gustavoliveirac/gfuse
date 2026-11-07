#!/usr/bin/env python
"""
modelo.py — dataset, fonte MW, rede e perdas do gfuse3.

Rede única para todas as variantes: codificador convolucional por quadro +
ConvLSTM (2 camadas) sobre os 6 quadros do GOES, fusão com os canais
ESTÁTICOS da janela e cabeça de HISTOGRAMA (K compartimentos por pixel).
P(>tau) sai da massa acima do limiar; o campo em mm é o valor esperado dos
pontos médios (usado só nas métricas de quantidade).

Canais dinâmicos (por quadro): 8 brutos (C08..C14 + RRQPE) e, na variante
diff, +3 diferenças Tb (C08-C13, C10-C13, C11-C13). Canais estáticos: splat
do contexto (valor log1p + máscara) e log1p do acumulado RRQPE da janela; a
variante interp acrescenta log1p(IDW), OK em log1p e o peso de informação do
kriging (1 junto de estação, 0 sem dado).
"""
from __future__ import annotations

import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import comum as C
from config3 import (BIN_MID, EDGES, MEDIAS_F, GOES_DIR, HID, HIST_LAGS, IMG_SIZE,
                     KM_PER_PX, LAMBDA_SECO, SECO_PASSADO_MIN, SECO_RAIO_PX,
                     SECO_T0,
                     LAMBDA_AMT, LAMBDA_NUC, LAMBDA_VIZ, MOV_FILE, NUCLEOS,
                     MAX_CTX, MAX_HLD, MAX_TGT, MW_DIR, MW_MIN_COBERTURA,
                     N_CH_RAW, N_FRAMES, DIFF_PARES, RRQPE_CH, STATS_F,
                     STEP_MIN, TARGET_FRAC)


N_PRIOR = 3          # rec: P(chuva), log1p(E[mm]) e indicador de prior válido


_MIDS = None


def valores_classe(agg):
    """Valor típico de cada compartimento: média observada (dados.py) ou,
    na falta dela, o ponto médio fixo do config."""
    global _MIDS
    if _MIDS is None:
        try:
            with open(MEDIAS_F) as f:
                _MIDS = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            _MIDS = {}
    return _MIDS.get(agg, BIN_MID[agg])


def n_canais(flags):
    c_dyn = N_CH_RAW + (len(DIFF_PARES) if flags["diff"] else 0)
    if flags.get("s1"):
        return c_dyn, 1                      # só log1p(RRQPE da janela)
    c_stat = (6 + (3 if flags["interp"] else 0)
              + (2 * HIST_LAGS if flags.get("hist") else 0)
              + (N_PRIOR if flags.get("rec") else 0))
    return c_dyn, c_stat


def carregar_stats():
    with open(STATS_F) as f:
        s = json.load(f)
    m = np.asarray(s["mean"], np.float32)[:, None, None]
    d = np.asarray(s["std"], np.float32)[:, None, None]
    d = np.where(d < 1e-3, 1.0, d)
    return m, d


# ============================================================
# Fonte MW (rótulo do pré-treino)
# ============================================================
class FonteMW:
    """Lê os npz do mw_prep.py. mmh = taxa mm/h da meia hora (carimbo =
    início, UTC), NaN fora do sobrevoo. janela_mm devolve o acumulado (mm)
    da janela de agg terminando em t0 BRT, reamostrado 64x64 nearest."""

    def __init__(self, base_dir=None):
        self.dir = Path(base_dir or MW_DIR)
        g = self.dir / "grade.npz"
        self.ok = g.exists() and len(list(self.dir.glob("mw_*.npz"))) > 0
        if not self.ok:
            return
        z = np.load(g)
        lat, lon = z["lat"].astype(float), z["lon"].astype(float)
        glat, glon = C.grade_latlon()
        self.ia = np.clip(np.round((glat - lat[0])
                                   / np.median(np.diff(lat))).astype(int),
                          0, len(lat) - 1)
        self.io = np.clip(np.round((glon - lon[0])
                                   / np.median(np.diff(lon))).astype(int),
                          0, len(lon) - 1)

    @lru_cache(maxsize=8192)
    def _meia_hora(self, stamp_utc):
        p = self.dir / f"mw_{pd.Timestamp(stamp_utc):%Y%m%d%H%M}.npz"
        if not p.exists():
            return None
        try:
            return np.load(p)["mmh"]
        except Exception:
            return None

    def janela_mm(self, t0_brt, agg):
        """Grade (64,64) em mm + máscara de pixels finitos, ou (None, None)."""
        if not self.ok:
            return None, None
        t0u = pd.Timestamp(t0_brt).round("10min") + pd.Timedelta(hours=3)
        if agg == 60:
            if t0u.minute != 0:
                return None, None
            r1 = self._meia_hora(t0u - pd.Timedelta(minutes=60))
            r2 = self._meia_hora(t0u - pd.Timedelta(minutes=30))
            # taxa média das meias horas com sobrevoo x 1 h; basta uma delas
            # (exigir as duas deixava o m60 com ~6% dos stems no pré-treino)
            meias = [r for r in (r1, r2) if r is not None]
            if not meias:
                return None, None
            pilha = np.stack(meias, 0)
            n_ok = np.isfinite(pilha).sum(0)
            mm = np.where(n_ok > 0,
                          np.nansum(pilha, 0) / np.maximum(n_ok, 1), np.nan)
        elif agg == 30:
            if t0u.minute not in (0, 30):
                return None, None
            r = self._meia_hora(t0u - pd.Timedelta(minutes=30))
            if r is None:
                return None, None
            mm = r * 0.5
        else:  # 10 min: meia hora que CONTÉM a janela [t0-10, t0)
            ini = (t0u - pd.Timedelta(minutes=10)).floor("30min")
            r = self._meia_hora(ini)
            if r is None:
                return None, None
            mm = r / 6.0
        g = mm[self.ia, self.io].astype(np.float32)
        mask = np.isfinite(g)
        if mask.mean() < MW_MIN_COBERTURA:
            return None, None
        return np.nan_to_num(g, nan=0.0), mask


class Movimento:
    """Vetor de movimento por stem (px por 10 min) lido de movimento.csv."""

    def __init__(self, arquivo=MOV_FILE):
        self.uv = {}
        self.ok = Path(arquivo).exists()
        if self.ok:
            d = pd.read_csv(arquivo, usecols=["goes_stem", "u", "v"])
            self.uv = {s: (float(u), float(v)) for s, u, v in
                       zip(d["goes_stem"].astype(str), d["u"], d["v"])}

    def __call__(self, stem):
        return self.uv.get(stem, (0.0, 0.0))


def alvo_fixo(cod, seed=42):
    """Divisão alvo/contexto fixa por estação (avaliação): a mesma estação é
    sempre alvo ou sempre contexto, em todos os stems e braços. Isso impede
    que a história ou a recursão carreguem o valor da própria estação-alvo."""
    return (zlib.crc32(f"{cod}|{seed}".encode()) & 1) == 0


# ============================================================
# Montagem do cubo e dos canais estáticos
# ============================================================
_IDX_GOES = None


def indice_goes():
    """Horário UTC arredondado a 10 min -> stem do arquivo (os segundos do
    scan variam, então o nome exato do quadro anterior nem sempre existe)."""
    global _IDX_GOES
    if _IDX_GOES is None:
        _IDX_GOES = {}
        for p in GOES_DIR.glob("*.npy"):
            try:
                t = pd.to_datetime(p.stem, format="%Y%m%d_%H%M%S").round("10min")
                _IDX_GOES.setdefault(t, p.stem)
            except Exception:
                pass
    return _IDX_GOES


def filtro_janela(campo, w, fn):
    if w <= 0:
        return campo
    H, W = campo.shape
    viz = -np.inf if fn is np.max else np.inf
    p = np.pad(campo, w, constant_values=viz)
    lin = fn([p[:, w + d:w + d + W] for d in range(-w, w + 1)], 0)
    return fn([lin[w + d:w + d + H, :] for d in range(-w, w + 1)], 0)


def carregar_cubo(stem, agg, ch_mean, ch_std, usar_diff):
    """(T, C, H, W) normalizado, acumulado RRQPE da janela (mm), e Tb 10,3 µm
    mínima e RRQPE máximo nos quadros usados pela regra do seco (m10: últimos
    SECO_PASSADO_MIN; m30/m60: os 6 quadros). Quadro ausente vira zero no
    cubo e não entra na regra do seco (que então fica mais conservadora)."""
    t0 = pd.to_datetime(stem, format="%Y%m%d_%H%M%S").round("10min")
    idx = indice_goes()
    A = max(1, agg // STEP_MIN)
    n_seco = (SECO_PASSADO_MIN // STEP_MIN + 1) if agg == 10 else N_FRAMES
    quadros, rrq, tbs, rrs = [], [], [], []
    for i in range(N_FRAMES):
        s_i = idx.get(t0 - pd.Timedelta(minutes=(N_FRAMES - 1 - i) * STEP_MIN))
        arr = None
        if s_i is not None:
            try:
                arr = np.load(GOES_DIR / f"{s_i}.npy").astype(np.float32)
            except Exception:
                arr = None
        presente = arr is not None
        if not presente:
            arr = np.zeros((N_CH_RAW, IMG_SIZE, IMG_SIZE), np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if i >= N_FRAMES - A:
            rrq.append(np.clip(arr[RRQPE_CH], 0.0, 200.0))
        if i >= N_FRAMES - n_seco:
            if presente:
                tbs.append(np.nan_to_num(C.rad_para_tb(arr[5], 5), nan=0.0))
                rrs.append(arr[RRQPE_CH])
            else:                       # sem quadro: não dá para afirmar seco
                tbs.append(np.zeros((IMG_SIZE, IMG_SIZE), np.float32))
                rrs.append(np.full((IMG_SIZE, IMG_SIZE), 1.0, np.float32))
        camadas = [arr]
        if usar_diff:
            tb = {c: C.rad_para_tb(arr[c], c) for par in DIFF_PARES for c in par}
            difs = np.stack([np.nan_to_num(tb[a] - tb[b], nan=0.0)
                             for a, b in DIFF_PARES], 0)
            camadas.append(difs)
        x = np.concatenate(camadas, 0)
        x = (x - ch_mean[:len(x)]) / ch_std[:len(x)]
        quadros.append(np.clip(np.nan_to_num(x, nan=0.0, posinf=4.0,
                                             neginf=-4.0), -4.0, 4.0))
    cubo = np.stack(quadros, 0).astype(np.float32)
    rrqpe_mm = (np.mean(rrq, 0) * (A * STEP_MIN / 60.0)).astype(np.float32)
    return cubo, rrqpe_mm, np.min(tbs, 0), np.max(rrs, 0)


def mascara_seco(tbmin, rrmax, px_chuva, py_chuva):
    """Pixels "seco certo": Tb mínima > SECO_T0 e RRQPE zero num raio de
    SECO_RAIO_PX, menos a vizinhança (mesmo raio) de estações com chuva."""
    seco = (filtro_janela(tbmin, SECO_RAIO_PX, np.min) > SECO_T0) & \
           (filtro_janela(rrmax, SECO_RAIO_PX, np.max) <= 0)
    if len(px_chuva):
        ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        d2 = ((xs[None] - np.asarray(px_chuva)[:, None, None]) ** 2
              + (ys[None] - np.asarray(py_chuva)[:, None, None]) ** 2).min(0)
        seco &= d2 > SECO_RAIO_PX ** 2
    return seco


def _splat(px, py, mm):
    G = IMG_SIZE
    v = np.zeros((G, G), np.float32); m = np.zeros((G, G), np.float32)
    for x, y, val in zip(px, py, mm):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < G and 0 <= yi < G:
            v[yi, xi] = max(v[yi, xi], np.log1p(max(val, 0.0)))
            m[yi, xi] = 1.0
    return v, m


def canais_estaticos(px_c, py_c, mm_c, px_e, py_e, mm_e, rrqpe_mm, vario,
                     usar_interp):
    """Ordem dos canais (o braço krig_res depende dela):
      0 splat log1p(mm) das estações de contexto DENTRO do recorte
      1 máscara do splat
      2 log1p(RRQPE da janela)
      3 krigagem de indicador (probabilidade de chuva > 0,2 mm)
      4 log1p(krigagem simples em mm)
      5 peso de informação da krigagem (1 junto a estação, 0 longe)
    As duas krigagens usam todas as estações de contexto, inclusive as de
    fora do recorte, e voltam para seco longe delas. log1p = ln(1 + x) aqui
    só ajusta a escala da entrada; a avaliação é sempre em mm."""
    sv, sm = _splat(px_c, py_c, mm_c)
    px_t = np.r_[px_c, px_e]; py_t = np.r_[py_c, py_e]; mm_t = np.r_[mm_c, mm_e]
    sk, ik, w = C.krigagem_simples(px_t, py_t, mm_t, vario)
    camadas = [sv, sm, np.log1p(np.clip(rrqpe_mm, 0, None)), ik, np.log1p(sk), w]
    if usar_interp:
        idw_mm, _, ok_log, wo = C.campos_interp(px_t, py_t, mm_t, vario)
        camadas += [np.log1p(idw_mm), ok_log.astype(np.float32), wo]
    return np.stack(camadas, 0).astype(np.float32)


CANAL_IK, CANAL_W = 3, 5


# ============================================================
# Datasets
# ============================================================
class DatasetGfuse(Dataset):
    """Amostras rotuladas a partir da base em matrizes (estacoes.Base).
    Alvo e holdout só entre estações de dentro do recorte. Treino: divisão
    alvo/contexto aleatória por stem (ou fixa, quando passada em `alvos`).
    Avaliação: divisão fixa por estação (alvo_fixo), igual para todos os
    braços."""

    def __init__(self, stems, base, agg, flags, holdout, vario, treino=True,
                 eval_seed=42, movimento=None):
        self.stems = list(stems)
        self.base = base
        self.agg = agg
        self.A = max(1, agg // STEP_MIN)
        self.flags = flags
        self.vario = vario
        self.treino = treino
        self.eval_seed = eval_seed
        self.W = base.janelas(agg)
        self.hold = np.array([c in set(holdout) for c in base.cod])
        self.fixo = np.array([alvo_fixo(c, eval_seed) for c in base.cod])
        self.ant = {k: base.deslocado(-k * self.A * STEP_MIN)
                    for k in range(1, HIST_LAGS + 1)}
        self.ch_mean, self.ch_std = carregar_stats()
        self.mov = movimento if movimento is not None else Movimento()

    def __len__(self):
        return len(self.stems)

    def info(self, stem, alvos=None):
        """Índices das estações do stem: liv (dentro, fora do holdout),
        tgt e ctx (divisão de liv), hld (holdout) e ext (fora do recorte)."""
        b = self.base
        i = b.pos[stem]
        w = self.W[i]
        val = np.isfinite(w)
        liv = np.flatnonzero(val & b.dentro & ~self.hold)
        hld = np.flatnonzero(val & b.dentro & self.hold)
        ext = np.flatnonzero(val & ~b.dentro)
        if alvos is not None:
            ehalvo = np.array([b.cod[j] in alvos for j in liv], bool)
        elif self.flags.get("s1") and self.treino:
            ehalvo = np.ones(len(liv), bool)  # s1 não vê estações: todas são alvo
        elif self.treino:
            ehalvo = np.zeros(len(liv), bool)
            k = int(round(len(liv) * TARGET_FRAC))
            ehalvo[np.random.default_rng().permutation(len(liv))[:k]] = True
        else:
            ehalvo = self.fixo[liv]
        return dict(i=i, w=w, liv=liv, tgt=liv[ehalvo], ctx=liv[~ehalvo],
                    hld=hld, ext=ext)

    def pers(self, i, idx):
        j = self.ant[1][i]
        if j < 0:
            return np.full(len(idx), np.nan, np.float32)
        return self.W[j][idx]

    def canais_historia(self, i, ctx):
        G = IMG_SIZE
        u, v = self.mov(self.base.stems[i])
        out = np.zeros((2 * HIST_LAGS, G, G), np.float32)
        for k in range(1, HIST_LAGS + 1):
            j = self.ant[k][i]
            if j < 0:
                continue
            wp = self.W[j][ctx]
            ok = np.isfinite(wp)
            px = self.base.px[ctx][ok] + k * self.A * u
            py = self.base.py[ctx][ok] + k * self.A * v
            out[2 * (k - 1)], out[2 * (k - 1) + 1] = _splat(px, py, wp[ok])
        return out

    def __getitem__(self, idx):
        return self.amostra(self.stems[idx])

    def amostra(self, stem, alvos=None):
        cubo, rrqpe_mm, tbmin, rrmax = carregar_cubo(
            stem, self.agg, self.ch_mean, self.ch_std, self.flags["diff"])
        inf = self.info(stem, alvos)
        b, w, i = self.base, inf["w"], inf["i"]
        ctx, ext = inf["ctx"], inf["ext"]
        # estações que podem informar a perda (nunca o holdout)
        conh = np.r_[ctx, inf["tgt"], ext]
        ch = conh[np.nan_to_num(w[conh]) > 0.2]
        seco = mascara_seco(tbmin, rrmax, b.px[ch], b.py[ch])
        estat = canais_estaticos(b.px[ctx], b.py[ctx], w[ctx],
                                 b.px[ext], b.py[ext], w[ext], rrqpe_mm,
                                 self.vario, self.flags["interp"])
        if self.flags.get("hist"):
            estat = np.concatenate([estat, self.canais_historia(i, ctx)], 0)
        if self.flags.get("s1"):
            estat = estat[2:3]                    # só o RRQPE (nenhuma estação)
        G = IMG_SIZE

        def bloco(idx, MAXN):
            pos = np.zeros((MAXN, 2), np.float32)
            mm = np.zeros(MAXN, np.float32)
            ok = np.zeros(MAXN, bool)
            pe = np.zeros(MAXN, np.float32)
            pev = np.zeros(MAXN, bool)
            n = min(len(idx), MAXN)
            if n:
                idx = idx[:n]
                pos[:n, 0] = b.px[idx] / G
                pos[:n, 1] = b.py[idx] / G
                mm[:n] = w[idx]
                ok[:n] = True
                p = self.pers(i, idx)
                pev[:n] = np.isfinite(p)
                pe[:n] = np.nan_to_num(p)
            return pos, mm, ok, pe, pev

        t_pos, t_mm, t_ok, t_pe, t_pev = bloco(inf["tgt"], MAX_TGT)
        h_pos, h_mm, h_ok, h_pe, h_pev = bloco(inf["hld"], MAX_HLD)
        # contexto completo (dentro + fora do recorte): usado na avaliação
        # pelas referências IDW/OK e pela distância à estação mais próxima
        todos = np.r_[ctx, ext]
        ctx_arr = np.zeros((MAX_CTX, 3), np.float32)
        n_ctx = min(len(todos), MAX_CTX)
        if n_ctx:
            ctx_arr[:n_ctx] = np.stack([b.px[todos], b.py[todos], w[todos]], 1)[:n_ctx]
        u, v = self.mov(stem)
        dentro_val = np.isfinite(w) & b.dentro
        chuvoso = bool((np.nan_to_num(w[dentro_val]) > 0.2).any())
        return {"cubo": torch.from_numpy(cubo),
                "estat": torch.from_numpy(estat),
                "rrqpe_mm": torch.from_numpy(rrqpe_mm),
                "uv": torch.tensor([u, v], dtype=torch.float32),
                "chuvoso": torch.tensor(chuvoso),
                "seco": torch.from_numpy(seco),
                "ctx": torch.from_numpy(ctx_arr),
                "n_ctx": torch.tensor(n_ctx),
                "tgt_pos": torch.from_numpy(t_pos),
                "tgt_mm": torch.from_numpy(t_mm),
                "tgt_ok": torch.from_numpy(t_ok),
                "tgt_pers": torch.from_numpy(t_pe),
                "tgt_pers_ok": torch.from_numpy(t_pev),
                "hld_pos": torch.from_numpy(h_pos),
                "hld_mm": torch.from_numpy(h_mm),
                "hld_ok": torch.from_numpy(h_ok),
                "hld_pers": torch.from_numpy(h_pe),
                "hld_pers_ok": torch.from_numpy(h_pev)}


def cadeias_10min(stems, ds=None):
    """Quebra uma lista de stems em cadeias de passos de 10 min contíguos."""
    ts = sorted(stems, key=lambda s: C.stem_para_brt(s))
    cad, atual, t_ant = [], [], None
    for s in ts:
        t = C.stem_para_brt(s).round("10min")
        if t_ant is not None and t - t_ant != pd.Timedelta(minutes=10):
            cad.append(atual); atual = []
        atual.append(s); t_ant = t
    if atual:
        cad.append(atual)
    return cad


class DatasetSeq(Dataset):
    """rec: sequências de REC_PASSOS stems consecutivos de 10 min, com a
    mesma divisão alvo/contexto em todos os passos (sorteada por sequência
    no treino; fixa por estação na validação)."""

    def __init__(self, base_ds, seqs):
        self.ds = base_ds
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        if self.ds.treino:
            liv = set()
            for s in seq:
                liv |= set(self.ds.base.cod[self.ds.info(s)["liv"]])
            cods = sorted(liv)
            np.random.default_rng().shuffle(cods)
            alvos = set(cods[:int(round(len(cods) * TARGET_FRAC))])
        else:
            alvos = None
        itens = [self.ds.amostra(s, alvos) for s in seq]
        return {k: torch.stack([it[k] for it in itens], 0) for k in itens[0]}


class DatasetMW(Dataset):
    """Pré-treino GOES -> MW: rótulo denso (64x64) do MWprecipitation onde
    houve sobrevoo; contexto = todas as estações válidas fora do holdout."""

    def __init__(self, stems, base, agg, flags, holdout, vario, fonte_mw):
        self.ds = DatasetGfuse(stems, base, agg, flags, holdout, vario,
                               treino=False, eval_seed=0)
        self.mw = fonte_mw
        self.agg = agg

    def __len__(self):
        return len(self.ds.stems)

    def __getitem__(self, idx):
        stem = self.ds.stems[idx]
        d, b = self.ds, self.ds.base
        cubo, rrqpe_mm, _, _ = carregar_cubo(stem, d.agg, d.ch_mean, d.ch_std,
                                             d.flags["diff"])
        inf = d.info(stem)
        w, liv, ext = inf["w"], inf["liv"], inf["ext"]
        estat = canais_estaticos(b.px[liv], b.py[liv], w[liv],
                                 b.px[ext], b.py[ext], w[ext], rrqpe_mm,
                                 d.vario, d.flags["interp"])
        if d.flags.get("s1"):
            estat = estat[2:3]                    # s1: só o RRQPE
        mw_mm, mask = self.mw.janela_mm(C.stem_para_brt(stem), self.agg)
        if mw_mm is None:
            mw_mm = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            mask = np.zeros((IMG_SIZE, IMG_SIZE), bool)
        return {"cubo": torch.from_numpy(cubo),
                "estat": torch.from_numpy(estat),
                "mw_mm": torch.from_numpy(mw_mm),
                "mw_ok": torch.from_numpy(mask)}


def filtrar_stems_mw(stems, fonte_mw, agg):
    return [s for s in stems
            if fonte_mw.janela_mm(C.stem_para_brt(s), agg)[0] is not None]


# ============================================================
# Rede
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2,
                              padding_mode="replicate")
        nn.init.xavier_uniform_(self.conv.weight, gain=0.5)

    def forward(self, x, h, c):
        i, f, g, o = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init(self, B, H, W, dev):
        z = torch.zeros(B, self.hid, H, W, device=dev)
        return z, z.clone()


def _bloco(cin, cout, dil=1):
    return nn.Sequential(
        # preenchimento por replicação: com zeros a rede via uma "borda"
        # artificial e tendia a pôr chuva nela
        nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil, bias=False,
                  padding_mode="replicate"),
        nn.GroupNorm(min(8, cout), cout), nn.GELU())


class RedeGfuse(nn.Module):
    """hurdle=False: cabeça única de histograma (K classes).
    hurdle=True: cabeça de ocorrência (logit de P>0,2) + cabeça de intensidade
    (K-1 classes condicionadas a chover), que recebe também o mapa de
    ocorrência e três máximos de vizinhança dele (5, 9 e 17 px), sem
    gradiente, como medida de proximidade ao núcleo. Nos dois casos forward
    devolve log-probabilidades (ou logits) das K classes, então a
    decodificação é a mesma."""

    def __init__(self, c_dyn, c_stat, k_bins, hid=HID, hurdle=False, krig=False):
        super().__init__()
        self.hurdle = hurdle
        self.krig = krig
        self.tronco = nn.Sequential(_bloco(c_dyn, hid), _bloco(hid, hid))
        self.lstm1 = ConvLSTMCell(hid, hid)
        self.lstm2 = ConvLSTMCell(hid, hid)
        self.fusao = nn.Sequential(_bloco(hid + c_stat, hid),
                                   _bloco(hid, hid, dil=2),
                                   _bloco(hid, hid, dil=4))
        if hurdle:
            self.cab_oc = nn.Conv2d(hid, 1, 1)
            self.cab_int = nn.Sequential(_bloco(hid + 4, hid),
                                         nn.Conv2d(hid, k_bins - 1, 1))
        else:
            self.cabeca = nn.Conv2d(hid, k_bins, 1)

    def forward(self, cubo, estat, detalhado=False):
        B, T, Ch, H, W = cubo.shape
        h1, c1 = self.lstm1.init(B, H, W, cubo.device)
        h2, c2 = self.lstm2.init(B, H, W, cubo.device)
        for t in range(T):
            x = self.tronco(cubo[:, t])
            h1, c1 = self.lstm1(x, h1, c1)
            h2, c2 = self.lstm2(h1, h2, c2)
        z = self.fusao(torch.cat([h2, estat], 1))
        if not self.hurdle:
            out = self.cabeca(z)                         # (B, K, H, W) logits
            return (out, None, None) if detalhado else out
        lp = self.cab_oc(z)                              # (B, 1, H, W)
        if self.krig:
            # parte da krigagem de indicador, ponderada pelo peso de
            # informação: junto às estações a rede só corrige a krigagem
            ik = estat[:, CANAL_IK:CANAL_IK + 1].clamp(0.02, 0.98)
            wk = estat[:, CANAL_W:CANAL_W + 1]
            lp = lp + wk * torch.log(ik / (1 - ik))
        p = torch.sigmoid(lp).detach()
        prox = [p] + [F.max_pool2d(p, k, 1, k // 2) for k in (5, 9, 17)]
        lq = self.cab_int(torch.cat([z] + prox, 1))      # (B, K-1, H, W)
        logp = torch.cat([F.logsigmoid(-lp),
                          F.logsigmoid(lp) + F.log_softmax(lq, 1)], 1)
        return (logp, lp, lq) if detalhado else logp


class RedeS2(nn.Module):
    """Estágio 2 aprendido: rede convolucional leve (sem sequência temporal)
    que recebe a saída congelada do estágio 1 e a informação das estações
    (splat, krigagens, peso de informação) e devolve o mesmo formato do
    hurdle (ocorrência + intensidade condicional)."""

    def __init__(self, c_in, k_bins, hid=HID):
        super().__init__()
        self.hurdle, self.krig = True, False
        self.corpo = nn.Sequential(_bloco(c_in, hid), _bloco(hid, hid, dil=2),
                                   _bloco(hid, hid, dil=4), _bloco(hid, hid))
        self.cab_oc = nn.Conv2d(hid, 1, 1)
        self.cab_int = nn.Sequential(_bloco(hid + 4, hid), nn.Conv2d(hid, k_bins - 1, 1))

    def forward(self, x, detalhado=False):
        z = self.corpo(x)
        lp = self.cab_oc(z)
        p = torch.sigmoid(lp).detach()
        prox = [p] + [F.max_pool2d(p, k, 1, k // 2) for k in (5, 9, 17)]
        lq = self.cab_int(torch.cat([z] + prox, 1))
        logp = torch.cat([F.logsigmoid(-lp), F.logsigmoid(lp) + F.log_softmax(lq, 1)], 1)
        return (logp, lp, lq) if detalhado else logp


def c_in_s2(agg, rk=False):
    n = 2 + len(NUCLEOS[agg])
    return n + (n if rk else 0) + 6


_VARIO = {}


def _vario(agg):
    if agg not in _VARIO:
        from config3 import PROC
        _VARIO[agg] = C.carregar_vario(agg, PROC)
    return _VARIO[agg]


@torch.no_grad()
def saida_s1_rk(sai1, agg, ctx, n_ctx):
    """s1_rk a partir da saída do s1: krigagem do erro do s1 nas estações de
    contexto (ctx: px, py, mm), para P(chuva), quantidade e cada núcleo.
    Mesmos canais que saida_s1, já corrigidos."""
    v = _vario(agg)
    s_np = sai1.float().cpu().numpy()
    ctx = ctx.cpu().numpy() if hasattr(ctx, "cpu") else np.asarray(ctx)
    n_ctx = n_ctx.cpu().numpy() if hasattr(n_ctx, "cpu") else np.asarray(n_ctx)
    limiares = [EDGES[agg][0], None] + list(NUCLEOS[agg])
    out = np.empty_like(s_np)
    for b in range(s_np.shape[0]):
        c = ctx[b, :int(n_ctx[b])]
        lam, _ = C.pesos_krigagem(c[:, 0], c[:, 1], v)
        for j, tau in enumerate(limiares):
            if tau is None:                                   # quantidade (log1p)
                mm2 = C.krigagem_residuo(np.expm1(s_np[b, j]), c[:, 0], c[:, 1],
                                         c[:, 2], v, 0.0, None, lam)
                out[b, j] = np.log1p(mm2)
            else:
                out[b, j] = C.krigagem_residuo(s_np[b, j], c[:, 0], c[:, 1],
                                               (c[:, 2] > tau).astype(float),
                                               v, 0.0, 1.0, lam)
    return torch.from_numpy(out).to(sai1.device)


@torch.no_grad()
def saida_s1(s1, cubo, estat, agg):
    """Canais do estágio 1 para o estágio 2: P(chuva), log1p(E[mm]) e
    P(>tau) de cada núcleo. O s1 recebe só o cubo e o canal de RRQPE."""
    prob = s1(cubo, estat[:, 2:3]).exp()
    m = torch.tensor(valores_classe(agg), device=cubo.device)[None, :, None, None]
    canais = [prob[:, 1:].sum(1, keepdim=True),
              torch.log1p((prob * m).sum(1, keepdim=True))]
    for tau in NUCLEOS[agg]:
        canais.append(prob[:, EDGES[agg].index(tau) + 1:].sum(1, keepdim=True))
    return torch.cat(canais, 1)


def prever(net, cubo, estat, flags, agg, s1=None, detalhado=False,
           ctx=None, n_ctx=None):
    """Saída de qualquer braço (exceto a recursão do rec). O s2rk precisa das
    estações de contexto (ctx, n_ctx do lote) para montar o s1_rk."""
    if flags.get("s2"):
        sai1 = saida_s1(s1, cubo, estat, agg)
        partes = [sai1]
        if flags.get("rk"):
            partes.append(saida_s1_rk(sai1, agg, ctx, n_ctx))
        x = torch.cat(partes + [estat[:, :6]], 1)
        return net(x, detalhado=detalhado)
    return net(cubo, estat, detalhado=detalhado) if detalhado else net(cubo, estat)


def criar_rede(flags, agg, device):
    """Rede do braço e, para o s2, o s1 congelado do mesmo fold (carregado
    à parte por quem chama, com carregar_s1)."""
    if flags.get("s2"):
        return RedeS2(c_in_s2(agg, flags.get("rk")), len(EDGES[agg]) + 1).to(device)
    c_dyn, c_stat = n_canais(flags)
    return RedeGfuse(c_dyn, c_stat, len(EDGES[agg]) + 1, hurdle=flags["hurdle"],
                     krig=flags["krig"]).to(device)


def carregar_s1(ckpt_dir, agg, fold, device):
    from config3 import ARM_FLAGS
    p = ckpt_dir / f"m{agg}" / f"s1_f{fold}.pt"
    if not p.exists():
        raise SystemExit(f"o s2 precisa do s1 do mesmo fold: {p} não existe")
    s1 = criar_rede(ARM_FLAGS["s1"], agg, device)
    s1.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    for q in s1.parameters():
        q.requires_grad_(False)
    return s1.eval()


# ============================================================
# Classes, perdas e decodificação
# ============================================================
def classes_de_mm(mm, edges_t):
    """classe k = número de limites estritamente abaixo de mm (obs > tau)."""
    return (mm.unsqueeze(-1) > edges_t).sum(-1).long()


def pesos_classes(mm_treino, agg, teto=10.0):
    edges = np.asarray(EDGES[agg])
    cls = (np.asarray(mm_treino)[:, None] > edges[None]).sum(1)
    cont = np.bincount(cls, minlength=len(edges) + 1).astype(float)
    cont = np.maximum(cont, 1.0)
    w = np.clip((cont.max() / cont) ** 0.5, 1.0, teto)
    return torch.tensor(w, dtype=torch.float32)


def _pesos_estacao(batch, ok):
    """Peso de importância de cada estação-alvo = peso da sua amostra."""
    p = batch.get("peso")
    if p is None:
        return torch.ones(int(ok.sum()), device=ok.device)
    return p[:, None].expand_as(ok)[ok]


def _media(v, w):
    return (v * w).sum() / w.sum().clamp(min=1e-8)


def _termo_seco_hist(logits, batch):
    """Pixels "seco certo" (máscara do seco): -log P(classe seca)."""
    seco = batch.get("seco")
    if seco is None or not seco.any():
        return logits.sum() * 0.0
    lp0 = F.log_softmax(logits.float(), 1)[:, 0]               # (B, H, W)
    w = batch.get("peso")
    w = (torch.ones(seco.shape[0], device=seco.device) if w is None else w)
    wpx = w[:, None, None].expand_as(seco)[seco]
    return _media(-lp0[seco], wpx)


def perda_alvos(logits, batch, agg, pesos, device):
    """Entropia cruzada nas estações-alvo (pesos de classe só se
    USAR_PESOS_CLASSE; pesos de importância por amostra), termo de quantidade
    e supervisão do seco nos pixels "seco certo"."""
    edges_t = torch.tensor(EDGES[agg], device=device)
    mids = torch.tensor(valores_classe(agg), device=device)
    B, K, H, W = logits.shape
    pos, mm, ok = batch["tgt_pos"], batch["tgt_mm"], batch["tgt_ok"]
    seco = LAMBDA_SECO * _termo_seco_hist(logits, batch)
    if not ok.any():
        return seco
    xi = (pos[..., 0] * W).round().clamp(0, W - 1).long()
    yi = (pos[..., 1] * H).round().clamp(0, H - 1).long()
    lg = _nas_estacoes(logits, xi, yi)[ok]
    mm_v = mm[ok]
    w = _pesos_estacao(batch, ok)
    cls = classes_de_mm(mm_v, edges_t)
    ce = F.cross_entropy(lg, cls, weight=pesos.to(device), reduction="none")
    perda = _media(ce, w)
    prob = F.softmax(lg.float(), -1)
    mm_pred = (prob * mids).sum(-1)
    umid = mm_v > 1.0
    if umid.any():
        se = (torch.log1p(mm_pred[umid].clamp(min=0)) - torch.log1p(mm_v[umid])) ** 2
        perda = perda + LAMBDA_AMT * _media(se, w[umid])
    return perda + seco


def perda_mw(logits, batch, agg, pesos, device):
    edges_t = torch.tensor(EDGES[agg], device=device)
    mm, ok = batch["mw_mm"], batch["mw_ok"]
    cls = classes_de_mm(mm, edges_t)                # (B, H, W)
    ce = F.cross_entropy(logits, cls, weight=pesos.to(device),
                         reduction="none")
    ok = ok.float()
    return (ce * ok).sum() / ok.sum().clamp(min=1.0)


def _nas_estacoes(campo, xi, yi):
    """campo (B, C, H, W) -> (B, M, C) nos pixels das estações."""
    B = campo.shape[0]
    bidx = torch.arange(B, device=campo.device)[:, None].expand_as(xi)
    return campo[bidx, :, yi, xi]


def media_gaussiana(campo, sigma):
    """Média ponderada gaussiana (normalizada) em volta de cada pixel. Um
    pico isolado pesa pouco, o que impede a rede de "acertar" a tolerância
    espacial com pixels soltos (falha observada com o máximo da janela)."""
    r = int(np.ceil(2 * sigma))
    x = torch.arange(-r, r + 1, device=campo.device, dtype=campo.dtype)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = (g[:, None] * g[None, :]); k = (k / k.sum())[None, None]
    return F.conv2d(F.pad(campo, (r, r, r, r), mode="replicate"), k)


def deslocar(campo, uv, passos=1.0):
    """Advecção por translação: saida(x) = campo(x - passos*uv), com
    interpolação bilinear e zero fora do domínio. campo (B,C,H,W), uv (B,2)
    em px por 10 min."""
    B, Cc, H, W = campo.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=campo.device, dtype=campo.dtype),
                            torch.arange(W, device=campo.device, dtype=campo.dtype),
                            indexing="ij")
    d = uv.to(campo.dtype) * passos
    xsrc = xs[None] - d[:, 0, None, None]
    ysrc = ys[None] - d[:, 1, None, None]
    grade = torch.stack([2 * (xsrc + 0.5) / W - 1, 2 * (ysrc + 0.5) / H - 1], -1)
    return F.grid_sample(campo, grade, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def prior_de_saida(logits, agg, uv, valido):
    """Canais do prior do rec a partir da saída anterior (sem gradiente):
    P(chuva), log1p(E[mm]) e indicador, já advectados 10 min."""
    with torch.no_grad():
        prob = F.softmax(logits.float(), 1)
        mids = torch.tensor(valores_classe(agg), device=logits.device)[None, :, None, None]
        p = prob[:, 1:].sum(1, keepdim=True)
        mm = torch.log1p((prob * mids).sum(1, keepdim=True))
        ind = torch.ones_like(p)
        pr = deslocar(torch.cat([p, mm, ind], 1), uv)
        return pr * valido.view(-1, 1, 1, 1).to(pr.dtype)


def _max_vizinhas(pos, mm, ok, r_px):
    """Máximo observado entre as estações-alvo da mesma amostra a até r_px
    (inclui a própria). pos em fração do domínio."""
    p = pos * IMG_SIZE
    d = torch.cdist(p, p)                                     # (B, M, M)
    viz = (d <= r_px) & ok[:, None, :]
    v = torch.where(viz, mm[:, None, :], torch.full_like(d, -1.0))
    return v.max(-1).values


def perda_hurdle(saida, batch, agg, device):
    """Ocorrência: entropia cruzada binária sem pesos de classe (pesos de
    importância por amostra). Tolerância espacial nos DOIS sentidos, com
    média gaussiana: estação com chuva puxa a probabilidade da vizinhança
    para cima; estação cuja vizinhança (estações-alvo a até ~4 km) está
    toda seca puxa para baixo. O mesmo vale para os núcleos (obs >= tau
    puxa P(>tau) para cima; vizinhança abaixo de tau/2 puxa para baixo).
    Intensidade: RPS no histograma condicional + MSE log1p (> 1 mm).
    Supervisão do seco nos pixels "seco certo"."""
    logp, lp, lq = saida
    edges_t = torch.tensor(EDGES[agg], device=device)
    mids = torch.tensor(valores_classe(agg)[1:], device=device)
    B, K, H, W = logp.shape
    pos, mm, ok = batch["tgt_pos"], batch["tgt_mm"], batch["tgt_ok"]
    seco = batch.get("seco")
    perda = lp.sum() * 0.0
    if seco is not None and seco.any():
        wb = batch.get("peso", torch.ones(B, device=device))
        wpx = wb[:, None, None].expand_as(seco)[seco]
        alvo0 = lp[:, 0][seco]
        perda = perda + LAMBDA_SECO * _media(
            F.binary_cross_entropy_with_logits(alvo0, torch.zeros_like(alvo0),
                                               reduction="none"), wpx)
    if not ok.any():
        return perda
    xi = (pos[..., 0] * W).round().clamp(0, W - 1).long()
    yi = (pos[..., 1] * H).round().clamp(0, H - 1).long()
    w = _pesos_estacao(batch, ok)
    mm_v = mm[ok]
    umid = mm_v > edges_t[0]
    ov = _max_vizinhas(pos, mm, ok, 2.0)[ok]

    lp_s = _nas_estacoes(lp, xi, yi)[..., 0][ok]
    perda = perda + _media(F.binary_cross_entropy_with_logits(
        lp_s, umid.float(), reduction="none"), w)
    pv = _nas_estacoes(media_gaussiana(torch.sigmoid(lp), 1.0), xi, yi)[..., 0][ok]
    pv = pv.clamp(1e-6, 1 - 1e-6)
    seca_viz = ov <= edges_t[0]
    termo = torch.where(umid, -torch.log(pv),
                        torch.where(seca_viz, -torch.log(1 - pv), torch.zeros_like(pv)))
    perda = perda + LAMBDA_VIZ * _media(termo, w)

    if umid.any():
        q = F.softmax(_nas_estacoes(lq, xi, yi)[ok][umid].float(), -1)
        cls_w = classes_de_mm(mm_v[umid], edges_t) - 1
        cdf_p = q.cumsum(-1)[:, :-1]
        j = torch.arange(K - 2, device=device)[None]
        cdf_o = (cls_w[:, None] <= j).float()
        rps = ((cdf_p - cdf_o) ** 2).sum(-1) / max(K - 2, 1)
        perda = perda + _media(rps, w[umid])
        e_mm = (q * mids).sum(-1)
        forte = mm_v[umid] > 1.0
        if forte.any():
            se = (torch.log1p(e_mm[forte]) - torch.log1p(mm_v[umid][forte])) ** 2
            perda = perda + LAMBDA_AMT * _media(se, w[umid][forte])

    prob = logp.exp()
    for tau in NUCLEOS[agg]:
        jt = EDGES[agg].index(tau)
        p_tau = prob[:, jt + 1:].sum(1, keepdim=True)
        pn = _nas_estacoes(media_gaussiana(p_tau, 1.5), xi, yi)[..., 0][ok]
        pn = pn.clamp(1e-6, 1 - 1e-6)
        sobe = mm_v >= tau
        desce = ov < tau / 2
        if not (sobe.any() or desce.any()):
            continue
        termo = torch.where(sobe, -torch.log(pn),
                            torch.where(desce, -torch.log(1 - pn), torch.zeros_like(pn)))
        perda = perda + LAMBDA_NUC * _media(termo, w)
    return perda


@torch.no_grad()
def decodificar(logits, agg):
    """probs, campo mm (E[mm]) e dict tau -> P(>tau), tudo numpy."""
    prob = F.softmax(logits.float(), 1).cpu().numpy()   # (B, K, H, W)
    mids = np.asarray(valores_classe(agg), np.float32)[None, :, None, None]
    mm = (prob * mids).sum(1)
    edges = EDGES[agg]
    p_tau = {}
    for j, tau in enumerate(edges):
        p_tau[tau] = prob[:, j + 1:].sum(1)
    return mm, p_tau
