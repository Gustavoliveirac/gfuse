#!/usr/bin/env python
"""
avalia.py — roda o checkpoint de UM braço/fold na validação e grava os
pares (uma linha por estação-alvo ou de holdout por stem).

Divisão alvo/contexto fixa por estação, igual para todos os braços.

Colunas principais: obs, pers, dist_ctx, n_ctx, px, py, mm_w*, p{tau}_w*,
idw/ok/rrq (baselines nos mesmos pontos), ter_* (mapa ternário) e
sd_nucleo_km.

Braço rec: roda a recursão em 10 min sobre a validação em ordem
cronológica (o prior é zerado quando há buraco na série). Em 30 e 60 min o
volume é a soma dos valores esperados das análises de 10 min; a chuva na
janela (> 0,2 mm) usa a maior P(chuva) entre os passos; os demais limiares
viram indicador (soma > tau), porque a soma não tem distribuição própria.

No m10, para todos os braços:
  <braço>_filtro  remove células (P(chuva) >= 0,5, conectividade 8) sem
                  antecessor no quadro de 10 min antes (advectado e
                  dilatado 1 px), exceto se atingirem P >= 0,7. Só usa o
                  passado, então vale em tempo real.
  coerência       por stem: diferença média |P(t) - P(t-10) advectado|,
                  número de células e fração com antecessor
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
# os workers do DataLoader passam os tensores por descritores de arquivo;
# em avaliações longas o limite de descritores abertos do sistema estoura
# ("received 0 items of ancdata"). Com "file_system" isso não ocorre.
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader

import comum as C
import modelo as M
from estacoes import Base
import verif as V
from config3 import (ARM_FLAGS, BATCH, C_DUVIDA, C_REF, CKPT_DIR, EDGES, FILTRO_ATIVO, MARGEM_KM,
                     FILTRO_P, FILTRO_P_FORTE, IMG_SIZE, KM_PER_PX, MODOS,
                     NUCLEOS, NUM_WORKERS, PARES_DIR, PROC, RAIOS_KM,
                     RES, TOL_WINS, VERSAO)

RAIOS_EXTRA = [4, 6]           # janelas extras (px) dos núcleos: 8 e 12 km

try:
    from scipy.ndimage import label as _label
except Exception:              # rotulação simples, se não houver scipy
    _label = None


def rotular(mask):
    if _label is not None:
        return _label(mask, structure=np.ones((3, 3)))
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1; pilha = [(y0, x0)]; lab[y0, x0] = n
        while pilha:
            y, x = pilha.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n; pilha.append((yy, xx))
    return lab, n


def deslocar_np(campo, u, v):
    """Translação inteira (arredondada) com zero fora do domínio."""
    dx, dy = int(round(u)), int(round(v))
    out = np.zeros_like(campo)
    H, W = campo.shape
    ys = slice(max(0, dy), min(H, H + dy)); yd = slice(max(0, -dy), min(H, H - dy))
    xs = slice(max(0, dx), min(W, W + dx)); xd = slice(max(0, -dx), min(W, W - dx))
    out[ys, xs] = campo[yd, xd]
    return out


def bloco_mais_chuvoso(stems, flag, n):
    """Trecho contíguo (em ordem temporal) de n stems com mais stems chuvosos;
    usado só no modo rápido, para a validação curta não cair num período seco."""
    if len(stems) <= n:
        return stems
    c = np.r_[0, np.cumsum(np.asarray(flag, int))]
    k = int(np.argmax(c[n:] - c[:-n]))
    return stems[k:k + n]


# ------------------------------------------------------------------
# linhas de pares
# ------------------------------------------------------------------
def estacoes_de_amostra(am):
    out = {}
    G = IMG_SIZE
    for lado, pre in (("alvo", "tgt"), ("holdout", "hld")):
        ok = np.asarray(am[f"{pre}_ok"], bool)
        if not ok.any():
            continue
        pos = np.asarray(am[f"{pre}_pos"])[ok]
        pers = np.asarray(am[f"{pre}_pers"])[ok]
        pok = np.asarray(am[f"{pre}_pers_ok"], bool)[ok]
        out[lado] = (pos[:, 0] * G, pos[:, 1] * G,
                     np.asarray(am[f"{pre}_mm"])[ok],
                     np.where(pok, pers, np.nan))
    n = int(am["n_ctx"])
    return out, np.asarray(am["ctx"])[:n]


def estacoes_de_info(ds, stem):
    inf = ds.info(stem)
    b, w, i = ds.base, inf["w"], inf["i"]
    out = {}
    for lado, idx in (("alvo", inf["tgt"]), ("holdout", inf["hld"])):
        if len(idx):
            out[lado] = (b.px[idx], b.py[idx], w[idx], ds.pers(i, idx))
    todos = np.r_[inf["ctx"], inf["ext"]]
    ctx = np.stack([b.px[todos], b.py[todos], w[todos]], 1).astype(np.float32)
    return out, ctx


def linhas_stem(stem, est, ctx, campos_mm, p_tau, agg, ternarios=None,
                nucleo=None, chuvoso=None):
    taus = EDGES[agg]
    nuc = NUCLEOS[agg]
    linhas = []
    for lado, (px, py, obs, pers) in est.items():
        if len(ctx):
            dist = np.hypot(px[:, None] - ctx[None, :, 0],
                            py[:, None] - ctx[None, :, 1]).min(1) * KM_PER_PX
        else:
            dist = np.full(len(px), np.nan)
        m_px = MARGEM_KM / KM_PER_PX
        margem = ((np.asarray(px) < m_px - 0.5) | (np.asarray(px) > IMG_SIZE - 0.5 - m_px)
                  | (np.asarray(py) < m_px - 0.5) | (np.asarray(py) > IMG_SIZE - 0.5 - m_px))
        cols = dict(goes_stem=stem, lado=lado, obs=obs, pers=pers,
                    dist_ctx=dist, n_ctx=len(ctx), stem_chuvoso=bool(chuvoso),
                    margem=margem,
                    px=np.asarray(px, np.float32), py=np.asarray(py, np.float32))
        for nome, campo in campos_mm.items():
            for w in TOL_WINS + RAIOS_EXTRA:
                cols[f"{nome}_w{w}"] = C.amostra_com_tolerancia(campo, px, py, w)
            cols[f"{nome}_min1"] = C.amostra_com_tolerancia(campo, px, py, 1,
                                                            fn=np.min)
        for tau in taus:
            if tau not in p_tau:
                continue
            for w in TOL_WINS + (RAIOS_EXTRA if tau in nuc else []):
                cols[f"p{tau:g}_w{w}"] = C.amostra_com_tolerancia(
                    p_tau[tau], px, py, w)
        if taus[0] in p_tau:
            cols[f"pmin{taus[0]:g}_w1"] = C.amostra_com_tolerancia(
                p_tau[taus[0]], px, py, 1, fn=np.min)
        if ternarios is not None:
            for (c, r_km), t in ternarios.items():
                cols[f"ter_c{c:g}_R{r_km}"] = C.amostra_com_tolerancia(
                    t, px, py, 0).astype(np.int8)
            cols["sd_nucleo_km"] = V.dist_sinalizada(nucleo, px, py, KM_PER_PX)
        linhas.append(pd.DataFrame(cols))
    return linhas


def ternario_e_areas(stem, p_chuva, areas):
    ternarios = {}
    for c in C_DUVIDA:
        for r_km in RAIOS_KM:
            t = V.ternario(p_chuva, c, int(round(r_km / KM_PER_PX)))
            ternarios[(c, r_km)] = t
            areas.append(dict(goes_stem=stem, c=c, R_km=r_km,
                              frac_chuvoso=float((t == V.CHUVOSO).mean()),
                              frac_duvida=float((t == V.DUVIDA).mean()),
                              frac_orfao=float((t == V.ORFAO).mean())))
    return ternarios, ternarios[(C_REF, RAIOS_KM[0])] == V.CHUVOSO


THR = 0.5                   # limiar de P(chuva) do braço (lido do checkpoint)


def campo_det(mm, p, thr=None):
    """Produto determinístico: chuva esperada CONDICIONADA a chover
    (E[mm]/P) onde P(chuva) >= limiar do braço, zero no resto. É o mapa que
    se entrega; o valor esperado incondicional (mm) suaviza os picos."""
    thr = THR if thr is None else thr
    p = np.asarray(p, np.float32)
    return np.where(p >= thr, np.asarray(mm) / np.maximum(p, 1e-6), 0.0).astype(np.float32)


@torch.no_grad()
def limiar_rk(net, flags, agg, fold, modo, device):
    """Limiar de P(chuva) do produto <braço>_rk (s1 + krigagem do erro), que
    maximiza o CSI nas estações-alvo de uma amostra estratificada do bloco de
    TREINO do fold (divisão fixa por estação)."""
    import treino as T
    from config3 import N_STEMS_LIMIAR
    prep = T.preparar(agg, fold, modo)
    pos = prep["base"].pos
    stems = T.amostra_estratificada(prep["tre"], [prep["chuv"][pos[s]] for s in prep["tre"]],
                                    N_STEMS_LIMIAR)
    # dataset com os 6 canais estáticos (saida_s1 pega dele só o RRQPE)
    ds = M.DatasetGfuse(stems, prep["base"], agg, dict(flags, s1=False, mw=False),
                        prep["holdout"], prep["vario"], treino=False,
                        movimento=M.Movimento())
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    h = m = f = np.zeros(len(V.GRADE_THR))
    G = IMG_SIZE
    for batch in dl:
        sai1 = M.saida_s1(net, batch["cubo"].to(device), batch["estat"].to(device), agg)
        p2 = M.saida_s1_rk(sai1, agg, batch["ctx"], batch["n_ctx"])[:, 0].cpu().numpy()
        pos_t = batch["tgt_pos"].numpy(); mm = batch["tgt_mm"].numpy()
        ok = batch["tgt_ok"].numpy()
        for b in range(len(p2)):
            o = ok[b]
            if not o.any():
                continue
            v = C.amostra_com_tolerancia(p2[b], pos_t[b, o, 0] * G, pos_t[b, o, 1] * G, 0)
            ob = mm[b, o] > EDGES[agg][0]
            pr = v[None] >= V.GRADE_THR[:, None]
            h = h + (pr & ob).sum(1); m = m + (~pr & ob).sum(1); f = f + (pr & ~ob).sum(1)
    _, thr, _ = V.csi_melhor(h.astype(float), m.astype(float), f.astype(float))
    return float(thr) if np.isfinite(thr) else 0.5


def estat_area(stem, p_chuva, ds, lista):
    """Fração do domínio com P(chuva) >= limiar (no todo, na faixa de 5 px da
    borda e no interior) e fração das estações dentro do recorte com chuva."""
    i = ds.base.pos[stem]
    w = ds.W[i][ds.base.dentro]
    w = w[np.isfinite(w)]
    G = IMG_SIZE; b = 5
    borda = np.ones((G, G), bool); borda[b:G - b, b:G - b] = False
    pr = np.asarray(p_chuva) >= THR
    lista.append(dict(goes_stem=stem, frac_prevista=float(pr.mean()),
                      frac_prevista_borda=float(pr[borda].mean()),
                      frac_prevista_interior=float(pr[~borda].mean()),
                      frac_estacoes_chuva=float((w > 0.2).mean()) if len(w) else np.nan,
                      n_estacoes=len(w)))


# ------------------------------------------------------------------
# filtro de células e coerência (m10)
# ------------------------------------------------------------------
def coerencia_e_filtro(stems, campos, mov):
    """campos: stem -> dict(p02, ...) em ordem cronológica.
    Devolve (tabela de coerência por stem, máscaras removidas por stem)."""
    por_t = {C.stem_para_brt(s).round("10min"): s for s in stems}
    coer, removidos = [], {}
    for s in stems:
        p = campos[s]["p02"].astype(np.float32)
        lab, n = rotular(p >= FILTRO_P)
        prev = por_t.get(C.stem_para_brt(s).round("10min") - pd.Timedelta(minutes=10))
        u, v = mov(s)
        rem = np.zeros(p.shape, bool)
        linha = dict(goes_stem=s, n_celulas=n, flicker=np.nan,
                     frac_com_antecessor=np.nan, n_removidas=0)
        if prev is not None:
            adv = deslocar_np(campos[prev]["p02"].astype(np.float32), u, v)
            linha["flicker"] = float(np.abs(p - adv).mean())
            m_prev = V.dilatar(adv >= FILTRO_P, 1)
            com = 0
            for k in range(1, n + 1):
                cel = lab == k
                if (cel & m_prev).any():
                    com += 1
                elif p[cel].max() < FILTRO_P_FORTE:
                    rem |= cel
                    linha["n_removidas"] += 1
            linha["frac_com_antecessor"] = com / n if n else np.nan
        coer.append(linha)
        removidos[s] = rem
    return pd.DataFrame(coer), removidos


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=int, required=True, choices=[10, 30, 60])
    ap.add_argument("--arm", required=True, choices=list(ARM_FLAGS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--modo", default="rapido", choices=list(MODOS))
    args = ap.parse_args()
    flags = ARM_FLAGS[args.arm]
    agg_ck = 10 if flags["rec"] else args.agg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = CKPT_DIR / f"m{agg_ck}" / f"{args.arm}_f{args.fold}.pt"
    pasta = PARES_DIR / f"m{args.agg}"
    pasta.mkdir(parents=True, exist_ok=True)
    saida = pasta / f"{args.arm}_f{args.fold}.parquet"
    rk_ok = True
    if flags["s1"]:                  # o produto <braço>_rk precisa do limiar próprio
        f_rk = pasta / f"{args.arm}_rk_f{args.fold}.parquet"
        try:
            import pyarrow.parquet as pq
            rk_ok = f_rk.exists() and "thr_rk" in pq.read_schema(f_rk).names
        except Exception:
            rk_ok = False
    if (saida.exists() and ck.exists() and rk_ok
            and saida.stat().st_mtime > ck.stat().st_mtime):
        print(f"[avalia m{args.agg} {args.arm} fold{args.fold}] pares já "
              f"gerados para este checkpoint, pulando")
        return
    if not ck.exists():
        raise SystemExit(f"checkpoint ausente: {ck} — rode treino.py antes")

    base = Base()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    mov = M.Movimento()
    cfg = MODOS[args.modo]

    net = M.criar_rede(flags, agg_ck, device)
    net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    net.eval()
    s1 = M.carregar_s1(CKPT_DIR, args.agg, args.fold, device) if flags["s2"] else None
    global THR
    THR_RK = 0.5
    try:
        with open(ck.with_suffix(".json")) as fj:
            THR = float(json.load(fj).get("thr_chuva", 0.5))
    except Exception:
        THR = 0.5
    print(f"[avalia] limiar de P(chuva) do braço: {THR:.2f}")
    THR_RK = THR
    if flags["s1"]:
        THR_RK = limiar_rk(net, flags, args.agg, args.fold, args.modo, device)
        print(f"[avalia] limiar de P(chuva) do {args.arm}_rk (bloco de treino): {THR_RK:.2f}")
    lin_rk = []                                # produto s1_rk (estágio 2 analítico)

    def val_de(agg):
        stems = C.stems_alinhados(list(base.stems), agg)
        fd = C.folds_temporais(stems)[args.fold]
        tem = base.tem_estacao(agg)
        return [s for s in fd["val"] if tem[base.pos[s]]]
    chuv = {a: base.chuvoso(a) for a in (10, args.agg)}

    linhas, areas, campos10, coer, area_st = [], [], {}, pd.DataFrame(), []
    taus10 = EDGES[10]

    if flags["rec"]:
        # ---------------- recursão em 10 min ----------------
        val10 = sorted(val_de(10), key=C.stem_para_brt)
        if cfg["max_stems_val"]:
            val10 = bloco_mais_chuvoso(val10, [chuv[10][base.pos[x]] for x in val10],
                                       cfg["max_stems_val"] * max(1, args.agg // 10))
        ds10 = M.DatasetGfuse(val10, base, 10, flags, holdout,
                              C.carregar_vario(10, PROC), treino=False,
                              movimento=mov)
        cadeias = M.cadeias_10min(val10, ds10)
        print(f"[avalia m{args.agg} rec fold{args.fold}] recursão em "
              f"{len(val10)} stems de 10 min ({len(cadeias)} cadeias)")
        A = max(1, args.agg // 10)
        ds_agg = (M.DatasetGfuse([], base, args.agg, flags, holdout,
                                 C.carregar_vario(args.agg, PROC), treino=False,
                                 movimento=mov) if A > 1 else None)
        alinh = set(C.stems_alinhados(val10, args.agg)) if A > 1 else set()
        with torch.no_grad():
            for cad in cadeias:
                prior = torch.zeros(1, M.N_PRIOR, IMG_SIZE, IMG_SIZE, device=device)
                hist_mm, hist_p, hist_d = [], [], []
                for s in cad:
                    am = ds10.amostra(s)
                    estat = torch.cat([am["estat"][None].to(device), prior], 1)
                    logits = net(am["cubo"][None].to(device), estat)
                    mm_f, p_tau = M.decodificar(logits, 10)
                    prior = M.prior_de_saida(logits, 10, am["uv"][None].to(device),
                                             torch.ones(1, device=device))
                    mm1 = mm_f[0]; hist_mm.append(mm1)
                    hist_p.append(p_tau[taus10[0]][0])
                    hist_d.append(campo_det(mm1, p_tau[taus10[0]][0]))
                    if A == 1:
                        pt = {t: p_tau[t][0] for t in taus10}
                        est, ctx = estacoes_de_amostra(am)
                        campos10[s] = dict(p02=pt[taus10[0]].astype(np.float16),
                                           p_nuc=pt[NUCLEOS[10][0]].astype(np.float16),
                                           mm=mm1.astype(np.float16),
                                           est=est, ctx=ctx,
                                           chuvoso=bool(am["chuvoso"]))
                        ter, nuc = ternario_e_areas(s, pt[taus10[0]], areas)
                        estat_area(s, pt[taus10[0]], ds10, area_st)
                        linhas += linhas_stem(s, est, ctx,
                                              {"mm": mm1, "det": hist_d[-1]}, pt, 10,
                                              ter, nuc, bool(am["chuvoso"]))
                    elif s in alinh and len(hist_mm) >= A:
                        # volume: soma dos valores esperados; chuva > 0,2 na
                        # janela: maior P(chuva) entre os passos (somar valores
                        # esperados espalharia garoa por todo o domínio);
                        # demais limiares: indicador da soma
                        soma = np.sum(hist_mm[-A:], 0)
                        pt = {t: (soma > t).astype(np.float32)
                              for t in EDGES[args.agg][1:]}
                        pt[EDGES[args.agg][0]] = np.max(hist_p[-A:], 0)
                        estat_area(s, pt[EDGES[args.agg][0]], ds_agg, area_st)
                        est, ctx = estacoes_de_info(ds_agg, s)
                        linhas += linhas_stem(s, est, ctx,
                                              {"mm": soma, "det": np.sum(hist_d[-A:], 0)}, pt,
                                              args.agg,
                                              chuvoso=chuv[args.agg][base.pos[s]])
    else:
        val = val_de(args.agg)
        if cfg["max_stems_val"] and len(val) > cfg["max_stems_val"]:
            if args.agg == 10:                       # contíguo, p/ coerência
                val = sorted(val, key=C.stem_para_brt)
                val = bloco_mais_chuvoso(val, [chuv[10][base.pos[x]] for x in val],
                                         cfg["max_stems_val"])
            else:
                val = val[:: max(1, len(val) // cfg["max_stems_val"])]
        vario = C.carregar_vario(args.agg, PROC)
        ds = M.DatasetGfuse(val, base, args.agg, flags, holdout, vario,
                            treino=False, movimento=mov)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS)
        print(f"[avalia m{args.agg} {args.arm} fold{args.fold}] "
              f"{len(val)} stems de validação")
        k = 0
        with torch.no_grad():
            for batch in dl:
                cubo = batch["cubo"].to(device); estat = batch["estat"].to(device)
                mm_f, p_tau = M.decodificar(
                    M.prever(net, cubo, estat, flags, args.agg, s1,
                             ctx=batch["ctx"], n_ctx=batch["n_ctx"]), args.agg)
                for b in range(cubo.shape[0]):
                    s = val[k]; k += 1
                    am = {kk: vv[b] for kk, vv in batch.items()}
                    est, ctx = estacoes_de_amostra(am)
                    idw_f, ok_f, _, _ = C.campos_interp(ctx[:, 0], ctx[:, 1],
                                                        ctx[:, 2], vario)
                    sk_f, _, _ = C.krigagem_simples(ctx[:, 0], ctx[:, 1], ctx[:, 2], vario)
                    pt = {t: p_tau[t][b] for t in EDGES[args.agg]}
                    campos = {"mm": mm_f[b], "det": campo_det(mm_f[b], pt[EDGES[args.agg][0]]),
                              "idw": idw_f, "ok": ok_f, "sk": sk_f,
                              "rrq": am["rrqpe_mm"].numpy()}
                    estat_area(s, pt[EDGES[args.agg][0]], ds, area_st)
                    ter, nuc = ternario_e_areas(s, pt[EDGES[args.agg][0]], areas)
                    linhas += linhas_stem(s, est, ctx, campos, pt, args.agg,
                                          ter, nuc, bool(am["chuvoso"]))
                    if flags["s1"]:
                        # estágio 2 analítico: krigagem do erro do s1 nas
                        # estações de contexto (mm e cada limiar)
                        lam, _ = C.pesos_krigagem(ctx[:, 0], ctx[:, 1], vario)
                        mm2 = C.krigagem_residuo(mm_f[b], ctx[:, 0], ctx[:, 1],
                                                 ctx[:, 2], vario, 0.0, None, lam)
                        pt2 = {t: C.krigagem_residuo(pt[t], ctx[:, 0], ctx[:, 1],
                                                     (ctx[:, 2] > t).astype(float),
                                                     vario, 0.0, 1.0, lam)
                               for t in EDGES[args.agg]}
                        lin_rk += linhas_stem(
                            s, est, ctx, {"mm": mm2,
                                          "det": campo_det(mm2, pt2[EDGES[args.agg][0]],
                                                           THR_RK)},
                            pt2, args.agg, chuvoso=bool(am["chuvoso"]))
                    if args.agg == 10:
                        campos10[s] = dict(
                            p02=pt[taus10[0]].astype(np.float16),
                            p_nuc=pt[NUCLEOS[10][0]].astype(np.float16),
                            mm=mm_f[b].astype(np.float16), est=est, ctx=ctx,
                            chuvoso=bool(am["chuvoso"]))

    if not linhas:
        raise SystemExit("nenhum par gerado — validação vazia?")
    pares = pd.concat(linhas, ignore_index=True)
    pares["agg"] = args.agg; pares["braco"] = args.arm; pares["fold"] = args.fold
    pares["versao"] = VERSAO
    pares["thr_chuva"] = np.float32(THR)
    est_tab = pd.DataFrame(areas)
    if lin_rk:
        pr = pd.concat(lin_rk, ignore_index=True)
        pr["agg"] = args.agg; pr["braco"] = f"{args.arm}_rk"; pr["fold"] = args.fold
        pr["versao"] = VERSAO; pr["thr_chuva"] = np.float32(THR_RK)
        pr["thr_rk"] = np.float32(THR_RK)
        pr.to_parquet(pasta / f"{args.arm}_rk_f{args.fold}.parquet", index=False)
        print(f"[avalia] produto {args.arm}_rk (s1 + krigagem do erro) gravado, "
              f"limiar {THR_RK:.2f}")

    # ---------------- filtro de células e coerência (m10) ----------------
    if args.agg == 10 and campos10:
        stems_ord = sorted(campos10, key=C.stem_para_brt)
        coer, rem = coerencia_e_filtro(stems_ord, campos10, mov)
        lin_f = []
        for s in (stems_ord if FILTRO_ATIVO else []):
            c = campos10[s]; r = rem[s]
            p02 = np.where(r, 0.0, c["p02"].astype(np.float32))
            pn = np.where(r, 0.0, c["p_nuc"].astype(np.float32))
            mmf = np.where(r, 0.0, c["mm"].astype(np.float32))
            lin_f += linhas_stem(s, c["est"], c["ctx"], {"mm": mmf},
                                 {taus10[0]: p02, NUCLEOS[10][0]: pn}, 10,
                                 chuvoso=c["chuvoso"])
        if lin_f:
            pf = pd.concat(lin_f, ignore_index=True)
            pf["agg"] = 10; pf["braco"] = f"{args.arm}_filtro"
            pf["fold"] = args.fold; pf["versao"] = VERSAO
            pf.to_parquet(pasta / f"{args.arm}_filtro_f{args.fold}.parquet",
                          index=False)
        print(f"[avalia] coerência: diferença média P(t) x P(t-10) advectado "
              f"{coer['flicker'].mean():.3f} | células com antecessor "
              f"{coer['frac_com_antecessor'].mean():.0%}")
        coer["braco"] = args.arm; coer["fold"] = args.fold
        coer.to_parquet(pasta / f"{args.arm}_f{args.fold}.coer.parquet",
                        index=False)
    if area_st:
        ar = pd.DataFrame(area_st); ar["braco"] = args.arm; ar["fold"] = args.fold
        ar.to_parquet(pasta / f"{args.arm}_f{args.fold}.area.parquet", index=False)
    if len(est_tab):
        est_tab["braco"] = args.arm; est_tab["fold"] = args.fold
        est_tab.to_parquet(pasta / f"{args.arm}_f{args.fold}.areas.parquet",
                           index=False)
    pares.to_parquet(saida, index=False)          # por último: marca "feito"

    al = pares[pares["lado"] == "alvo"]
    print(f"[avalia] {len(pares):,} pares ({len(al):,} alvo, "
          f"{len(pares) - len(al):,} holdout) -> {saida}")
    q = C.metricas_quantidade(al["mm_w0"].values, al["obs"].values)
    c02 = C.contingencia(al[f"p{EDGES[args.agg][0]:g}_w0"] >= THR,
                         al["obs"] > EDGES[args.agg][0])
    print(f"  r {q['r']:.3f} | chuva (P>={THR:.2f}) POD {c02['POD']:.2f} "
          f"FAR {c02['FAR']:.2f} CSI {c02['CSI']:.2f}")


if __name__ == "__main__":
    main()
