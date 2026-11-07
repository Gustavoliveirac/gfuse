#!/usr/bin/env python
"""
treino.py — treina UM braço numa janela e num fold do k-fold temporal.

Critério de escolha do checkpoint: a menor perda na validação (subamostra
fixa), sem métrica composta. A cada época o log mostra também POD, FAR e
CSI de P(chuva) >= 0,5 nas estações-alvo, só para acompanhamento.

Braço rec: treinado sempre em 10 min, com sequências de REC_PASSOS stems
consecutivos; a estimativa de cada passo entra no seguinte como prior
advectado pelo vetor de movimento. Braço mw_di: pré-treino GOES->MW antes
do ajuste fino.

Uso:
  python treino.py --agg 60 --arm base --fold 0 --modo rapido
  python treino.py --agg 10 --arm rec  --fold 2 --modo completo
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
# os workers do DataLoader passam os tensores por descritores de arquivo;
# em avaliações longas o limite de descritores abertos do sistema estoura
# ("received 0 items of ancdata"). Com "file_system" isso não ocorre.
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader, WeightedRandomSampler

import comum as C
import modelo as M
from estacoes import Base
from config3 import (ARM_FLAGS, BATCH, CKPT_DIR, EDGES, GRAD_CLIP, IMG_SIZE,
                     LR, MODOS, NUM_WORKERS, PROC, REC_BATCH,
                     REC_PASSOS, RES, VERSAO, FRAC_CHUVOSO, N_STEMS_LIMIAR,
                     USAR_PESOS_CLASSE)


def pesos_proporcao(flag_chuva):
    """Pesos de sorteio para que FRAC_CHUVOSO das amostras venham de stems
    com chuva e o resto dos demais, independentemente de quantos stems secos
    a base tenha."""
    f = np.asarray(flag_chuva, bool)
    n1, n0 = f.sum(), (~f).sum()
    if n1 == 0 or n0 == 0:
        return np.ones(len(f))
    return np.where(f, FRAC_CHUVOSO / n1, (1 - FRAC_CHUVOSO) / n0)


def amostra_estratificada(stems, flag, n):
    """Subamostra determinística com metade de stems chuvosos (se houver)."""
    stems = np.asarray(stems); flag = np.asarray(flag, bool)
    if len(stems) <= n:
        return list(stems)
    out = []
    for grupo in (stems[flag], stems[~flag]):
        k = min(len(grupo), n // 2 if flag.any() and (~flag).any() else n)
        if k:
            out += list(grupo[np.linspace(0, len(grupo) - 1, k).astype(int)])
    return sorted(out)


def preparar(agg, fold, modo, subamostrar=True):
    base = Base()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    hold_m = np.array([c in holdout for c in base.cod])
    stems = C.stems_alinhados(list(base.stems), agg)
    fd = C.folds_temporais(stems)[fold]
    vario = C.carregar_vario(agg, PROC)

    com_alvo = base.tem_estacao(agg, hold_m)
    com_algo = base.tem_estacao(agg)
    chuv = base.chuvoso(agg)
    pos = base.pos
    tre = [s for s in fd["treino"] if com_alvo[pos[s]]]
    val = [s for s in fd["val"] if com_algo[pos[s]]]
    cfg = MODOS[modo]
    if subamostrar and cfg["max_stems"] and len(tre) > cfg["max_stems"]:
        tre = amostra_estratificada(tre, [chuv[pos[s]] for s in tre], cfg["max_stems"])
    if subamostrar and cfg["max_stems_val"] and len(val) > cfg["max_stems_val"]:
        val = amostra_estratificada(val, [chuv[pos[s]] for s in val],
                                    cfg["max_stems_val"])
    flag_tre = np.array([chuv[pos[s]] for s in tre])
    todos_tre = [s for s in fd["treino"] if com_alvo[pos[s]]]
    pi = float(np.mean([chuv[pos[s]] for s in todos_tre])) if todos_tre else 0.5
    # pesos de importância: o sorteio leva a FRAC_CHUVOSO de stems chuvosos,
    # a base real tem pi; cada amostra pesa pi/FRAC (chuvosa) ou
    # (1-pi)/(1-FRAC) (demais), e a perda volta a medir a distribuição real
    imp = (pi / FRAC_CHUVOSO, (1 - pi) / (1 - FRAC_CHUVOSO))
    W = base.janelas(agg)
    m_liv = base.dentro & ~hold_m
    idx_tre = np.array([pos[s] for s in tre])
    obs_tre = W[np.ix_(idx_tre, np.flatnonzero(m_liv))].ravel()
    pesos_cls = M.pesos_classes(obs_tre[np.isfinite(obs_tre)], agg)
    print(f"  stems com chuva: {pi:.0%} da base de treino | sorteio leva a "
          f"{FRAC_CHUVOSO:.0%} | pesos de importância {imp[0]:.2f} (chuvoso) "
          f"e {imp[1]:.2f} (demais)")
    return dict(base=base, holdout=holdout, vario=vario, tre=tre, val=val,
                fd=fd, chuv=chuv, pesos_stem=pesos_proporcao(flag_tre),
                pesos_cls=pesos_cls, cfg=cfg, imp=imp)


IMP = (1.0, 1.0)            # pesos de importância (definidos no main)
S1 = None                   # estágio 1 congelado (só no braço s2)


def calc_perda(net, batch, agg, flags, pesos, device, prior=None):
    for k in ("cubo", "estat", "tgt_pos", "tgt_mm", "tgt_ok", "seco", "chuvoso"):
        batch[k] = batch[k].to(device)
    batch["peso"] = torch.where(batch["chuvoso"], torch.tensor(IMP[0], device=device),
                                torch.tensor(IMP[1], device=device)).float()
    estat = batch["estat"]
    if prior is not None:
        estat = torch.cat([estat, prior], 1)
    if flags["hurdle"]:
        saida = M.prever(net, batch["cubo"], estat, flags, agg, S1, detalhado=True,
                         ctx=batch["ctx"], n_ctx=batch["n_ctx"])
        return M.perda_hurdle(saida, batch, agg, device), saida[0]
    logits = net(batch["cubo"], estat)
    return M.perda_alvos(logits, batch, agg, pesos, device), logits


@torch.no_grad()
def avaliar_val(net, loader, device, agg, flags, pesos):
    """Perda média na validação (critério de escolha do checkpoint) e, só
    para acompanhar, POD/FAR/CSI de P(chuva) >= 0,5 nas estações-alvo."""
    net.eval()
    tot = n = 0.0
    h = m = f = 0.0
    G = IMG_SIZE
    for batch in loader:
        loss, logits = calc_perda(net, batch, agg, flags, pesos, device)
        tot += loss.item(); n += 1
        _, p_tau = M.decodificar(logits, agg)
        p02 = p_tau[EDGES[agg][0]]
        pos = batch["tgt_pos"].cpu().numpy(); mm = batch["tgt_mm"].cpu().numpy()
        ok = batch["tgt_ok"].cpu().numpy()
        for b in range(len(p02)):
            o = ok[b]
            if not o.any():
                continue
            v = C.amostra_com_tolerancia(p02[b], pos[b, o, 0] * G, pos[b, o, 1] * G, 0)
            pr, ob = v >= 0.5, mm[b, o] > EDGES[agg][0]
            h += (pr & ob).sum(); m += (~pr & ob).sum(); f += (pr & ~ob).sum()
    info = dict(POD=h / (h + m) if h + m else np.nan,
                FAR=f / (h + f) if h + f else np.nan,
                CSI=h / (h + m + f) if h + m + f else np.nan)
    return tot / max(n, 1), info


@torch.no_grad()
def avaliar_val_rec(net, loader, device, pesos):
    net.eval()
    tot = n = 0.0
    for seq in loader:
        prior = None
        for i in range(seq["cubo"].shape[1]):
            batch = {k: v[:, i] for k, v in seq.items()}
            valido = torch.full((batch["cubo"].shape[0],), float(i > 0), device=device)
            if prior is None:
                prior = torch.zeros(batch["cubo"].shape[0], M.N_PRIOR, IMG_SIZE,
                                    IMG_SIZE, device=device)
            loss, logits = calc_perda(net, batch, 10, ARM_FLAGS["rec"], pesos,
                                      device, prior)
            tot += loss.item(); n += 1
            prior = M.prior_de_saida(logits, 10, batch["uv"].to(device),
                                     torch.ones_like(valido))
    return tot / max(n, 1), {}


@torch.no_grad()
def escolher_limiar(net, prep, flags, agg, device, mov):
    """Limiar de P(chuva) que maximiza o CSI nas estações-alvo de uma
    amostra estratificada do bloco de TREINO (divisão fixa por estação)."""
    import verif as V
    stems = amostra_estratificada(
        prep["tre"], [prep["chuv"][prep["base"].pos[s]] for s in prep["tre"]],
        N_STEMS_LIMIAR)
    ds = M.DatasetGfuse(stems, prep["base"], agg, flags, prep["holdout"],
                        prep["vario"], treino=False, movimento=mov)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    net.eval()
    h = m = f = np.zeros(len(V.GRADE_THR))
    G = IMG_SIZE
    for batch in dl:
        cubo = batch["cubo"].to(device); estat = batch["estat"].to(device)
        _, p_tau = M.decodificar(M.prever(net, cubo, estat, flags, agg, S1,
                                          ctx=batch["ctx"], n_ctx=batch["n_ctx"]), agg)
        p02 = p_tau[EDGES[agg][0]]
        pos = batch["tgt_pos"].numpy(); mm = batch["tgt_mm"].numpy()
        ok = batch["tgt_ok"].numpy()
        for b in range(len(p02)):
            o = ok[b]
            if not o.any():
                continue
            v = C.amostra_com_tolerancia(p02[b], pos[b, o, 0] * G, pos[b, o, 1] * G, 0)
            ob = mm[b, o] > EDGES[agg][0]
            pr = v[None] >= V.GRADE_THR[:, None]
            h = h + (pr & ob).sum(1); m = m + (~pr & ob).sum(1); f = f + (pr & ~ob).sum(1)
    csi, thr, _ = V.csi_melhor(h.astype(float), m.astype(float), f.astype(float))
    return float(thr) if np.isfinite(thr) else 0.5


def sequencias(stems, ds, passo=None):
    """Janelas de REC_PASSOS stems consecutivos de 10 min."""
    passo = passo or max(1, REC_PASSOS // 2)
    out = []
    for cad in M.cadeias_10min(stems, ds):
        for i in range(0, len(cad) - REC_PASSOS + 1, passo):
            out.append(cad[i:i + REC_PASSOS])
    return out


def pre_treinar(agg, fold, flags, prep, device, epochs, modo):
    """Pré-treino GOES->MW do fold; devolve o caminho do checkpoint."""
    tipo = "s1" if flags.get("s1") else "di"
    p_ck = CKPT_DIR / f"m{agg}" / f"pre_{tipo}_{modo}_{VERSAO}_f{fold}.pt"
    p_ck.parent.mkdir(parents=True, exist_ok=True)
    if p_ck.exists():
        print(f"  pré-treino já existe: {p_ck}")
        return p_ck
    fonte = M.FonteMW()
    if not fonte.ok:
        raise SystemExit("sem npz de MW em dados/mw — rode mw_prep.py "
                         "(ou tire mw_di de ARMS)")
    stems_pre = M.filtrar_stems_mw(prep["fd"]["treino"], fonte, agg)
    if len(stems_pre) < 50:
        raise SystemExit(f"só {len(stems_pre)} stems com MW — pré-treino inviável")
    print(f"  pré-treino: {len(stems_pre)} stems com sobrevoo MW")
    ds = M.DatasetMW(stems_pre, prep["base"], agg, flags, prep["holdout"],
                     prep["vario"], fonte)
    n_ep = min(len(stems_pre), MODOS[modo]["amostras_pre"])
    amostrador = torch.utils.data.RandomSampler(ds, replacement=False,
                                                num_samples=n_ep)
    dl = DataLoader(ds, batch_size=BATCH, sampler=amostrador,
                    num_workers=NUM_WORKERS, drop_last=True)
    net = M.criar_rede(flags, agg, device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    for ep in range(epochs):
        net.train(); tot = n = 0; t_ep = time.time()
        for batch in dl:
            cubo = batch["cubo"].to(device); estat = batch["estat"].to(device)
            batch["mw_mm"] = batch["mw_mm"].to(device)
            batch["mw_ok"] = batch["mw_ok"].to(device)
            loss = M.perda_mw(net(cubo, estat), batch, agg,
                              prep["pesos_cls"], device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item(); n += 1
        print(f"    pre ep{ep + 1}/{epochs}  loss={tot / max(n, 1):.4f}  "
              f"({(time.time() - t_ep) / 60:.1f} min)")
    torch.save(net.state_dict(), p_ck)
    return p_ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=int, required=True, choices=[10, 30, 60])
    ap.add_argument("--arm", required=True, choices=list(ARM_FLAGS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--modo", default="rapido", choices=list(MODOS))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--forcar", action="store_true",
                    help="retreina mesmo se já existe checkpoint deste modo")
    args = ap.parse_args()
    flags = ARM_FLAGS[args.arm]
    if flags["rec"]:
        args.agg = 10           # o rec é sempre treinado em 10 min
    ck_json = (CKPT_DIR / f"m{args.agg}" / f"{args.arm}_f{args.fold}.json")
    if ck_json.exists() and not args.forcar:
        with open(ck_json) as fj:
            feito = json.load(fj)
        if feito.get("modo") == args.modo and feito.get("versao") == VERSAO:
            print(f"[treino m{args.agg} {args.arm} fold{args.fold}] já concluído "
                  f"no modo {args.modo}, pulando")
            return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_ini = time.time()
    print(f"[treino m{args.agg} {args.arm} fold{args.fold} {args.modo}] "
          f"device={device.type}")

    # o rec precisa de stems contíguos: sem subamostrar (limita sequências)
    prep = preparar(args.agg, args.fold, args.modo,
                    subamostrar=not flags["rec"])
    cfg = prep["cfg"]
    epochs = args.epochs or cfg["epochs"]
    print(f"  treino {len(prep['tre'])} stems | val {len(prep['val'])} stems")
    if not prep["tre"] or not prep["val"]:
        raise SystemExit("fold sem stems suficientes")
    mov = M.Movimento()
    if (flags.get("hist") or flags["rec"]) and not mov.ok:
        print("  AVISO: movimento.csv ausente, história/recursão sem advecção")

    ds_tre = M.DatasetGfuse(prep["tre"], prep["base"], args.agg, flags,
                            prep["holdout"], prep["vario"], treino=True,
                            movimento=mov)
    val_sub = prep["val"]
    if cfg["val_sub"] and len(val_sub) > cfg["val_sub"]:
        pos = prep["base"].pos
        val_sub = amostra_estratificada(
            val_sub, [prep["chuv"][pos[s]] for s in val_sub], cfg["val_sub"])
    ds_val = M.DatasetGfuse(val_sub, prep["base"], args.agg, flags,
                            prep["holdout"], prep["vario"], treino=False,
                            movimento=mov)

    if flags["rec"]:
        seq_tre = sequencias(prep["tre"], ds_tre)
        ds_val.stems = prep["val"]
        seq_val = sequencias(prep["val"], ds_val, passo=REC_PASSOS)
        n_sv = max(1, cfg["val_sub"] // REC_PASSOS)
        if len(seq_val) > n_sv:
            seq_val = seq_val[:: max(1, len(seq_val) // n_sv)]
        if not seq_tre or not seq_val:
            raise SystemExit("sem sequências de 10 min contíguos suficientes")
        pos = prep["base"].pos
        pesos_seq = pesos_proporcao(
            [any(prep["chuv"][pos[s]] for s in q) for q in seq_tre])
        n_ep = min(cfg["seq_epoca"] or len(seq_tre), len(seq_tre) * 4)
        amostrador = WeightedRandomSampler(torch.tensor(pesos_seq, dtype=torch.double),
                                           num_samples=n_ep, replacement=True)
        dl_tre = DataLoader(M.DatasetSeq(ds_tre, seq_tre), batch_size=REC_BATCH,
                            sampler=amostrador, num_workers=NUM_WORKERS,
                            drop_last=True)
        dl_val = DataLoader(M.DatasetSeq(ds_val, seq_val), batch_size=REC_BATCH,
                            shuffle=False, num_workers=NUM_WORKERS)
        print(f"  rec: {len(seq_tre)} sequências de {REC_PASSOS} passos "
              f"({n_ep} por época) | val {len(seq_val)} sequências")
    else:
        n_ep = cfg["amostras_epoca"] or len(prep["tre"])
        sampler = WeightedRandomSampler(torch.tensor(prep["pesos_stem"], dtype=torch.double),
                                        num_samples=n_ep, replacement=True)
        dl_tre = DataLoader(ds_tre, batch_size=BATCH, sampler=sampler,
                            num_workers=NUM_WORKERS, drop_last=True)
        dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False,
                            num_workers=NUM_WORKERS)
        print(f"  {n_ep} amostras por época | {NUM_WORKERS} workers")

    net = M.criar_rede(flags, args.agg, device)
    if flags["s2"]:
        global S1
        S1 = M.carregar_s1(CKPT_DIR, args.agg, args.fold, device)
        print(f"  s2: estágio 1 congelado de s1_f{args.fold}.pt")
    if flags["mw"]:
        p_ck = pre_treinar(args.agg, args.fold, flags, prep, device,
                           cfg["pre_epochs"], args.modo)
        net.load_state_dict(torch.load(p_ck, map_location=device, weights_only=True))
        print(f"  pesos do pré-treino carregados de {p_ck.name}")
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    global IMP
    IMP = prep["imp"]
    pesos = (prep["pesos_cls"] if USAR_PESOS_CLASSE
             else torch.ones_like(prep["pesos_cls"]))
    if flags["rec"]:
        # o rec soma as análises de 10 min para formar 30 e 60 min; com pesos
        # de classe o valor esperado de cada passo sai inflado e a soma
        # multiplica o viés (razão de volume 3,8 no m60 do rápido). Sem pesos,
        # a probabilidade de cada passo fica calibrada.
        pesos = torch.ones_like(pesos)

    ck = CKPT_DIR / f"m{args.agg}" / f"{args.arm}_f{args.fold}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    melhor, melhor_ep, sem_melhora = np.inf, -1, 0
    for ep in range(epochs):
        net.train(); tot = n = 0; t_ep = time.time()
        for batch in dl_tre:
            if flags["rec"]:
                # desenrola a sequência; a estimativa anterior entra como
                # prior advectado (sem gradiente entre passos)
                prior = None; loss = 0.0
                L = batch["cubo"].shape[1]
                for i in range(L):
                    b_i = {k: v[:, i] for k, v in batch.items()}
                    if prior is None:
                        prior = torch.zeros(b_i["cubo"].shape[0], M.N_PRIOR,
                                            IMG_SIZE, IMG_SIZE, device=device)
                    l_i, logits = calc_perda(net, b_i, 10, flags, pesos,
                                             device, prior)
                    loss = loss + l_i / L
                    prior = M.prior_de_saida(
                        logits, 10, b_i["uv"].to(device),
                        torch.ones(b_i["cubo"].shape[0], device=device))
            else:
                loss, _ = calc_perda(net, batch, args.agg, flags, pesos, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item(); n += 1
        if flags["rec"]:
            perda_v, info = avaliar_val_rec(net, dl_val, device, pesos)
        else:
            perda_v, info = avaliar_val(net, dl_val, device, args.agg, flags, pesos)
        marca = ""
        if np.isfinite(perda_v) and perda_v < melhor:
            melhor, melhor_ep, sem_melhora = perda_v, ep, 0
            torch.save(net.state_dict(), ck)
            marca = "  <- melhor"
        else:
            sem_melhora += 1
        txt = (f"  P(chuva)>=0,5: POD {info['POD']:.2f} FAR {info['FAR']:.2f} "
               f"CSI {info['CSI']:.2f}" if info else "")
        print(f"  ep{ep + 1}/{epochs}  perda treino {tot / max(n, 1):.4f}  "
              f"perda val {perda_v:.4f}{txt}  "
              f"({(time.time() - t_ep) / 60:.1f} min){marca}")
        if sem_melhora >= cfg["paciencia"]:
            print("  parada antecipada"); break
    if melhor_ep < 0:
        torch.save(net.state_dict(), ck)

    # limiar de decisão da chuva, escolhido no bloco de TREINO do fold
    net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    thr = 0.5 if flags["rec"] else escolher_limiar(net, prep, flags, args.agg,
                                                   device, mov)
    print(f"  limiar de P(chuva) que maximiza o CSI no treino: {thr:.2f}")

    linha = dict(agg=args.agg, braco=args.arm, fold=args.fold, modo=args.modo,
                 versao=VERSAO, thr_chuva=thr,
                 perda_val=round(float(melhor), 5) if np.isfinite(melhor) else np.nan,
                 epoca=melhor_ep + 1, n_treino=len(prep["tre"]),
                 n_val=len(prep["val"]),
                 minutos=round((time.time() - t_ini) / 60, 1))
    f = RES / "treinos.csv"
    ant = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pd.concat([ant, pd.DataFrame([linha])], ignore_index=True).to_csv(
        f, index=False)
    with open(ck.with_suffix(".json"), "w") as fj:
        json.dump(linha, fj)
    print(f"[treino] melhor perda de validação {melhor:.4f} "
          f"(ep {melhor_ep + 1}) -> {ck}")


if __name__ == "__main__":
    main()
