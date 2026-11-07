#!/usr/bin/env python
"""
mapas_eventos.py — painéis comparativos dos eventos mais intensos da
validação, para ver como cada fonte representou a chuva.

Para cada janela (m10/m30/m60), escolhe os N stems de validação com maior
chuva observada numa estação dentro do recorte (separados por pelo menos
6 h) e gera uma figura com:
  linhas   instantes consecutivos (m10: t-20, t-10, t; m30/m60: janela
           anterior e janela atual), para ver a continuidade no tempo
  colunas  Tb 10,3 µm (último quadro; em branco a área que a regra do seco
           marca como "seco certo"), RRQPE acumulado, IDW e krigagem simples
           das estações de contexto (mm, a mesma referência "sk" das
           métricas), e cada braço treinado. Nos painéis dos braços, a chuva
           em mm só aparece onde a probabilidade de chuva é >= 0,5
           (branco = previsto seco).
Sobre os painéis, todas as estações com a cor da chuva observada nelas, na
mesma escala (branco = seco): círculo = o modelo viu (contexto), triângulo =
o modelo não viu (alvo), quadrado = nunca visto (holdout). Um x marca as
estações da faixa de MARGEM_KM junto à borda (fora das métricas); o
retângulo tracejado delimita essa faixa.

Cada evento usa os checkpoints do fold em que ele está na validação, então
nenhum modelo viu o evento no treino.

Saída: resultados/mapas/m{agg}_ev{k}_{stem}.png e resultados/mapas_eventos.csv

Uso:  python mapas_eventos.py --modo rapido --aggs 60 30 10
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import torch                                          # noqa: E402
# os workers do DataLoader passam os tensores por descritores de arquivo;
# em avaliações longas o limite de descritores abertos do sistema estoura
# ("received 0 items of ancdata"). Com "file_system" isso não ocorre.
torch.multiprocessing.set_sharing_strategy("file_system")
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

import comum as C                                     # noqa: E402
import modelo as M                                    # noqa: E402
from config3 import (ARM_FLAGS, ARMS, CKPT_DIR, EDGES, GOES_DIR, IMG_SIZE,  # noqa: E402
                     KM_PER_PX, MARGEM_KM, MODOS, PROC, RES)
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
from estacoes import Base                             # noqa: E402

NIVEIS = {10: [0.2, 1, 2, 5, 10, 20], 30: [0.2, 1, 5, 10, 20, 30],
          60: [0.2, 1, 5, 10, 20, 30, 50]}
SEP_H = 6


def paleta(agg):
    niv = NIVEIS[agg]
    cores = plt.get_cmap("turbo")(np.linspace(0.1, 0.95, len(niv) - 1))
    cmap = ListedColormap(cores)
    cmap.set_under("white"); cmap.set_over(plt.get_cmap("turbo")(0.99))
    return cmap, BoundaryNorm(niv, cmap.N)


def escolher_eventos(base, agg, folds_usar, n):
    W = base.janelas(agg)
    stems = C.stems_alinhados(list(base.stems), agg)
    folds = C.folds_temporais(stems)
    cand = []
    for f in folds_usar:
        for s in folds[f]["val"]:
            i = base.pos[s]
            w = np.where(base.dentro, W[i], np.nan)
            if np.isfinite(w).any() and np.nanmax(w) > 0:
                j = int(np.nanargmax(w))
                cand.append((float(w[j]), s, f, base.cod[j],
                             int((np.nan_to_num(w) > 0.2).sum())))
    cand.sort(reverse=True)
    esc = []
    for c in cand:
        t = C.stem_para_brt(c[1])
        if all(abs((t - C.stem_para_brt(e[1])).total_seconds()) >= SEP_H * 3600
               for e in esc):
            esc.append(c)
        if len(esc) >= n:
            break
    return esc


def carregar_rede(arm, agg, fold, device):
    """(rede, s1 congelado ou None, limiar de P(chuva)) ou None."""
    import json
    flags = ARM_FLAGS[arm]
    agg_ck = 10 if flags["rec"] else agg
    ck = CKPT_DIR / f"m{agg_ck}" / f"{arm}_f{fold}.pt"
    if not ck.exists():
        return None
    net = M.criar_rede(flags, agg_ck, device)
    net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    s1 = M.carregar_s1(CKPT_DIR, agg, fold, device) if flags["s2"] else None
    try:
        thr = float(json.load(open(ck.with_suffix(".json"))).get("thr_chuva", 0.5))
    except Exception:
        thr = 0.5
    return net.eval(), s1, thr


def thr_rk(arm, agg, fold, padrao):
    """Limiar do produto <braço>_rk gravado pela avaliação (ou o do braço)."""
    try:
        from config3 import PARES_DIR
        d = pd.read_parquet(PARES_DIR / f"m{agg}" / f"{arm}_rk_f{fold}.parquet",
                            columns=["thr_rk"])
        return float(d["thr_rk"].iloc[0])
    except Exception:
        return padrao


@torch.no_grad()
def campo_rec(net, ds10, base, stem_fim, A, n_aquec, device):
    """Roda a recursão do rec em 10 min terminando em stem_fim e devolve
    dict stem -> soma, nos últimos A passos, da chuva condicional onde
    P(chuva) >= 0,5 (mesmo produto <braço>_det das métricas)."""
    i_fim = base.pos[stem_fim]
    passos = []
    for k in range(n_aquec + A, -1, -1):
        j = base.deslocado(-10 * k)[i_fim]
        passos.append(base.stems[j] if j >= 0 else None)
    prior = torch.zeros(1, M.N_PRIOR, IMG_SIZE, IMG_SIZE, device=device)
    hist, hp, out = [], [], {}
    for s in passos:
        if s is None:                                  # buraco: reinicia
            prior.zero_(); hist = []; hp = []; continue
        am = ds10.amostra(s)
        logits = net(am["cubo"][None].to(device),
                     torch.cat([am["estat"][None].to(device), prior], 1))
        mm, pt = M.decodificar(logits, 10)
        prior = M.prior_de_saida(logits, 10, am["uv"][None].to(device),
                                 torch.ones(1, device=device))
        p0 = pt[EDGES[10][0]][0]
        hist.append(np.where(p0 >= 0.5, mm[0] / np.maximum(p0, 1e-6), 0.0))
        if len(hist) >= A:
            out[s] = np.sum(hist[-A:], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="rapido", choices=list(MODOS))
    ap.add_argument("--aggs", nargs="*", type=int, default=[60, 30, 10])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Base()
    holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
    mov = M.Movimento()
    pasta = RES / "mapas"
    pasta.mkdir(parents=True, exist_ok=True)
    folds_usar = MODOS[args.modo]["folds"]
    tabela = []

    for agg in args.aggs:
        A = max(1, agg // 10)
        cmap, norm = paleta(agg)
        vario = C.carregar_vario(agg, PROC)
        eventos = escolher_eventos(base, agg, folds_usar, args.n)
        print(f"[mapas m{agg}] {len(eventos)} eventos")
        for k, (mx, stem, fold, cod_mx, n_chuva) in enumerate(eventos, 1):
            i = base.pos[stem]
            if agg == 10:
                linhas = [base.deslocado(-20)[i], base.deslocado(-10)[i], i]
            else:
                linhas = [base.deslocado(-agg)[i], i]
            linhas = [base.stems[j] for j in linhas if j >= 0]
            redes = {a: carregar_rede(a, agg, fold, device) for a in args.arms}
            redes = {a: r for a, r in redes.items() if r is not None}
            ds = {a: M.DatasetGfuse([], base, agg, ARM_FLAGS[a], holdout, vario,
                                    treino=False, movimento=mov)
                  for a in redes if not ARM_FLAGS[a]["rec"]}
            ds_ref = M.DatasetGfuse([], base, agg, ARM_FLAGS["base"], holdout,
                                    vario, treino=False, movimento=mov)
            rec = {}
            if "rec" in redes:
                ds10 = M.DatasetGfuse([], base, 10, ARM_FLAGS["rec"], holdout,
                                      C.carregar_vario(10, PROC), treino=False,
                                      movimento=mov)
                rec = campo_rec(redes["rec"][0], ds10, base, stem, A,
                                n_aquec=12 + (A * (len(linhas) - 1)), device=device)
            colunas = (["Tb 10,3 µm", "RRQPE", "IDW", "krigagem"] + list(redes)
                       + [f"{a}_rk" for a in redes if ARM_FLAGS[a]["s1"]])
            fig, axs = plt.subplots(len(linhas), len(colunas),
                                    figsize=(2.3 * len(colunas), 2.4 * len(linhas) + 0.6),
                                    squeeze=False)
            for r, s in enumerate(linhas):
                am = ds_ref.amostra(s)
                inf = ds_ref.info(s)
                w = inf["w"]
                todos = np.r_[inf["ctx"], inf["ext"]]
                sk, ik, _ = C.krigagem_simples(base.px[todos], base.py[todos],
                                               w[todos], vario)
                try:
                    tb = C.rad_para_tb(np.load(GOES_DIR / f"{s}.npy")[5], 5)
                except Exception:
                    tb = np.full((IMG_SIZE, IMG_SIZE), np.nan)
                mascara = lambda mm, p, thr=0.5: np.where(np.asarray(p) >= thr,
                                                          np.asarray(mm) / np.maximum(p, 1e-6), 0.0)
                idw = C.idw(np.stack([base.px[todos], base.py[todos]], 1) * KM_PER_PX,
                            w[todos], C.grade_km()).reshape(IMG_SIZE, IMG_SIZE) \
                    if len(todos) else np.zeros((IMG_SIZE, IMG_SIZE))
                campos = {"RRQPE": am["rrqpe_mm"].numpy(), "IDW": idw, "krigagem": sk}
                seco_px = am["seco"].numpy()
                with torch.no_grad():
                    for a, (net, s1, thr) in redes.items():
                        if ARM_FLAGS[a]["rec"]:
                            campos[a] = rec.get(s)
                            continue
                        x = ds[a].amostra(s)
                        mm, pt = M.decodificar(M.prever(
                            net, x["cubo"][None].to(device), x["estat"][None].to(device),
                            ARM_FLAGS[a], agg, s1, ctx=x["ctx"][None],
                            n_ctx=x["n_ctx"][None]), agg)
                        p0 = pt[EDGES[agg][0]][0]
                        campos[a] = mascara(mm[0], p0, thr)
                        if ARM_FLAGS[a]["s1"]:
                            # estágio 2 analítico: krigagem do erro do s1
                            cx = np.r_[inf["ctx"], inf["ext"]]
                            lam, _ = C.pesos_krigagem(base.px[cx], base.py[cx], vario)
                            mm2 = C.krigagem_residuo(mm[0], base.px[cx], base.py[cx],
                                                     w[cx], vario, 0.0, None, lam)
                            p2 = C.krigagem_residuo(p0, base.px[cx], base.py[cx],
                                                    (w[cx] > EDGES[agg][0]).astype(float),
                                                    vario, 0.0, 1.0, lam)
                            campos[f"{a}_rk"] = mascara(mm2, p2, thr_rk(a, agg, fold, thr))
                for c, nome in enumerate(colunas):
                    ax = axs[r, c]
                    ax.set_xticks([]); ax.set_yticks([])
                    if nome == "Tb 10,3 µm":
                        # só onde a regra do seco NÃO descarta chuva
                        # (branco = "seco certo" pelo satélite)
                        ax.imshow(np.ma.masked_where(seco_px, tb), cmap="gray_r",
                                  vmin=190, vmax=300)
                    else:
                        campo = campos.get(nome)
                        if campo is None:
                            ax.text(0.5, 0.5, "sem dado", ha="center",
                                    transform=ax.transAxes); continue
                        im = ax.imshow(campo, cmap=cmap, norm=norm)
                        mp = MARGEM_KM / KM_PER_PX
                        ax.add_patch(Rectangle((mp - 0.5, mp - 0.5), IMG_SIZE - 2 * mp,
                                               IMG_SIZE - 2 * mp, fill=False, ls="--",
                                               lw=0.6, ec="0.3"))
                        for grupo, mk in (("ctx", "o"), ("tgt", "^"), ("hld", "s")):
                            idx = inf[grupo]
                            if not len(idx):
                                continue
                            ax.scatter(base.px[idx], base.py[idx], c=w[idx],
                                       cmap=cmap, norm=norm, s=30, marker=mk,
                                       edgecolors="k", linewidths=0.6)
                            px_, py_ = base.px[idx], base.py[idx]
                            na_borda = ((px_ < mp - 0.5) | (px_ > IMG_SIZE - 0.5 - mp)
                                        | (py_ < mp - 0.5) | (py_ > IMG_SIZE - 0.5 - mp))
                            if na_borda.any():
                                ax.scatter(px_[na_borda], py_[na_borda], marker="x",
                                           c="k", s=14, linewidths=0.7)
                        ax.set_xlim(-0.5, IMG_SIZE - 0.5); ax.set_ylim(IMG_SIZE - 0.5, -0.5)
                    if r == 0:
                        ax.set_title(nome, fontsize=9)
                    if c == 0:
                        ax.set_ylabel(C.stem_para_brt(s).strftime("%d/%m/%y %H:%M"),
                                      fontsize=8)
            fig.suptitle(f"m{agg}  |  {C.stem_para_brt(stem):%d/%m/%Y %H:%M} BRT  |  "
                         f"máx. estação {mx:.1f} mm ({cod_mx})  |  "
                         f"{n_chuva} estações com chuva  |  fold {fold}", fontsize=10, y=1.0)
            fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.04,
                         pad=0.04, label=f"mm em {agg} min (cor das estações = "
                         f"chuva observada; branco = seco)", extend="max")
            leg = [Line2D([], [], ls="", marker="o", mfc="w", mec="k", label="o modelo viu (contexto)"),
                   Line2D([], [], ls="", marker="^", mfc="w", mec="k", label="o modelo não viu (alvo)"),
                   Line2D([], [], ls="", marker="s", mfc="w", mec="k", label="nunca visto (holdout)"),
                   Line2D([], [], ls="", marker="x", color="k", label=f"faixa de {MARGEM_KM:g} km da borda (fora das métricas)")]
            fig.legend(handles=leg, loc="upper center", ncol=4, fontsize=8,
                       bbox_to_anchor=(0.5, 0.965), frameon=False)
            arq = pasta / f"m{agg}_ev{k}_{stem}.png"
            fig.savefig(arq, dpi=110, bbox_inches="tight")
            plt.close(fig)
            tabela.append(dict(agg=agg, ordem=k, goes_stem=stem,
                               hora_brt=C.stem_para_brt(stem), fold=fold,
                               obs_max_mm=mx, estacao_max=cod_mx,
                               n_estacoes_chuva=n_chuva, arquivo=arq.name))
            print(f"  {arq.name}")
    pd.DataFrame(tabela).to_csv(RES / "mapas_eventos.csv", index=False)


if __name__ == "__main__":
    main()
