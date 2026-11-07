#!/usr/bin/env python
"""
adveccao.py — teste de advecção pura, sem treino.

Parte de uma análise do campo de chuva de 10 min em t0 feita com TODAS as
estações (krigagem simples; e o estágio 2 "s2", se treinado), projeta esse
campo para t0+10, +20, +30 e +60 min SEM estações e SEM GOES, e verifica nas
estações no instante futuro. Mede se o movimento estimado ajuda a seguir a
chuva e por quanto tempo um campo advectado continua parecido com o real.

Projeções comparadas
  persistencia   campo parado
  goes_global    vetor único do movimento.csv (topos frios entre t0-10 e t0)
  goes_local     campo de movimento por blocos de 32 px (passo 16), com o
                 vetor global onde o bloco não tem topo frio suficiente
  vento          média vetorial do vento das estações meteorológicas em t0
                 (o vento sopra DE wind_dir; o deslocamento é no sentido oposto)
  goes_vento     média do goes_global com o vento (quando os dois existem)
  estacoes       movimento estimado SÓ pelas estações: deslocamento que melhor
                 leva o campo krigado de t0-10 ao de t0 (vale em tempo real)
  oraculo        deslocamento que melhor leva o campo krigado de t0 ao de
                 t0+k (usa o futuro; é o limite superior do que a advecção
                 poderia render se o movimento fosse conhecido)
A projeção é semi-lagrangiana: campo(t0+k)(x) = campo(t0)(x - k*V(x)).

Saídas em resultados/: adveccao_resumo.csv (método x antecedência x
análise), adveccao_vento.csv (comparação entre o movimento do GOES e o vento),
adveccao_movimento.csv (velocidade e rumo de cada fonte de movimento, fração
de casos parados e concordância com o oráculo)

Uso:  python adveccao.py [--analises krig s2] [--passo 2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import comum as C
from config3 import (ADV_BLOCO_PX, ADV_PASSO_BLOCO, ADV_PASSOS, CKPT_DIR, GOES_DIR,
                     IMG_SIZE, KM_PER_PX, MARGEM_KM, MET_DIR, MET_FONTES, MOV_FILE,
                     PROC, RES)
from estacoes import Base
from movimento import campo_frio, deslocamento

LIMIARES = [0.2, 2.0, 5.0]
G = IMG_SIZE


def deslocar_campo(campo, U, V, k):
    """Semi-lagrangiano bilinear: saida(x) = campo(x - k*V(x)); zero fora."""
    ys, xs = np.mgrid[0:G, 0:G].astype(np.float32)
    xo = xs - k * U; yo = ys - k * V
    x0 = np.floor(xo).astype(int); y0 = np.floor(yo).astype(int)
    fx = xo - x0; fy = yo - y0
    out = np.zeros((G, G), np.float32)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < G) & (yi >= 0) & (yi < G)
        out[ok] += (w * campo[np.clip(yi, 0, G - 1), np.clip(xi, 0, G - 1)])[ok]
    return out


def movimento_local(f0, f1, u_g, v_g):
    """Campo (U, V) por blocos, interpolado bilinearmente entre os centros."""
    cent = list(range(ADV_BLOCO_PX // 2, G - ADV_BLOCO_PX // 2 + 1, ADV_PASSO_BLOCO))
    n = len(cent)
    Ub = np.full((n, n), u_g, np.float32); Vb = np.full((n, n), v_g, np.float32)
    h = ADV_BLOCO_PX // 2
    for i, cy in enumerate(cent):
        for j, cx in enumerate(cent):
            a, b = f0[cy - h:cy + h, cx - h:cx + h], f1[cy - h:cy + h, cx - h:cx + h]
            if (a > 0).mean() < 0.02 or (b > 0).mean() < 0.02:
                continue
            u, v, corr = deslocamento(a, b, max_px=6)
            if np.isfinite(corr) and corr >= 0.3:
                Ub[i, j], Vb[i, j] = u, v
    eixo = np.arange(G, dtype=float)
    def interp(B):
        lin = np.array([np.interp(eixo, cent, B[i]) for i in range(n)])
        return np.array([np.interp(eixo, cent, lin[:, j]) for j in range(G)]).T
    return interp(Ub).astype(np.float32), interp(Vb).astype(np.float32)


def vento_estacoes(base):
    """(T, 2) deslocamento médio em px/10 min dado pelo vento das estações
    meteorológicas (NaN sem dado)."""
    meta_f = MET_DIR / "metadados.csv"
    if not meta_f.exists():
        print(f"[adveccao] AVISO: {meta_f} não existe, sem projeção por vento")
        return np.full((base.T, 2), np.nan)
    meta = pd.read_csv(meta_f, sep=";")
    meta = meta[meta["Tipo"].astype(str).str.lower().str.startswith("met")
                & meta["Fonte"].isin(MET_FONTES)]
    U, Vv, N = np.zeros(base.T), np.zeros(base.T), np.zeros(base.T)
    for _, r in meta.iterrows():
        arq = MET_DIR / str(r["Fonte"]) / f"{str(r['Codigo']).strip()}.csv"
        if not arq.exists():
            continue
        try:
            d = pd.read_csv(arq, sep=";", usecols=lambda c: c in ("date", "wind_speed", "wind_dir"))
            d["date"] = pd.to_datetime(d["date"])
        except Exception:
            continue
        if not {"wind_speed", "wind_dir"} <= set(d.columns):
            continue
        d = d.dropna().sort_values("date").drop_duplicates("date")
        d["date"] = d["date"].astype("datetime64[ns]")
        tol = pd.Timedelta("12min") if "10" in str(r["Frequencia"]) else pd.Timedelta("35min")
        a = pd.DataFrame({"t": base.t.astype("datetime64[ns]"),
                          "k": np.arange(base.T)}).sort_values("t")
        m = pd.merge_asof(a, d, left_on="t", right_on="date", tolerance=tol,
                          direction="nearest").dropna(subset=["wind_speed"])
        ws = m["wind_speed"].values.astype(float); wd = np.radians(m["wind_dir"].values.astype(float))
        px_10 = ws * 600.0 / (KM_PER_PX * 1000.0)        # m/s -> px por 10 min
        # sopra DE wd: desloca para wd+180 -> leste = -sin, sul = +cos
        U[m["k"].values] += -px_10 * np.sin(wd)
        Vv[m["k"].values] += px_10 * np.cos(wd)
        N[m["k"].values] += 1
    out = np.full((base.T, 2), np.nan)
    ok = N > 0
    out[ok, 0], out[ok, 1] = U[ok] / N[ok], Vv[ok] / N[ok]
    print(f"[adveccao] vento das estações em {ok.mean():.0%} dos stems "
          f"({len(meta)} estações meteorológicas no metadados)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analises", nargs="*", default=["krig", "s2"])
    ap.add_argument("--passo", type=int, default=1, help="usa 1 a cada N stems chuvosos")
    ap.add_argument("--max", type=int, default=0, help="limite de t0 (teste)")
    args = ap.parse_args()
    base = Base()
    W = base.janelas(10)
    vario = C.carregar_vario(10, PROC)
    m_px = MARGEM_KM / KM_PER_PX
    verif = base.dentro & (base.px >= m_px - 0.5) & (base.px <= G - 0.5 - m_px) \
        & (base.py >= m_px - 0.5) & (base.py <= G - 0.5 - m_px)
    mov = pd.read_csv(MOV_FILE).set_index("goes_stem") if MOV_FILE.exists() else None
    vento = vento_estacoes(base)
    chuv = base.chuvoso(10)
    t0s = np.flatnonzero(chuv)[::args.passo]
    if args.max:
        t0s = t0s[:args.max]
    print(f"[adveccao] {len(t0s):,} instantes iniciais com chuva")

    # ---------------- análise s2 (opcional) ----------------
    s2 = None
    if "s2" in args.analises:
        try:
            s2 = AnaliseS2(base, vario)
        except Exception as e:
            print(f"[adveccao] análise s2 indisponível ({e}); seguindo só com a krigagem")

    acc = {}
    comp_vento, movs = [], []
    passos = ADV_PASSOS
    cache_sk = {}

    def krig(kk):
        """Krigagem simples com todas as estações válidas no stem kk."""
        if kk not in cache_sk:
            wk = W[kk]; vk = np.isfinite(wk)
            cache_sk[kk] = (C.krigagem_simples(base.px[vk], base.py[vk], wk[vk], vario)[0]
                            if vk.sum() >= 3 else None)
            if len(cache_sk) > 64:
                cache_sk.pop(next(iter(cache_sk)))
        return cache_sk[kk]

    def desloc_campos(a, b, max_px):
        if a is None or b is None or a.max() <= 0 or b.max() <= 0:
            return np.nan, np.nan, np.nan
        return deslocamento(a.astype(np.float32), b.astype(np.float32), max_px=max_px)

    for n_t, k in enumerate(t0s):
        s = base.stems[k]
        w0 = W[k]
        val = np.isfinite(w0)
        if val.sum() < 3:
            continue
        sk = krig(k)
        analises = {"krig": sk}
        if s2 is not None:
            a = s2(s)
            if a is not None:
                analises["s2"] = a
        # movimentos
        ug = vg = 0.0
        if mov is not None and s in mov.index:
            ug, vg = float(mov.at[s, "u"]), float(mov.at[s, "v"])
        Ul = Vl = None
        kp = base.deslocado(-10)[k]
        if kp >= 0:
            try:
                f0 = campo_frio(np.load(GOES_DIR / f"{base.stems[kp]}.npy", mmap_mode="r")[5])
                f1 = campo_frio(np.load(GOES_DIR / f"{s}.npy", mmap_mode="r")[5])
                Ul, Vl = movimento_local(f0, f1, ug, vg)
            except Exception:
                pass
        um = {"persistencia": (0.0, 0.0), "goes_global": (ug, vg)}
        ue = ve = ce = np.nan
        if kp >= 0:
            ue, ve, ce = desloc_campos(krig(kp), sk, 6)
        if np.isfinite(ce) and ce >= 0.3:
            um["estacoes"] = (ue, ve)
        else:
            um["estacoes"] = (0.0, 0.0)                   # sem estimativa: parado
        j1 = base.deslocado(10)[k]
        uo = vo = co = np.nan
        if j1 >= 0:
            uo, vo, co = desloc_campos(sk, krig(j1), 6)
        movs.append(dict(goes_stem=s, u_goes=ug, v_goes=vg,
                         u_vento=vento[k, 0], v_vento=vento[k, 1],
                         u_est=ue, v_est=ve, corr_est=ce,
                         u_orac=uo, v_orac=vo, corr_orac=co))
        if Ul is not None:
            um["goes_local"] = (Ul, Vl)
        if np.isfinite(vento[k]).all():
            um["vento"] = (vento[k, 0], vento[k, 1])
            um["goes_vento"] = ((ug + vento[k, 0]) / 2, (vg + vento[k, 1]) / 2)
            comp_vento.append(dict(goes_stem=s, u_goes=ug, v_goes=vg,
                                   u_vento=vento[k, 0], v_vento=vento[k, 1]))
        for passo in passos:
            j = base.deslocado(10 * passo)[k]
            if j < 0:
                continue
            obs = W[j]
            vv = np.isfinite(obs) & verif
            if not vv.any():
                continue
            xi = np.round(base.px[vv]).astype(int); yi = np.round(base.py[vv]).astype(int)
            o = obs[vv]
            mov_k = dict(um)
            dxo, dyo, co_k = desloc_campos(sk, krig(j), min(6 * passo, 24))
            mov_k["oraculo"] = ((dxo / passo, dyo / passo) if np.isfinite(co_k)
                                else (0.0, 0.0))
            for nome_a, campo in analises.items():
                for nome_m, (U, V) in mov_k.items():
                    Uf = np.full((G, G), U, np.float32) if np.isscalar(U) else U
                    Vf = np.full((G, G), V, np.float32) if np.isscalar(V) else V
                    proj = deslocar_campo(campo, Uf, Vf, passo)[yi, xi]
                    a = acc.setdefault((nome_a, nome_m, passo), dict(
                        n=0, sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0,
                        **{f"{c}{t:g}": 0 for t in LIMIARES for c in "hmf"}))
                    a["n"] += len(o); a["sx"] += proj.sum(); a["sy"] += o.sum()
                    a["sxx"] += (proj ** 2).sum(); a["syy"] += (o ** 2).sum()
                    a["sxy"] += (proj * o).sum()
                    for t in LIMIARES:
                        p_, o_ = proj > t, o > t
                        a[f"h{t:g}"] += int((p_ & o_).sum()); a[f"m{t:g}"] += int((~p_ & o_).sum())
                        a[f"f{t:g}"] += int((p_ & ~o_).sum())
        if n_t % 5000 == 0:
            print(f"  ... {n_t}/{len(t0s)}")

    linhas = []
    for (na, nm, p), a in sorted(acc.items()):
        n = a["n"]
        cov = a["sxy"] / n - a["sx"] * a["sy"] / n ** 2
        vx = a["sxx"] / n - (a["sx"] / n) ** 2; vy = a["syy"] / n - (a["sy"] / n) ** 2
        row = dict(analise=na, metodo=nm, antecedencia_min=10 * p, n=n,
                   r=cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan,
                   razao_volume=a["sx"] / a["sy"] if a["sy"] > 0 else np.nan)
        for t in LIMIARES:
            h, m, f = a[f"h{t:g}"], a[f"m{t:g}"], a[f"f{t:g}"]
            row[f"POD{t:g}"] = h / (h + m) if h + m else np.nan
            row[f"FAR{t:g}"] = f / (h + f) if h + f else np.nan
            row[f"CSI{t:g}"] = h / (h + m + f) if h + m + f else np.nan
        linhas.append(row)
    rs = pd.DataFrame(linhas)
    rs.to_csv(RES / "adveccao_resumo.csv", index=False)
    if comp_vento:
        cv = pd.DataFrame(comp_vento)
        vel = lambda u, v: np.hypot(u, v) * KM_PER_PX * 6
        rumo = lambda u, v: (np.degrees(np.arctan2(u, -v)) + 360) % 360
        cv["vel_goes_kmh"] = vel(cv.u_goes, cv.v_goes); cv["vel_vento_kmh"] = vel(cv.u_vento, cv.v_vento)
        cv["rumo_goes"] = rumo(cv.u_goes, cv.v_goes); cv["rumo_vento"] = rumo(cv.u_vento, cv.v_vento)
        dif = (cv["rumo_goes"] - cv["rumo_vento"] + 180) % 360 - 180
        resumo_v = pd.DataFrame([dict(
            n=len(cv), vel_goes_mediana_kmh=cv["vel_goes_kmh"].median(),
            vel_vento_mediana_kmh=cv["vel_vento_kmh"].median(),
            rumo_goes_mediano=cv["rumo_goes"].median(), rumo_vento_mediano=cv["rumo_vento"].median(),
            dif_rumo_mediana_abs=float(np.abs(dif).median()),
            corr_u=float(np.corrcoef(cv.u_goes, cv.u_vento)[0, 1]) if len(cv) > 2 else np.nan,
            corr_v=float(np.corrcoef(cv.v_goes, cv.v_vento)[0, 1]) if len(cv) > 2 else np.nan)])
        resumo_v.to_csv(RES / "adveccao_vento.csv", index=False)
    if movs:
        mv = pd.DataFrame(movs)
        vel = lambda u, v: np.hypot(u, v) * KM_PER_PX * 6
        rumo = lambda u, v: (np.degrees(np.arctan2(u, -v)) + 360) % 360
        lin = []
        for nome, (cu, cv, cc) in {"goes": ("u_goes", "v_goes", None),
                                   "vento": ("u_vento", "v_vento", None),
                                   "estacoes": ("u_est", "v_est", "corr_est"),
                                   "oraculo_10min": ("u_orac", "v_orac", "corr_orac")}.items():
            d = mv[np.isfinite(mv[cu]) & np.isfinite(mv[cv])]
            if cc:
                d = d[d[cc] >= 0.3]
            if not len(d):
                continue
            o = mv.loc[d.index]
            ok_o = np.isfinite(o["u_orac"]) & (o["corr_orac"] >= 0.3)
            lin.append(dict(fonte=nome, n=len(d),
                            vel_mediana_kmh=float(vel(d[cu], d[cv]).median()),
                            vel_p90_kmh=float(vel(d[cu], d[cv]).quantile(.9)),
                            rumo_mediano=float(rumo(d[cu], d[cv]).median()),
                            frac_parado=float((np.hypot(d[cu], d[cv]) < 0.5).mean()),
                            corr_u_com_oraculo=float(np.corrcoef(d.loc[ok_o, cu], o.loc[ok_o, "u_orac"])[0, 1])
                            if ok_o.sum() > 2 else np.nan,
                            corr_v_com_oraculo=float(np.corrcoef(d.loc[ok_o, cv], o.loc[ok_o, "v_orac"])[0, 1])
                            if ok_o.sum() > 2 else np.nan))
        pd.DataFrame(lin).to_csv(RES / "adveccao_movimento.csv", index=False)
        print("[adveccao] movimento por fonte:")
        print(pd.DataFrame(lin).round(2).to_string(index=False))
    if len(rs):
        print("[adveccao] r e CSI(>0,2 mm) por antecedência:")
        piv = rs.pivot_table(index=["analise", "metodo"], columns="antecedencia_min",
                             values=["r", "CSI0.2"])
        print(piv.round(3).to_string())
    print(f"[adveccao] csv em {RES}")


class AnaliseS2:
    """Análise do s2 (estágio 2) em t0 com TODAS as estações como contexto,
    usando o checkpoint do fold em que t0 está na validação."""

    def __init__(self, base, vario):
        import torch
        import modelo as M
        from config3 import ARM_FLAGS, EDGES
        self.torch, self.M, self.EDGES = torch, M, EDGES
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stems = C.stems_alinhados(list(base.stems), 10)
        self.fold_de = {s: fd["fold"] for fd in C.folds_temporais(stems) for s in fd["val"]}
        self.flags = ARM_FLAGS["s2"]
        holdout = set(pd.read_csv(RES / "holdout.csv")["station_code"].astype(str))
        self.ds = M.DatasetGfuse([], base, 10, self.flags, holdout, vario,
                                 treino=False, movimento=M.Movimento())
        self.redes = {}
        if not any((CKPT_DIR / "m10" / f"s2_f{f}.pt").exists() for f in range(5)):
            raise RuntimeError("nenhum checkpoint do s2 no m10")

    def _rede(self, fold):
        if fold not in self.redes:
            ck = CKPT_DIR / "m10" / f"s2_f{fold}.pt"
            if not ck.exists():
                self.redes[fold] = None
            else:
                net = self.M.criar_rede(self.flags, 10, self.dev)
                net.load_state_dict(self.torch.load(ck, map_location=self.dev, weights_only=True))
                s1 = self.M.carregar_s1(CKPT_DIR, 10, fold, self.dev)
                import json
                try:
                    thr = float(json.load(open(ck.with_suffix(".json"))).get("thr_chuva", 0.5))
                except Exception:
                    thr = 0.5
                self.redes[fold] = (net.eval(), s1, thr)
        return self.redes[fold]

    def __call__(self, stem):
        fold = self.fold_de.get(stem)
        r = self._rede(fold) if fold is not None else None
        if r is None:
            return None
        net, s1, thr = r
        am = self.ds.amostra(stem, alvos=set())       # todas as estações no contexto
        with self.torch.no_grad():
            mm, pt = self.M.decodificar(self.M.prever(
                net, am["cubo"][None].to(self.dev), am["estat"][None].to(self.dev),
                self.flags, 10, s1), 10)
        p = pt[self.EDGES[10][0]][0]
        return np.where(p >= thr, mm[0] / np.maximum(p, 1e-6), 0.0).astype(np.float32)


if __name__ == "__main__":
    main()
