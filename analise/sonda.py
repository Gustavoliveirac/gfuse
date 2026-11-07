#!/usr/bin/env python
"""
sonda.py — levantamento somente leitura do estado do gfuse3 no servidor.

Não treina, não avalia e não altera nada em resultados/ nem em processados/.
Junta num único zip em logs/ tudo o que é preciso para decidir o que entra no
capítulo:

  1. identificação   versão do código (VERSAO do config3.py e git describe)
  2. execução        rodadas no resultados/master.log, braços e folds concluídos,
                     falhas e avisos, processos do gfuse3 rodando agora
  3. artefatos       checkpoints e pares por janela, braço e fold (artefatos.csv)
  4. estações        cada estação da base com fonte, tipo, dentro/fora do recorte,
                     excluída, holdout, papel na avaliação, margem e nº de leituras
                     (estacoes.csv), e o total do metadados.csv por fonte
  5. resultados      cópia dos CSV/TXT consolidados que existem hoje em resultados/
  6. consolidação    roda o resumo.py numa pasta temporária sobre os pares que já
                     existem (rodada em andamento incluída), sem tocar em resultados/
  7. antigos         código e CSVs pequenos de ~/Documentos/gfuse e gfuse2 e
                     inventário das pastas de dados de comparação (IMERG, MERGE,
                     PDIR, GSMaP, Hydro-Estimator)

Uso (de dentro da pasta do gfuse3):
    python analise/sonda.py               completo
    python analise/sonda.py --sem-resumo  pula a etapa 6 (mais leve em memória)

Devolver: o zip logs/sonda_AAAAMMDD_HHMM.zip (o caminho sai na última linha).
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback
import zipfile
import zlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
LOGS = RAIZ / "logs"
RES = RAIZ / "resultados"
PROC = RAIZ / "processados"
HOME = Path.home()
ANTIGOS = [HOME / "Documentos" / "gfuse", HOME / "Documentos" / "gfuse2"]
MERGE_DIR = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/chuvas/merge")
BUSCA_DADOS = [Path("/media/gustavo")]
PADRAO_DADOS = re.compile(r"(pdir|gsmap|hydro|merge|imerg|persiann|cmorph)", re.I)
LIM_ANTIGO_MB = 40          # teto total de CSV/TXT copiados dos projetos antigos
LIM_ARQ_ANTIGO_MB = 3       # teto por arquivo

_buf = io.StringIO()


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _buf.write(s + "\n")


def secao(t):
    p("")
    p("=" * 70)
    p(t)
    p("=" * 70)


def quando(f):
    try:
        return dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def roda(cmd, cwd=RAIZ, timeout=60):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"(falhou: {e})"


def valor_config(nome):
    """Lê uma atribuição simples do config3.py sem importá-lo."""
    txt = (RAIZ / "config3.py").read_text(encoding="utf-8")
    m = re.search(rf'^{nome}\s*=\s*(.+?)\s*(#.*)?$', txt, re.M)
    return m.group(1).strip() if m else None


def caminho_config(nome):
    v = valor_config(nome)
    m = re.search(r'Path\("([^"]+)"\)', v or "")
    return Path(m.group(1)) if m else None


# ----------------------------------------------------------------- 1
def identificacao():
    secao("1. IDENTIFICAÇÃO")
    p("data:", dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "| máquina:", socket.gethostname())
    p("pasta:", RAIZ)
    p("python:", sys.version.split()[0], sys.executable)
    p("VERSAO no config3.py:", valor_config("VERSAO"))
    p("git describe:", roda(["git", "describe", "--tags", "--always", "--dirty"]))
    p("espaço livre:", roda(["df", "-h", str(RAIZ)]).splitlines()[-1])


# ----------------------------------------------------------------- 2
def execucao(z):
    secao("2. EXECUÇÃO (resultados/master.log)")
    ml = RES / "master.log"
    if not ml.exists():
        p("master.log não existe")
    else:
        txt = ml.read_text(encoding="utf-8", errors="replace").splitlines()
        p(f"master.log: {len(txt):,} linhas, modificado em {quando(ml)}")
        cab = [i for i, l in enumerate(txt) if l.startswith("== gfuse3 |")]
        p(f"rodadas registradas: {len(cab)}")
        for i in cab:
            p("   ", txt[i])
        ult = txt[cab[-1]:] if cab else txt
        feitos = [l.strip("- ") for l in ult if re.match(r"^---- m\d+ ", l)]
        p(f"última rodada: {len(feitos)} treinos iniciados")
        p("   " + ", ".join(feitos[-60:]))
        prob = [l for l in ult if re.search(r"PULANDO|FALHA|Traceback|Error|AVISO", l)]
        p(f"última rodada: {len(prob)} linhas com falha ou aviso (até 40)")
        for l in prob[:40]:
            p("   ", l[:200])
        p("últimas 60 linhas:")
        for l in txt[-60:]:
            p("   ", l[:200])
        z.write(ml, "resultados_atuais/master.log")
    p("")
    p("processos do gfuse3 rodando agora:")
    ps = roda(["bash", "-c", "ps -eo pid,etime,pcpu,pmem,args | grep -E 'treino.py|avalia.py|run_all|resumo.py|adveccao.py|mapas_eventos' | grep -v grep"])
    p(ps or "    nenhum")
    gpu = roda(["bash", "-c", "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"])
    if gpu:
        p("GPU:", gpu)


# ----------------------------------------------------------------- 3
def artefatos(z):
    secao("3. ARTEFATOS (checkpoints e pares)")
    linhas = []
    for tipo, base, ext in (("ckpt", RES / "ckpt", ".pt"), ("pares", RES / "pares", ".parquet")):
        if not base.exists():
            p(f"{base} não existe")
            continue
        for f in sorted(base.rglob(f"*{ext}")):
            nome = f.name
            if tipo == "pares" and nome.count(".") > 1:
                continue                              # .areas, .coer, .area
            m = re.match(r"(.+)_f(\d+)" + re.escape(ext) + "$", nome)
            if not m:
                continue
            linhas.append(dict(tipo=tipo, janela=f.parent.name, braco=m.group(1),
                               fold=int(m.group(2)), modificado=quando(f),
                               mb=round(f.stat().st_size / 1e6, 2)))
    if not linhas:
        p("nenhum artefato encontrado")
        return
    import pandas as pd
    d = pd.DataFrame(linhas)
    z.writestr("artefatos.csv", d.to_csv(index=False))
    for (tipo, jan), g in d.groupby(["tipo", "janela"]):
        p(f"{tipo} {jan}:")
        for br, gg in g.groupby("braco"):
            p(f"    {br:<16} folds {sorted(gg.fold.tolist())}  último em {gg.modificado.max()}")
    tr = RES / "treinos.csv"
    if tr.exists():
        z.write(tr, "resultados_atuais/treinos.csv")
        p(f"treinos.csv copiado ({quando(tr)})")


# ----------------------------------------------------------------- 4
def estacoes(z):
    secao("4. ESTAÇÕES")
    import numpy as np
    import pandas as pd
    meta = caminho_config("META_CSV")
    if meta and meta.exists():
        try:
            m = pd.read_csv(meta, sep=None, engine="python")
            cols = [c for c in m.columns if c.lower() in ("fonte", "tipo")]
            p(f"metadados.csv ({meta}): {len(m)} linhas")
            if cols:
                p(m.groupby(cols).size().to_string())
            z.write(meta, "estacoes_metadados.csv")
        except Exception as e:
            p("não consegui ler o metadados.csv:", e)
    npz = PROC / "estacoes.npz"
    if not npz.exists():
        p("processados/estacoes.npz não existe")
        return
    zz = np.load(npz, allow_pickle=False)
    cod = zz["cod"].astype(str); tipo = zz["tipo"].astype(int)
    fonte = zz["fonte"].astype(str); px = zz["px"]; py = zz["py"]
    G = int(valor_config("IMG_SIZE") or 64)
    km = float(valor_config("KM_PER_PX") or 2.0)
    marg = float((valor_config("MARGEM_KM") or "10").split()[0])
    dentro = (px >= -0.5) & (px < G - 0.5) & (py >= -0.5) & (py < G - 0.5)
    dist_borda = np.minimum.reduce([px + 0.5, G - 0.5 - px, py + 0.5, G - 0.5 - py]) * km
    excl_txt = (RAIZ / "config3.py").read_text(encoding="utf-8")
    excl = np.array([f'"{c}"' in excl_txt.split("ESTACOES_EXCLUIDAS", 1)[1][:3000] for c in cod])
    hold = set()
    hf = RES / "holdout.csv"
    if hf.exists():
        hold = set(pd.read_csv(hf).iloc[:, 0].astype(str))
    eh_hold = np.isin(cod, list(hold))
    alvo = np.array([(zlib.crc32(f"{c}|42".encode()) & 1) == 0 for c in cod])
    papel = np.where(~dentro, "contexto (fora)",
             np.where(eh_hold, "holdout",
             np.where(alvo, "alvo", "contexto")))
    n_leit = np.full(len(cod), -1)
    try:
        i0 = np.flatnonzero(tipo == 0); i1 = np.flatnonzero(tipo == 1)
        R10 = zz["R10"]
        n_leit[i0] = np.isfinite(R10[:, :, -1]).sum(0)
        del R10
        RH = zz["RH"]
        n_leit[i1] = np.isfinite(RH).sum(0)
        T = len(zz["stems"])
        p(f"base: {T:,} stems x {len(cod)} estações")
    except Exception as e:
        p("contagem de leituras falhou:", e)
    d = pd.DataFrame(dict(cod=cod, fonte=fonte, tipo=np.where(tipo == 0, "10 min", "horaria"),
                          dentro=dentro, dist_borda_km=np.round(dist_borda, 1),
                          na_margem=dentro & (dist_borda < marg), excluida=excl,
                          papel=papel, n_stems_com_leitura=n_leit))
    z.writestr("estacoes.csv", d.to_csv(index=False))
    p("")
    p("por fonte e tipo (dentro / fora do recorte):")
    p(d.groupby(["fonte", "tipo"]).agg(dentro=("dentro", "sum"),
                                       fora=("dentro", lambda s: int((~s).sum()))).to_string())
    p("")
    p("papel na avaliação (estações de dentro, sem excluídas):")
    dd = d[d.dentro & ~d.excluida]
    p(dd.groupby(["tipo", "papel"]).size().to_string())
    p(f"de dentro na margem de {marg:g} km (fora das métricas): {int(dd.na_margem.sum())}")
    p(f"excluídas pelo config3: {', '.join(d.cod[d.excluida]) or 'nenhuma'}")
    p("observação: no treino a divisão alvo/contexto é sorteada por stem; o papel acima é o da avaliação")


# ----------------------------------------------------------------- 5
def resultados_atuais(z):
    secao("5. RESULTADOS CONSOLIDADOS QUE EXISTEM HOJE EM resultados/")
    if not RES.exists():
        p("resultados/ não existe")
        return
    arqs = sorted([f for f in RES.iterdir() if f.is_file() and f.suffix in (".csv", ".txt")])
    for f in arqs:
        z.write(f, f"resultados_atuais/{f.name}")
    p(f"{len(arqs)} arquivos copiados para resultados_atuais/")
    for f in arqs:
        p(f"    {quando(f)}  {f.stat().st_size / 1e3:>9.1f} kB  {f.name}")
    for f in sorted(PROC.glob("*.csv")) + sorted(PROC.glob("*.json")):
        if f.stat().st_size < 5e6:
            z.write(f, f"processados/{f.name}")
    env = RES / "envio_gfuse3.zip"
    if env.exists():
        p(f"envio_gfuse3.zip existe ({quando(env)}, {env.stat().st_size / 1e6:.1f} MB); os CSVs dele são os de cima")


# ----------------------------------------------------------------- 6
def consolidacao(z):
    secao("6. CONSOLIDAÇÃO DOS PARES QUE JÁ EXISTEM (resumo.py numa pasta temporária)")
    tmp = Path(tempfile.mkdtemp(prefix="sonda_resumo_"))
    codigo = ("import sys; sys.path.insert(0, '.'); from pathlib import Path; "
              "import config3; config3.RES = Path(sys.argv[1]); "
              "sys.argv = ['resumo.py']; import resumo; resumo.main()")
    p(f"pasta temporária: {tmp} (resultados/ não é alterada)")
    try:
        r = subprocess.run([sys.executable, "-c", codigo, str(tmp)], cwd=RAIZ,
                           capture_output=True, text=True, timeout=6 * 3600)
        saida = (r.stdout + r.stderr).splitlines()
        p(f"resumo.py terminou com código {r.returncode}; últimas 40 linhas:")
        for l in saida[-40:]:
            p("   ", l[:200])
        z.writestr("consolidado_agora/resumo_stdout.txt", "\n".join(saida))
        n = 0
        for f in sorted(tmp.iterdir()):
            if f.is_file() and f.suffix in (".csv", ".txt"):
                z.write(f, f"consolidado_agora/{f.name}"); n += 1
        p(f"{n} arquivos em consolidado_agora/")
    except Exception as e:
        p("consolidação falhou:", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------- 7
def antigos(z):
    secao("7. PROJETOS ANTIGOS E DADOS DE COMPARAÇÃO")
    usado = 0.0
    for raiz in ANTIGOS:
        if not raiz.exists():
            p(f"{raiz} não existe")
            continue
        p(f"{raiz}:")
        for dirpath, dirnames, files in os.walk(raiz):
            prof = Path(dirpath).relative_to(raiz).parts
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))
                           and d not in ("ckpt", "checkpoints", "cache", "node_modules")]
            if len(prof) >= 3:
                dirnames[:] = []
            for fn in sorted(files):
                f = Path(dirpath) / fn
                rel = f.relative_to(raiz)
                try:
                    mb = f.stat().st_size / 1e6
                except OSError:
                    continue
                if f.suffix in (".py", ".sh", ".md") and mb < 0.5:
                    z.write(f, f"antigos/{raiz.name}/{rel}")
                    p(f"    código  {quando(f)}  {rel}")
                elif f.suffix in (".csv", ".txt", ".log") and mb <= LIM_ARQ_ANTIGO_MB \
                        and usado + mb <= LIM_ANTIGO_MB:
                    z.write(f, f"antigos/{raiz.name}/{rel}"); usado += mb
                    p(f"    tabela  {quando(f)}  {mb:6.2f} MB  {rel}")
                elif f.suffix in (".csv", ".txt", ".log", ".parquet", ".zip"):
                    p(f"    (não copiado) {mb:8.2f} MB  {rel}")
    p("")
    p("pastas de dados de comparação:")
    vistos = set()
    alvos = [caminho_config("IMERG_DIR"), MERGE_DIR]
    for base in BUSCA_DADOS:
        if not base.exists():
            continue
        for dirpath, dirnames, files in os.walk(base):
            prof = len(Path(dirpath).relative_to(base).parts)
            if prof >= 5:
                dirnames[:] = []
                continue
            if len(files) > 2000:
                dirnames[:] = []               # pasta de dados: não descer mais
            for d in dirnames:
                if PADRAO_DADOS.search(d):
                    alvos.append(Path(dirpath) / d)
    for a in alvos:
        if a is None or a in vistos or not a.exists():
            continue
        vistos.add(a)
        try:
            nomes = sorted(e.name for e in os.scandir(a) if e.is_file())
            sub = sorted(e.name for e in os.scandir(a) if e.is_dir())
        except OSError as e:
            p(f"    {a}: erro {e}")
            continue
        p(f"    {a}")
        p(f"        {len(nomes):,} arquivos, {len(sub)} subpastas"
          + (f" ({', '.join(sub[:8])}{' ...' if len(sub) > 8 else ''})" if sub else ""))
        if nomes:
            p(f"        primeiro {nomes[0]} | último {nomes[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-resumo", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    carimbo = dt.datetime.now().strftime("%Y%m%d_%H%M")
    zpath = LOGS / f"sonda_{carimbo}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        etapas = [identificacao, lambda: execucao(z), lambda: artefatos(z),
                  lambda: estacoes(z), lambda: resultados_atuais(z)]
        if not a.sem_resumo:
            etapas.append(lambda: consolidacao(z))
        etapas.append(lambda: antigos(z))
        for e in etapas:
            try:
                e()
            except Exception:
                p("ERRO nesta etapa (a sonda continua):")
                p(traceback.format_exc())
        secao("FIM")
        p(f"devolver: {zpath}")
        z.writestr("sonda.log", _buf.getvalue())
    (LOGS / "sonda.log").write_text(_buf.getvalue(), encoding="utf-8")
    print(f"\nDEVOLVER: {zpath}")


if __name__ == "__main__":
    main()
