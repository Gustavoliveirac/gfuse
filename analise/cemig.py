"""Leitura do radar da Cemig (Mateus Leme) para os mapas: índice dos arquivos por hora de fim (UTC)."""
import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import radar_cemig_tiff as RC

RAIZ_CEMIG = Path(os.environ.get("CEMIG_DIR", "/dados/hidromet/logs/radar_cemig"))
_IDX = None


def indice():
    """{fim da hora em UTC: arquivo}; o instante do nome é o fim do acumulado; dois arquivos, vale o maior."""
    global _IDX
    if _IDX is None:
        _IDX = {}
        if RAIZ_CEMIG.exists():
            for f in RAIZ_CEMIG.rglob("elefante-analysis*"):
                m = RC.NOME.search(f.name)
                if not m or not f.is_file():
                    continue
                t = pd.Timestamp(pd.to_datetime(m.group(1), format="%y%m%d%H%M%S"))
                if t not in _IDX or f.stat().st_size > _IDX[t].stat().st_size:
                    _IDX[t] = f
    return _IDX


def hora(fim_utc):
    """(campo mm em 1 h, lat2d, lon2d) da hora que termina em fim_utc; None se não houver arquivo legível."""
    f = indice().get(pd.Timestamp(fim_utc).round("h"))
    if f is None:
        return None
    try:
        info, a = RC.ler(f)
    except Exception:
        return None
    if RC.LINHAS_DE_SUL_PARA_NORTE:
        a = a[::-1]
    ln, ls, lw, le = RC.caixa(info)
    nj, ni = a.shape
    lat = ln - (np.arange(nj) + 0.5) * (ln - ls) / nj
    lon = lw + (np.arange(ni) + 0.5) * (le - lw) / ni
    lo2, la2 = np.meshgrid(lon, lat)
    return a.astype(float), la2, lo2


def disponivel(fim_utc):
    return pd.Timestamp(fim_utc).round("h") in indice()
