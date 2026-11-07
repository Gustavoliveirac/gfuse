#!/usr/bin/env python
"""
config3.py — caminhos e constantes do gfuse3. Tudo que os demais módulos
precisam saber está aqui; nenhum módulo importa código de pastas antigas.

Únicos caminhos fora desta pasta (edite aqui embaixo se algum estiver
errado na sua máquina — nada mais no código precisa mudar):
  GOES_DIR   .npy 64x64 do GOES (8 canais: C08..C14 + RRQPE), 1 por scan
  IMERG_DIR  HDF5 do IMERG Early (só lido UMA vez pelo mw_prep.py)
  EST_DIR    pasta com <FONTE>/<cod>.csv das séries processadas (sep ';')
  META_CSV   o metadados.csv do projeto (sep ';')
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# ---------- caminhos externos (edite se necessário) ----------
GOES_DIR = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/canada/numpy_64px")
IMERG_DIR = Path("/media/gustavo/Elements/imerg/imerg_data/early")
# recorte de 128 px (mesmo centro do de 64), gerado pelo preProcess; usado só
# para estimar o vetor de movimento. Se não existir, usa o de 64 px.
GOES128_DIR = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/canada/numpy_128px")
# Saída do QA/QC (run_qc.py): mesmas séries de "processados", com as colunas
# de flag e o status_final por registro (0 válido, 1 atenção, 2 inválido).
# Até a v12 o gfuse3 lia "processados" (antes do QA/QC) e ignorava as flags,
# por isso entraram valores como 234,6 mm em 10 min.
EST_DIR  = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados_qaqc")
META_CSV = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados_qaqc/metadados.csv")
# registros com status_final acima disto são descartados (2 = inválido)
QAQC_STATUS_MAX = 1

# ---------- caminhos locais (só o que este código gera) ----------
DADOS     = RAIZ / "dados"
MW_DIR    = DADOS / "mw"                    # npz gravados pelo mw_prep.py
PROC      = RAIZ / "processados"            # parquet, stats, variogramas
EST_NPZ   = PROC / "estacoes.npz"         # matrizes tempo x estação (dados.py)
QC_NPZ    = PROC / "qc_estacoes.npz"      # registros suspeitos (qc_estacoes.py)
MEDIAS_F  = PROC / "medias_classes.json"  # média observada por compartimento
STATS_F   = PROC / "goes_stats.json"
RES       = RAIZ / "resultados"             # checkpoints, pares, csv, logs
CKPT_DIR  = RES / "ckpt"
PARES_DIR = RES / "pares"

# ---------- estações excluídas ----------
# Estações retiradas da base (não entram em treino, contexto nem avaliação).
# Critério: diag_satelite.py (10/2026, base de 135.223 stems). Com a regra
# "seco certo" (Tb 10,3 µm mínima num raio de 12 km acima de 270 K e RRQPE
# zero no mesmo raio), a taxa mediana de chuva registrada nas estações foi de
# 0,24%. As duas abaixo ficaram muito acima disso, o que indica registro
# espúrio ou problema de horário, não chuva quente local (as estações
# seguintes da lista ficaram entre 1% e 2%). Revisar a série original antes
# de reincluir. A lista completa está em resultados/seco_violacoes_estacao.csv
# e, a partir da v11, também em resultados/qc_estacoes.csv.
ESTACOES_EXCLUIDAS = {
    "315780724H": "CEMADEN; 17,2% de chuva sob céu limpo (1.559 de 9.040 "
                  "casos), cerca de 70 vezes a mediana das estações",
    "314610710A": "CEMADEN; 5,9% de chuva sob céu limpo (1.400 de 23.712 "
                  "casos), cerca de 25 vezes a mediana das estações",
}

# ---------- limite físico dos registros ----------
# Rede de segurança além do QA/QC: valores acima disso viram dado ausente no
# dados.py. Motivo: o rápido da v11 escolheu como "eventos extremos"
# registros de 234,6 mm e 113 mm em 10 min em estações isoladas (só 1 a 3
# estações com chuva no domínio), fisicamente impossíveis. O QA/QC marca 10
# min acima de 50 mm como limite físico, mas, se a hora for corroborada por
# vizinhas, o registro fica com status 1 (atenção) e não 2; por isso o
# limite é mantido aqui também. 50 mm em 10 min equivale a 300 mm/h.
LIMITE_10MIN_MM = 50.0
LIMITE_HORA_MM = 150.0

# ---------- controle de qualidade dos registros (qc_estacoes.py) ----------
# FORA DO PIPELINE desde a v12: os dados já passaram por controle de
# qualidade; o script fica como ferramenta para a rotina de QA/QC futura e só
# é aplicado se rodado à mão e QC_APLICAR = True.
QC_APLICAR = False
# Chuva suspeita: chuva > 0,2 mm em 10 min com Tb mínima > QC_T0 num raio de
# QC_RAIO_PX em todos os quadros de t-QC_PASSADO_MIN até t, RRQPE zero no
# mesmo raio e período, e nenhuma estação a até QC_RAIO_PX com chuva em t.
# Esses registros viram dado ausente (não entram em treino nem avaliação).
# Zero suspeito (pluviômetro entupido): zero com Tb < 220 K no pixel, RRQPE
# >= 5 mm/h e pelo menos 3 vizinhas a até QC_RAIO_PX com >= 2 mm em 10 min.
# Só é reportado, não removido.
QC_T0 = 270.0
QC_RAIO_PX = 6
QC_PASSADO_MIN = 40

# ---------- supervisão do seco ----------
# Pixel "seco certo" recebe alvo seco na perda, inclusive na borda e longe
# das estações. Regra escolhida pelo diag_satelite.py: Tb mínima > 270 K num
# raio de 12 km e RRQPE zero no mesmo raio violou 0,35% dos registros e
# marcou 41% do domínio. Para cobrir o adiantamento de 20 a 30 min do
# satélite, a condição vale para os quadros dos últimos SECO_PASSADO_MIN
# (m10) ou todos os 6 quadros (m30/m60). Pixels a até SECO_RAIO_PX de uma
# estação de contexto com chuva saem da máscara (a estação prevalece).
SECO_T0 = 270.0
SECO_RAIO_PX = 6
SECO_PASSADO_MIN = 30
LAMBDA_SECO = 0.5

# ---------- limiar de decisão da chuva ----------
# Probabilidade calibrada de um evento com ~5% de frequência raramente passa
# de 0,5, e a decisão em 0,5 subestimava a área com chuva (FBIAS 0,6 a 0,8 no
# completo da v13). Depois de cada treino, o limiar que maximiza o CSI de
# P(chuva) é escolhido em N_STEMS_LIMIAR stems do bloco de TREINO do fold
# (nunca na validação) e usado nos mapas, no produto <braço>_det e na decisão
# da ocorrência.
N_STEMS_LIMIAR = 1500

# ---------- teste de advecção pura (adveccao.py) ----------
# Análise em t0 com todas as estações (krigagem simples; e o s2 se treinado),
# projetada para t0+k sem estações nem GOES, verificada nas estações em t0+k.
ADV_PASSOS = [1, 2, 3, 6]                  # x 10 min
ADV_BLOCO_PX = 32                          # blocos do movimento local
ADV_PASSO_BLOCO = 16
# Vento das estações meteorológicas (série bruta, com as colunas wind_speed
# em m/s e wind_dir em graus, de onde o vento sopra)
MET_DIR = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados")
MET_FONTES = ["INMET", "SUDECAP_METEO"]

# ---------- avaliação ----------
MARGEM_KM = 10.0           # estações a menos disso da borda não entram nas métricas

# ---------- domínio ----------
BH_LAT, BH_LON = -19.9068, -43.9765
IMG_SIZE   = 64
KM_PER_PX  = 2.0
N_FRAMES   = 6
STEP_MIN   = 10
N_CH_RAW   = 8            # 7 radiâncias IR (C08..C14) + RRQPE (mm/h)
RRQPE_CH   = 7
SENTINEL   = -9999.0

# número de onda central nominal (cm-1) das bandas do cubo, índice 0..6
# (GOES-16 ABI; usado na conversão radiância -> Tb, correção de banda omitida
#  por ser idêntica em todos os arquivos)
NU_BANDAS = [1615.75, 1447.87, 1364.28, 1184.51, 1041.43, 968.18, 894.29]
# pares de diferença espectral (índices no cubo): C08-C13, C10-C13, C11-C13
DIFF_PARES = [(0, 5), (2, 5), (3, 5)]
N_CH_DIFF  = len(DIFF_PARES)

# ---------- janelas, escada e nota ----------
AGGS = [10, 30, 60]
# limites dos compartimentos do histograma = escada de limiares (P(>tau) exata)
EDGES = {10: [0.2, 1.0, 2.0, 5.0, 10.0, 20.0],
         30: [0.2, 1.0, 5.0, 10.0, 20.0, 30.0],
         60: [0.2, 1.0, 5.0, 10.0, 20.0, 30.0]}
# ponto médio (mm) de cada classe, p/ campo E[mm] (última classe: valor típico)
# BIN_MID: valor típico de cada compartimento, usado só se ainda não existir
# processados/medias_classes.json (gerado pelo dados.py com a média observada
# de cada compartimento; o ponto médio fixo superestimava o volume)
BIN_MID = {10: [0.0, 0.5, 1.5, 3.5, 7.5, 15.0, 30.0],
           30: [0.0, 0.5, 3.0, 7.5, 15.0, 25.0, 45.0],
           60: [0.0, 0.5, 3.0, 7.5, 15.0, 25.0, 50.0]}
LADDER   = EDGES                              # escada de avaliação = limites
TOL_WINS = [0, 1, 2]                          # exato, 3x3 (~2 km), 5x5 (~4 km)
MIN_EVENTOS = 30                              # regra dos 30 casos

# ---------- amostragem / holdout / folds ----------
MAX_CTX, MAX_TGT, MAX_HLD = 220, 96, 60
TARGET_FRAC   = 0.5
HOLDOUT_FRAC  = 0.2
HOLDOUT_SEED  = 42
EVAL_SEED     = 42
K_FOLDS       = 5
GAP_HORAS     = 6.0        # folga temporal entre treino e bloco de validação
FRAC_CHUVOSO  = 0.5        # fração das amostras de treino tiradas de stems com
                           # chuva (> 0,2 mm) em alguma estação; o resto, dos demais

# ---------- variantes ----------
# braços (v14): base, hurdle, krig_res, s1 e s2 por padrão; hist, rec e o
# filtro de células saíram da lista padrão (sem ganho no completo da v13).
# braços (descrição):
#   base   rede de histograma, GOES + splat das estações de contexto
#   hist   base + janelas anteriores das estações de contexto, deslocadas
#          pelo vetor de movimento (estratégia 1)
#   hurdle base + duas cabeças (ocorrência calibrada e intensidade
#          condicional), tolerância espacial por média gaussiana
#   rec    análise recursiva de 10 min: recebe a própria estimativa anterior
#          advectada; 30 e 60 min saem da soma das análises de 10 min
#          (estratégia 2)
#   krig_res  hurdle com a ocorrência partindo da krigagem (ver ARM_FLAGS)
# O filtro de células sem antecessor (estratégia 3) é aplicado na avaliação
# do m10 a todos os braços e aparece como <braço>_filtro.
# interp, di, diff e mw_di continuam disponíveis via ARMS=...
# v15 mantém VERSAO = "v14": os checkpoints já treinados são reaproveitados e
# só os braços novos (s1_mw e s2rk) são treinados.
ARMS = ["base", "hurdle", "krig_res", "s1", "s2", "s1_mw", "s2rk"]
_F = dict(diff=False, interp=False, mw=False, hurdle=False, hist=False,
          rec=False, krig=False, s1=False, s2=False, rk=False)
ARM_FLAGS = {"base":     dict(_F),
             "hurdle":   dict(_F, hurdle=True),
             # krig_res: estrutura do hurdle, com o logit da ocorrência
             # partindo da krigagem de indicador ponderada pelo peso de
             # informação (perto das estações a rede só corrige a krigagem;
             # longe delas decide pelo satélite)
             "krig_res": dict(_F, hurdle=True, krig=True),
             # s1: estágio 1, SÓ o GOES na entrada (nenhuma estação), cabeças
             # do hurdle; todas as estações fora do holdout são alvo no treino.
             # Na avaliação gera também o produto s1_rk (estágio 2 analítico:
             # krigagem simples do erro do s1 nas estações de contexto).
             "s1":       dict(_F, hurdle=True, s1=True),
             # s2: estágio 2 aprendido, rede leve que recebe a saída congelada
             # do s1 do mesmo fold, a krigagem e as estações de contexto
             "s2":       dict(_F, hurdle=True, s2=True),
             # s1_mw: s1 com pré-treino denso GOES -> MWprecipitation (IMERG),
             # que foi pensado para um estimador só de satélite
             "s1_mw":    dict(_F, hurdle=True, s1=True, mw=True),
             # s2rk: como o s2, mas recebendo também o s1_rk pronto (s1 +
             # krigagem do erro), para a rede refinar a ocorrência sem perder o
             # que a correção clássica preserva nos núcleos
             "s2rk":     dict(_F, hurdle=True, s2=True, rk=True),
             "hist":     dict(_F, hist=True),
             "rec":      dict(_F, rec=True),
             "diff":     dict(_F, diff=True),
             "interp":   dict(_F, interp=True),
             "di":       dict(_F, diff=True, interp=True),
             "mw_di":    dict(_F, diff=True, interp=True, mw=True)}

# ---------- movimento, história e recursão ----------
MOV_FILE = Path(__file__).resolve().parent / "processados" / "movimento.csv"
MOV_MAX_PX = 6             # deslocamento máximo por 10 min (12 km, ~72 km/h)
MOV_TB_FRIO = 255.0        # K; o movimento é estimado nos topos mais frios
HIST_LAGS = 3              # janelas anteriores das estações de contexto
REC_PASSOS = 4             # passos de 10 min por sequência no treino do rec
REC_BATCH = 2              # sequências por lote (4 passos cada): mesma memória dos demais braços
FILTRO_ATIVO = False       # produto <braço>_filtro (sem ganho no completo da v13)
FILTRO_P = 0.5             # célula = P(chuva) >= FILTRO_P
FILTRO_P_FORTE = 0.7       # célula nova sem antecessor só fica se atingir isso

# ---------- núcleos intensos, nota v8 e mapa ternário ----------
# núcleos em mm acumulados na janela (m10 5 mm = 30 mm/h; m30 10 mm = 20 mm/h;
# m60 20 e 30 mm/h). Todos são limites do histograma (P(>tau) exata).
NUCLEOS = {10: [5.0], 30: [10.0], 60: [20.0, 30.0]}
RAIOS_KM = [4, 8, 12]                 # tolerância espacial dos núcleos
CUSTO_MISS = 1.5                      # dizer seco e chover
CUSTO_FA = 1.0                        # dizer chuva e não chover
C_DUVIDA = [0.2, 0.3, 0.4]            # custo da dúvida (largura da faixa)
C_REF = 0.3                           # faixa usada p/ núcleo do mapa ternário
VERSAO = "v14"                         # checkpoints/pares de versões antigas são refeitos

# ---------- treino ----------
BATCH      = 8
LR         = 1e-4
HID        = 64
LAMBDA_AMT = 0.5           # peso do termo de quantidade (MSE em log1p, alvos >1mm)
# Pesos de classe desligados desde a v11: com eles, deixar de prever chuva
# custava até 10 vezes mais que prever chuva errada, e os modelos passaram a
# colorir o domínio inteiro (P média 0,48 contra frequência observada de 0,23
# no m10). O sorteio com FRAC_CHUVOSO de stems chuvosos é compensado por pesos
# de importância na perda, para a probabilidade refletir a frequência real.
USAR_PESOS_CLASSE = False
LAMBDA_VIZ = 0.3           # hurdle: tolerância 3x3 para estações chuvosas
LAMBDA_NUC = 0.5           # hurdle: tolerância 5x5 para estações com núcleo
GRAD_CLIP  = 1.0
# amostras_epoca: stems sorteados por época (com WET_BOOST); 0 = todos.
# No completo, uma "época" é uma amostra de 4000 stems, não o conjunto inteiro,
# para o tempo por época não depender do tamanho do fold.
MODOS = {"rapido":   dict(epochs=6,  pre_epochs=2, paciencia=99, folds=[0],
                          max_stems=500, max_stems_val=180, val_sub=120,
                          amostras_epoca=0, amostras_pre=800, seq_epoca=150),
         "completo": dict(epochs=25, pre_epochs=8, paciencia=5,
                          folds=list(range(K_FOLDS)),
                          max_stems=0, max_stems_val=0, val_sub=800,
                          amostras_epoca=4000, amostras_pre=3000,
                          seq_epoca=1200)}

# processos que montam as amostras (leitura dos npy + kriging); é o gargalo,
# a GPU fica ociosa esperando. Ajuste se a máquina tiver poucos núcleos.
import os as _os
NUM_WORKERS = max(2, min(8, (_os.cpu_count() or 4) - 2))

# fração mínima de pixels finitos do MW no domínio para o stem entrar no pré-treino
MW_MIN_COBERTURA = 0.4

VARIO_PADRAO = {10: dict(nugget=0.030, sill=0.074, range=10.6),
                30: dict(nugget=0.066, sill=0.153, range=17.2),
                60: dict(nugget=0.095, sill=0.146, range=13.0)}

for _d in (DADOS, MW_DIR, PROC, RES, CKPT_DIR, PARES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
