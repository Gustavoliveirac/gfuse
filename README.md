# gfuse3

Rodada limpa do nowcasting GOES-16 + pluviômetros. Código novo, sem importar
nada das pastas antigas.

## Variantes (braços) da v14

| braço | o que é |
|---|---|
| base | rede de referência (ConvLSTM sobre 6 quadros do GOES + estações de contexto e krigagem como entrada) |
| hurdle | base com duas cabeças: probabilidade de chuva calibrada e intensidade condicional |
| krig_res | hurdle com a ocorrência partindo da krigagem de indicador |
| s1 | estágio 1: só o GOES na entrada, nenhuma estação; é um estimador de chuva por satélite treinado para a região |
| s1_rk | estágio 2 analítico (sem treino): s1 + krigagem do erro do s1 nas estações de contexto; respeita o valor medido perto das estações |
| s2 | estágio 2 aprendido: rede leve que recebe a saída congelada do s1 e a krigagem |
| s1_mw (+ s1_mw_rk) | s1 com pré-treino denso GOES -> MWprecipitation (IMERG) |
| s2rk | s2 que recebe também o s1_rk pronto |

A v15 mantém VERSAO = "v14": checkpoints e pares já existentes são
reaproveitados; só s1_mw e s2rk são treinados, e o s1 é reavaliado para
gravar o s1_rk com limiar próprio (escolhido no bloco de treino).

hist, rec, filtro de células, interp, di e mw_di continuam disponíveis via `ARMS=...`.

Cada treino escolhe, no bloco de treino do fold, o limiar de P(chuva) que
maximiza o CSI; ele é usado na ocorrência, nos mapas e no produto `_det`.

## Teste de advecção pura (adveccao.py)

Análise em t0 com todas as estações (krigagem e s2), projetada para +10, +20,
+30 e +60 min sem estações nem GOES e verificada nas estações. Compara campo
parado, vetor global do GOES, movimento local por blocos, vento das estações
meteorológicas (`MET_DIR`), a média GOES + vento, o movimento estimado só
pelas estações (campo krigado de t0-10 para t0) e um oráculo que usa o
futuro (limite superior). Saídas
`adveccao_resumo.csv`, `adveccao_vento.csv` e `adveccao_movimento.csv`.

## Etapas novas

- `movimento.py` estima o vetor de movimento por correlação cruzada dos topos
  frios (C13) entre quadros consecutivos, no recorte de 128 px
  (`GOES128_DIR`; se não existir, usa o de 64 px e avisa no log).
- `diag_rrqpe.py` mede o atraso e o excesso de área do RRQPE frente às
  estações (por evento e por stem).

## Base de dados (v10)

O `dados.py` grava as estações em matrizes tempo x estação
(`processados/estacoes.npz`), no lugar do parquet longo. Ele refaz a base
sozinho quando o número de arquivos do GOES muda, então basta rodar o
`run_all.sh` depois de completar os `.npy`. O antigo
`processados/dataset.parquet` não é mais usado e pode ser apagado.

Estações fora do recorte de 64 px ficam na base só como contexto: entram
no canal de IDW (que respeita a distância real até a borda), nunca como
alvo, holdout ou splat. O log do `dados.py` mostra quantas ficaram dentro e
fora por fonte.

No treino, metade das amostras de cada época vem de stems com chuva em
alguma estação e metade dos demais (`FRAC_CHUVOSO`), qualquer que seja a
proporção de dias secos na base. Na avaliação, as métricas saem em duas
leituras: todos os stems (operação contínua, inclui alarmes falsos em dia
seco) e só os stems com chuva.

## Controle de qualidade e regra do seco (v11)

- `ESTACOES_EXCLUIDAS` no config3.py lista as estações retiradas da base,
  com o motivo de cada uma.
- `qc_estacoes.py` NÃO roda no pipeline (os dados já passaram por QA/QC).
  Fica como ferramenta para a rotina futura: marca chuva sob céu limpo e
  zeros suspeitos (pluviômetro entupido), com taxas por estação em
  `qc_estacoes.csv`. Só é aplicado se `QC_APLICAR = True`.
- Séries lidas da saída do QA/QC (`processados_qaqc`, v13): registros com
  `status_final` 2 (inválido) são descartados; o total por estação vai para
  `resultados/dados_descartes.csv`.
- Limite físico (v12): registros acima de 50 mm em 10 min ou 150 mm em 1 h
  viram dado ausente (`LIMITE_10MIN_MM`, `LIMITE_HORA_MM`).
- O valor de cada compartimento do histograma é a média observada nele
  (`processados/medias_classes.json`), não mais o ponto médio fixo.
- `<braço>_det` é o mapa entregue: chuva condicional onde P(chuva) >= 0,5 e
  zero no resto. É avaliado ao lado do valor esperado do braço.
- Supervisão do seco: pixels com Tb > 270 K e RRQPE zero num raio de 12 km
  nos últimos 30 min recebem alvo seco na perda, menos perto de estações com
  chuva. Parâmetros `SECO_*` no config3.py, escolhidos pelo
  `diag_satelite.py`.
- Sem pesos de classe; o sorteio 50/50 é compensado por pesos de
  importância. Convoluções com preenchimento por replicação na borda.
- As entradas de krigagem são a krigagem simples em mm e a de indicador de
  chuva, ambas com média zero (voltam para seco longe das estações) e com as
  estações de fora do recorte. O braço `krig_res` parte da krigagem de
  indicador e a rede corrige.
- Estações a menos de `MARGEM_KM` (10 km) da borda ficam fora das métricas.

## Mapas dos eventos

`mapas_eventos.py` (etapa 5 do `run_all.sh`) gera, por janela, painéis dos 6
eventos mais intensos da validação: linhas com instantes consecutivos e
colunas com Tb 10,3 µm, RRQPE, IDW e cada braço, com as estações por cima
(contexto, alvo e holdout). Os PNG vão para `resultados/mapas/` e entram no
zip de envio.

## Métricas

Sem métrica composta. O checkpoint é escolhido pela menor perda na
validação. Os CSVs trazem r e POD, FAR e CSI por limiar, por tolerância
espacial e para os núcleos (m10 5 mm, m30 10 mm, m60 20 e 30 mm). A
avaliação usa a mesma divisão alvo/contexto por estação em todos os braços.

## Nada é copiado — tudo é lido de onde já está

Não há pasta `dados/estacoes` nem `dados/metadados.csv` aqui dentro. Os
únicos caminhos fora desta pasta ficam no topo do `config3.py`:

```python
GOES_DIR = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/canada/numpy_64px")
IMERG_DIR = Path("/media/gustavo/Elements/imerg/imerg_data/early")
EST_DIR  = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados_qaqc")
META_CSV = Path("/media/gustavo/8e275a83-51ed-4fe7-8a40-b8801f6efed5/ufmg/gfuse/processados_qaqc/metadados.csv")
```

`EST_DIR` é a pasta com `<FONTE>/<cod>.csv` das séries processadas; `IMERG_DIR`
só é lido (uma vez, pelo `mw_prep.py`) se for rodar o braço `mw_di`. Se algum
desses caminhos estiver errado na sua máquina, edite só essas quatro linhas —
nada mais no código precisa mudar.

ANADEM não é usado (era topografia, de um braço fora do escopo). O parquet,
stats, variogramas, folds e holdout são gerados aqui dentro
(`processados/` e `resultados/`).

## Como rodar

```bash
bash run_all.sh rapido      # valida tudo: fold 0, poucas épocas, subamostra
bash run_all.sh completo    # k-fold temporal k=5, todas as variantes
```

Escopo parcial: `AGGS="60" ARMS="base di" bash run_all.sh completo`

Retomada: se a rodada for interrompida, basta rodar o mesmo comando de novo. Treinos já concluídos no mesmo modo e avaliações já feitas são pulados.

## O que enviar de volta para análise

Só o `resultados/envio_gfuse3.zip` (CSVs, resumos e o master.log).
