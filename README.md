# VaR e Expected Shortfall condicionais ao regime — Setor Agro/Commodities (HMM + LSTM)

**TCC — Engenharia de Produção, Universidade Presbiteriana Mackenzie**
Orientador: Prof. Gustavo Rocha da Silva

## Tese central

Este trabalho testa se condicionar o forecast de uma rede **LSTM** ao regime de mercado
detectado por um **Hidden Markov Model (HMM)** produz estimativas de **VaR** e **Expected
Shortfall (ES)** mais acuradas do que o benchmark econométrico **MS-GARCH**, em particular
em regimes de estresse, aplicado a uma carteira representativa do **setor agro/commodities
brasileiro** (equities do agronegócio, futuros de commodities agrícolas e USD/BRL).

A distribuição preditiva dos retornos é construída via **Monte Carlo Dropout** (Gal e
Ghahramani, 2016) e as estimativas são validadas por backtesting formal (Kupiec 1995,
Christoffersen 1998, Acerbi-Székely 2014, Diebold-Mariano 1995). Uma análise complementar
de interpretabilidade (SHAP) identifica os drivers de risco por regime.

## Carteira (7 ativos) — ver `config.yaml`

| Categoria | Ativo | Ticker |
|---|---|---|
| Equity agro (grãos) | SLC Agrícola | `SLCE3.SA` |
| Equity agro (proteína) | JBS | `JBSS32.SA` (BDR; `JBSS3.SA` descontinuado — reincorporação para JBS N.V.) |
| Equity agro (proteína/pecuária) | Marfrig | `MBRF3.SA` (`MRFG3.SA` descontinuado — fusão Marfrig-BRF) |
| Equity agro (açúcar/etanol) | São Martinho | `SMTO3.SA` |
| Commodity | Soja (futuros) | `ZS=F` |
| Commodity | Café Arábica (futuros) | `KC=F` |
| Câmbio | USD/BRL | `BRL=X` |

Lista ajustável em `config.yaml`.

O agregado atual é um **índice proxy de preços**, com pesos iguais e rebalanceamento
diário: converte log-retornos em retornos simples, agrega e reaplica `log1p`.
Se faltar qualquer componente, o retorno agregado fica ausente. Não representa
o P&L de contratos futuros com margem, colateral e custos de rolagem.

<<<<<<< HEAD
Soja (contrato de novembro) e café (Mar/Mai/Jul/Set/Dez) vêm da série pública de
preços múltiplos do pysystemtrade, com rolagem por razão novo/antigo nos preços da mesma
data (`scripts/build_public_futures.py`). Saltos não são removidos. A fonte só tem
fechamento, e por isso Garman–Klass, Corwin–Schultz e volume ficam de fora para os futuros.
A base termina em 28/03/2024, fim da série do café. O autor deixou de atualizá-la citando
licenciamento dos dados; o uso aqui é acadêmico, com citação.
=======
Os futuros são usados sem remoção automática de saltos: a identificação de
contratos e os fatores de rolagem ainda precisam ser validados. `adjust_ohlc_roll`
só ajusta preços quando recebe fatores explícitos `novo/antigo` da mesma data.
Os resultados permanecem exploratórios até essa validação.
>>>>>>> 82a779c875e26e294f16e974965f241e480e5012

## Estrutura do repositório

```
hmm_lstm_agro/
├── config.yaml            # ativos, datas, hiperparâmetros, seeds
├── requirements.txt       # dependências Python
├── r/                      # benchmark MS-GARCH (R, pacote MSGARCH)
├── data/{raw,interim,processed}
├── notebooks/              # pipeline numerado, ponta a ponta
├── src/{data,models,risk,interpretability}
├── reports/{figures,tables}
├── tests/
└── docs/                   # pré-projeto original e material de apoio
```

## Como reproduzir

1. `python -m venv .venv && .venv\Scripts\activate` (Windows) e `pip install -r requirements.txt`
<<<<<<< HEAD
   (`requirements-lock.txt` tem as versões exatas usadas).
2. Instalar R ≥ 4.6 e os pacotes `MSGARCH`, `arrow` e `jsonlite`; as versões usadas ficam
   em `reports/tables/msgarch/versions.txt`.
3. Dados: `data/raw/` não é versionado. Os futuros de soja e café vêm da reconstrução
   pública com rolagem (`scripts/build_public_futures.py`, arquivos conferidos por SHA256).
4. Rodar, da raiz do projeto:

| etapa | comando | tempo aprox. |
|---|---|---|
| dados e features | notebooks 01 → 03 (ou `scripts/rebuild_local.py` para 02/03) | minutos |
| LSTM e HMM-LSTM no walk-forward | `python scripts/run_walk_forward.py` | ~30 min |
| MS-GARCH | `python scripts/msgarch_inputs.py`, `Rscript r/msgarch_benchmark.R`, `python scripts/msgarch_forecasts.py` | ~5 min |
| backtests | `python scripts/evaluate_walk_forward.py` | ~2 min |
| robustez ativo a ativo | as mesmas etapas com `--target-asset ATIVO` (ver `scripts/summarize_univariate.py`) | ~1 h em paralelo |
| tabelas finais | `python scripts/build_final_tables.py` → `reports/tables/final/` | segundos |
| análises e figuras | notebooks 04 → 08 → `reports/figures/` | ~15 min |

O treino é determinístico em CPU (semente fixa): repetir `run_walk_forward.py` reproduz as
previsões bit a bit. O teste é usado uma única vez, sem ajuste posterior de hiperparâmetros.

O preenchimento usa somente os primeiros três pregões de cada ausência. Seleção de variáveis,
padronização e HMM são ajustados dentro do treino de cada janela, e o regime entra na rede pelo
filtro Forward. O arquivo `hmm_regimes_insample.parquet` (notebook 03) é só descritivo e não
deve ser usado como entrada fora da amostra.
=======
2. Instalar R + pacote `MSGARCH` (ver `r/msgarch_benchmark.R`) para o benchmark econométrico
3. Ajustar `config.yaml` se necessário (ativos, janela de datas, hiperparâmetros)
4. Executar os notebooks implementados em `notebooks/` (01 → 03).

Com os dados locais já disponíveis, execute `.venv\Scripts\python scripts/rebuild_local.py`
na raiz do projeto para reexecutar 02/03. Os artefatos anteriores são copiados
para `reports/archive/`; `reports/tables/rebuild_manifest.json` registra status,
hashes dos dados e o caráter in-sample do experimento. `requirements-lock.txt`
registra as versões do ambiente que executou a reconstrução (Windows/Python 3.13).

O preenchimento usa somente os primeiros três pregões de cada ausência, com
preço passado e volume zero nas barras preenchidas. Volume de FX é excluído
explicitamente. Demais ausências são preservadas e reportadas em
`feature_missingness.csv`; a seleção/imputação para a LSTM deve ser aprendida
dentro de cada janela de treino, ainda não implementada.

O HMM trata trechos separados por ausências como sequências independentes.
O Forward não usa observações futuras, mas seus parâmetros atuais são ajustados
in-sample; o arquivo de regimes não deve ser usado como feature fora da amostra.
>>>>>>> 82a779c875e26e294f16e974965f241e480e5012

## Status e plano de trabalho

Ver [`PLANO.md`](PLANO.md) para o checklist detalhado e o cronograma semanal (26/08 a 01/12).
