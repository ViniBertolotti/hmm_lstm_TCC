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

Os futuros são usados sem remoção automática de saltos: a identificação de
contratos e os fatores de rolagem ainda precisam ser validados. `adjust_ohlc_roll`
só ajusta preços quando recebe fatores explícitos `novo/antigo` da mesma data.
Os resultados permanecem exploratórios até essa validação.

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

## Status e plano de trabalho

Ver [`PLANO.md`](PLANO.md) para o checklist detalhado e o cronograma semanal (26/08 a 01/12).
