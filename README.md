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
| Equity agro (proteína) | JBS | `JBSS3.SA` |
| Equity agro (proteína/pecuária) | Marfrig | `MRFG3.SA` |
| Equity agro (açúcar/etanol) | São Martinho | `SMTO3.SA` |
| Commodity | Soja (futuros) | `ZS=F` |
| Commodity | Café Arábica (futuros) | `KC=F` |
| Câmbio | USD/BRL | `BRL=X` |

Lista ajustável em `config.yaml` (nenhum ticker é hard-coded no código-fonte).

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
4. Executar os notebooks em `notebooks/` na ordem numérica (01 → 07)

## Status e plano de trabalho

Ver [`PLANO.md`](PLANO.md) para o checklist detalhado e o cronograma semanal (26/08 a 01/12).
