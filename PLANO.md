# Plano de trabalho — TCC (setor Agro/Commodities)

Período: **26/08/2026 → 01/12/2026** (14 semanas). Objetivo: ter o pipeline completo
(dados → HMM → LSTM → MC Dropout → VaR/ES → benchmark MS-GARCH → backtesting) **funcionando**
até o início de dezembro, além do documento do projeto/artigo atualizado para o setor
agro/commodities.

Cada semana lista entregáveis objetivos (arquivos que devem existir e rodar) para facilitar
acompanhamento. Marque `[x]` ao concluir.

---

## Semana 1 — 26/08 a 01/09 — Setup e ajuste do documento
- [ ] Criar/validar ambiente Python (`requirements.txt`) e ambiente R (`MSGARCH`)
- [ ] Confirmar/ajustar lista de 7 ativos agro em `config.yaml`
- [ ] Atualizar título do TCC para refletir o setor agro/commodities
- [ ] Reescrever Cap. 1 (problema de pesquisa, objetivos, justificativa) trocando a carteira
      genérica pela carteira agro — manter H1/H2/H3 e a estrutura de pergunta de pesquisa
- [ ] Copiar o pré-projeto original para `docs/pre_projeto_original.docx`

## Semana 2 — 02 a 08/09 — Coleta de dados
- [ ] `src/data/collectors.py`: funções para Yahoo Finance (equities/commodities/FX),
      BCB-SGS (Selic, IPCA, IBC-Br), FRED (Treasuries 10y, DXY), Ipeadata (EMBI+)
- [ ] `notebooks/01_data_collection.ipynb`: baixar e salvar `data/raw/` para os 7 ativos + macro
- [ ] Validar cobertura temporal 2005–2024 e ausência de gaps críticos por ativo

## Semana 3 — 09 a 15/09 — Engenharia de features
- [ ] `src/data/features.py`: Garman-Klass, GARCH(1,1) univariado (`arch`), lags (1/5/21d),
      skew/curtose móveis (21/63d), bid-ask spread normalizado, volume normalizado
- [ ] Tratamento de missing (interpolação linear até 3 dias úteis, senão descarta janela)
- [ ] Testes de estacionariedade (ADF, KPSS) e filtro de multicolinearidade (VIF > 10)
- [ ] `notebooks/02_feature_engineering.ipynb`: gerar `data/processed/features.parquet`

## Semana 4 — 16 a 22/09 — Detecção de regimes (HMM)
- [ ] `src/models/hmm.py`: HMM gaussiano (`hmmlearn`), grade de 2–4 estados, seleção por BIC/AIC
- [ ] Viterbi (in-sample) e Forward (prospectivo, sem lookahead bias)
- [ ] Robustez via bootstrap em blocos
- [ ] `notebooks/03_hmm_regimes.ipynb`: caracterizar regimes (persistência, médias, covariâncias)

## Semana 5 — 23 a 29/09 — Arquitetura LSTM
- [ ] `src/models/lstm.py`: LSTM 2 camadas (64/32), dropout 0,3, one-hot do regime como feature,
      lookback de 42 dias, saída densa, loss MSE, otimizador Adam
- [ ] Grid search restrito: lookback ∈ {21,42,63}, dropout ∈ {0.2,0.3,0.4},
      batch ∈ {32,64}, lr ∈ {0.001,0.0005}
- [ ] `notebooks/04_lstm_training.ipynb` (parte 1 — arquitetura e grid search)

## Semana 6 — 30/09 a 06/10 — Treinamento
- [ ] Split 70/15/15 (hiperparâmetros) + walk-forward (treino 5a / teste 6m, passo 6m)
- [ ] Early stopping (paciência 15), seeds fixas, `requirements.txt`/versões documentadas
- [ ] `notebooks/04_lstm_training.ipynb` (parte 2 — treino final e conjunto de teste lacrado)

## Semana 7 — 07 a 13/10 — MC Dropout e VaR/ES
- [ ] `src/models/mc_dropout.py`: 1000 passagens com dropout ativo por instante de previsão
- [ ] `src/risk/var_es.py`: VaR (quantil empírico) e ES (média das perdas abaixo do quantil),
      condicionais ao regime e não condicionais (95%/99%)
- [ ] `notebooks/05_mc_dropout_var_es.ipynb`

## Semana 8 — 14 a 20/10 — Benchmark MS-GARCH (R)
- [ ] `r/msgarch_benchmark.R`: MS-GARCH(1,1) Student-t, 2–3 estados, seleção por BIC
- [ ] VaR/ES via FHS; ponte com o pipeline Python (rpy2 ou exportação CSV)
- [ ] Comparar ativo a ativo (univariado) e em nível de carteira (variâncias ponderadas)

## Semana 9 — 21 a 27/10 — Backtesting
- [ ] `src/risk/backtesting.py`: Kupiec (1995), Christoffersen (1998),
      Christoffersen-Pelletier (2004, duração), Acerbi-Székely (2014, ES)
- [ ] Funções de perda: Gonzalez-Rivera/Lee/Mishra (VaR) e Fissler-Ziegel (VaR+ES conjunto)
- [ ] Teste Diebold-Mariano com correção Harvey-Leybourne-Newbold
- [ ] `notebooks/06_backtesting.ipynb`

## Semana 10 — 28/10 a 03/11 — Interpretabilidade e consolidação
- [ ] `src/interpretability/shap_analysis.py`: DeepSHAP sobre a LSTM treinada, ranking por regime
- [ ] `notebooks/07_shap_interpretability.ipynb`
- [ ] Consolidar tabelas/figuras finais em `reports/tables` e `reports/figures`

## Semana 11 — 04 a 10/11 — Resultados e hipóteses
- [ ] Testar H1 (cobertura Kupiec), H2 (cobertura condicional em regime de estresse),
      H3 (Diebold-Mariano em regime de calmaria)
- [ ] Robustez univariada (ativo a ativo)
- [ ] Rascunho do capítulo de resultados/discussão

## Semana 12 — 11 a 17/11 — Revisão da Metodologia/Literatura
- [ ] Revisar Cap. 3 (Metodologia) para refletir decisões reais de implementação
- [ ] Complementar Revisão da Literatura com referências específicas de risco em
      agronegócio/commodities, se houver lacuna identificada durante a execução

## Semana 13 — 18 a 24/11 — Revisão geral
- [ ] Revisão ABNT completa (formatação, referências, citações)
- [ ] Revisão dos apêndices (fichamentos)
- [ ] Preparar conteúdo do pôster

## Semana 14 — 25/11 a 01/12 — Buffer e entrega
- [ ] Revisão final (leitura completa, ortografia, coerência entre capítulos)
- [ ] Entrega do projeto/artigo
- [ ] Entrega/apresentação do pôster

--
- `rpy2` requer R instalado e compatível com a versão do Python/venv — testar a ponte
  Python↔R cedo (Semana 8) para não perder tempo depois.
- O cronograma está agressivo (14 semanas para todo o pipeline). Semana 14 é buffer —
  se atrasar, é a que deve absorver o atraso, não a entrega final.
