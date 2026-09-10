# Plano de trabalho — TCC (setor Agro/Commodities)

Período: **26/08/2026 → 01/12/2026** (14 semanas). Objetivo: ter o pipeline completo
(dados → HMM → LSTM → MC Dropout → VaR/ES → benchmark MS-GARCH → backtesting) **funcionando**
até o início de dezembro, além do documento do projeto/artigo atualizado para o setor
agro/commodities.

Cada semana lista entregáveis objetivos (arquivos que devem existir e rodar) para facilitar
acompanhamento. Marque `[x]` ao concluir.

---

## Semana 1 — 26/08 a 01/09 — Setup e ajuste do documento
- [x] Criar/validar ambiente Python (`requirements.txt`) — ambiente R (`MSGARCH`) fica para a Semana 8
- [x] Confirmar/ajustar lista de 7 ativos agro em `config.yaml`
- [ ] Atualizar título do TCC para refletir o setor agro/commodities
- [ ] Reescrever Cap. 1 (problema de pesquisa, objetivos, justificativa) trocando a carteira
      genérica pela carteira agro — manter H1/H2/H3 e a estrutura de pergunta de pesquisa
- [x] Copiar o pré-projeto original para `docs/pre_projeto_original.docx`

## Semana 2 — 02 a 08/09 — Coleta de dados
- [x] `src/data/collectors.py`: funções para Yahoo Finance (equities/commodities/FX),
      BCB-SGS (Selic, IPCA, IBC-Br), FRED (Treasuries 10y, DXY via proxy `DTWEXBGS`) —
      Ipeadata (EMBI+) adiado, ver Pendências
- [x] `notebooks/01_data_collection.ipynb`: baixar e salvar `data/raw/` para os 7 ativos + macro
- [x] Validar cobertura temporal 2005–2024 e ausência de gaps críticos por ativo
      (`src/data/validation.py::check_coverage`)

## Semana 3 — 09 a 15/09 — Engenharia de features
- [x] `src/data/features.py`: Garman-Klass, GARCH(1,1) univariado expansivo (`arch`,
      refit mensal), retornos multi-horizonte (1/5/21d), skew/curtose móveis (21/63d),
      spread de Corwin-Schultz (proxy de bid-ask, ver Pendências), volume normalizado,
      vol. implícita (`^VIX` + Ibov/Bloomberg)
- [x] Tratamento de missing (`align_and_fill_ohlc`: interpolação linear até 3 dias úteis,
      senão mantém `NaN` estrutural — trim final fica para o dataset de modelagem)
- [x] Testes de estacionariedade (ADF, KPSS) e filtro de multicolinearidade (VIF > 10)
      em `src/data/feature_selection.py`, rodado no painel *pooled* (todos os ativos)
- [x] `notebooks/02_feature_engineering.ipynb`: gera `data/processed/features.parquet`
      (4568 linhas × 112 colunas = 16 features × 7 ativos) e os relatórios de
      estacionariedade/correlação/VIF em `reports/tables/`

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

## Pendências / Backlog

- **Ipeadata / EMBI+**: não implementado nesta rodada. Não há pacote Python maduro e a
  fonte é a mais frágil das quatro (site sem API estável). Falta: escolher entre
  `ipeadatapy` e scraping manual, e adicionar `collect_ipeadata_embi()` em
  `src/data/collectors.py` seguindo o mesmo padrão dos demais coletores.
- **Decisão pendente com o orientador — soja/café, Yahoo (USD, futuros) vs. CEPEA (BRL,
  spot)**: a carteira já tem USD/BRL como ativo explícito; usar futuros de soja/café em
  USD embute risco cambial dentro do preço, duplicando (double counting) o risco cambial
  já capturado por USD/BRL. `collectors.py` hoje coleta as duas séries lado a lado
  (`collect_futures_raw`/`adjust_futures_roll` para Yahoo, `collect_cepea_spot` para
  CEPEA) exatamente para permitir essa decisão com dado real em mãos. Falta: validar com
  o Prof. Gustavo qual série entra oficialmente na carteira antes da Semana 3
  (engenharia de features).
- **Proxy do DXY**: o FRED gratuito não publica o índice DXY (ICE). Usamos `DTWEXBGS`
  (Broad Trade-Weighted U.S. Dollar Index) como proxy — metodologicamente equivalente
  na literatura, mas não é o mesmo índice/base. Documentar essa ressalva no Cap. 3
  (Metodologia) quando o documento for revisado.
- **Ajuste de rolagem dos futuros — trocado de aditivo para multiplicativo (Semana 3)**:
  `adjust_futures_roll`/`adjust_ohlc_roll` usavam back-adjustment *aditivo* (Panama).
  Rodando de verdade em ZS=F/KC=F (2005-2024, ~19 anos, dezenas de rolagens), o
  deslocamento aditivo acumulado empurrou preços do início da amostra para **valores
  negativos** — inviabilizando Garman-Klass/Corwin-Schultz (precisam de log(H/L) e
  log(C/O) positivos). Trocado para ajuste **multiplicativo (por razão)**: nunca produz
  preço negativo e preserva os retornos percentuais/log exatamente (mais coerente com
  o resto do pipeline, que é todo baseado em log-retornos). `data/raw/futures_*.parquet`
  já foi regenerado com o ajuste corrigido. O limiar fixo de log-retorno (8%) para
  detectar rolagem continua o mesmo — vale revisitar se uma rolagem real produzir um
  salto menor que isso.
- **Cap. 1 do `.docx` (pivô agro)**: `docs/pre_projeto_original.docx` ainda está na
  versão genérica ("carteiras brasileiras"), sem menção a agro/commodities/tickers.
  Reescrita adiada de propósito para depois da coleta de dados (decisão desta rodada).
- **FRED pendente de chave**: `collect_fred` está pronto e os `series_id` (`DGS10`,
  `DTWEXBGS`) foram confirmados no site do FRED, mas não foi possível rodar de ponta a
  ponta nesta sessão por faltar `FRED_API_KEY` no `.env` local. Copie `.env.example`
  para `.env`, preencha a chave e rode a última célula de `01_data_collection.ipynb`
  (ou o notebook inteiro) para gerar `data/raw/macro_fred.parquet`.
- **Gap real encontrado em USD/BRL**: a validação aponta um gap crítico genuíno de 17
  dias úteis em USD/BRL (`BRL=X`) entre 01/08/2008 e 25/08/2008 (crise financeira de
  2008) — não é um problema do código, mas vale checar se esse buraco afeta o HMM na
  Semana 4 (pode exigir interpolação ou nota metodológica).
- **Cobertura parcial de equities/CEPEA-soja é esperada, não um bug**: `covers_requested_range=False`
  para SLCE3.SA/JBSS32.SA/MBRF3.SA/SMTO3.SA e para a série CEPEA-soja só reflete que
  esses ativos/BDRs foram listados ou a série começou depois de 2005 (ex.: JBSS32.SA
  só existe como BDR a partir de 2007) — a tabela de validação registra a data real de
  início de cada série (`coverage_start`) em `reports/tables/data_coverage_summary.csv`.
- **Tickers corrigidos nesta rodada**: `JBSS3.SA` e `MRFG3.SA` (do pré-projeto/config
  original) estão descontinuados na Yahoo Finance por corporate actions recentes —
  JBS reincorporou-se como JBS N.V. (NYSE: JBS, BDR na B3: `JBSS32.SA`) e Marfrig
  fundiu-se com a BRF (nova entidade: `MBRF3.SA`). `config.yaml`/`README.md` já
  atualizados; vale mencionar essa mudança se o Cap. 1/3 do `.docx` citar os tickers
  antigos. `JBSS32.SA` é BDR (razão de conversão própria) — atentar a isso na
  metodologia de ponderação da carteira.
- **Início comum do painel = 2007-06-29**: definido pela equity/BDR mais tardia
  (MBRF3.SA). O painel de features (`data/processed/features.parquet`) começa nessa
  data; 2005-2007 dos futuros/câmbio (que têm histórico mais longo) fica de fora do
  painel multivariado por causa disso — se isso for uma perda relevante de amostra
  para o HMM, é uma decisão a revisitar com o orientador (trade-off amostra vs. ativos).
- **`implied_vol_br` (Bloomberg) precisa de atualização manual periódica**:
  `data/raw/ibov_vol_bloomberg.xlsx` (IBOV Index, `30DAY_IMPVOL_100.0%MNY_DF`) foi
  exportado uma vez do terminal Bloomberg, cobre até 2024-12-30, e **não é atualizado
  automaticamente** — não está no git (licenciamento Bloomberg). Se o período de
  amostra avançar, reexportar do terminal e sobrescrever o arquivo.
- **IVol-BR do NEFIN — só robustez, não é input do modelo**: gratuito, mas
  descontinuado em 2022-04 (cobertura real: 01/08/2011–29/04/2022 — CSV direto em
  `https://nefin.com.br/resources/volatility_index/IVol-BR.csv`). Deixaria ~2,7 anos
  de `NaN` bem no período mais recente/de teste se fosse usado como feature principal;
  por isso `implied_vol_br` no pipeline vem do Bloomberg (cobertura completa até 2024).
  Guardar a URL aqui para uma eventual análise de robustez/comparação futura.
- **`garman_klass_vol_63d` descartado por VIF (Semana 3)**: o filtro de
  multicolinearidade (limiar 10) removeu essa coluna do conjunto de features retido —
  redundante com `garman_klass_vol_21d`/`realized_vol_*d`/`garch_cond_vol`. Ver
  `reports/tables/feature_vif_selection.csv` para o relatório completo (mantidas vs.
  descartadas) e `feature_stationarity.csv`/`feature_correlation.csv` para os demais
  diagnósticos da Passada 2.
