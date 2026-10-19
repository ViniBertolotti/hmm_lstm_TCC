# Plano de trabalho — TCC (setor Agro/Commodities)

Período: **26/08/2026 → 01/12/2026** (14 semanas). Objetivo: ter o pipeline completo
(dados → HMM → LSTM → MC Dropout → VaR/ES → benchmark MS-GARCH → backtesting) **funcionando**
até o início de dezembro, além do documento do projeto/artigo atualizado para o setor
agro/commodities.

Cada semana lista entregáveis objetivos (arquivos que devem existir e rodar) para facilitar
acompanhamento. Marque `[x]` ao concluir.

## Estado verificado e prioridades (19/10/2026)

Esta avaliação qualifica as marcações do cronograma abaixo. Código escrito e
testes simulados não demonstram validade das séries ou desempenho financeiro.

- **Futuros e rolagens:** a fonte definitiva permanece pendente. A lógica por
  razão usa preços simultâneos dos contratos e preserva o retorno de mercado;
  foram verificados casos simulados, sem corrigir os arquivos Yahoo/Bloomberg.
  O candidato público não tem aprovação final de procedência/licenciamento,
  OHLCV, calendário ex ante e liquidez. Não remover retornos extremos sem prova.
  Dois boletins oficiais do USDA confirmaram vencimentos distintos da soja e
  coincidência pontual do Yahoo com o primeiro contrato listado. Cobertura
  diária de 2005-2024 e solução para o café continuam sem confirmação. Evidência
  em `reports/tables/futures_source_audit.json`; não é uma base final corrigida.
- **Modelos e risco:** módulos mínimos de LSTM, MC Dropout, VaR/ES, calibração
  de erros e backtests têm verificações sintéticas. O executor separa treino,
  seleção/early stopping, calibração posterior e teste. Reservar 252 sessões
  para calibração dentro dos cinco anos de desenvolvimento reduz o treino
  efetivo; esse protocolo precisa ser explicitado. A cauda empírica a 99% é
  pequena e não sustenta conclusões fortes com essa amostra. Dispersão do
  dropout não equivale à distribuição das perdas. Uma rodada exploratória já
  produziu as previsões descritas abaixo; a validação final da base e a comparação
  completa com MS-GARCH continuam pendentes.
- **R/MS-GARCH:** adiado pelo usuário até obter liberação do TI. O piloto
  preparado ainda não foi executado; não declarar o benchmark validado.
- **Fonte e escopo:** usar alternativas sem custo, preservar a especificação
  com futuros e avaliar CEPEA separadamente. Mudanças de mercado, vencimentos,
  features e período exigem justificativa e discussão com o orientador.
  O ensaio separado CEPEA executou LSTM e HMM-LSTM na primeira janela de teste
  (29/06 a 28/12/2012), com informações defasadas em um pregão e alvo preservado.
  Restaram 278 sequências de treino, 129 erros de calibração e 122 previsões de
  teste. Houve 94 retornos zero da soja nos 123 pregões do bloco. Ambos os
  modelos tiveram MSE de retorno ligeiramente maior que a referência zero.
  A disponibilidade de amostras e a cauda de 99% exigem revisão; não houve
  substituição da versão com futuros nem conclusão sobre H1-H3. Evidência em
  `reports/tables/cepea_pilot_interpretation.json`.
  A nova consulta Yahoo reproduziu três datas sem cotações das quatro ações.
  Para SLCE3, a ausência também está na resposta bruta do provedor. O ofício
  072/2020-PRE da B3 confirma pregão normal em 20/11/2020; não transformar
  ausência de preço em feriado nem preencher automaticamente. As outras duas
  datas seguem sem confirmação primária de calendário nesta auditoria.
  Evidência em `reports/tables/equity_gap_probe.json`.
- **Resultados e texto:** notebooks 02/03 precisam ser reexecutados após validar
  a base final. Experimentos financeiros, comparação dos modelos, discussão
  H1-H3, revisão ABNT e pôster continuam pendentes.
  H1-H3 foram conferidas no documento: a interpretação de H3 precisa distinguir
  ausência de diferença detectada de evidência de não inferioridade. Também é
  necessário remover a exigência de sustentar uma hipótese como condição de
  sucesso científico. Auditoria em `reports/tables/hypotheses_audit.json`;
  documento original preservado.

## Rodada exploratória fora da amostra (19/10/2026)

`scripts/run_walk_forward.py` + `scripts/evaluate_walk_forward.py`: 23 janelas, 2.717 dias de
teste (01/10/2012-27/03/2024), sobre a base disponível cuja validação final dos futuros e
rolagens permanece pendente. VaR/ES por simulação histórica
filtrada (erros da calibração padronizados pela vol EWMA λ=0,94 da carteira; na HMM-LSTM,
cauda do regime atual quando há ≥100 erros — usada em 40% dos dias, nos outros 60% cai na
cauda conjunta). Referência: média zero + mesma FHS-EWMA. **O MS-GARCH ainda não entrou.**
Tabelas em `reports/tables/walk_forward/`.

| | HMM-LSTM | LSTM | FHS-EWMA |
|---|---|---|---|
| Violações VaR 95%, geral (esperado 5%) | 5,2% (Kupiec p=0,59) | 5,2% | 5,1% |
| Violações VaR 99%, geral (esperado 1%) | 1,1% (p=0,59) | 1,1% | 1,1% |
| VaR 95%, regime de estresse (289 dias) | 8,0% (p=0,03) | 7,6% (p=0,06) | 6,6% (p=0,24) |
| VaR 99%, regime de estresse | 2,4% (p=0,04) | 2,4% (p=0,04) | 2,8% (p=0,01) |

- **H1, leitura exploratória:** cobertura geral próxima da nominal e ausência de rejeição
  pelos testes empregados. Isso não prova calibração correta, independência ou equivalência.
  A 99%, contagens esparsas limitam a aproximação assintótica. A conclusão final depende da base.
- **H2, leitura exploratória:** as taxas de violações aumentam no estresse. A referência de
  média zero tem menor taxa que a HMM-LSTM a 95%; essa comparação não substitui o MS-GARCH.
  Estatísticas Acerbi–Székely negativas indicam um diagnóstico a investigar, sem p-valor porque
  trajetórias sob H0 não foram geradas. H2 frente ao benchmark continua pendente.
- **H3 não foi formalmente avaliada:** nos demais regimes, o escore quantílico da HMM-LSTM
  a 95% é maior que o da LSTM e o da referência (DM bilateral p≈0,006 e p≈0,005).
  A 99%, não se detectou diferença pelos critérios empregados. Sem margem e procedimento
  predefinidos, não cabe declarar confirmação ou rejeição de não inferioridade econômica.
- Hipótese explicativa a examinar: a previsão da média pode acrescentar ruído ao risco.
  O MSE é maior que o da média zero em 36 dos 46 ajustes registrados. Isso não isola o
  efeito da rede, pois a HMM-LSTM também usa calibração por regime, diferente da LSTM.
- **Não ajustar o modelo olhando para o teste** para fazer as hipóteses passarem: invalida a
  conclusão. Qualquer variação (sem cauda por regime, outro mínimo, LSTM prevendo
  volatilidade em vez de média) entra como análise de sensibilidade declarada como tal.

A avaliação auditada está em `reports/tables/walk_forward/evaluation_v2/`. Confere
manifesto, janelas, unidades, horizonte e pareamento, preserva lacunas e inclui ICs,
contagens de transições e limitações. Os 27 arquivos anteriores e suas métricas foram
preservados. A verificação relevante passou em 20 testes; não autentica os preços,
os horários de publicação ou a execução histórica do treinamento.

## Rodada com o benchmark MS-GARCH (19/10/2026)

R 4.6.1 + MSGARCH instalados. `scripts/msgarch_inputs.py` → `r/msgarch_benchmark.R` →
`scripts/msgarch_forecasts.py` → `scripts/evaluate_walk_forward.py` (saída em
`reports/tables/walk_forward/evaluation_v3/`). MS-GARCH(1,1) Student-t ajustado por janela só
nos 5 anos de desenvolvimento, K∈{2,3} pelo BIC (K=2 sempre), parâmetros congelados no teste,
previsão de t com dados até t−1 (verificado). VaR/ES por FHS com os resíduos padronizados do
desenvolvimento, como no pré-projeto; o paramétrico Student-t fica como sensibilidade (mais
conservador: 4,2% e 0,74% de violações). Mesmos 2.717 alvos dos outros modelos.
Re-execução do walk-forward reproduziu as previsões bit a bit e salvou os erros da calibração,
usados para o p-valor Acerbi–Székely (999 trajetórias sob H0, VaR/ES fixos) e o escore FZ0.

| | HMM-LSTM | LSTM | FHS-EWMA | MS-GARCH |
|---|---|---|---|---|
| VaR 95%, geral | 5,2% (p=0,59) | 5,2% | 5,1% | 4,7% (p=0,43) |
| VaR 99%, geral | 1,1% (p=0,59) | 1,1% | 1,1% | 1,0% (p=0,82) |
| VaR 95%, estresse (289 dias) | 8,0% (p=0,03) | 7,6% | 6,6% | 6,6% (p=0,24) |
| VaR 99%, estresse | 2,4% (p=0,04) | 2,4% | 2,8% | 2,1% (p=0,11) |
| ES 99%, geral: p A-S | 0,005 | 0,008 | 0,006 | 0,61 |

- **H1 (original: cobertura compatível e desempenho superior ou equivalente ao MS-GARCH):**
  cobertura compatível nos quatro modelos; mas o MS-GARCH tem escore quantílico e FZ0
  significativamente melhores que a HMM-LSTM (DM p≈0,001 a 95% e 99%). Parte de cobertura
  sustentada, parte comparativa não.
- **H2 (estresse, menor erro de cobertura que o MS-GARCH):** não sustentada. HMM-LSTM 8,0% vs
  6,6% (95%) e 2,4% vs 2,1% (99%); Kupiec rejeita a HMM-LSTM e não o MS-GARCH. Diferenças de
  escore no estresse não significativas (DM p≈0,5): poucos dias.
- **H3 (calmaria, sem perda significativa frente ao benchmark pelo DM):** rejeitada. A HMM-LSTM
  é pior que o MS-GARCH (DM p<0,001 a 95% e 99%, nos dois escores).
- ES: só o MS-GARCH não subestima o ES no geral; no estresse, todos subestimam (p<0,04).
- Leitura: o que mais pesa é a dinâmica de volatilidade. O MS-GARCH também supera a FHS-EWMA
  de média zero, e a previsão de média da LSTM só piora o escore.

## Pôster da Mostra TCC I preparado em 19/10/2026

- [x] Preencher o template fornecido em cópia separada, mantendo a página A1 vertical,
  o mestre, o tema, as fontes e os elementos institucionais.
- [x] Salvar `docs/poster_mostraTCC1_2026-2_agro.pptx`, com gráficos e fluxograma editáveis,
  resultados explicitamente exploratórios e pendências dos futuros/MS-GARCH visíveis.
- [x] Conferir a página renderizada, os dados dos gráficos, as planilhas embutidas e
  a preservação do template. Registro: `reports/tables/poster_mostraTCC1_2026-2_verificacao.json`.
- [ ] Acrescentar o e-mail do orientador quando ele enviar, conforme resposta do usuário.
- [ ] Obter a revisão do orientador e a liberação da versão de apresentação.

## Revisão de literatura preparada em 19/10/2026

- [x] Redigir e conferir um capítulo separado com 11 seções e 34 referências:
  `docs/revisao_literatura_agro_rev2.docx`. Originais preservados.
- [x] Conferir fontes, alcance das consultas, fórmulas e integridade do texto;
  verificar visualmente a prévia do DOCX. Registro em
  `reports/tables/revisao_literatura_verificacao.json`.
- [x] Integrar o capítulo em nova cópia do projeto e harmonizar introdução,
  metodologia, remissões e lista geral de referências; substituir os dez
  apêndices por sínteses conferidas. Arquivo:
  `docs/projeto_tcc_agro_rev2_literatura.docx`. Sumário nativo preparado para
  atualização no Word; ver `docs/projeto_tcc_agro_rev2_literatura_notas.txt`.
  Verificação: `reports/tables/projeto_tcc_agro_rev2_verificacao.json`.
- [ ] Conferir a paginação final no Word e o manual vigente do curso.

Esta entrega fundamenta o estudo e não constitui conclusão dos experimentos,
validação dos futuros ou aprovação das hipóteses H1–H3. Notas de integração em
`docs/revisao_literatura_notas.txt`.

## Replanejamento (19/10/2026)

Em 19/10 as Semanas 1-4 estavam prontas (exceto o documento) e as Semanas 5-8 não tinham
começado: cerca de 4 semanas de atraso, sem buffer até 01/12. O cronograma original abaixo
fica como referência; vale este:

| Período | Entregas |
|---|---|
| 19-25/10 | Re-export dos futuros ajustados + rodar 02/03 de novo; dataset walk-forward (`src/data/dataset.py`, feito); instalar R + MSGARCH e fazer um teste; título + Cap. 1 (rev1 feita, aguardando revisão) |
| 26/10-01/11 | LSTM walk-forward com grid reduzido; MC Dropout → VaR/ES |
| 02-08/11 | MS-GARCH + backtesting (Kupiec, Christoffersen, Acerbi-Székely, Diebold-Mariano) |
| 09-15/11 | Resultados H1-H3; SHAP só se sobrar tempo (primeiro corte) |
| 16-22/11 | Cap. 3 revisado conforme a implementação |
| 23/11-01/12 | ABNT, revisão final, pôster |

---

## Semana 1 — 26/08 a 01/09 — Setup e ajuste do documento
- [x] Criar/validar ambiente Python (`requirements.txt`) — ambiente R (`MSGARCH`) fica para a Semana 8
- [x] Confirmar/ajustar lista de 7 ativos agro em `config.yaml`
- [~] Atualizar título do TCC para refletir o setor agro/commodities — proposta em
      `docs/projeto_tcc_agro_rev1.docx` (controle de alterações), falta revisar e validar com o orientador
- [~] Reescrever Cap. 1 (problema de pesquisa, objetivos, justificativa) trocando a carteira
      genérica pela carteira agro — rev1 no mesmo arquivo; 3 comentários pendentes (fontes do
      PIB/exportações do agro, número do câmbio em 2020, mudanças societárias JBS/Marfrig)
- [x] Copiar o pré-projeto original para `docs/pre_projeto_original.docx`

## Semana 2 — 02 a 08/09 — Coleta de dados
- [x] `src/data/collectors.py`: funções para Yahoo Finance (equities/commodities/FX),
      BCB-SGS (Selic, IPCA, IBC-Br), FRED (Treasuries 10y, DXY via proxy `DTWEXBGS`) —
      Ipeadata (EMBI+) adiado, ver Pendências
- [x] `notebooks/01_data_collection.ipynb`: baixar e salvar `data/raw/` para os 7 ativos + macro
- [x] Validar cobertura temporal 2005–2024 e ausência de gaps críticos por ativo
      (`src/data/validation.py::check_coverage`)

## Semana 3 — 09 a 15/09 — Engenharia de features
- [x] `src/data/features.py`: Garman-Klass, GARCH(1,1) (refit periódico dos parâmetros +
      atualização diária da variância condicional com retornos observados, sem lookahead),
      retornos multi-horizonte (1/5/21d), skew/curtose móveis (21/63d), spread de
      Corwin-Schultz (proxy de bid-ask), volume normalizado, vol. implícita
      (`^VIX` + Ibov/Bloomberg)
- [x] Tratamento de missing **corrigido** (`align_and_fill_ohlc`: forward-fill causal até
      3 primeiros dias de cada ausência, nunca interpolação/backward-fill — ver Pendências) sobre o
      **calendário real de pregão da B3** (`pandas_market_calendars`), não dias-úteis
      genéricos
- [x] Soja/café (`ZS=F`/`KC=F`) **convertidos para BRL-equivalente** (câmbio do próprio
      dia) antes de qualquer feature — carteira medida na perspectiva de um investidor
      brasileiro (ver Pendências)
- [x] Testes de estacionariedade (ADF, KPSS) **por ativo separadamente** e filtro de
      multicolinearidade/correlação (VIF > 10, pooled) em `src/data/feature_selection.py`
      — diagnóstico exploratório na amostra completa, seleção final entra por split
      walk-forward (Semana 6)
- [x] `notebooks/02_feature_engineering.ipynb`: gera `data/processed/features.parquet`
      (calendário B3; volume normalizado excluído para FX) +
      `data/processed/portfolio_return.parquet` (retorno agregado equal-weighted, BRL,
      preparação para o HMM) + relatórios em `reports/tables/`

## Semana 4 — 16 a 22/09 — Detecção de regimes (HMM)
- [x] `src/models/hmm.py`: HMM gaussiano covariância cheia (`hmmlearn`), grade de 2–4
      estados via BIC/AIC (`select_n_states`), Baum-Welch com múltiplas inicializações
      (`fit_gaussian_hmm`)
- [x] Viterbi in-sample (`viterbi_decode`) e Forward prospectivo sem lookahead
      (`forward_filter`, implementado na mão — `hmmlearn` só expõe smoothing
      forward-backward publicamente)
- [x] Robustez via bootstrap em blocos (`block_bootstrap_stability`): estabilidade do nº
      de estados (BIC) + estabilidade da sequência de regimes (Viterbi realinhado por
      volatilidade, % concordância + Adjusted Rand Index)
- [x] `notebooks/03_hmm_regimes.ipynb`: fit único/exploratório (in-sample) nas duas
      especificações (3D: retorno+vol21+vol63; 2D robustez: retorno+vol21), caracteriza
      regimes (persistência, médias, duração esperada) — walk-forward de verdade fica
      para a Semana 6, ver Pendências

## Semana 5 — 23 a 29/09 — Arquitetura LSTM
- [x] `src/models/lstm.py`: LSTM 2 camadas (64/32), dropout 0,3, one-hot do regime como feature,
      lookback de 42 dias, saída densa, loss MSE, otimizador Adam
- [ ] (não executado; configuração fixa declarada) Grid search restrito: lookback ∈ {21,42,63}, dropout ∈ {0.2,0.3,0.4},
      batch ∈ {32,64}, lr ∈ {0.001,0.0005}
- [ ] `notebooks/04_lstm_training.ipynb` (parte 1 — arquitetura e grid search)

## Semana 6 — 30/09 a 06/10 — Treinamento
- [x] Walk-forward (treino 5a / teste 6m, passo 6m) em `src/data/dataset.py`: 25 janelas
      (testes de 2012-09 a 2024-12). Em vez do split 70/15/15 global, a validação é o trecho
      final (15%) de cada janela de treino. Seleção de features, padronização e HMM (inclusive o
      nº de estados) são ajustados só no treino de cada janela; o regime entra como one-hot do
      filtro Forward.
- [x] Early stopping (paciência 15), seeds fixas, `requirements.txt`/versões documentadas
- [ ] `notebooks/04_lstm_training.ipynb` (parte 2 — treino final e conjunto de teste lacrado)

## Semana 7 — 07 a 13/10 — MC Dropout e VaR/ES
- [x] `src/models/mc_dropout.py`: 1000 passagens com dropout ativo por instante de previsão
- [x] `src/risk/var_es.py`: VaR (quantil empírico) e ES (média das perdas abaixo do quantil),
      condicionais ao regime e não condicionais (95%/99%)
- [ ] `notebooks/05_mc_dropout_var_es.ipynb`

## Semana 8 — 14 a 20/10 — Benchmark MS-GARCH (R)
- [x] `r/msgarch_benchmark.R`: MS-GARCH(1,1) Student-t, 2–3 estados, seleção por BIC (19/10;
      K=2 em todas as 23 janelas; K=3 falhou em 9 por erro do pacote nos valores iniciais e
      perdeu no BIC nas 14 em que convergiu)
- [x] VaR/ES via FHS (vol MS-GARCH × resíduos padronizados do desenvolvimento); ponte por
      parquet (`scripts/msgarch_inputs.py` → R → `scripts/msgarch_forecasts.py`); paramétrico
      Student-t guardado como sensibilidade
- [ ] Comparar ativo a ativo (univariado). Nível de carteira feito sobre o retorno agregado
      (não por variâncias ponderadas, que ignorariam covariâncias)

## Semana 9 — 21 a 27/10 — Backtesting
- [x] `src/risk/backtests.py`: Kupiec (1995), Christoffersen (1998), Acerbi-Székely com p-valor
      por simulação sob H0 (`src/risk/null_paths.py`); duração (C-P 2004) não feita —
      Christoffersen-Pelletier (2004, duração), Acerbi-Székely (2014, ES)
- [x] Funções de perda: escore quantílico (VaR) e FZ0 de Fissler-Ziegel (VaR+ES conjunto)
- [x] Teste Diebold-Mariano (HAC Bartlett, 5 defasagens; HLN disponível)
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
  spot)**: a carteira tem USD/BRL explícito e futuros convertidos para BRL; essas exposições
  precisam ser justificadas pelo objetivo do índice e não constituem automaticamente
  duplicação contábil. `collectors.py` hoje coleta as duas séries lado a lado
  (`collect_futures_raw`/`adjust_futures_roll` para Yahoo, `collect_cepea_spot` para
  CEPEA) exatamente para permitir essa decisão com dado real em mãos. Falta: validar com
  o Prof. Gustavo qual série entra oficialmente na carteira antes da Semana 3
  (engenharia de features).
- **Proxy do DXY**: o FRED gratuito não publica o índice DXY (ICE). Usamos `DTWEXBGS`
  (Broad Trade-Weighted U.S. Dollar Index) como proxy — metodologicamente equivalente
  na literatura, mas não é o mesmo índice/base. Documentar essa ressalva no Cap. 3
  (Metodologia) quando o documento for revisado.
- **Rolagem dos futuros — resolvida com a base pública do pysystemtrade (19/10).** O Yahoo e
  o genérico `S 1 COMB` / `KC1 COMB` do Bloomberg não são ajustados (idênticos em ~100% das
  datas). `scripts/build_public_futures.py` baixa os CSVs "multiple prices" do pysystemtrade
  (commit a62269d2, conferido por SHA256), que trazem o contrato em uso e o seguinte no mesmo
  dia, e reconstrói o índice por razão novo/antigo (`src/data/public_futures_chain.py`).
  Soja: contrato de novembro, 19 rolagens; café: mar/mai/jul/set/dez, 85 rolagens. Validação:
  a queda de -23% da soja em 15/09/2008 no Yahoo vira -2,4%; no café os maiores movimentos
  (seca de 2014, geadas de jul/2021) são reais e permanecem. Ressalvas para o Cap. 3:
  (1) só há fechamento — Garman-Klass, Corwin-Schultz e volume ficam de fora para soja e café;
  (2) a base termina em 28/03/2024, então a amostra passa a ir de 29/06/2007 a 28/03/2024;
  (3) o autor parou de atualizar a base citando dúvidas sobre licenciamento dos dados
  redistribuídos — uso acadêmico com citação, vale comentar com o orientador; (4) o horário
  23:00 da fonte é nocional. Os exports do Bloomberg ficam em `data/raw/` só como registro.
- **Cap. 1 do `.docx` (pivô agro)**: `docs/pre_projeto_original.docx` ainda está na
  versão genérica ("carteiras brasileiras"), sem menção a agro/commodities/tickers.
  Reescrita adiada de propósito para depois da coleta de dados (decisão desta rodada).
- **FRED**: o arquivo local existe; atualizar pela fonte exige `FRED_API_KEY`.
- **Ausência de câmbio em 2008**: o agregado não renormaliza pesos. Quando falta
  qualquer constituinte, mantém `NaN` e registra `portfolio_coverage.csv`.
  Soja/café dependem do mesmo câmbio. O HMM recebe comprimentos das sequências
  contíguas, sem inventar transições através de lacunas.
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
- **VIF descarta uma feature de vol redundante (varia entre rodadas)**: o filtro de
  multicolinearidade (limiar 10, diagnóstico exploratório) sempre derruba uma das
  medidas de volatilidade (já foi `garman_klass_vol_63d`, depois da correção do
  calendário/câmbio passou a ser `realized_vol_21d`) — esperado, várias medidas de vol
  são naturalmente correlacionadas entre si. Ver `reports/tables/feature_vif_selection.csv`
  para o resultado da rodada mais recente; a seleção final de verdade é por split
  walk-forward (Semana 6), não este diagnóstico.
- **Correções de causalidade encontradas rodando de verdade (antes da Semana 4)**:
  1) `align_and_fill_ohlc` usava interpolação linear (usa o preço FUTURO conhecido para
     preencher o passado — lookahead bias real). Trocado por forward-fill causal puro
     (nunca back-fill), limitado aos três primeiros pregões ausentes. A regra tudo-ou-nada foi
     removida porque dependia da duração futura da ausência.
  2) Painel alinhado a `pd.bdate_range` (dias-úteis genéricos) tratava feriados de
     B3/CBOT/ICE como "dado faltando". Trocado pelo calendário real de pregão da B3
     (`pandas_market_calendars`) como calendário mestre do painel (carteira na
     perspectiva de um investidor brasileiro).
  3) ADF/KPSS rodavam num painel de 7 ativos concatenados linha-a-linha — quebra a
     ordenação temporal que os testes assumem. Corrigido para rodar por ativo
     separadamente (`stationarity_by_asset`, índice `(asset, feature)`).
  4) GARCH usava forecast multi-passos-à-frente entre refits (não reage a choques
     realizados até o próximo refit mensal). Trocado por refit periódico dos parâmetros
     + atualização DIÁRIA da variância condicional via recursão `h_t = ω + α·(r_{t-1}-μ)² +
     β·h_{t-1}` com os retornos realmente observados — ainda sem lookahead, mais fiel
     ao caminho real.
- **Conversão cambial de soja/café para BRL (Semana 3, upstream)**: `ZS=F`/`KC=F` (USD)
  são convertidos para BRL-equivalente (`convert_ohlc_to_brl`, câmbio do próprio dia)
  ANTES de qualquer feature — não só no retorno agregado da carteira. Garman-Klass é
  invariante ao fator cambial diário; Corwin-Schultz usa extremos de dois dias e
  pode mudar com o câmbio,
  mas retornos/GARCH/skew/curtose desses 2 ativos mudam de verdade (passam a incorporar
  a volatilidade cambial), pois o objetivo é medir risco na perspectiva de um investidor
  brasileiro. `BRL=X` como fonte do câmbio é a mesma usada como ativo da carteira.
- **Retorno da carteira**: `log1p(mean(expm1(log_retornos)))`, com sete pesos
  iguais rebalanceados diariamente e exigência de todos os constituintes. É índice
  proxy em BRL, não P&L de contratos com margem/colateral/custos.
- **Resultados HMM**: números anteriores à correção de dados estão arquivados em
  `reports/archive/`. Consultar CSVs regenerados e `rebuild_manifest.json`; não
  reutilizar os antigos 98%/75% como resultados corrigidos.
  **Rodada com futuros ajustados (19/10, in-sample, 2007-06 a 2024-03):** BIC e AIC escolhem
  4 estados; o bootstrap confirma 4 estados em 98% das reamostragens, mas a sequência de
  regimes é só moderadamente estável (concordância ~69%, ARI ~0,47). O estado de maior
  volatilidade (127 dias) cobre 24/11/2008-13/02/2009 e 09/03/2020-19/06/2020 e tem o
  **maior** retorno médio (+0,19%/dia). Não é artefato de rolagem: o HMM usa vol realizada
  olhando para trás, então o regime inclui a recuperação pós-crise (mede turbulência, não
  direção), e a carteira em reais tem hedge natural — dólar e commodities convertidas
  sobem nesses episódios, e SLC/JBS ganharam ~0,3%/dia. Ponto para a discussão: na
  carteira do agro, "estresse" não equivale a perda.
- **Bootstrap**: blocos não atravessam lacunas e são sequências independentes no fit.
  Compara os modelos na cronologia original; concordância direta usa somente o
  mesmo número de estados, ARI inclui todos os sucessos. Falhas são contadas.
  As métricas não são diretamente comparáveis às anteriores, que escolhiam a
  última ocorrência de datas duplicadas na reamostragem.
- **Convergência HMM**: exige ganho finito, não negativo e inferior à tolerância
  entre as últimas iterações; atingir n_iter sozinho não basta.
- **HMM é exploratório in-sample**: os parâmetros ainda usam a amostra completa.
  Walk-forward e preparação final da LSTM permanecem na Semana 6.
- **Ausências nas features**: volume de FX foi excluído por não ser aplicável.
  Demais ausências (burn-in, volatilidade implícita pré-2015, gaps) são preservadas.
  Seleção/imputação deverá usar exclusivamente o treino de cada split.
