# Risco de mercado no agronegócio com HMM e LSTM

TCC de Engenharia de Produção — Universidade Presbiteriana Mackenzie.
Orientador: Prof. Gustavo Rocha da Silva.

O projeto compara previsões de VaR e Expected Shortfall (ES) de uma LSTM e de
uma HMM-LSTM com o MS-GARCH e uma referência FHS-EWMA de média zero. O objetivo
é avaliar se a informação sobre o regime de mercado melhora a previsão de risco,
principalmente nos períodos de maior volatilidade.

A LSTM prevê o retorno médio. O VaR e o ES são calculados por simulação histórica
filtrada (FHS), usando erros de calibração padronizados pela volatilidade EWMA.
Na HMM-LSTM, os erros são separados por regime quando há pelo menos 100 observações;
caso contrário, usa-se o conjunto de erros. O MC Dropout é um diagnóstico de
incerteza da rede e não fornece, sozinho, a distribuição de perdas.

## Dados e carteira

| Exposição | Ativo | Identificador no projeto |
|---|---|---|
| Ações | SLC Agrícola | `SLCE3.SA` |
| Ações/BDR | JBS | `JBSS32.SA` |
| Ações | Marfrig/MBRF | `MBRF3.SA` |
| Ações | São Martinho | `SMTO3.SA` |
| Futuros | Soja | `ZS=F` |
| Futuros | Café arábica | `KC=F` |
| Câmbio | USD/BRL | `BRL=X` |

O agregado é um índice de preços em reais com pesos iguais e rebalanceamento
diário: `log1p(mean(expm1(log_retornos)))`. O retorno fica ausente quando falta
qualquer componente. Esse índice não inclui margem, colateral ou custos de
rolagem dos contratos futuros.

As séries de soja e café são reconstruídas com os preços de contratos do
pysystemtrade, fixados por commit e conferidos por SHA256. A rolagem usa a razão
entre os preços dos dois contratos na mesma data. A fonte só contém fechamento;
por isso, medidas que exigem abertura, máxima, mínima ou volume ficam de fora
para esses ativos. Os futuros são convertidos para reais antes do cálculo das
variáveis. A amostra comum disponível vai de junho de 2007 a março de 2024,
limitada pelo histórico do café.

Os parâmetros estão em [config.yaml](config.yaml). As datas de coleta ali
configuradas são mais amplas que a cobertura efetiva dos dados.

## Instalação

Na raiz do projeto, em PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O arquivo `requirements-lock.txt` registra as versões do ambiente usado.
Para o benchmark, instale R e os pacotes `MSGARCH` e `arrow`.
As versões da execução local ficam em `reports/tables/msgarch/versions.txt`.

Os dados não são versionados. O notebook 01 faz a coleta; a atualização do FRED
exige `FRED_API_KEY` no `.env` (ver `.env.example`). A volatilidade implícita
brasileira depende do arquivo local `data/raw/ibov_vol_bloomberg.xlsx`, exportado
do terminal Bloomberg. Essa entrada precisa ser obtida separadamente para
reproduzir o experimento completo. Consulte também as condições de uso das
fontes antes de redistribuir os dados.

## Execução

Execute os comandos a partir da raiz. Primeiro rode o notebook
`notebooks/01_data_collection.ipynb` e reconstrua os futuros:

```powershell
python scripts/build_public_futures.py
python scripts/rebuild_local.py
```

`rebuild_local.py` executa os notebooks 02 e 03 com os dados locais, guarda cópias
das saídas anteriores em `reports/archive/` e atualiza `requirements-lock.txt`.
O HMM do notebook 03 é uma análise descritiva da amostra completa; seus regimes
não são usados como entrada da avaliação fora da amostra.

Depois, execute o experimento e a avaliação:

```powershell
python scripts/run_walk_forward.py
python scripts/msgarch_inputs.py
Rscript r/msgarch_benchmark.R
python scripts/msgarch_forecasts.py
python scripts/evaluate_walk_forward.py
python scripts/evaluate_density.py
```

Cada janela tem cinco anos de desenvolvimento e seis meses de teste, com passo
de seis meses. Os últimos 252 pregões do desenvolvimento são reservados à
calibração; a validação ocupa 15% do período restante. Seleção de variáveis,
padronização e HMM são ajustados no treino. O regime usado na previsão vem do
filtro Forward. A configuração da LSTM é fixa, sem busca em grade.

Para a robustez por ativo, repita as etapas com `--target-asset` conforme os
exemplos em `scripts/summarize_univariate.py`. Após concluir essas avaliações:

```powershell
python scripts/summarize_univariate.py
python scripts/build_final_tables.py
python -m pytest
```

As previsões ficam em `reports/tables/walk_forward/`, os backtests em
`walk_forward/evaluation_v3/` e os diagnósticos de densidade em
`walk_forward/density/`. As tabelas consolidadas são gravadas em
`reports/tables/final/`. Os notebooks 04 a 09 apresentam as análises e geram
figuras em `reports/figures/`.

## Organização

- `src/data/`: coleta, tratamento e preparação das janelas.
- `src/models/`: HMM, LSTM e MC Dropout.
- `src/risk/`: cenários, VaR/ES, backtests e avaliação de densidade.
- `scripts/` e `r/`: execução do experimento e benchmark MS-GARCH.
- `notebooks/`: exploração dos dados e análise dos resultados.
- `tests/`: testes dos dados, modelos e avaliação.
- `docs/`: versões do texto acadêmico e pôster; `pesquisa/` contém a revisão em
  texto e `apoio/` reúne notas, fontes consultadas e uma exportação HTML.
- `data/` e `reports/`: entradas e saídas locais, fora do versionamento.

O documento integrado mais recente é
`docs/projeto_tcc_agro_rev2_literatura.docx`. O pré-projeto e a revisão anterior
permanecem como histórico. As tarefas para a entrega e a leitura dos resultados
estão em [PLANO.md](PLANO.md).
