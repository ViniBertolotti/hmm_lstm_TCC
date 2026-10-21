# Acompanhamento do TCC

Atualizado em 21/10/2026. Entrega prevista: 01/12/2026.

## Situação do experimento

O fluxo de dados, HMM, LSTM/HMM-LSTM, FHS, MS-GARCH e backtests já foi executado.
A rodada disponível tem 23 janelas e 2.717 dias de teste, de 01/10/2012 a
27/03/2024. Os notebooks 04 a 09 apresentam os resultados, incluindo robustez
por ativo, SHAP, PIT e CRPS. Os comandos estão no [README](README.md).

As tabelas em `reports/tables/final/` são a referência para a redação.
Rodadas anteriores e registros de conferência ficam em
`reports/archive/`. As correções de execução e documentação de 21/10 ficam em
`reports/archive/correcoes_20261021/`.

## Resultados a discutir

A leitura abaixo segue as hipóteses do pré-projeto, registradas em
`reports/tables/final/T5_hipoteses.csv`.

| Hipótese | Resultado da rodada |
|---|---|
| H1: cobertura compatível e desempenho superior ou equivalente ao MS-GARCH | Parcial: cobertura não rejeitada no total, mas escores piores que os do MS-GARCH. |
| H2: melhor desempenho no estresse | Não sustentada: a HMM-LSTM apresenta mais violações que o MS-GARCH. |
| H3: ausência de perda significativa nos demais regimes | Rejeitada pelos testes DM dos escores quantílico e FZ0. |

No estresse (289 dias), a taxa de violações do VaR de 95% é 8,0% na HMM-LSTM
e 6,6% no MS-GARCH; a 99%, é 2,4% e 2,1%, respectivamente. No total, o teste
de ES a 99% rejeita as duas redes e a FHS-EWMA, mas não o MS-GARCH. No
estresse, rejeita os quatro modelos.

O MS-GARCH tem o menor CRPS médio e PIT marginal mais próximo da uniforme
(KS p = 0,96), mas o teste de Berkowitz rejeita sua especificação no total.
Portanto, os diagnósticos de densidade também precisam ser discutidos em
conjunto. Não rejeitar cobertura não demonstra equivalência entre modelos.

Na rodada de referência, os 289 dias de estresse usaram a cauda conjunta por
falta de erros do estado. A comparação com calibração conjunta mantém as
previsões das redes e altera apenas a regra de seleção dos erros. Essa análise
separa a comparação das entradas de regime da mudança na calibração da cauda.
O MS-GARCH usa resíduos do desenvolvimento; as redes usam o bloco de calibração.

## Decisões metodológicas para o texto

- Descrever o agregado como índice de preços em reais com sete pesos iguais,
  exigindo todos os componentes; distinguir esse índice do P&L de futuros.
- Explicar a conversão de soja/café para reais e a exposição adicional ao USD/BRL.
- Registrar a fonte dos futuros, o ajuste de rolagem, a ausência de OHLCV,
  o horário nocional da fonte e o fim da série em março de 2024.
- Documentar o calendário B3 e o preenchimento causal dos primeiros três
  pregões de cada ausência. As demais lacunas são preservadas.
- Distinguir o HMM descritivo do notebook 03 dos ajustes por janela. No
  walk-forward, a rede recebe o regime filtrado, sem informação futura.
- Descrever a LSTM com configuração fixa (42 pregões, camadas 64/32,
  dropout 0,3), sem busca em grade. MC Dropout é diagnóstico; VaR/ES usam FHS.
- Explicar a EWMA com λ = 0,94 e o mínimo de 100 erros para a cauda por regime.
- Descrever o MS-GARCH Student-t com seleção de 2 ou 3 estados por BIC no
  desenvolvimento. O VaR/ES principal usa FHS; o paramétrico é sensibilidade.
- Identificar `DTWEXBGS` como proxy do dólar, diferente do DXY, e registrar a
  dependência da exportação Bloomberg para a volatilidade implícita brasileira.

## Pendências para a entrega

- [x] Retomar H1–H3 do pré-projeto no documento integrado, com registro da
  formulação intermediária de H3. Não foi realizado teste de não inferioridade econômica.
- [ ] Revisar com o orientador o título e a interpretação de H1–H3.
- [x] Redigir resultados e discussão a partir das tabelas consolidadas (seção 3.12).
- [x] Atualizar a metodologia com FHS, calibração, MS-GARCH e SHAP executados.
- [x] Resolver os conflitos de merge e validar a suíte: 118 testes passaram.
- [x] Registrar configuração, versões, hashes e checkpoints nas novas execuções;
  impedir retomada incompatível e conferir a integridade dos artefatos.
- [x] Corrigir Berkowitz para não ligar sessões separadas por lacunas.
- [ ] Confirmar com o orientador a escolha dos futuros e as condições de uso
  da base pública. Manter CEPEA como análise separada (`scripts/pilot_cepea.py`).
- [ ] Discutir a interpretação dos quatro estados do HMM descritivo. O estado
  de maior volatilidade inclui recuperações; estresse não significa queda.
- [ ] Revisar citações, referências, apêndices e formatação pelo manual do curso.
- [x] Atualizar o sumário no Word e conferir a paginação do documento integrado (45 páginas).
- [x] Atualizar o pôster com o benchmark MS-GARCH executado e conferir o layout.
- [ ] Incluir o e-mail do orientador no pôster, após confirmação do endereço.
- [ ] Obter a revisão do orientador e preparar os arquivos de entrega.

## Prazo de revisão

| Período | Trabalho restante |
|---|---|
| Até 01/11 | Validar decisões com o orientador e atualizar a metodologia. |
| 02–15/11 | Redigir resultados, discussão e limitações. |
| 16–24/11 | Revisar texto, referências, apêndices e pôster. |
| 25/11–01/12 | Conferir os arquivos finais e entregar. |
