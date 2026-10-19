# 2 REVISÃO DA LITERATURA

A estimação do risco de mercado exige mais do que prever o movimento médio dos preços. Para apoiar decisões de exposição, proteção e reserva de recursos, é necessário avaliar a possibilidade de perdas e a intensidade que elas podem atingir. Essa distinção ganha relevância em uma carteira ligada ao agronegócio brasileiro, na qual ações de empresas, commodities agrícolas e câmbio respondem a fatores parcialmente comuns, mas representam exposições econômicas diferentes. A análise precisa, portanto, aproximar os fundamentos do mercado das propriedades estatísticas dos retornos e dos limites dos modelos empregados.

Este capítulo articula essas questões a partir da literatura sobre volatilidade condicional, mudanças de regime, redes recorrentes, incerteza preditiva e avaliação de risco. O argumento que orienta a revisão é que a combinação entre Hidden Markov Models (HMM) e redes Long Short-Term Memory (LSTM) constitui uma possibilidade de modelagem a ser investigada, cuja utilidade depende da qualidade dos dados e de evidências fora da amostra. A presença de uma arquitetura mais flexível, por si só, não demonstra melhor previsão de Value at Risk (VaR) ou Expected Shortfall (ES). Os conceitos e estudos examinados a seguir estabelecem os fundamentos necessários para avaliar essa contribuição.

## 2.1 Delimitação e organização da revisão

A revisão tem caráter narrativo e crítico. Seu propósito é relacionar contribuições teóricas e empíricas ao problema de estimar o risco de uma carteira do agronegócio, identificando o que cada abordagem permite representar e quais questões permanecem abertas. A seleção reúne trabalhos fundadores dos métodos utilizados, aplicações financeiras e documentos metodológicos das instituições responsáveis pelos indicadores de mercado. A inclusão de estudos clássicos é necessária porque conceitos como volatilidade condicional, estados latentes e cobertura de quantis continuam sustentando as aplicações recentes.

O exame das aplicações distingue três objetos de previsão: o retorno esperado, a volatilidade e o risco de cauda. Embora relacionados, esses objetos exigem critérios diferentes de avaliação. Um estudo sobre acerto da direção dos preços pode justificar o interesse por uma rede neural, mas não constitui evidência suficiente de que essa rede produz quantis bem calibrados. Da mesma maneira, reduzir o erro de previsão da volatilidade não assegura, isoladamente, a adequação das perdas extremas estimadas. Essa distinção orienta a leitura dos resultados publicados e evita transportar conclusões entre problemas que não são equivalentes.

As fontes são discutidas conforme sua contribuição para o desenho da pesquisa, considerando o mercado analisado, a variável prevista, o tratamento da ordem temporal, os modelos de comparação e o alcance das conclusões. Documentos de metodologia esclarecem o significado dos dados; estudos empíricos permitem examinar aplicações e limitações; trabalhos teóricos fundamentam as propriedades dos estimadores e dos testes. A síntese não pretende medir a produção científica do campo nem estabelecer a inexistência de trabalhos semelhantes. Procura construir uma base argumentativa que permita justificar as decisões do TCC e interpretar seus resultados sem antecipá-los.

## 2.2 Risco de mercado em uma carteira do agronegócio brasileiro

### 2.2.1 Exposições a commodities ações e câmbio

Uma carteira relacionada ao agronegócio não corresponde a uma exposição uniforme ao preço dos produtos agrícolas. Para uma empresa produtora, a elevação do preço de venda pode favorecer a receita; para uma atividade que utiliza grãos como insumo, o mesmo movimento pode pressionar custos. Nas companhias abertas, os efeitos ainda dependem de endividamento, composição das receitas, estrutura de custos, decisões de investimento e condições do mercado acionário. Por isso, a associação setorial entre os ativos é um ponto de partida para a análise, e não uma garantia de comportamento semelhante ou de diversificação estável.

A literatura de commodities ajuda a compreender por que essas relações variam. Gorton, Hayashi e Rouwenhorst (2013) relacionam os prêmios de risco dos contratos futuros às condições dos estoques e mostram a relevância da base e de outras informações de preços para esse mercado. A interpretação econômica é que a disponibilidade física do produto altera o valor de mantê-lo e a remuneração exigida pelos participantes. Assim, uma mudança na curva de vencimentos não representa apenas um detalhe de construção da série: pode refletir condições distintas de oferta, demanda e armazenamento.

Choques climáticos acrescentam outra dimensão. Ubilava (2012) investiga os efeitos do El Niño e da La Niña sobre os preços mundiais do café, utilizando dados mensais entre março de 1989 e dezembro de 2010 e uma estrutura autorregressiva com transição suave. O estudo identifica efeitos de curto prazo associados ao fenômeno climático e oferece evidência de que a dinâmica de preços pode apresentar não linearidades. Seu alcance, entretanto, é o mercado de café e a frequência examinada. Os resultados não estabelecem o desempenho de um modelo diário de VaR para uma carteira brasileira, mas ajudam a justificar a atenção a mudanças nas condições de mercado.

Para o investidor que mede patrimônio em reais, o câmbio modifica a exposição aos ativos cotados em dólares. Se P em USD representa o preço de referência de uma commodity e E representa a quantidade de reais por dólar, o preço equivalente em reais é obtido pelo produto dessas duas grandezas. Consequentemente, o retorno logarítmico dessa referência em reais reúne a variação do preço em dólares e a variação cambial. Essa identidade contábil não elimina a necessidade de sincronizar os horários das cotações, pois preços associados à mesma data podem ter sido observados em momentos diferentes.

A presença adicional de uma posição cambial na carteira deve ser interpretada à luz dessa conversão. A moeda pode influenciar tanto a referência em reais das commodities quanto o componente de câmbio considerado separadamente. Isso não constitui necessariamente um erro, mas significa que os pesos nominais dos componentes não resumem toda a exposição cambial do conjunto. A análise do risco precisa reconhecer essa sobreposição, em vez de presumir que sete séries distintas representam sete fontes independentes de incerteza.

### 2.2.2 Contratos futuros séries contínuas e mercado físico

Os contratos futuros possuem vencimento, o que exige uma regra para acompanhar uma exposição ao longo de vários anos. Segundo a CME Group (s.d.), a rolagem envolve encerrar a posição em um vencimento e estabelecer outra em um contrato posterior; o volume negociado pode auxiliar a escolha do momento da transição. A construção de uma série contínua, porém, requer decisões adicionais sobre qual vencimento utilizar, quando substituí-lo e como tratar a diferença entre os preços dos contratos. Também exige observar as regras de entrega e os limites operacionais específicos de cada instrumento.

Quando uma série troca simplesmente o contrato antigo pelo novo, o retorno calculado entre as duas datas pode combinar a mudança efetiva do mercado com a diferença entre vencimentos. Essa descontinuidade pode contaminar medidas de volatilidade, a identificação de regimes e a estimação das caudas. Ajustes por diferença ou por razão procuram tratar o problema, mas não têm os mesmos efeitos sobre níveis e variações percentuais. A escolha precisa corresponder à variável analisada e ser documentada com preços simultâneos dos contratos, sem apagar movimentos extremos apenas por parecerem inconvenientes.

Uma regra baseada em liquidez também exige atenção à informação disponível no momento da decisão. Se o volume total do dia define qual contrato será escolhido para a posição que supostamente já estava aberta naquele mesmo dia, a análise pode incorporar informação posterior à decisão. O mesmo cuidado vale para indicadores calculados com preços de fechamento de bolsas diferentes. A validade de uma regra de rolagem depende, portanto, de sua interpretação temporal, e não apenas de produzir uma trajetória visualmente mais contínua.

Há ainda uma diferença entre um índice de referências de preços e o resultado financeiro de contratos efetivamente negociados. O resultado de uma posição futura depende de quantidade de contratos, multiplicadores, ajustes diários, recursos mantidos como garantia, custos e regra de rolagem. Um índice que combina variações percentuais de séries contínuas pode representar uma aproximação útil da exposição conjunta, mas não reproduz automaticamente essa contabilidade. A relação entre estoques, curva e retornos discutida por Gorton, Hayashi e Rouwenhorst (2013) reforça a necessidade de separar essas interpretações.

Os indicadores do Centro de Estudos Avançados em Economia Aplicada (Cepea) representam outra natureza de informação. A metodologia da soja de Paranaguá define produto, praça, condições comerciais e procedimentos de coleta; por isso, o indicador se refere a uma referência do mercado físico, ainda que possa se relacionar à liquidação de instrumentos financeiros (Cepea, s.d.). Empregar um indicador físico em lugar de um futuro altera a exposição estudada e envolve diferenças de qualidade, localização, prazo e formação de preços. Essa alternativa pode ser pertinente em uma análise própria, mas não deve ser apresentada como simples correção técnica de uma série futura.

## 2.3 Retornos volatilidade e informação de mercado

### 2.3.1 Retorno da carteira e fatos estilizados

O retorno simples de um ativo mede a variação relativa do preço entre dois instantes, enquanto o retorno logarítmico corresponde ao logaritmo da razão entre esses preços. Para preços positivos, as duas medidas se relacionam por uma transformação conhecida. A distinção importa porque os retornos logarítmicos são aditivos no tempo, mas não são diretamente aditivos entre ativos de uma carteira. Em uma carteira com pesos definidos antes do período, o retorno simples é a soma ponderada dos retornos simples dos componentes; o logaritmo do retorno bruto dessa carteira deve ser calculado depois da agregação.

Rₚ,ₜ = Σᵢ wᵢ,ₜ₋₁Rᵢ,ₜ     e     rₚ,ₜ = ln(1 + Rₚ,ₜ)

Nessa expressão, R representa o retorno simples, r o retorno logarítmico e w o peso do ativo no início do período. A representação pressupõe uma definição compatível de patrimônio e de retorno para todos os componentes. Quando parte das séries representa referências contínuas de futuros, a interpretação como carteira operacional depende das convenções financeiras discutidas anteriormente. A fórmula organiza a mensuração do índice, mas não substitui a definição econômica da exposição.

Cont (2001) reúne propriedades recorrentes em séries de retornos, entre elas caudas pesadas, agrupamento de volatilidade e formas de dependência que não se resumem à autocorrelação linear. A baixa previsibilidade da direção de um retorno não implica ausência de previsibilidade em sua intensidade. Períodos de grandes variações podem ser seguidos por outros períodos de instabilidade, mesmo quando o sinal do movimento seguinte permanece difícil de antecipar. Essa diferença explica por que prever risco e prever a valorização de um ativo são tarefas relacionadas, mas distintas.

Também é necessário separar características condicionais e incondicionais. Uma distribuição observada ao longo de muitos anos pode reunir períodos de baixa e alta volatilidade. Parte de sua dispersão decorre dessa mistura de condições. Ainda assim, identificar mudanças de variância não garante que toda a espessura das caudas tenha sido explicada. Para o presente estudo, isso significa que a classificação de regimes e o tratamento das inovações extremas precisam ser avaliados conjuntamente, sem tomar a presença de estados como solução automática para qualquer forma de risco.

### 2.3.2 Volatilidade condicional e medidas auxiliares

Engle (1982) introduz a estrutura ARCH, na qual a variância condicional depende de informações passadas dos erros. Bollerslev (1986) amplia essa ideia com o GARCH, incorporando variâncias condicionais defasadas. No caso GARCH(1,1), uma formulação usual combina uma constante positiva, o choque quadrático anterior e a variância do período anterior. A estrutura permite representar persistência e atualização do risco de maneira relativamente parcimoniosa.

hₜ = ω + αε²ₜ₋₁ + βhₜ₋₁

A equação expressa uma variância condicional h, e não o retorno esperado. Na especificação convencional, restrições sobre ω, α e β asseguram positividade; condições adicionais são necessárias para a existência de momentos e para a estabilidade do processo. Mesmo com parâmetros constantes, o GARCH produz volatilidade variável e constitui um modelo não linear para a distribuição dos retornos. Sua limitação não deve, portanto, ser descrita como incapacidade geral de lidar com não linearidades. A questão relevante é se a forma de atualização e a distribuição escolhidas representam adequadamente o mercado considerado.

Outras medidas fornecem informações complementares. Garman e Klass (1980) propõem estimadores que utilizam abertura, máxima, mínima e fechamento, aproveitando a amplitude observada dentro do período. Essas medidas dependem de hipóteses sobre o processo de preços e da consistência das cotações. Uma estimativa obtida de OHLC diário não deve ser identificada automaticamente com a variância realizada calculada a partir de retornos intradiários. Além de medirem aspectos distintos, ambas podem ser afetadas por horários de negociação, movimentos entre sessões e qualidade dos registros.

No mercado brasileiro, Campani e Durães (2018) analisam a previsão de volatilidade do USD/BRL com modelos GARCH ampliados por variância realizada de alta frequência e pelo índice de volatilidade implícita FXVol. Os resultados favorecem a incorporação dessas informações no contexto examinado. A contribuição do estudo é mostrar que diferentes fontes podem acrescentar conteúdo preditivo ao histórico de retornos. Isso não permite concluir que qualquer indicador de volatilidade terá o mesmo efeito em ações, commodities ou em uma carteira agregada.

A volatilidade implícita também precisa ser associada ao mercado de origem. A Cboe (s.d.) define o VIX como uma medida de volatilidade esperada para o S&P 500 nos 30 dias seguintes, extraída de preços de opções. Ele pode servir como informação sobre condições financeiras externas, mas não mede diretamente a volatilidade futura de uma carteira do agronegócio brasileiro. Da mesma forma, o estimador de Corwin e Schultz (2012) utiliza máximas e mínimas diárias para inferir o spread de negociação. Trata-se de uma aproximação de liquidez, e não de uma observação direta de cotações de compra e venda.

## 2.4 Value at Risk Expected Shortfall e risco de cauda

### 2.4.1 Definição das medidas e interpretação da perda

A mensuração do risco começa pela definição da variável de perda, do horizonte e do conjunto de informações disponível. Se o objeto for o retorno simples de um índice de preços, pode-se representar a perda proporcional por L = −R. Valores positivos indicam perdas e valores negativos indicam ganhos. Essa convenção evita misturar o sinal de um retorno com o sinal da exposição ao risco e não requer transformar todos os ganhos em zero. Caso o objeto seja o resultado monetário de uma carteira operacional, a definição precisa acompanhar sua contabilidade específica.

Para um nível de confiança α, o VaR é um quantil da distribuição de perdas condicionada às informações disponíveis na origem da previsão. Um VaR a 99% procura delimitar um ponto abaixo do qual se encontra aproximadamente 99% dessa distribuição, sob as condições do modelo. Ele não representa a maior perda possível nem informa, sozinho, a magnitude do que ocorre além do quantil. Uma carteira pode apresentar o mesmo VaR que outra e, ainda assim, concentrar perdas muito mais intensas na região extrema.

VaRα,ₜ = inf{ℓ : P(Lₜ₊₁ ≤ ℓ | ℱₜ) ≥ α}

O Expected Shortfall complementa essa informação ao resumir a região mais desfavorável da distribuição. Acerbi e Tasche (2002) discutem sua definição e coerência, destacando a importância de tratar corretamente distribuições com descontinuidades. Uma formulação geral corresponde à média dos quantis da cauda superior das perdas. Em distribuições contínuas, ela pode ser relacionada à perda esperada além do VaR; na presença de massa concentrada no quantil, a regra de ponderação dessa massa precisa ser preservada.

ESα,ₜ = [1/(1 − α)] ∫ₐ¹ VaRu,ₜ du

Na expressão, a integral começa em α e reúne os quantis superiores até 1. A propriedade de subaditividade do ES, nas condições de integrabilidade pertinentes, oferece uma justificativa para seu uso na análise de diversificação. Essa propriedade não significa, porém, que sua estimativa seja simples ou estável em qualquer amostra. Como o ES depende da intensidade das perdas extremas, a escassez de observações na cauda pode torná-lo especialmente sensível à escolha do período e à especificação da distribuição (Acerbi; Tasche, 2002).

### 2.4.2 Formas de estimação e informação disponível na cauda

Há diferentes caminhos para obter as medidas. Modelos paramétricos especificam uma distribuição condicional; a simulação histórica utiliza realizações passadas; procedimentos semiparamétricos combinam uma estrutura dinâmica com informações empíricas sobre os erros. Hull e White (1998), por exemplo, propõem atualizar a volatilidade na simulação histórica, permitindo que observações passadas sejam consideradas à luz das condições correntes de risco. A escolha entre esses caminhos envolve hipóteses distintas sobre estabilidade dos resíduos, capacidade de adaptação e tratamento dos extremos.

A previsão direta dos quantis constitui outra possibilidade. Engle e Manganelli (2004) apresentam o CAViaR, que modela a evolução do quantil condicional por regressão quantílica, sem exigir que toda a distribuição dos retornos seja descrita por uma única família paramétrica. Taylor (2019) desenvolve uma abordagem semiparamétrica para a previsão conjunta de VaR e ES. Essas contribuições mostram que a modelagem do risco pode ser orientada diretamente às características da cauda, em vez de decorrer necessariamente de uma previsão pontual seguida por uma hipótese adicional de distribuição.

O nível de confiança influencia a quantidade de informação efetivamente disponível. Em uma amostra de 125 observações, uma cauda de 1% corresponde a uma massa de apenas 1,25 observação. A obtenção de milhares de cenários computacionais a partir dessa amostra não equivale à observação de milhares de crises independentes. A simulação pode reduzir o erro numérico de um cálculo, mas a qualidade da estimativa continua limitada pela informação histórica e pelas hipóteses que geram os cenários.

A unidade da variável também precisa ser mantida. Se uma rede prevê retornos logarítmicos, a perda simples correspondente é obtida por L = 1 − exp(r). Para estimar o ES nessa escala, cada cenário deve ser transformado antes da média na cauda. Em geral, transformar a média de uma variável não produz a média de sua transformação. Essa diferença matemática afeta a coerência entre o alvo previsto, os cenários simulados e a interpretação econômica das medidas de risco.

## 2.5 Mudanças de regime e Hidden Markov Models

Hamilton (1989) propõe uma abordagem para representar mudanças discretas e não observadas na dinâmica de séries econômicas, com aplicação ao ciclo de negócios dos Estados Unidos. A ideia central é permitir que o comportamento da série dependa de um estado latente que evolui segundo uma cadeia de Markov. Em uma aplicação financeira, esse mecanismo pode representar condições com diferentes níveis de retorno e volatilidade. A transição entre estados passa a integrar o modelo, em vez de ser definida exclusivamente por datas de crise escolhidas após a observação dos acontecimentos.

Os HMM organizam essa ideia em dois componentes: um processo de estados não observados e uma distribuição das observações condicionada ao estado. A matriz de transição descreve as probabilidades de passagem entre estados, enquanto as distribuições de emissão representam o comportamento dos dados em cada um deles. Rabiner (1989) sistematiza os problemas de avaliação, inferência e estimação associados a essa estrutura, oferecendo a base algorítmica para seu emprego em diferentes domínios.

Na formulação gaussiana, cada estado pode apresentar média e matriz de covariância próprias. A mistura resultante amplia a representação possível da distribuição agregada, mas não elimina as hipóteses adotadas dentro de cada estado. Dependências remanescentes, caudas condicionais muito espessas ou durações incompatíveis com a estrutura markoviana podem continuar mal representadas. Um HMM deve, portanto, ser avaliado como uma aproximação estatística útil, e não como recuperação inequívoca de um mecanismo econômico verdadeiro.

A distinção entre probabilidades filtradas e suavizadas é decisiva para previsão. A filtragem utiliza somente observações disponíveis até o instante considerado. A suavização incorpora também observações posteriores, podendo ser adequada à interpretação retrospectiva de episódios históricos. Embora ambas sejam úteis, apenas a primeira corresponde à informação que estaria disponível para uma decisão tomada naquele momento. O algoritmo de Viterbi, por sua vez, procura uma trajetória conjunta mais provável; aplicá-lo à sequência completa não transforma seus estados históricos em informações prospectivas (Rabiner, 1989).

πₜ(k) ∝ fₖ(yₜ) Σⱼ pⱼₖπₜ₋₁(j)

Essa forma resumida da recursão Forward combina a probabilidade anterior, as transições entre estados e a plausibilidade da observação corrente sob cada estado. A normalização faz as probabilidades somarem um. Na aplicação preditiva, os parâmetros envolvidos também precisam ter sido estimados com dados anteriores ao período avaliado. Uma filtragem calculada apenas com o passado ainda pode carregar informação futura se a matriz de transição ou as distribuições de emissão tiverem sido ajustadas com a amostra completa.

A escolha do número de estados envolve um compromisso entre representação e precisão. Mais estados permitem descrever diferenças adicionais, mas aumentam a quantidade de parâmetros e podem fragmentar episódios raros. Critérios de informação, estabilidade da estimação e interpretação das distribuições ajudam a avaliar esse compromisso; a escolha final não deve ser orientada pelo desempenho já observado no teste. Além disso, os números atribuídos aos estados não têm significado econômico intrínseco. Comparações entre janelas exigem critérios consistentes de identificação, como a ordenação por uma medida de volatilidade estimada no treinamento.

Para integrar o HMM a outro modelo, pode-se utilizar um estado selecionado ou suas probabilidades. A seleção de um único estado simplifica a representação, mas descarta parte da incerteza de classificação. Probabilidades preservam essa informação, embora acrescentem entradas e possíveis dependências ao modelo seguinte. Não há razão teórica suficiente para presumir que uma dessas opções dominará sempre a outra. O ganho deve ser investigado na comparação com uma arquitetura equivalente que não recebe a informação de regime.

## 2.6 Modelos MS GARCH como referência econométrica

O MS-GARCH combina mudanças de regime com dinâmicas de variância condicional. Haas, Mittnik e Paolella (2004) apresentam uma formulação que procura lidar com dificuldades de estimação e de caracterização dinâmica de modelos anteriores. Sua proposta permite que processos de variância associados a diferentes regimes evoluam em paralelo. O estado latente determina a distribuição relevante para as observações, preservando a atualização da volatilidade a partir do histórico disponível.

Essa estrutura evita interpretar o MS-GARCH como a simples seleção diária de um entre vários GARCH ajustados de forma independente após a classificação dos dados. A modelagem envolve a dinâmica das variâncias, as probabilidades dos estados e a distribuição condicional resultante. Dependendo da especificação, mudanças nos parâmetros de persistência e na forma das inovações podem representar comportamentos que um modelo de regime único descreve de maneira menos flexível. O aumento da flexibilidade, contudo, também pode dificultar a estimação e exigir mais informação da amostra.

Ardia et al. (2019) apresentam o pacote MSGARCH, com ferramentas para estimação por máxima verossimilhança e inferência bayesiana, simulação, previsão de densidades e cálculo de VaR e ES. A contribuição computacional amplia a possibilidade de aplicações reproduzíveis. O artigo não deve ser interpretado como demonstração de superioridade universal do MS-GARCH em qualquer período ou mercado. Sua relevância para este TCC está em oferecer uma referência que também contempla regimes e heterocedasticidade, tornando a comparação com a arquitetura neural conceitualmente mais exigente.

Uma consequência importante da distribuição em mistura é que o quantil agregado não corresponde, em geral, à média ponderada dos quantis de cada estado. As probabilidades precisam compor a distribuição preditiva antes da obtenção do VaR e do ES. Da mesma forma, escolher inovações Student-t constitui uma hipótese paramétrica sobre a distribuição condicional; isso não equivale a utilizar simulação histórica filtrada. A nomenclatura dos procedimentos precisa refletir como os cenários foram efetivamente gerados, pois métodos com nomes semelhantes podem impor hipóteses diferentes (Ardia et al., 2019).

## 2.7 Redes LSTM e modelos híbridos em finanças

### 2.7.1 Aprendizado temporal e objeto da previsão

Hochreiter e Schmidhuber (1997) introduzem a LSTM para enfrentar dificuldades de aprendizado de dependências longas em redes recorrentes. A arquitetura utiliza um estado de memória e mecanismos de controle do fluxo de informação, reduzindo problemas associados à propagação do gradiente ao longo da sequência. Gers, Schmidhuber e Cummins (2000) acrescentam a porta de esquecimento adaptativa, componente importante das formulações modernas. Essa evolução permite que a rede aprenda quando preservar, incorporar ou descartar informações, sem fixar antecipadamente uma única regra de dependência temporal.

Em finanças, uma LSTM pode receber sequências de retornos, medidas de volatilidade e outras variáveis disponíveis na origem da previsão. Sua flexibilidade permite representar interações que seriam trabalhosas de especificar manualmente. Entretanto, a quantidade de observações efetivamente aproveitável diminui com a exigência de janelas completas, defasagens e divisões cronológicas. Acrescentar camadas ou variáveis pode aumentar a capacidade de ajuste sem produzir melhora fora da amostra, especialmente quando os episódios relevantes para a cauda são poucos.

A função de perda também delimita o que a rede procura aprender. Sob condições usuais, minimizar o erro quadrático médio orienta a previsão para a média condicional do alvo. Se esse alvo for o retorno, o resultado não será automaticamente um quantil extremo nem uma previsão de volatilidade. Uma arquitetura voltada à variância, a quantis ou a uma distribuição completa resolve problemas diferentes. A justificativa do modelo precisa, portanto, combinar a estrutura da rede com o alvo, a função de perda e o procedimento que transforma a saída em uma medida de risco.

### 2.7.2 Evidências empíricas e limites de transferência

Fischer e Krauss (2018) aplicam LSTM à previsão fora da amostra da direção dos movimentos de ações componentes do S&P 500 entre 1992 e 2015, comparando-a a métodos de classificação sem a mesma estrutura de memória. O estudo constitui uma referência para o uso de redes recorrentes em finanças, mas seu objeto é distinto da estimação de perdas extremas de uma carteira brasileira. A evidência de desempenho em uma estratégia direcional não comprova, por si, a cobertura de VaR ou a precisão de ES em outro universo de ativos.

A integração entre regimes e redes neurais já aparece em trabalhos voltados ao risco. De Mel (2026) apresenta uma combinação de HMM gaussiano e LSTM para dados da Microsoft entre 2016 e 2022, separando os períodos de treinamento e teste. Conforme a descrição publicada, a rede prevê uma aproximação de volatilidade, o regime é utilizado como entrada e uma distribuição Student-t de resíduos padronizados fornece o tratamento da cauda. Esse desenho mostra que a informação de estado e a distribuição das inovações desempenham funções distintas na obtenção das medidas de risco.

O trabalho de De Mel (2026) também ilustra um limite de interpretação dos backtests: a não rejeição da cobertura incondicional é um resultado específico, que não substitui a avaliação da dependência das violações, da severidade das perdas ou da comparação pareada entre modelos. Sua aplicação a uma ação norte-americana oferece um antecedente metodológico, mas não resolve a questão do desempenho em uma carteira que reúne ações brasileiras, commodities agrícolas e câmbio.

Arimond et al. (2026) estudam redes neurais e VaR em carteiras de índices de ações e títulos de diferentes mercados, incluindo inicializações baseadas em HMM. Os autores relatam benefícios de uma LSTM em determinadas configurações e destacam a dependência em relação à disponibilidade de históricos longos. A forma de incorporar os estados e a finalidade de gestão de ativos diferem de uma rede que recebe probabilidades filtradas como variáveis explicativas. O estudo reforça o interesse pela combinação, ao mesmo tempo que impede tratar HMM-LSTM como uma arquitetura sem antecedentes na literatura financeira.

A contribuição de uma combinação específica precisa ser identificada por comparação. Uma LSTM sem regime ajuda a verificar se o HMM acrescenta informação útil além das mesmas variáveis e defasagens. O MS-GARCH permite examinar se os eventuais ganhos dependem apenas de contemplar estados e volatilidade variável. Essas comparações não isolam um efeito causal em sentido econômico, mas tornam a interpretação preditiva mais clara. Se os modelos utilizarem informações ou períodos diferentes, parte da diferença observada poderá decorrer dessas escolhas, e não da arquitetura em si.

## 2.8 Incerteza preditiva e Monte Carlo Dropout

Gal e Ghahramani (2016) apresentam uma interpretação do dropout como aproximação de inferência bayesiana sob determinadas condições. Mantê-lo ativo em sucessivas avaliações da rede permite obter saídas diferentes para uma mesma entrada e examinar a sensibilidade da previsão às configurações de parâmetros implicitamente representadas. A técnica oferece uma forma computacionalmente acessível de investigar incerteza do modelo. Sua aplicação, entretanto, precisa respeitar a arquitetura e o esquema de dropout; diferentes formas de aplicar máscaras às conexões não são automaticamente equivalentes.

Kendall e Gal (2017) distinguem incerteza epistêmica, associada ao conhecimento limitado sobre o modelo, e incerteza aleatória, ligada à variabilidade inerente às observações. Embora seu estudo seja desenvolvido em visão computacional, a distinção conceitual é útil para a previsão financeira. Duas redes podem concordar sobre o retorno médio esperado e, ainda assim, o retorno realizado apresentar grande dispersão. A concordância entre previsões de média não implica ausência de risco no mercado.

Essa diferença limita o uso direto do Monte Carlo Dropout para calcular VaR e ES. Se cada avaliação da rede gera uma estimativa da média condicional, a distribuição dessas estimativas descreve principalmente variações na previsão do modelo. Ela não representa necessariamente a distribuição de todos os retornos que podem ocorrer no período seguinte. Para transformar a previsão em uma distribuição de perdas, é necessário representar também a variabilidade das observações e examinar se essa representação continua adequada quando as condições de mercado mudam (Gal; Ghahramani, 2016; Kendall; Gal, 2017).

Um caminho é combinar uma previsão de localização com uma escala condicional e uma distribuição de resíduos. Em termos esquemáticos, o retorno seguinte pode ser representado pela soma da localização prevista e de um erro padronizado multiplicado pela escala. A ideia se aproxima da adaptação da simulação histórica às condições correntes de volatilidade discutida por Hull e White (1998). A interpretação depende de verificar se os resíduos padronizados são suficientemente estáveis; utilizar uma escala constante, por exemplo, não introduz adaptação diária da dispersão apenas por haver uma rede neural no componente de média.

Quando os resíduos são utilizados para calibrar a distribuição após a escolha da rede, o período de calibração precisa ser separado do teste. Ajustar a cauda com os próprios erros que depois serão apresentados como desempenho fora da amostra compromete a avaliação. A calibração também consome observações que deixam de estar disponíveis para treinar ou selecionar o modelo. Em séries com episódios raros de estresse, o compromisso entre quantidade de dados, atualidade e precisão dos extremos precisa ser discutido expressamente.

O número de avaliações de dropout deve ser entendido como uma escolha de precisão computacional. Mil avaliações podem ajudar a estabilizar médias ou quantis de uma aproximação, mas não justificam a qualidade estatística dessa aproximação por si mesmas. Há ainda risco de contar a mesma incerteza duas vezes ao combinar a dispersão das saídas com uma distribuição de erros que já incorpora seus efeitos. A passagem da rede para VaR e ES precisa ser validada como parte do método, com hipóteses claras sobre o significado de cada componente.

## 2.9 Validação temporal backtesting e comparação das previsões

### 2.9.1 Avaliação fora da amostra e informação disponível

Hyndman e Athanasopoulos (2021) descrevem a avaliação por origens móveis, na qual cada previsão é construída a partir de observações anteriores ao instante previsto. Essa organização aproxima a avaliação da situação enfrentada por quem precisa decidir sem conhecer o futuro. Janelas móveis e janelas expansivas representam compromissos diferentes: as primeiras priorizam informação recente e descartam parte do histórico; as segundas acumulam observações, mas podem preservar períodos menos representativos das condições atuais.

A separação temporal deve alcançar todo o procedimento. Seleção de variáveis, padronização, escolha de hiperparâmetros, ajuste do HMM e estimação de parâmetros de distribuição precisam respeitar as informações permitidas em cada janela. Proibir o uso de dados futuros apenas na etapa final de treinamento da LSTM não é suficiente. Também importa verificar se a última observação de um bloco usa um alvo que já pertence ao bloco seguinte, especialmente quando se prevê o retorno do próximo pregão.

A disponibilidade econômica da informação não se resume ao índice de datas do arquivo. Um indicador divulgado depois do fechamento ou uma variável macroeconômica revisada posteriormente pode não ter sido conhecido na forma empregada pelo modelo. Quando os mercados possuem calendários e horários diferentes, uma regra de alinhamento precisa explicitar a origem da informação. Esses cuidados ajudam a distinguir habilidade preditiva de vantagens artificiais produzidas pela construção da base.

### 2.9.2 Cobertura independência e severidade das perdas

Kupiec (1995) propõe procedimentos para examinar a precisão de modelos de risco e destaca a necessidade de amostras suficientemente grandes para avaliar eventos de baixa probabilidade. No teste de cobertura incondicional, compara-se a proporção observada de violações com a probabilidade nominal da cauda. Se o VaR é calculado com confiança de 99%, a referência é uma proporção de violações de 1%. O teste avalia compatibilidade estatística, e não a capacidade de antecipar cada episódio extremo.

Christoffersen (1998) amplia a análise ao considerar a estrutura temporal das violações. Um modelo pode apresentar a frequência total esperada e concentrar suas falhas em uma sequência curta de dias turbulentos. Nessa situação, a cobertura incondicional pode ocultar uma deficiência relevante para a gestão do risco. A análise de independência e de cobertura condicional procura identificar esse problema, embora também dependa da quantidade de observações e de transições efetivamente disponíveis.

Uma avaliação por regime deve preservar essa dimensão temporal. Selecionar apenas dias de estresse e tratá-los como uma nova sequência contínua pode criar adjacências que não existiram no calendário original. Além disso, classificar o regime com probabilidades suavizadas ou com o retorno que se deseja prever introduz informação incompatível com uma decisão prospectiva. O estado utilizado para comparar previsões precisa corresponder à informação conhecida na origem de cada uma delas.

A avaliação do ES exige examinar a severidade das perdas. Acerbi e Székely (2014) apresentam procedimentos de backtesting específicos e distinguem o problema de testar uma medida do problema de comparar modelos por uma função de pontuação. A compatibilidade da frequência de violações do VaR não demonstra, sozinha, que o ES descreve adequadamente a cauda. A escolha de um teste de ES deve considerar os dados requeridos, a definição da estatística e a construção de sua referência sob a hipótese nula; calcular apenas uma estatística não fornece automaticamente um p-valor válido.

### 2.9.3 Comparação preditiva e interpretação das hipóteses

Diebold e Mariano (1995) desenvolvem um teste de igualdade de precisão preditiva a partir da diferença entre perdas de previsão de modelos concorrentes. A aplicação exige previsões comparáveis, realizadas para as mesmas datas e avaliadas pela mesma função de perda. A dependência temporal dessa diferença precisa ser considerada na estimação da incerteza. A escolha do critério é substantiva: comparar o erro quadrático de retornos previstos responde a uma pergunta diferente de comparar a qualidade de quantis de perda.

Fissler e Ziegel (2016) demonstram, sob condições apropriadas, a elicitabilidade conjunta de VaR e ES, fornecendo fundamento para funções de pontuação consistentes para o par de medidas. A contribuição permite comparar previsões de cauda sem reduzir a análise à contagem de violações. Engle e Manganelli (2004), ao orientarem a estimação para quantis, também evidenciam a importância de alinhar o critério de avaliação ao objeto previsto. Um limiar de VaR excessivamente conservador pode produzir poucas violações e, mesmo assim, ser pouco informativo para decisões de exposição.

Esses fundamentos orientam a interpretação das hipóteses do TCC. A cobertura nominal, central em H1, precisa ser examinada juntamente com a precisão das previsões; em H2, a análise do estresse requer atenção à concentração temporal e ao número reduzido de eventos; em H3, não rejeitar igualdade de desempenho não demonstra equivalência ou ausência de deterioração economicamente relevante. Uma alegação de não inferioridade exige definir previamente o que constitui uma diferença aceitável e utilizar um procedimento compatível com essa pergunta.

A leitura dos resultados também deve considerar o número de janelas, modelos e regimes examinados. Escolher a especificação depois de observar quais comparações foram favoráveis produz uma avaliação otimista. É mais informativo apresentar diferenças de desempenho, medidas de incerteza, comportamento em diferentes períodos e limites de poder estatístico do que declarar um vencedor com base no maior p-valor. Um resultado desfavorável à arquitetura híbrida continua sendo cientificamente relevante se decorrer de uma comparação coerente e reproduzível.

## 2.10 Interpretabilidade e limites das explicações

Lundberg e Lee (2017) apresentam o SHAP como uma estrutura de atribuição de contribuições das variáveis a previsões específicas, reunindo métodos de explicação em uma formulação aditiva. A abordagem pode auxiliar a investigação de quais informações estão associadas às saídas de um modelo e de como essa associação varia entre condições de mercado. Para uma rede recorrente, a definição do objeto explicado deve esclarecer se as contribuições correspondem a variáveis, defasagens, grupos de períodos ou a alguma agregação desses elementos.

Os valores atribuídos dependem da distribuição de referência e da forma como se representa a ausência de uma variável. Janzing, Minorics e Blöbaum (2020) discutem justamente as diferenças entre condicionamento observacional e interpretação de intervenções nesse problema. Em dados financeiros, variáveis de volatilidade e suas defasagens costumam compartilhar informação, de modo que a contribuição atribuída a uma delas pode mudar conforme o tratamento das demais. Explicações produzidas por SHAP devem ser lidas como descrições do comportamento do modelo sob uma referência definida, sem atribuição automática de efeitos causais no mercado.

A interpretabilidade pode complementar a análise das previsões, mas não substitui calibração e backtesting. Uma explicação economicamente plausível não torna correto um quantil mal estimado; uma variável com grande atribuição não precisa ser uma causa da perda observada. No escopo deste trabalho, o exame das explicações é mais útil quando esclarece a coerência das relações aprendidas e suas limitações, preservando a avaliação do risco como critério principal da contribuição empírica.

## 2.11 Síntese crítica e posicionamento da pesquisa

A literatura examinada oferece fundamentos complementares. A família ARCH-GARCH representa a atualização da volatilidade; os modelos de regime introduzem estados latentes e transições; as redes LSTM permitem aprender relações temporais flexíveis; os métodos de risco e de avaliação delimitam o que deve ser previsto e como examinar sua qualidade. A combinação desses elementos é conceitualmente justificável, mas acrescenta pontos de decisão e fontes de incerteza. O ganho de representação precisa compensar os custos de estimação, a redução da amostra efetiva e a maior dificuldade de interpretar cada componente.

Os estudos de Fischer e Krauss (2018), De Mel (2026) e Arimond et al. (2026) não respondem à mesma pergunta empírica. O primeiro trata de previsão direcional; o segundo combina regimes, previsão de volatilidade e uma hipótese para a cauda em uma ação; o terceiro relaciona redes, estados e gestão de carteiras de índices. Lidos em conjunto, eles sustentam o interesse pela pesquisa sem autorizar a conclusão de que uma arquitetura híbrida será superior em qualquer aplicação. A contribuição deste TCC deve ser localizada na avaliação de uma configuração definida para uma exposição ao agronegócio brasileiro, com comparações e restrições de informação claramente estabelecidas.

Esse posicionamento também aproxima a fundamentação do problema econômico. Ações, futuros e câmbio não podem ser reunidos apenas por conveniência de disponibilidade. A construção das séries precisa preservar o significado dos preços, das rolagens e da conversão monetária. Na mesma direção, a passagem de uma previsão de retorno para VaR e ES exige uma representação defensável da distribuição de perdas. Esses aspectos são parte do problema de pesquisa, pois um ganho aparente de precisão pode decorrer tanto de uma escolha de modelagem quanto de inconsistências na base ou na avaliação.

O Quadro 1 reúne os principais vínculos entre a revisão e as decisões que precisam ser justificadas no estudo. A síntese organiza a argumentação; ela não substitui a apresentação dos procedimentos e resultados no capítulo metodológico e na análise empírica.

| Questão de pesquisa | Fundamentação principal | Implicação para o estudo |

| O que a carteira representa | Gorton et al. (2013); Ubilava (2012); CME Group (s.d.) | Distinguir exposição financeira, referência contínua e preço físico. |

| Como representar volatilidade e estados | Bollerslev (1986); Hamilton (1989); Haas et al. (2004) | Comparar dinâmicas de variância e regimes sem presumir superioridade. |

| O que a rede efetivamente prevê | Fischer e Krauss (2018); De Mel (2026) | Separar retorno, volatilidade e quantil na escolha do alvo. |

| Como passar da previsão para o risco | Acerbi e Tasche (2002); Gal e Ghahramani (2016); Hull e White (1998) | Representar a variabilidade dos retornos e justificar a cauda. |

| Como avaliar previsões concorrentes | Kupiec (1995); Christoffersen (1998); Fissler e Ziegel (2016) | Examinar cobertura, dependência e qualidade das medidas de cauda. |

| Como interpretar ganhos e limites | Diebold e Mariano (1995); Janzing et al. (2020) | Distinguir precisão preditiva, não inferioridade e explicação causal. |

A questão que permanece aberta é se a informação de regime acrescenta capacidade de previsão de risco em relação a referências que utilizam informações comparáveis. Uma resposta convincente depende de dados consistentes, regras temporais verificáveis e avaliação conjunta de frequência, dependência e severidade das perdas. A contribuição pode estar na identificação de ganhos em condições específicas, na demonstração de limites da combinação ou na constatação de que uma solução mais simples oferece desempenho semelhante. Em qualquer desses casos, o valor acadêmico decorre da coerência entre a pergunta, o método e as evidências produzidas.

## Referências

ACERBI, Carlo; TASCHE, Dirk. On the coherence of expected shortfall. Journal of Banking & Finance, v. 26, n. 7, p. 1487–1503, 2002. DOI: https://doi.org/10.1016/S0378-4266(02)00283-2. Disponível em: https://arxiv.org/abs/cond-mat/0104295. Acesso em: 19 out. 2026.

ACERBI, Carlo; SZÉKELY, Balázs. Backtesting expected shortfall. New York: MSCI, 2014. Disponível em: https://www.msci.com/documents/10199/22aa9922-f874-4060-b77a-0f0e267a489b. Acesso em: 19 out. 2026.

ARDIA, David; BLUTEAU, Keven; BOUDT, Kris; CATANIA, Leopoldo; TROTTIER, Denis-Alexandre. Markov-switching GARCH models in R: the MSGARCH package. Journal of Statistical Software, v. 91, n. 4, p. 1–38, 2019. DOI: https://doi.org/10.18637/jss.v091.i04. Disponível em: https://www.jstatsoft.org/article/view/v091i04. Acesso em: 19 out. 2026.

ARIMOND, Alexander et al. Neural networks and value at risk in asset management. Review of Quantitative Finance and Accounting, v. 67, p. 277–316, 2026. DOI: https://doi.org/10.1007/s11156-025-01460-y. Disponível em: https://link.springer.com/article/10.1007/s11156-025-01460-y. Acesso em: 19 out. 2026.

BOLLERSLEV, Tim. Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, v. 31, n. 3, p. 307–327, 1986. DOI: https://doi.org/10.1016/0304-4076(86)90063-1. Disponível em: https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf. Acesso em: 19 out. 2026.

CAMPANI, Carlos Heitor; DURÃES, Assis Gustavo da Silva. Forecasting USD-BRL currency rate volatility using realized and implied volatilities data. Estudos Econômicos, v. 48, n. 4, p. 687–719, 2018. DOI: https://doi.org/10.1590/0101-41614845cca. Disponível em: https://revistas.usp.br/ee/article/view/131900. Acesso em: 19 out. 2026.

CBOE GLOBAL MARKETS. Cboe VIX FAQ. Chicago: Cboe, [s.d.]. Disponível em: https://www.cboe.com/tradable_products/vix/faqs. Acesso em: 19 out. 2026.

CENTRO DE ESTUDOS AVANÇADOS EM ECONOMIA APLICADA (CEPEA). Metodologia da Soja CEPEA/ESALQ — Paranaguá. Piracicaba: Esalq/USP, [s.d.]. Disponível em: https://cepea.org.br/br/metodologia/metodologia-da-soja-esalq-bm-fbovespa-paranagua.aspx. Acesso em: 19 out. 2026.

CHRISTOFFERSEN, Peter F. Evaluating interval forecasts. International Economic Review, v. 39, n. 4, p. 841–862, 1998. DOI: https://doi.org/10.2307/2527341. Disponível em: https://christoffersen.com/academic-papers/. Acesso em: 19 out. 2026.

CME GROUP. Understanding futures expiration and contract roll. Chicago: CME Group, [s.d.]. Disponível em: https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-futures-expiration-contract-roll. Acesso em: 19 out. 2026.

CONT, Rama. Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance, v. 1, n. 2, p. 223–236, 2001. DOI: https://doi.org/10.1080/713665670. Disponível em: https://doi.org/10.1080/713665670. Acesso em: 19 out. 2026.

CORWIN, Shane A.; SCHULTZ, Paul. A simple way to estimate bid-ask spreads from daily high and low prices. The Journal of Finance, v. 67, n. 2, p. 719–760, 2012. DOI: https://doi.org/10.1111/j.1540-6261.2012.01729.x. Disponível em: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01729.x. Acesso em: 19 out. 2026.

DE MEL, W. A. R. Value at risk of stock market returns using a hidden Markov–LSTM hybrid model. Sri Lankan Journal of Applied Statistics, v. 27, n. 1, p. 118–131, 2026. DOI: https://doi.org/10.4038/sljas.v27i1.8231. Disponível em: https://reference-global.com/article/10.4038/sljas.v27i1.8231. Acesso em: 19 out. 2026.

DIEBOLD, Francis X.; MARIANO, Roberto S. Comparing predictive accuracy. Journal of Business & Economic Statistics, v. 13, n. 3, p. 253–263, 1995. DOI: https://doi.org/10.1080/07350015.1995.10524599. Disponível em: https://ink.library.smu.edu.sg/soe_research/347/. Acesso em: 19 out. 2026.

ENGLE, Robert F. Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, v. 50, n. 4, p. 987–1007, 1982. DOI: https://doi.org/10.2307/1912773. Disponível em: https://www.jstor.org/stable/1912773. Acesso em: 19 out. 2026.

ENGLE, Robert F.; MANGANELLI, Simone. CAViaR: conditional autoregressive value at risk by regression quantiles. Journal of Business & Economic Statistics, v. 22, n. 4, p. 367–381, 2004. DOI: https://doi.org/10.1198/073500104000000370. Disponível em: https://www.simonemanganelli.org/Simone/Research_files/caviarPublished.pdf. Acesso em: 19 out. 2026.

FISCHER, Thomas; KRAUSS, Christopher. Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, v. 270, n. 2, p. 654–669, 2018. DOI: https://doi.org/10.1016/j.ejor.2017.11.054. Disponível em: https://cris.fau.de/publications/208534319/. Acesso em: 19 out. 2026.

FISSLER, Tobias; ZIEGEL, Johanna F. Higher order elicitability and Osband’s principle. The Annals of Statistics, v. 44, n. 4, p. 1680–1707, 2016. DOI: https://doi.org/10.1214/16-AOS1439. Disponível em: https://arxiv.org/abs/1503.08123. Acesso em: 19 out. 2026.

GAL, Yarin; GHAHRAMANI, Zoubin. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 33., 2016, New York. Proceedings [...]. PMLR, v. 48, p. 1050–1059, 2016. Disponível em: https://proceedings.mlr.press/v48/gal16.html. Acesso em: 19 out. 2026.

GARMAN, Mark B.; KLASS, Michael J. On the estimation of security price volatilities from historical data. The Journal of Business, v. 53, n. 1, p. 67–78, 1980. DOI: https://doi.org/10.1086/296072. Disponível em: https://doi.org/10.1086/296072. Acesso em: 19 out. 2026.

GERS, Felix A.; SCHMIDHUBER, Jürgen; CUMMINS, Fred. Learning to forget: continual prediction with LSTM. Neural Computation, v. 12, n. 10, p. 2451–2471, 2000. DOI: https://doi.org/10.1162/089976600300015015. Disponível em: https://doi.org/10.1162/089976600300015015. Acesso em: 19 out. 2026.

GORTON, Gary B.; HAYASHI, Fumio; ROUWENHORST, K. Geert. The fundamentals of commodity futures returns. Review of Finance, v. 17, n. 1, p. 35–105, 2013. DOI: https://doi.org/10.1093/rof/rfs019. Disponível em: https://faculty.som.yale.edu/garygorton/commodity-futures/. Acesso em: 19 out. 2026.

HAAS, Markus; MITTNIK, Stefan; PAOLELLA, Marc S. A new approach to Markov-switching GARCH models. Journal of Financial Econometrics, v. 2, n. 4, p. 493–530, 2004. DOI: https://doi.org/10.1093/jjfinec/nbh020. Disponível em: https://doi.org/10.1093/jjfinec/nbh020. Acesso em: 19 out. 2026.

HAMILTON, James D. A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, v. 57, n. 2, p. 357–384, 1989. DOI: https://doi.org/10.2307/1912559. Disponível em: https://www.jstor.org/stable/1912559. Acesso em: 19 out. 2026.

HOCHREITER, Sepp; SCHMIDHUBER, Jürgen. Long short-term memory. Neural Computation, v. 9, n. 8, p. 1735–1780, 1997. DOI: https://doi.org/10.1162/neco.1997.9.8.1735. Disponível em: https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory. Acesso em: 19 out. 2026.

HULL, John; WHITE, Alan. Incorporating volatility updating into the historical simulation method for value-at-risk. Journal of Risk, v. 1, n. 1, p. 5–19, 1998. DOI: https://doi.org/10.21314/JOR.1998.001. Disponível em: https://www-2.rotman.utoronto.ca/~hull/downloadablepublications/VaR_VolUpdating.pdf. Acesso em: 19 out. 2026.

HYNDMAN, Rob J.; ATHANASOPOULOS, George. Forecasting: principles and practice. 3. ed. Melbourne: OTexts, 2021. Disponível em: https://otexts.com/fpp3/tscv.html. Acesso em: 19 out. 2026.

JANZING, Dominik; MINORICS, Lenon; BLÖBAUM, Patrick. Feature relevance quantification in explainable AI: a causal problem. In: INTERNATIONAL CONFERENCE ON ARTIFICIAL INTELLIGENCE AND STATISTICS, 23., 2020. Proceedings [...]. PMLR, v. 108, p. 2907–2916, 2020. Disponível em: https://proceedings.mlr.press/v108/janzing20a.html. Acesso em: 19 out. 2026.

KENDALL, Alex; GAL, Yarin. What uncertainties do we need in Bayesian deep learning for computer vision? In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, v. 30, 2017. Proceedings [...]. Curran Associates, 2017. Disponível em: https://arxiv.org/abs/1703.04977. Acesso em: 19 out. 2026.

KUPIEC, Paul H. Techniques for verifying the accuracy of risk measurement models. The Journal of Derivatives, v. 3, n. 2, p. 73–84, 1995. DOI: https://doi.org/10.3905/jod.1995.407942. Disponível em: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7065. Acesso em: 19 out. 2026.

LUNDBERG, Scott M.; LEE, Su-In. A unified approach to interpreting model predictions. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, v. 30, 2017. Proceedings [...]. Curran Associates, 2017. Disponível em: https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html. Acesso em: 19 out. 2026.

RABINER, Lawrence R. A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, v. 77, n. 2, p. 257–286, 1989. DOI: https://doi.org/10.1109/5.18626. Disponível em: https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf. Acesso em: 19 out. 2026.

TAYLOR, James W. Forecasting value at risk and expected shortfall using a semiparametric approach based on the asymmetric Laplace distribution. Journal of Business & Economic Statistics, v. 37, n. 1, p. 121–133, 2019. DOI: https://doi.org/10.1080/07350015.2017.1281815. Disponível em: https://users.ox.ac.uk/~mast0315/. Acesso em: 19 out. 2026.

UBILAVA, David. El Niño, La Niña, and world coffee price dynamics. Agricultural Economics, v. 43, n. 1, p. 17–26, 2012. DOI: https://doi.org/10.1111/j.1574-0862.2011.00562.x. Disponível em: https://onlinelibrary.wiley.com/doi/10.1111/j.1574-0862.2011.00562.x. Acesso em: 19 out. 2026.
