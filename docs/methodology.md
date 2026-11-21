# Metodologia

Este documento descreve como o experimento foi desenhado, por que foi desenhado assim e o que os números podem e não podem dizer. Os resultados da rodada final estão em [results.md](results.md).

## Pergunta

Para um assistente que responde perguntas sobre documentos internos, qual abordagem entrega a melhor qualidade, a que custo, e como cada uma se comporta quando o conhecimento muda: ajustar um modelo com LoRA, recuperar trechos no momento da consulta, ou combinar as duas coisas?

## Hipóteses

As hipóteses foram registradas antes de gerar o dataset. Elas podem ser refutadas, e o desenho foi feito para que isso seja possível.

**H1.** O fine-tuning com LoRA aprende os fatos presentes no treino e generaliza mal para fatos que não estiveram lá. A diferença entre os regimes `seen` e `unseen` deve ser grande.

**H2.** O RAG responde bem sobre fatos novos, mas seu teto é a qualidade da recuperação. Quando o trecho certo não entra no top-k, a resposta tende a falhar, mesmo com um modelo capaz.

**H3.** A combinação de fine-tuning e RAG tem a menor taxa de alucinação em fatos conhecidos, ao custo de maior latência e de um pipeline mais difícil de manter.

**H4.** Depois de uma mudança de fato, o modelo ajustado tende a repetir o valor antigo. O RAG reflete a mudança assim que o índice é reconstruído, sem novo treino.

## Desenho experimental

O conjunto de teste é dividido em quatro regimes, e essa divisão é o centro do desenho. Uma comparação feita sobre uma única lista de perguntas esconde a diferença que mais importa: se a abordagem aprendeu o conhecimento ou apenas aprendeu a responder perguntas parecidas com as de treino.

| Regime | Fatos | Itens | O que mede |
|---|---|---|---|
| `seen` | Treino | Uma pergunta reservada por fato | Recuperação de conhecimento treinado, com a pergunta reformulada |
| `unseen` | Held-out | Todas as perguntas geradas | Conhecimento que o modelo nunca viu, mas que está no corpus |
| `update` | Treino, depois alterados | Sondas escritas à mão | Comportamento depois de uma mudança de fato |
| `out_of_scope` | Não se aplica | Oito perguntas | Recusa diante de perguntas sem resposta na base |

A divisão é feita por fato, e não por pergunta. Se fosse por pergunta, duas perguntas diferentes sobre o mesmo fato poderiam cair uma no treino e outra no teste, e o teste mediria memorização de um fato que o modelo já tinha visto. Com a divisão por fato, os fatos held-out não têm nenhuma linha no treino.

Algumas variáveis são controladas: o mesmo modelo base, o mesmo corpus, o mesmo juiz, o mesmo conjunto de teste, a mesma seed e decodificação gulosa. A única coisa que muda entre as variantes é o caminho até a resposta: contexto recuperado, adaptador LoRA, ou nenhum dos dois.

Os quatro regimes respondem a perguntas diferentes, e não devem ser colapsados em um único número. Uma abordagem pode ser excelente no `seen` e ruim no `unseen`, e isso é justamente o que a decisão de arquitetura precisa saber.

## Geração do dataset

O dataset é gerado por um LLM maior a partir de cada fato isolado. O gerador recebe o título do documento e o texto de um único fato, e devolve pares de pergunta e resposta. Trabalhar com fatos isolados mantém cada resposta ancorada em uma fonte. Quando o gerador recebe um documento inteiro, é mais fácil que ele misture fatos ou invente uma relação entre eles.

Os filtros de qualidade são aplicados nesta ordem:

1. Forma. A pergunta precisa ter pelo menos três palavras, e a resposta precisa ter entre 10 e 400 caracteres.
2. Consistência. A resposta precisa conter pelo menos um dos termos-chave do fato de origem. É um filtro grosseiro, mas pega a falha mais comum, que é o gerador inventar um prazo ou um valor.
3. Quase-duplicatas. Perguntas com similaridade de sequência igual ou acima de 0,9 são descartadas, mantendo a primeira ocorrência.

O relatório de cada filtro é salvo em `artifacts/dataset/quality_report.json`. Além disso, 30 pares são sorteados para revisão manual, com a resposta de referência ao lado. A revisão é pequena, mas serve para saber o que o modelo está aprendendo antes de treinar, e não só depois.

A limitação que mais pesa é o viés do gerador. As perguntas são escritas por um modelo, e o estilo delas pode não se parecer com o de um cliente real. Isso afeta os números absolutos e deve ser lido como limitação, e não como detalhe.

## Treinamento

O modelo base é o Qwen2.5-1.5B-Instruct. Ele é pequeno o bastante para treinar em GPU de consumo ou em um Colab, e é aberto, o que facilita a reprodução. A pergunta do projeto é metodológica, e não depende de o modelo ser grande.

O adaptador LoRA tem rank 16, alpha 32 e dropout de 0,05, aplicado a todas as projeções lineares de atenção e de MLP. Com GPU, o modelo base é carregado em 4 bits NF4 (QLoRA), o que reduz a memória sem mudar a arquitetura do adaptador.

O treino dura 3 épocas, com taxa de aprendizado de 2e-4, decaimento cosseno, warmup de 5% e batch efetivo de 16. A perda é calculada apenas sobre os tokens da resposta, porque o formato é prompt/completion. Isso impede que o adaptador gaste capacidade aprendendo a reproduzir a pergunta.

Tempo de treino, curva de perda e GPU utilizada são gravados em `training_summary.json`, junto ao adaptador.

## Recuperação

A unidade recuperada é o fato, com o título do documento como prefixo. Essa escolha está documentada em [architecture.md](architecture.md). Com um corpus de 24 fatos curtos e autocontidos, um corte por caracteres quebraria fatos ao meio e tornaria ambígua a métrica de recall: o trecho recuperado continha a resposta ou não?

O encoder é o `intfloat/multilingual-e5-small`, que exige os prefixos `query:` e `passage:`. Os vetores são normalizados e o índice é um FAISS `IndexFlatIP`, que calcula a similaridade cosseno de forma exata. Para um corpus muito maior, o tipo de índice mudaria; aqui a busca exata é simples e não é o gargalo.

## Avaliação

As métricas são separadas em determinísticas e de julgamento.

**Key-term recall** é a fração dos termos-chave do fato presentes na resposta, depois de normalização. É a métrica factual principal, porque é tolerante a diferenças de redação e exige os valores certos. Um prazo de 30 dias não conta como 7 dias, e "189" não conta como "89".

**Exact match e token F1** comparam a resposta com a referência gerada. São mantidas para comparabilidade com a literatura, mas penalizam respostas corretas com redação diferente. Não são a métrica de decisão.

**Judge correct** é o julgamento de correção feito por um LLM com rubrica explícita. A rubrica pede que valores, prazos e condições coincidam, mas que a redação seja livre.

**Judge supported** é o julgamento de suporte: toda afirmação factual da resposta é sustentada pelos fatos de referência? A taxa de alucinação é `1 - supported`. A pergunta é sobre groundedness, e não sobre se a resposta é útil.

**Refusal rate** mede, no regime `out_of_scope`, a fração de respostas que recusam ou dizem não saber. Qualquer resposta substantiva a uma pergunta sem resposta na base é contada como alucinação.

**Retrieval hit** mede, nas abordagens com RAG, se o fato correto está entre os top-k recuperados. Ele separa dois tipos de erro: o modelo não usou um trecho que estava lá, ou o trecho nem chegou ao modelo.

**Latência e tokens** são medidos por requisição. A latência inclui a recuperação, quando ela existe. O custo monetário não é reportado, porque depende do provedor; o custo de treino é reportado em tempo de GPU.

### Validação do juiz

O juiz é outro LLM, e a confiança nele precisa ser demonstrada. A rodada de avaliação exporta 40 respostas em `judge_review_sample.csv`, com o julgamento do juiz e a referência ao lado. Eu rotulo essas linhas à mão nas colunas `human_correct` e `human_supported`. Quando a amostra está rotulada, o pipeline calcula a concordância por critério. O limite que defini antes de rodar é de 85%. Abaixo disso, a rubrica é revista antes de qualquer conclusão ser tirada dos números.

## Ameaças à validade

O dataset é sintético. O gerador escreve as perguntas, e o estilo delas pode não representar clientes reais. Os números absolutos devem ser lidos com isso em mente, e as comparações entre abordagens são mais confiáveis do que os valores em si.

O conjunto de teste é pequeno. Com algumas dezenas de itens por regime, diferenças de poucos pontos percentuais podem ser ruído. Recomendo que a rodada final seja lida com intervalos de confiança por bootstrap, que ainda não estão implementados.

Há uma única seed e um único modelo base. O resultado mostra o comportamento de um ponto do espaço de desenho, e não uma lei geral.

A variante híbrida é treinada sem contexto e avaliada com contexto. Isso significa que o adaptador nunca viu o formato do prompt com o trecho recuperado. Uma alternativa seria aumentar o treino com contextos recuperados, o que mudaria a comparação e fica como próximo passo.

O regime `update` depende de quatro sondas escritas à mão. É um teste pequeno, mas direto: ele mostra se o modelo repete o valor antigo.

## Como ler os resultados

Os números devem ser interpretados em conjunto. Uma abordagem com alta taxa de acerto no `seen` e baixa no `unseen` confirma H1. Uma queda de qualidade do RAG concentrada em casos em que o retrieval hit é zero aponta para H2, e não para o modelo. A taxa de alucinação do híbrido precisa ser comparada à do fine-tuning puro para testar H3, levando em conta a latência adicional. O regime `update` testa H4 diretamente.

Uma hipótese refutada também é resultado. O relatório final deve dizer, para cada uma, se ela foi confirmada, refutada ou se o experimento não tem poder suficiente para decidir.
