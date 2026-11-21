# Resultados

**Status:** rodada final pendente. As tabelas abaixo devem ser preenchidas a partir de `artifacts/results/` depois de executar `make all`. Os valores ainda não foram medidos, e nenhum número deve ser citado antes da execução.

Leia os resultados junto com as ameaças à validade descritas em [methodology.md](methodology.md).

## Configuração da rodada

| Item | Valor |
|---|---|
| Modelo base | Qwen2.5-1.5B-Instruct |
| Gerador de dataset | |
| Juiz | |
| Encoder de recuperação | intfloat/multilingual-e5-small |
| Fatos na base (treino / held-out) | 24 (—  /  —) |
| Pares gerados / após filtros | — / — |
| Itens de teste por regime | seen: — · unseen: — · update: 4 · out_of_scope: 8 |
| Seed | 42 |
| GPU e tempo de treino | — |

## Qualidade por regime

| Abordagem | Regime | n | Key-term recall | Exact match | Token F1 | Judge correct | Taxa de alucinação |
|---|---|---|---|---|---|---|---|
| base | seen | | | | | | |
| base | unseen | | | | | | |
| base | update | | | | | | |
| rag | seen | | | | | | |
| rag | unseen | | | | | | |
| rag | update | | | | | | |
| finetuned | seen | | | | | | |
| finetuned | unseen | | | | | | |
| finetuned | update | | | | | | |
| hybrid | seen | | | | | | |
| hybrid | unseen | | | | | | |
| hybrid | update | | | | | | |

## Recusa fora do escopo

| Abordagem | Itens | Taxa de recusa |
|---|---|---|
| base | 8 | |
| rag | 8 | |
| finetuned | 8 | |
| hybrid | 8 | |

## Latência e tokens

| Abordagem | Latência p50 (s) | Latência p95 (s) | Tokens de resposta (média) | Recuperação com acerto |
|---|---|---|---|---|
| base | | | | — |
| rag | | | | |
| finetuned | | | | — |
| hybrid | | | | |

## Validação do juiz

| Critério | Itens rotulados | Concordância |
|---|---|---|
| Correção | | |
| Suporte | | |

O limite definido antes da rodada é de 85%. Abaixo disso, a rubrica deve ser revista antes de qualquer conclusão.

## Leitura dos resultados

Esta seção será escrita depois que os números estiverem disponíveis. Para cada hipótese, deve constar se ela foi confirmada, refutada ou se o experimento não tem poder suficiente para decidir.

**H1 (memorização não generaliza):** —

**H2 (RAG limitado pela recuperação):** —

**H3 (híbrido com menor alucinação, com custo):** —

**H4 (fatos alterados e o modelo ajustado):** —

## Recomendação

A recomendação final deve dizer quando usar cada abordagem, com base nas métricas e nos custos medidos. Ela também deve dizer o que mudaria a conclusão, por exemplo um corpus maior, um gerador diferente ou uma taxa de atualização alta.
