# Custom LLM: fine-tuning com LoRA ou RAG?

Comparação empírica entre fine-tuning com LoRA, RAG e a combinação dos dois, medida com o mesmo modelo base, o mesmo corpus e o mesmo conjunto de teste.

Toda equipe que quer um modelo que saiba sobre o próprio negócio chega, mais cedo ou mais tarde, à mesma dúvida: vale treinar um modelo com esses documentos, ou basta buscar o trecho certo no momento da pergunta? Em geral a decisão sai de intuição ou de um artigo genérico. Este projeto tenta trocar a intuição por medição.

O pipeline gera um dataset sintético a partir de uma base de conhecimento estruturada, ajusta um modelo de 1,5 bilhão de parâmetros com LoRA, constrói um índice vetorial sobre o mesmo corpus e avalia quatro variantes em condições iguais. A avaliação separa perguntas sobre fatos vistos no treino, fatos nunca vistos, fatos alterados depois do treino e perguntas fora do escopo do produto. Essa separação é o que torna a comparação útil: sem ela, o número final mistura memorização com capacidade de aprendizado.

O domínio é inventado de propósito. O Fluxo é um sistema de faturamento fictício, com planos, prazos e regras de nota fiscal que nenhum modelo pré-treinado conhece. Assim o resultado mede o que cada abordagem faz com o conhecimento fornecido, e não o que o modelo já sabia de antemão.

## O que o projeto responde

- Fine-tuning com LoRA memoriza fatos de domínio com precisão suficiente para dispensar a recuperação?
- O que acontece com cada abordagem quando um fato muda depois do treino?
- A combinação de fine-tuning e RAG compensa a complexidade extra que adiciona?
- Quanto custam, em latência e tokens, as abordagens por requisição?

A recomendação final e os números que a sustentam estão em [docs/results.md](docs/results.md). A metodologia, as hipóteses testadas e as ameaças à validade estão em [docs/methodology.md](docs/methodology.md). A organização técnica do código está em [docs/architecture.md](docs/architecture.md).

## Visão geral

```
base de conhecimento (24 fatos, 6 documentos)
        │
        ▼
geração de pares pergunta/resposta ──► filtros de qualidade ──► split por fato
                                                                  │            │
                                                                  ▼            ▼
                                                          train.jsonl     test_set.jsonl
                                                                  │            │
                                                           LoRA / QLoRA        │
                                                                  ▼            │
                                                          adaptador LoRA ──────┤
                                                                               ▼
base de conhecimento ──► embeddings + FAISS ─────────►  base · rag · finetuned · hybrid
                                                                               │
                                                     métricas + LLM-as-judge com rubrica
```

## Domínio e regimes de avaliação

A base de conhecimento tem seis documentos com quatro fatos cada: planos e cobrança, reembolsos, emissão de notas fiscais, segurança, integrações e suporte. Cada fato é uma afirmação atômica com os termos-chave que uma resposta correta precisa conter, por exemplo o prazo de 72 horas para transmissão em contingência ou o limite de 120 requisições por minuto na API.

O conjunto de teste é dividido em quatro regimes, porque cada um responde a uma pergunta diferente:

| Regime | Pergunta que responde | Como é construído |
|---|---|---|
| `seen` | O modelo recupera conhecimento do treino quando a pergunta muda de forma? | Uma pergunta por fato de treino fica reservada para teste |
| `unseen` | O modelo responde fatos que nunca viu? | Fatos held-out, sem nenhuma linha no treino |
| `update` | O modelo acompanha uma mudança feita depois do treino? | Sondas escritas à mão sobre quatro fatos alterados |
| `out_of_scope` | O modelo recusa o que não está na base? | Oito perguntas fora do produto ou sem resposta documentada |

## Reproduzindo o experimento

Requisitos: Python 3.10 ou superior. Para o treino em QLoRA, uma GPU NVIDIA com 8 GB de memória basta; um Colab com T4 também funciona. A geração do dataset e o julgamento das respostas usam uma API compatível com a da OpenAI, que pode ser a própria OpenAI, um vLLM ou um Ollama.

```bash
git clone https://github.com/<seu-usuario>/custom-llm-lora-vs-rag.git
cd custom-llm-lora-vs-rag

python -m venv .venv && source .venv/bin/activate
make install                 # pacote em modo editável, com dependências de desenvolvimento
pip install bitsandbytes     # opcional, apenas com GPU NVIDIA (QLoRA)

cp .env.example .env         # preencha LLM_API_KEY e, se usar endpoint próprio, LLM_BASE_URL
set -a; source .env; set +a

make test                    # testes unitários das métricas, filtros e splits
make all                     # dataset -> treino -> índice -> avaliação
```

Cada etapa também pode ser executada separadamente: `make dataset`, `make train`, `make index` e `make evaluate`. Os artefatos são gravados em `artifacts/` e incluem o relatório de filtros de qualidade, a amostra de revisão do dataset, o adaptador LoRA, o índice FAISS, os registros de cada resposta e os resumos por regime.

Todos os parâmetros ficam em `configs/default.yaml`. Para trocar o modelo base, o gerador ou o juiz, basta editar o YAML.

## Estrutura do repositório

```
.
├── configs/default.yaml          parâmetros de todo o pipeline
├── data/
│   ├── knowledge_base/           base de conhecimento v1.0 e atualizações (v1.1)
│   └── evaluation/               perguntas fora do escopo
├── docs/                         metodologia, arquitetura e resultados
├── scripts/                      um script por etapa do pipeline
├── src/custom_llm/
│   ├── config.py                 configuração tipada (pydantic)
│   ├── knowledge.py              modelo da base de conhecimento e das atualizações
│   ├── dataset_builder.py        splits por fato, linhas de treino, itens de teste
│   ├── generation/               geração de pares, filtros de qualidade, cliente LLM
│   ├── training.py               LoRA / QLoRA com TRL
│   ├── retrieval.py              embeddings e índice FAISS
│   ├── inference.py              as quatro abordagens comparadas
│   └── evaluation/               métricas determinísticas, juiz e runner
└── tests/                        testes de métricas, filtros e splits
```

## Decisões de projeto

Os splits são feitos por fato, e não por pergunta. Se fossem por pergunta, duas perguntas diferentes sobre o mesmo fato poderiam cair uma no treino e outra no teste, e o teste mediria memorização. Com o split por fato, um fato held-out não tem nenhuma linha no treino.

A unidade de recuperação é o fato, não um trecho de tamanho fixo. O corpus é pequeno e cada fato é autocontido. Um corte por caracteres partiria fatos ao meio e tornaria ambígua a pergunta sobre se o trecho recuperado continha a resposta.

O juiz não é tratado como verdade. A rubrica pede dois julgamentos separados, correção e suporte factual, e a amostra de respostas julgadas é exportada para revisão humana. A concordância entre o juiz e o revisor é calculada pelo próprio pipeline quando a amostra está rotulada.

O treino calcula a perda apenas sobre a resposta, usando o formato prompt/completion do TRL. Assim o adaptador gasta capacidade aprendendo o conteúdo, e não a reproduzir a pergunta.

A variante híbrida é treinada sem contexto e avaliada com contexto. Essa diferença de distribuição é uma escolha deliberada e aparece como limitação na metodologia. Uma alternativa seria treinar com contexto recuperado, o que mudaria a comparação.

## Limitações

O dataset é gerado por um LLM, e as perguntas refletem o estilo desse gerador, não necessariamente o de clientes reais. Os números absolutos devem ser lidos com esse viés em mente.

O conjunto de teste é pequeno, com algumas dezenas de itens em regimes de tamanho variável. Diferenças pequenas entre abordagens não devem ser interpretadas como vitórias sem uma análise de incerteza.

Cada experimento usa um modelo base e uma seed. Não há repetição com várias seeds nem intervalos de confiança por bootstrap nesta versão.

O juiz é outro LLM e pode ter vieses de estilo. A calibragem com revisão humana reduz esse risco, mas não o elimina.

## Contexto no portfólio

Este projeto complementa o [LegacyLens](#), que usa RAG em uma aplicação de onboarding sobre código legado. O LegacyLens mostra como RAG funciona em produção. Este projeto responde à pergunta que vem depois: em que casos RAG não basta, e quando vale a pena treinar um modelo.

## Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).

Autor: Seu Nome
