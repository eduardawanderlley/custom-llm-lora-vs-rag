# Custom LLM: LoRA Fine-Tuning vs. RAG (`custom-llm-lora-vs-rag`)

[English](#english) | [Português](#português)

---

## English

An empirical benchmark comparing LoRA fine-tuning, Retrieval-Augmented Generation (RAG), and a hybrid approach using the exact same base model, knowledge corpus, and evaluation suite.

Every team building an LLM application around internal business knowledge eventually hits the same fork in the road: should we fine-tune a model on our documents, or just retrieve the right passages at query time? Most of the time, that decision comes down to gut feeling or generic blog posts. I built this project to replace guesswork with controlled measurements.

The pipeline generates a synthetic dataset from a structured knowledge base, fine-tunes a 1.5B-parameter model with LoRA, builds a FAISS vector index over the same corpus, and evaluates four variants side by side (`base`, `rag`, `finetuned`, and `hybrid`). Instead of collapsing everything into a single accuracy score, the evaluation splits test cases across seen facts, unseen facts, facts updated after training, and out-of-scope queries. Without that split, benchmark numbers mix pure memorization with actual generalization.

The domain itself is intentionally fictional. **Fluxo** is a made-up billing and invoicing platform with custom plans, deadlines, and tax rules that no pretrained model has ever seen. That guarantees the evaluation measures how well each method handles newly injected knowledge rather than pretraining recall.

### Questions this project answers

* Can LoRA fine-tuning memorize domain facts accurately enough to skip retrieval altogether?
* How does each approach behave when business rules change after training?
* Does combining fine-tuning with RAG justify the extra engineering and operational complexity?
* What is the actual trade-off in latency and token consumption per request?

Final recommendations and benchmark numbers are documented in [docs/results.md](docs/results.md). Methodology, hypotheses, and threats to validity are in [docs/methodology.md](docs/methodology.md), and the codebase layout is detailed in [docs/architecture.md](docs/architecture.md).

### Pipeline overview

```text
knowledge base (24 atomic facts, 6 documents)
        │
        ▼
synthetic Q&A generation ──► quality filters ──► fact-level split
                                                    │          │
                                                    ▼          ▼
                                            train.jsonl   test_set.jsonl
                                                    │          │
                                             LoRA / QLoRA      │
                                                    ▼          │
                                             LoRA adapter ─────┤
                                                               ▼
knowledge base ──► embeddings + FAISS ───────► base · rag · finetuned · hybrid
                                                               │
                                                metrics + rubric LLM-as-judge
```

### Domain & evaluation regimes

The knowledge base contains six documents with four atomic facts each, covering billing plans, refunds, invoice issuance, security, integrations, and customer support. Every fact is self-contained and includes concrete values that a valid answer must preserve, such as a 72-hour contingency transmission window or a 120 requests-per-minute API rate limit.

The test suite is split into four regimes, each isolating a specific behavior:

| Regime | Question it answers | How it is built |
| :--- | :--- | :--- |
| `seen` | Can the model recall training knowledge when the question is phrased differently? | One question per training fact is held out for testing |
| `unseen` | How does the model handle facts it was never trained on? | Held-out facts with zero rows in the training set |
| `update` | Does the model keep up when a rule changes after training? | Hand-written probes targeting four modified facts in `v1.1` |
| `out_of_scope` | Does the model abstain when the answer is missing from the corpus? | Eight questions outside the product scope or undocumented |

### Running the experiment

Requires Python 3.10 or higher. For QLoRA training, an NVIDIA GPU with 8 GB VRAM (or a free Google Colab T4 instance) is enough. Dataset generation and rubric judging use any OpenAI-compatible API endpoint (OpenAI, vLLM, or Ollama).

```bash
git clone [https://github.com/eduardawanderlley/custom-llm-lora-vs-rag.git](https://github.com/eduardawanderlley/custom-llm-lora-vs-rag.git)
cd custom-llm-lora-vs-rag

python -m venv .venv && source .venv/bin/activate
make install                 # editable install with dev dependencies
pip install bitsandbytes     # optional, only needed on NVIDIA GPUs for QLoRA

cp .env.example .env         # set LLM_API_KEY and optional LLM_BASE_URL
set -a; source .env; set +a

make test                    # unit tests for metrics, quality filters, and splits
make all                     # dataset -> train -> index -> evaluate
```

You can also run each stage individually via `make dataset`, `make train`, `make index`, and `make evaluate`. All outputs are saved under `artifacts/`, including quality filter reports, human review samples, the trained LoRA adapter, the FAISS index, raw per-query logs, and regime summaries.

All hyperparameters and model choices live in `configs/default.yaml`. Switching the base model, generator, or judge only requires editing that file.

### Repository structure

```text
.
├── configs/default.yaml          pipeline hyperparameters and model settings
├── data/
│   ├── knowledge_base/           v1.0 corpus and v1.1 post-training updates
│   └── evaluation/               out-of-scope test queries
├── docs/                         methodology, architecture, and results
├── scripts/                      standalone entrypoint scripts per stage
├── src/custom_llm/
│   ├── config.py                 typed configuration via pydantic
│   ├── knowledge.py              knowledge base and update domain models
│   ├── dataset_builder.py        fact-level splits, training rows, and test items
│   ├── generation/               Q&A generation, quality filters, and LLM client
│   ├── training.py               LoRA / QLoRA training loop with TRL
│   ├── retrieval.py              embeddings and FAISS vector index
│   ├── inference.py              execution logic for all four variants
│   └── evaluation/               deterministic metrics, LLM judge, and runner
└── tests/                        unit tests for splits, filters, and metrics
```

### Key design decisions

* **Fact-level splits instead of question-level splits:** Splitting by question would allow two paraphrases of the same fact to land in both train and test sets, inflating scores through surface memorization. Splitting by fact guarantees that held-out facts never appear during training.
* **Atomic facts as retrieval units:** Because the corpus is small and each fact is self-contained, indexing individual facts avoids arbitrary character chunking that could split a rule in half and blur retrieval attribution.
* **Calibrated LLM judge:** The rubric scores factual correctness and context support separately, and exports a sample of judged responses for manual annotation. When labeled, the pipeline computes inter-annotator agreement between the judge and human reviewer.
* **Completion-only loss masking:** Training uses TRL's prompt/completion format so loss is computed exclusively on the target answer, spending adapter capacity on domain knowledge rather than memorizing question phrasing.
* **Distribution shift in the hybrid variant:** The hybrid variant is fine-tuned without retrieved context and evaluated with context injected at inference time. This mismatch is intentional and explicitly tracked as a methodological trade-off rather than hidden.

### Limitations

* **Synthetic query distribution:** Questions are generated by an LLM and reflect its phrasing patterns rather than noisy real-world user queries. Absolute scores should be read with that bias in mind.
* **Small evaluation sample:** The test set contains a few dozen items across unevenly sized regimes, so small margins between variants should not be over-interpreted without uncertainty estimates.
* **Single seed per run:** Each experiment runs with a single base model and random seed, without multi-seed variance or bootstrap confidence intervals in this version.
* **Judge style bias:** Using an LLM as a judge introduces stylistic preferences; human calibration mitigates this risk but does not remove it completely.

### Portfolio context

This project complements [LegacyLens](https://github.com/eduardawanderlley/legacylens), which applies RAG to legacy codebase onboarding. While LegacyLens focuses on building a production RAG pipeline, this repository answers the follow-up engineering question: when does RAG fall short, and when is fine-tuning actually worth the cost?

### License

MIT. See [LICENSE](LICENSE).

**Author:** Eduarda Wanderlley

---

## Português

Comparação empírica entre fine-tuning com LoRA, RAG e a combinação dos dois, medida com o mesmo modelo base, o mesmo corpus e o mesmo conjunto de teste.

Toda equipe que precisa adaptar um modelo de linguagem ao próprio negócio chega, cedo ou tarde, à mesma dúvida: vale a pena treinar o modelo com esses documentos ou basta buscar o trecho certo na hora da pergunta? Na maioria das vezes, essa decisão sai de intuição ou de algum artigo genérico. Construí este projeto para trocar o achismo por medição controlada.

O pipeline gera um dataset sintético a partir de uma base de conhecimento estruturada, ajusta um modelo de 1,5 bilhão de parâmetros com LoRA, monta um índice vetorial FAISS sobre o mesmo corpus e avalia quatro abordagens em condições iguais (`base`, `rag`, `finetuned` e `hybrid`). A avaliação separa perguntas sobre fatos vistos no treino, fatos inéditos, regras alteradas depois do treino e perguntas fora do escopo do produto. Sem essa divisão, a métrica final mistura memorização com capacidade real de generalização.

O domínio foi inventado de propósito. O **Fluxo** é um sistema fictício de faturamento com planos, prazos e regras de emissão de nota fiscal que nenhum modelo pré-treinado conhece. Assim, o experimento mede o que cada técnica faz com o conhecimento novo fornecido, e não o que o modelo já trazia do pré-treino.

### O que o projeto responde

* Fine-tuning com LoRA memoriza fatos do domínio com precisão suficiente para dispensar a busca em documentos?
* Como cada abordagem se comporta quando uma regra de negócio muda depois do treino?
* Combinar fine-tuning com RAG compensa a complexidade extra de engenharia e operação?
* Qual é o custo real, em latência e tokens, de cada abordagem por requisição?

A recomendação final e os números completos estão em [docs/results.md](docs/results.md). A metodologia, as hipóteses testadas e as ameaças à validade estão em [docs/methodology.md](docs/methodology.md), e a arquitetura do código está detalhada em [docs/architecture.md](docs/architecture.md).

### Visão geral do pipeline

```text
base de conhecimento (24 fatos atômicos, 6 documentos)
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

### Domínio e regimes de avaliação

A base de conhecimento tem seis documentos com quatro fatos cada, cobrindo planos e cobrança, reembolsos, emissão de notas fiscais, segurança, integrações e suporte. Cada fato é autocontido e traz regras objetivas que a resposta precisa acertar, como o prazo de 72 horas para transmissão em contingência ou o limite de 120 requisições por minuto na API.

O conjunto de teste é dividido em quatro regimes, cada um isolando um comportamento específico:

| Regime | Pergunta que responde | Como é construído |
| :--- | :--- | :--- |
| `seen` | O modelo recupera o conhecimento do treino quando a pergunta muda de forma? | Uma pergunta por fato de treino fica reservada para teste |
| `unseen` | O modelo responde sobre fatos que nunca viu no treino? | Fatos held-out, sem nenhuma linha no conjunto de treino |
| `update` | O modelo acompanha uma regra alterada depois do treino? | Sondas escritas à mão sobre quatro fatos modificados na `v1.1` |
| `out_of_scope` | O modelo recusa responder o que não está documentado? | Oito perguntas fora do escopo do produto ou sem resposta na base |

### Como reproduzir o experimento

Requisitos: Python 3.10 ou superior. Para o treino em QLoRA, basta uma GPU NVIDIA com 8 GB de VRAM (ou um Google Colab gratuito com T4). A geração do dataset e o julgamento das respostas usam qualquer API compatível com o padrão da OpenAI (OpenAI, vLLM ou Ollama).

```bash
git clone [https://github.com/eduardawanderlley/custom-llm-lora-vs-rag.git](https://github.com/eduardawanderlley/custom-llm-lora-vs-rag.git)
cd custom-llm-lora-vs-rag

python -m venv .venv && source .venv/bin/activate
make install                 # instala o pacote em modo editável com dependências de dev
pip install bitsandbytes     # opcional, apenas para treino QLoRA em GPU NVIDIA

cp .env.example .env         # preencha LLM_API_KEY e, se usar servidor próprio, LLM_BASE_URL
set -a; source .env; set +a

make test                    # testes unitários de métricas, filtros e splits
make all                     # dataset -> treino -> índice -> avaliação
```

Cada etapa também pode ser rodada separadamente com `make dataset`, `make train`, `make index` e `make evaluate`. Os artefatos são salvos em `artifacts/`, incluindo o relatório dos filtros de qualidade, a amostra para revisão humana, o adaptador LoRA treinado, o índice FAISS, os logs de cada resposta e os resumos por regime.

Todos os parâmetros ficam centralizados em `configs/default.yaml`. Para trocar o modelo base, o gerador ou o juiz, basta editar esse arquivo.

### Estrutura do repositório

```text
.
├── configs/default.yaml          parâmetros de todo o pipeline
├── data/
│   ├── knowledge_base/           base de conhecimento v1.0 e atualizações v1.1
│   └── evaluation/               perguntas fora do escopo
├── docs/                         metodologia, arquitetura e resultados
├── scripts/                      um script executável por etapa do pipeline
├── src/custom_llm/
│   ├── config.py                 configuração tipada com pydantic
│   ├── knowledge.py              modelos da base de conhecimento e atualizações
│   ├── dataset_builder.py        splits por fato, linhas de treino e itens de teste
│   ├── generation/               geração de pares, filtros de qualidade e cliente LLM
│   ├── training.py               treino LoRA / QLoRA com TRL
│   ├── retrieval.py              embeddings e índice vetorial FAISS
│   ├── inference.py              execução das quatro abordagens comparadas
│   └── evaluation/               métricas determinísticas, juiz LLM e runner
└── tests/                        testes unitários de métricas, filtros e splits
```

### Decisões de projeto

* **Split por fato, e não por pergunta:** Dividir por pergunta permitiria que duas variações sobre o mesmo fato caíssem uma no treino e outra no teste, medindo apenas memorização superficial. Com o split por fato, um item held-out nunca aparece no treino.
* **Fato atômico como unidade de busca:** Como o corpus é enxuto e cada fato é autocontido, indexar por fato evita cortes arbitrários por caracteres que partiriam regras ao meio e deixariam ambíguo se o trecho recuperado continha a resposta.
* **Juiz calibrado com revisão humana:** A rubrica separa correção factual de aderência ao contexto e exporta uma amostra para revisão manual. Quando rotulada, o próprio pipeline calcula a concordância entre o juiz e o revisor humano.
* **Cálculo de perda apenas na resposta:** O treino usa o formato prompt/completion do TRL para calcular a loss somente sobre a resposta esperada, concentrando a capacidade do adaptador no conteúdo do domínio em vez de decorar perguntas.
* **Diferença de distribuição na variante híbrida:** O modelo híbrido é treinado sem contexto recuperado e avaliado com contexto injetado na inferência. Essa escolha foi intencional e está documentada como limitação metodológica.

### Limitações

* **Dataset sintético:** As perguntas são geradas por um LLM e refletem o estilo de escrita do gerador, não necessariamente o ruído de usuários reais.
* **Amostra de teste reduzida:** O conjunto de teste tem algumas dezenas de exemplos distribuídos em regimes de tamanhos diferentes; diferenças pequenas entre variantes não devem ser lidas como conclusivas sem análise de incerteza.
* **Seed única por execução:** Cada experimento roda com um modelo base e uma seed, sem repetição em múltiplas seeds nem intervalos de confiança por bootstrap nesta versão.
* **Viés de estilo do juiz:** Usar outro LLM como avaliador traz preferências de estilo; a calibração humana reduz esse efeito, mas não o elimina por completo.

### Contexto no portfólio

Este projeto complementa o [LegacyLens](https://github.com/eduardawanderlley/legacylens), que aplica RAG em um fluxo de onboarding sobre código legado. Enquanto o LegacyLens mostra como estruturar um RAG em produção, este repositório responde à pergunta seguinte: em quais cenários o RAG não basta e quando realmente vale a pena treinar um modelo.

### Licença

Distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE).

**Autora:** Eduarda Wanderlley
