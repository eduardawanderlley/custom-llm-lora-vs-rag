# Arquitetura

O projeto é um pipeline com quatro etapas que se encadeiam por arquivos. Cada etapa lê artefatos da anterior e grava os seus, o que permite rodar uma parte do experimento sem repetir as outras.

## Fluxo de dados

A base de conhecimento (`data/knowledge_base/fluxo_v1.json`) é a única fonte de verdade. A partir dela, a etapa de dataset gera pares pergunta/resposta, aplica os filtros, divide os fatos entre treino e held-out e grava `train.jsonl` e `test_set.jsonl`. A etapa de treino consome `train.jsonl` e grava o adaptador LoRA. A etapa de índice lê a base de conhecimento com as atualizações aplicadas e grava o índice FAISS. A etapa de avaliação carrega o modelo base com o adaptador, o índice e o juiz, executa as quatro abordagens sobre o conjunto de teste e grava os registros e os resumos.

## Módulos

`config.py` carrega `configs/default.yaml` em modelos pydantic. Os caminhos derivados (dataset, adaptador, índice, resultados) são propriedades de `PathsConfig`, para que nenhum script monte caminhos à mão.

`knowledge.py` define `Fact`, `Document`, `FactUpdate` e `KnowledgeBase`. Os objetos são imutáveis. A aplicação de atualizações gera uma nova base, e nunca altera a anterior. Atualizações que referenciam fatos inexistentes falham na carga, e não são ignoradas em silêncio.

`dataset_builder.py` concentra as regras de divisão e de montagem do dataset: o split por fato, a reserva de uma pergunta por fato de treino para o regime `seen`, a exclusão dos fatos alterados dos regimes `seen` e `unseen`, e a verificação de vazamento entre treino e teste.

`generation/` contém o cliente para qualquer endpoint compatível com a API da OpenAI, o gerador de pares por fato e os filtros de qualidade. O cliente tenta novamente em caso de erro de rede ou limite de taxa.

`models.py` e `training.py` cuidam do carregamento do modelo base e do treino. O treino usa `SFTTrainer` com `prompt`/`completion`, e prepara o modelo para k-bit apenas quando ele foi carregado quantizado.

`retrieval.py` encapsula o encoder, a indexação e a busca. Os prefixos exigidos pela família E5 são aplicados internamente, e não pelos chamadores, para que ninguém esqueça deles.

`inference.py` define as quatro abordagens como configurações de um mesmo modelo: `base` não usa adaptador nem contexto, `rag` desliga o adaptador e usa contexto, `finetuned` usa o adaptador sem contexto, e `hybrid` usa o adaptador com contexto. Um único modelo na memória serve às quatro, o que torna a comparação mais justa e a execução mais barata.

`evaluation/` separa as métricas determinísticas (`metrics.py`), o juiz com rubrica (`judge.py`) e a execução com agregação (`runner.py`). O runner grava um registro por resposta, para que qualquer número possa ser rastreado até o texto que o produziu.

## Artefatos

| Arquivo | Conteúdo |
|---|---|
| `artifacts/dataset/train.jsonl` | Linhas de treino no formato prompt/completion |
| `artifacts/dataset/test_set.jsonl` | Itens de teste com regime, referência e termos-chave |
| `artifacts/dataset/split_manifest.json` | Fatos de treino e held-out, para reprodução |
| `artifacts/dataset/quality_report.json` | Contagem de pares removidos por cada filtro |
| `artifacts/dataset/review_sample.csv` | Amostra para revisão manual do dataset |
| `artifacts/adapter/` | Adaptador LoRA e `training_summary.json` |
| `artifacts/index/` | Índice FAISS e lista de passagens |
| `artifacts/results/records.csv` | Uma linha por resposta, com todas as métricas |
| `artifacts/results/summary_by_regime.csv` | Métricas agregadas por abordagem e regime |
| `artifacts/results/summary_out_of_scope.csv` | Taxa de recusa por abordagem |
| `artifacts/results/judge_review_sample.csv` | Amostra do juiz para rotulagem humana |

## Decisões

### ADR-001: split por fato

**Contexto.** Perguntas diferentes sobre o mesmo fato podem cair em treino e teste. Nesse caso, o teste mede memorização, e não capacidade de responder sobre conhecimento novo.

**Decisão.** O split é feito sobre os fatos. Fatos held-out não têm nenhuma linha no treino. Fatos de treino têm uma pergunta reservada, que é reformulada e entra no regime `seen`.

**Consequências.** O regime `unseen` testa generalização de verdade. O regime `seen` testa recuperação de conhecimento treinado. O custo é que o dataset de treino perde uma pergunta por fato, o que é pouco com 24 fatos, mas aceitável para a comparação.

### ADR-002: unidade de recuperação é o fato

**Contexto.** O corpus tem 24 fatos curtos e autocontidos. Chunking por caracteres partiria fatos e tornaria o retrieval hit ambíguo.

**Decisão.** Cada fato é uma passagem, com o título do documento como prefixo.

**Consequências.** A métrica de recuperação é exata: o fato correto está ou não está no top-k. Para uma base grande, com fatos que dependem uns dos outros, essa decisão precisará ser revista, e a interface do `Retriever` já separa a unidade da busca.

### ADR-003: juiz com rubrica e calibragem humana

**Contexto.** Métricas lexicais não capturam bem uma resposta correta com redação diferente, e um juiz sem calibragem pode dar uma aparência de rigor que não existe.

**Decisão.** O juiz recebe os fatos de referência, a resposta de referência e a resposta avaliada, e responde com dois critérios separados: correção e suporte. Uma amostra é rotulada à mão, e a concordância é reportada.

**Consequências.** O juiz é usado para o que ele faz bem, que é avaliar sentido. Os termos-chave e o exact match continuam como âncoras determinísticas, e divergências entre eles e o juiz são sinal para revisar a rubrica.

### ADR-004: a variante híbrida é treinada sem contexto

**Contexto.** Treinar o adaptador com contextos recuperados mudaria a comparação, porque o fine-tuning passaria a ter acesso ao RAG durante o treino.

**Decisão.** O adaptador é o mesmo usado na variante `finetuned`. A diferença entre `finetuned` e `hybrid` é apenas a presença de contexto na inferência.

**Consequências.** O híbrido pode estar subotimizado, porque o adaptador não foi treinado para ler o formato de contexto. Isso aparece como limitação no relatório.

### ADR-005: índice exato em vez de aproximado

**Contexto.** O corpus tem 24 passagens.

**Decisão.** `IndexFlatIP` com vetores normalizados, que calcula a similaridade cosseno de forma exata.

**Consequências.** Nenhuma perda de recall por aproximação. Com corpus maiores, um índice aproximado como HNSW passa a fazer sentido, e a troca fica isolada em `retrieval.py`.
