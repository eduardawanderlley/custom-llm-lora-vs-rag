"""Generates question/answer pairs grounded in one fact at a time."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from custom_llm.generation.llm_client import ChatClient
from custom_llm.knowledge import Fact
from custom_llm.utils import extract_json

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """Você está montando dados de treinamento para o assistente de suporte do Fluxo, um sistema de faturamento e notas fiscais.

Com base somente no trecho abaixo, escreva {n} pares de pergunta e resposta, como um cliente real perguntaria.

Regras:
- As perguntas devem ser naturais e variadas. Algumas são diretas; outras trazem um pouco do contexto de quem usa o sistema. Não copie frases do trecho.
- As respostas devem ter de uma a três frases, em português, e usar apenas informações do trecho.
- Não invente valores, prazos ou condições que não estejam no trecho.

Responda somente com JSON, neste formato:
{{"pairs": [{{"question": "...", "answer": "..."}}]}}

Trecho:
{context}"""


@dataclass(frozen=True)
class QAPair:
    fact_id: str
    question: str
    answer: str


def generate_for_fact(client: ChatClient, fact: Fact, n: int) -> list[QAPair]:
    prompt = GENERATION_PROMPT.format(n=n, context=fact.passage)
    payload = extract_json(client.complete([{"role": "user", "content": prompt}]))
    if payload is None:
        logger.warning("Unparseable generator output for fact %s", fact.id)
        return []

    pairs: list[QAPair] = []
    for item in payload.get("pairs", []):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            pairs.append(QAPair(fact.id, question, answer))
    return pairs


def generate_all(client: ChatClient, facts: list[Fact], n: int) -> list[QAPair]:
    pairs: list[QAPair] = []
    for index, fact in enumerate(facts, start=1):
        pairs.extend(generate_for_fact(client, fact, n))
        logger.info("[%d/%d] generated pairs for %s", index, len(facts), fact.id)
    return pairs
