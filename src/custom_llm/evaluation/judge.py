"""LLM-as-judge with an explicit rubric.

A judge is only as good as its agreement with a person, so the evaluation script exports a sample
for manual labelling and the runner reports the agreement (see runner.judge_agreement).
"""

from __future__ import annotations

from dataclasses import dataclass

from custom_llm.dataset_builder import EvalItem
from custom_llm.generation.llm_client import ChatClient
from custom_llm.utils import extract_json

JUDGE_PROMPT = """Você avalia respostas de um assistente de suporte do produto Fluxo.

Fatos de referência (fonte da verdade):
{reference}

Pergunta: {question}
Resposta de referência: {gold}
Resposta avaliada: {prediction}

Critérios:
1. correct: 1 se a resposta avaliada transmite a mesma informação central da resposta de referência, e 0 caso contrário. Diferenças de redação não contam. Valores, prazos e condições precisam coincidir.
2. supported: 1 se todas as afirmações factuais da resposta avaliada estão sustentadas pelos fatos de referência, e 0 se houver alguma afirmação não sustentada, mesmo que plausível. Admitir que não sabe não é uma afirmação factual.

Responda somente com JSON, sem texto adicional:
{{"correct": 0 ou 1, "supported": 0 ou 1, "reason": "frase curta"}}"""


@dataclass(frozen=True)
class Verdict:
    correct: bool
    supported: bool
    reason: str


def _flag(value: object) -> bool:
    return str(value).strip() in {"1", "true", "True"}


class Judge:
    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def evaluate(self, item: EvalItem, prediction: str) -> Verdict | None:
        prompt = JUDGE_PROMPT.format(
            reference=item.reference,
            question=item.question,
            gold=item.gold or "",
            prediction=prediction,
        )
        payload = extract_json(self.client.complete([{"role": "user", "content": prompt}]))
        if payload is None:
            return None
        return Verdict(
            correct=_flag(payload.get("correct", 0)),
            supported=_flag(payload.get("supported", 0)),
            reason=str(payload.get("reason", ""))[:300],
        )
